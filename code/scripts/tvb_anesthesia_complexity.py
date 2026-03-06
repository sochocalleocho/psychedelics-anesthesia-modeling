"""
tvb_anesthesia_complexity.py
-----------------------------
Runs TVB whole-brain simulations under four conditions and computes
Lempel-Ziv Complexity (LZc) and Perturbational Complexity Index (PCI),
across N_SEEDS random seeds in parallel to obtain means and error bars.

Model: Martin et al.'s Zerlaut_gK_gNa (second order, 8 state variables)
       which natively accepts per-region g_K_e/g_K_i arrays and uses separate
       E_K/E_Na leak reversal potentials. This captures both the mu_V (mean voltage)
       and mu_G (membrane conductance) effects of 5-HT2A receptor activation,
       unlike the single-g_L formulation which loses the mu_G channel.

Connectivity: Martin's DK68 connectivity (connectivity_68_QL20120814.zip)
5-HT2A map: Pre-ordered for Martin's connectivity region labelling.

Conditions  (Sacha et al. 2025 anesthetic parameterisation + Martin et al. DOI)
----------
  Awake:        b_e=5,  tau_e=5.0, tau_i=5.0  (wake baseline)
  Propofol:     b_e=30, tau_e=5.0, tau_i=7.0  (GABAergic anesthesia)
  Propofol+DOI: b_e=30, tau_e=5.0, tau_i=7.0  + per-region g_K reduction via 5-HT2A
  DOI only:     b_e=5,  tau_e=5.0, tau_i=5.0  + per-region g_K reduction via 5-HT2A

Key parameters:
  coupling a = 0.3 (Linear), weight normalisation: column-sum
  noise nsig = [0,0,0,0,0,0,0,1.0] (only on ou_drift state variable)
  P_e/P_i: Sacha/Fede (b_e=30, EL_e=-64) — best propofol collapse + DOI rescue
  E_K = -90 mV, E_Na = 50 mV, g_L = 10 nS
  Baseline E_L_e = -64.0 mV, E_L_i = -65.0 mV (Sacha's native operating point)

DOI/psychedelic parameterisation (Martin et al. 2025):
  Region-specific g_K reduction via 5-HT2A receptor density.
  g_Na stays fixed per cell type; only g_K varies across regions.
  DOI end-points: E_L_e_end=-61.2, E_L_i_end=-64.4 (Martin 5HT2A_heterogeneity_study, DK68).

LZc: computed on spontaneous activity (last 1s at 0.1ms resolution).
PCI: computed on EVOKED (TMS-stimulated) activity -- stimulus to region 5,
     +/-300ms analysis window, trial-averaged binarisation.

Usage:
  python tvb_anesthesia_complexity.py          # all 4 conditions, LZc + PCI
  python tvb_anesthesia_complexity.py --debug  # Awake vs DOI only, LZc only
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PIPELINE_HUB = os.path.join(PROJECT_ROOT, "paper_pipeline_hub")
MARTIN_REPO  = os.path.join(PROJECT_ROOT, "simulated_serotonergic_receptors_tvb")
SACHA_TOOLS  = os.path.join(PIPELINE_HUB, "TVB", "tvb_model_reference", "src")

# Martin's Zerlaut_gK_gNa model source directory
MARTIN_MODEL_SRC = os.path.join(
    MARTIN_REPO, "tvbsim", "TVB", "tvb_model_reference", "src")

# Martin's connectivity
MARTIN_CONN_DIR  = os.path.join(MARTIN_REPO, "data", "connectivity", "DK68")
MARTIN_CONN_FILE = "connectivity_68_QL20120814.zip"

# ---------------------------------------------------------------------------
# Biophysical constants (Martin et al. g_K / g_Na formulation)
# ---------------------------------------------------------------------------
E_NA   = 50.0    # mV -- sodium leak reversal potential
E_K    = -90.0   # mV -- potassium leak reversal potential
G_L    = 10.0    # nS -- baseline total leak conductance (g_K + g_Na = g_L)

# Baseline E_L values — Sacha's native operating point (E_L_e=-64.0).
# TF (P_e/P_i) = CONFIG1, fitted at b_e=0 pA, tau_i=5ms, E_L_e=-64 mV (universal baseline).
# Drug states change b_e/tau_i/EL_e in mean-field ODEs; polynomial P stays fixed.
# Validated: propofol LZc collapse (slow up-down oscillations, 42% down-state time).
E_L_E_BASE = -64.0   # mV -- excitatory neurons at rest (Sacha's native operating point)
E_L_I_BASE = -65.0   # mV -- inhibitory neurons at rest (same in both Sacha and Martin)


def _conversion(E_Na, E_K, E_L, g_L=None, g_Na=None):
    """
    Derive (g_K, g_Na) from leak reversal potentials.
    Matches Martin's receptors.py:conversion() exactly.
    Provide either g_L (to compute both) or g_Na (to keep g_Na fixed).
    """
    if g_L is not None:
        g_K = g_L * (E_L - E_Na) / (E_K - E_Na)
        g_Na_out = g_L - g_K
        return g_K, g_Na_out
    g_L_new = g_Na * (E_Na - E_K) / (E_L - E_K)
    g_K = g_L_new - g_Na
    return g_K, g_Na


# Baseline conductance decomposition
G_K_E_BASE, G_NA_E = _conversion(E_NA, E_K, E_L_E_BASE, g_L=G_L)
G_K_I_BASE, G_NA_I = _conversion(E_NA, E_K, E_L_I_BASE, g_L=G_L)

# Drug end-point E_L values (Martin et al. 2025, full 5-HT2A activation, DK68)
# Values from Martin's main 5HT2A_heterogeneity_study.ipynb, matched to E_L_e_start=-63.
E_L_E_PSI = -61.2   # mV -- excitatory at maximum psychedelic effect
E_L_I_PSI = -64.4   # mV -- inhibitory at maximum psychedelic effect

# DOI end-point conductances (g_Na stays FIXED, only g_K changes)
G_K_E_PSI, _ = _conversion(E_NA, E_K, E_L_E_PSI, g_Na=G_NA_E)
G_K_I_PSI, _ = _conversion(E_NA, E_K, E_L_I_PSI, g_Na=G_NA_I)


def _load_5ht2a_for_tvb():
    """Load 68-region DK 5-HT2A z-scored receptor density, clip negatives to 0."""
    density_file = os.path.join(
        MARTIN_REPO, "data", "receptors", "DK68", "5HT2a_reordered.txt"
    )
    densities = np.loadtxt(density_file)
    return np.clip(densities, 0, None)


def _compute_gK_map(densities):
    """Convert non-negative 5-HT2A density -> per-region (g_K_e, g_K_i) arrays.

    Matches Martin's get_g_K_values() logic: linear interpolation from
    [0, d_max] -> [g_K_base, g_K_psi].

    Returns:
        (g_K_e, g_K_i): arrays of shape (68,) with per-region conductances.
    """
    d_max = densities.max()
    n = len(densities)
    if d_max == 0:
        return np.full(n, G_K_E_BASE), np.full(n, G_K_I_BASE)

    # Higher receptor density -> lower g_K (more depolarised)
    g_K_e = np.interp(densities, [0, d_max], [G_K_E_BASE, G_K_E_PSI])
    g_K_i = np.interp(densities, [0, d_max], [G_K_I_BASE, G_K_I_PSI])

    return g_K_e, g_K_i


# Pre-compute per-region g_K maps
_DENSITIES_68 = _load_5ht2a_for_tvb()
_GK_E_DOI, _GK_I_DOI = _compute_gK_map(_DENSITIES_68)

# Compute effective E_L for display purposes
_gL_e_doi = _GK_E_DOI + G_NA_E
_EL_e_doi = (_GK_E_DOI * E_K + G_NA_E * E_NA) / _gL_e_doi
_gL_i_doi = _GK_I_DOI + G_NA_I
_EL_i_doi = (_GK_I_DOI * E_K + G_NA_I * E_NA) / _gL_i_doi

print(f"5-HT2A density map: {_DENSITIES_68.shape[0]} regions (DK68), "
      f"max={_DENSITIES_68.max():.3f}, mean={_DENSITIES_68.mean():.3f}")
print(f"  g_K_e range (DOI): {_GK_E_DOI.min():.4f} - {_GK_E_DOI.max():.4f} nS  "
      f"(baseline {G_K_E_BASE:.4f})")
print(f"  g_K_i range (DOI): {_GK_I_DOI.min():.4f} - {_GK_I_DOI.max():.4f} nS  "
      f"(baseline {G_K_I_BASE:.4f})")
print(f"  Effective E_L_e range: {_EL_e_doi.min():.2f} - {_EL_e_doi.max():.2f} mV  "
      f"(baseline {E_L_E_BASE})")
print(f"  Effective E_L_i range: {_EL_i_doi.min():.2f} - {_EL_i_doi.max():.2f} mV  "
      f"(baseline {E_L_I_BASE})")


# ---------------------------------------------------------------------------
# Conditions -- Sacha et al. 2025 anesthetic params + Martin DOI
# ---------------------------------------------------------------------------
# g_K_e / g_K_i: scalar for uniform, 68-element array for DOI heterogeneity
CONDITIONS = {
    "Awake": {
        "b_e": 5.0, "tau_e": 5.0, "tau_i": 5.0,
        "g_K_e": np.array([G_K_E_BASE]),
        "g_K_i": np.array([G_K_I_BASE]),
    },
    "Propofol": {
        "b_e": 30.0, "tau_e": 5.0, "tau_i": 7.0,
        "g_K_e": np.array([G_K_E_BASE]),
        "g_K_i": np.array([G_K_I_BASE]),
    },
    "Propofol+DOI": {
        "b_e": 30.0, "tau_e": 5.0, "tau_i": 7.0,
        "g_K_e": _GK_E_DOI,
        "g_K_i": _GK_I_DOI,
    },
    "DOI only": {
        "b_e": 5.0, "tau_e": 5.0, "tau_i": 5.0,
        "g_K_e": _GK_E_DOI,
        "g_K_i": _GK_I_DOI,
    },
}

# Debug subset: just Awake vs DOI for quick validation
CONDITIONS_DEBUG = {
    "Awake":    CONDITIONS["Awake"],
    "DOI only": CONDITIONS["DOI only"],
}

# Propofol subset: Propofol and Propofol+DOI (LZc only, no PCI)
CONDITIONS_PROPOFOL = {
    "Propofol":     CONDITIONS["Propofol"],
    "Propofol+DOI": CONDITIONS["Propofol+DOI"],
}

CONDITION_COLORS = {
    "Awake":        "#2ca02c",
    "Propofol":     "#d62728",
    "Propofol+DOI": "#1f77b4",
    "DOI only":     "#9467bd",
}

# ---------------------------------------------------------------------------
# Transfer Function Polynomial Coefficients
# ---------------------------------------------------------------------------
# Reference: Martin/Fayçal (b_e=0, EL_e=-63) — good DOI sensitivity, poor propofol
P_E_MARTIN = np.array([-0.04983106, 0.00506355, -0.02347012, 0.00229515, -0.00041053,
                        0.01054705, -0.03659253, 0.00743749,  0.00126506, -0.04072161])
P_I_MARTIN = np.array([-0.05149122, 0.00400369, -0.00835201, 0.00024142, -0.00050706,
                        0.00143454, -0.01468669, 0.00450271,  0.00284722, -0.01535780])

# Reference: Sacha/Fede (b_e=30, EL_e=-64) — good propofol collapse, no DOI signal
P_E_SACHA = np.array([-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
                       0.00341614, -0.01156433, 0.00194753,  0.00274079, -0.01066769])
P_I_SACHA = np.array([-0.05184978, 0.00615930, -0.01403522, 0.00166511, -0.00205590,
                       0.00318432, -0.03112775, 0.00656668,  0.00171829, -0.04516385])

# Custom fit: b_e=5, EL_e=-63 — fitted to capture BOTH DOI and propofol sensitivity.
# Fitted using make_fit_from_data() from Sacha's Tf_calc pipeline (10 loop iterations,
# 50x50 grid, 5s AdEx sim per point, range_exc/inh=(0,5), w_prec=False).
# RS mean error: 2.07 Hz, FS mean error: 7.56 Hz.
P_E_CUSTOM = np.array([-0.04942773, 0.00249781, -0.01155355, 0.00460832, 0.00083928,
                        0.00192923, -0.02032910, 0.00304681, -0.00241909, -0.01295879])
P_I_CUSTOM = np.array([-0.04971462, -0.00188472, 0.03682690, -0.00518017, 0.00126867,
                        0.02196745, 0.07574954, -0.00358921, -0.00286524, 0.06042949])

# Propofol-fit TF: b_e=30, tau_i=7, E_L=-64 — fitted at the ACTUAL propofol operating point.
# Key innovation: tau_i=7 (vs Sacha's tau_i=5) amplifies sigma_V in training data,
# producing sigma_V polynomial coefficients 4-12× larger than Sacha's (even > Martin's).
# This should give BOTH propofol collapse AND DOI sensitivity.
P_E_PROP = np.array([-0.05010789, 0.00281572, -0.01223011, 0.01172682, 0.00109498,
                      0.04324058, -0.03698233, 0.00574098, -0.00486640, -0.04846384])
P_I_PROP = np.array([-0.05541512, 0.01490904, -0.06014388, 0.04245892, -0.00520429,
                     -0.02425036, -0.12614890, 0.02465346, -0.00313130, -0.07453547])

# Combined 3-regime fit: awake (b_e=5, EL=-64) + propofol (b_e=30, tau_i=7, EL=-64)
#                      + DOI-adjacent (b_e=5, EL=-63).
# Generated with FIXED tf_simulation_fast.py (w decays during refractoriness).
# Balanced error: RS=3.31 Hz (3.20/3.52/3.20 awake/prop/doi-adj), FS=10.46 Hz.
# The third regime (EL=-63) provides moment-space coverage between awake (EL=-64)
# and the DOI endpoint (EL=-61.2), enabling interpolation rather than extrapolation.
# P[5] (sigma_V^2): 0.0312 — reflects genuine sigma_V variation across EL range.
P_E_COMBINED = np.array([-0.04834604, -0.00046560,  0.00262377, -0.00344897,  0.00145982,
                           0.03122459,  0.02661126,  0.00293044, -0.00002222, -0.00270516])
P_I_COMBINED = np.array([-0.04853483, -0.00156604,  0.00440974,  0.01627825,  0.00051037,
                           0.03907078,  0.03995453,  0.00864973, -0.02007251, -0.00020276])

# Fresh reproducible fit: b_e=0, tau_i=5, EL_e=-64 (same operating point as CONFIG1/P_E_SACHA).
# Generated with FIXED tf_simulation_fast.py (FS-RS_0 cell, refractory bug corrected).
# Same stochastic class as CONFIG1; different realization (CONFIG1 had no fixed seed).
# Max coef diff vs CONFIG1: RS=0.025, FS=0.021 — same order of magnitude.
# BROKEN: loop_n=10 drives P[5] to 0.028 (8× too large) → wrong TVB dynamics.
P_E_B0_EL64 = np.array([-0.05007537, 0.00376876, -0.01389398, 0.00691960, 0.00012585,
                          0.02833698, -0.02263122, 0.00400189, -0.00312780, -0.01547013])
P_I_B0_EL64 = np.array([-0.05258031, 0.00619224, -0.01370992, 0.00604868, -0.00114091,
                          0.00589041, -0.02770389, 0.00365163, -0.00054324, -0.02446988])

# REPRODUCED CONFIG1: fit from Sacha's stored data (ExpTF_50x50_b_e_0_RS.npy + trial_FS.npy)
# using loop_n=1 (Sacha's notebook hyperparameter — the crucial difference from our loop_n=10).
# P[5] (sigma_V^2): 0.003596 vs CONFIG1's 0.003416 — essentially identical (0.5% diff).
# Key: loop_n=1 leaves optimizer partially converged, keeping P[5] near zero.
#      loop_n=10 fully converges to high-P[5] polynomial that breaks DOI dynamics.
P_E_REPRO = np.array([-0.04815773, 0.00077078,  0.00914011, -0.01530795, 0.00102783,
                       0.00359615,  0.01254575, -0.00258750,  0.00574350, -0.02075684])
P_I_REPRO = np.array([-0.05376053,  0.01062306, -0.04301243,  0.01639271, -0.00250266,
                       0.00244531, -0.06732433,  0.01162860, -0.00623785, -0.04571834])

# ---------------------------------------------------------------------------
# ══════════════════════════  EDIT HERE  ════════════════════════════════════
# ---------------------------------------------------------------------------
# Active TF coefficients — swap to change which polynomial is used.
# CONFIG1 (P_E_SACHA): fit at b_e=0 pA, tau_i=5ms, EL_e=-64 mV — the UNIVERSAL baseline.
#   Drug states (b_e, tau_i, EL_e) enter only through moment equations (mu_V, sigma_V, tau_V).
#   Correctly produces propofol LZc collapse (slow up-down oscillations).
# P_E_REPRO: FAILED — P[5] matches CONFIG1 (0.0036) but P[2],P[3],P[6] differ (Nelder-Mead
#   converges to different local minimum on different platform/scipy version). Propofol LZc ≠ collapse.
# P_E_B0_EL64: loop_n=10 version — P[5] too large, broken dynamics. Do not use.
# P_E_COMBINED: WRONG operating point (b_e=5/30, tau_i=7). Do not use.
P_E = P_E_SACHA
P_I = P_I_SACHA

N_SEEDS  = 16         # ≥15 required for reliable DOI>Awake detection (Δ≈0.002)
N_TRIALS = 5          # PCI trial-averaging blocks per seed (Sacha uses 5)
SIM_LEN  = 5000.0     # ms total simulation length
DT       = 0.1        # ms integration step
MON_DT   = 0.1        # ms monitor sampling period
# ---------------------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

# LZc analysis
ANALYSIS_LAST_MS = 1000.0   # use last 1s for LZc

# PCI stimulus parameters (Sacha et al.)
STIM_REGION  = 5        # region index for TMS pulse
STIM_VAL     = 1e-3     # kHz stimulus amplitude
STIM_DUR     = 50.0     # ms pulse duration
CUT_TRANSIENT = 2000.0  # ms transient to discard
T_ANALYSIS   = 300.0    # ms pre/post stimulus window


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _run_one_sim(params, seed, conn, n_regions, with_stimulus=True):
    """Run a single TVB simulation and return (times, fr_kHz, stim_onset)."""
    import numpy as np
    from tvb.simulator import simulator, coupling, integrators, monitors, noise
    from tvb.simulator.lab import equations, patterns
    import Zerlaut_gK_gNa as custom_zerlaut

    g_K_e = params["g_K_e"]
    g_K_i = params["g_K_i"]

    model = custom_zerlaut.Zerlaut_adaptation_second_order(
        # Leak conductances (g_K varies per region for DOI, g_Na fixed)
        g_K_e  = g_K_e,
        g_Na_e = np.array([G_NA_E]),
        g_K_i  = g_K_i,
        g_Na_i = np.array([G_NA_I]),
        # Leak reversal potentials
        E_K_e  = np.array([E_K]),
        E_Na_e = np.array([E_NA]),
        E_K_i  = np.array([E_K]),
        E_Na_i = np.array([E_NA]),
        # Cell parameters
        C_m    = np.array([200.0]),
        b_e    = np.array([params["b_e"]]),
        a_e    = np.array([0.0]),
        b_i    = np.array([0.0]),
        a_i    = np.array([0.0]),
        tau_w_e= np.array([500.0]),
        tau_w_i= np.array([1.0]),
        # Synaptic parameters
        tau_e  = np.array([params["tau_e"]]),
        tau_i  = np.array([params["tau_i"]]),
        E_e    = np.array([0.0]),
        E_i    = np.array([-80.0]),
        Q_e    = np.array([1.5]),
        Q_i    = np.array([5.0]),
        # Network parameters
        N_tot       = np.array([10000]),
        p_connect_e = np.array([0.05]),
        p_connect_i = np.array([0.05]),
        g           = np.array([0.2]),
        T           = np.array([20.0]),
        K_ext_e = np.array([400]),
        K_ext_i = np.array([0]),
        # External inputs
        external_input_ex_ex = np.array([0.315e-3]),
        external_input_ex_in = np.array([0.000]),
        external_input_in_ex = np.array([0.315e-3]),
        external_input_in_in = np.array([0.000]),
        # Noise / OU
        tau_OU       = np.array([5.0]),
        weight_noise = np.array([1e-4]),
        # Transfer function coefficients
        P_e = P_E,
        P_i = P_I,
        # Long-range inhibitory factor (default 1.0, no effect)
        inh_factor = np.array([1.0]),
    )

    coupl = coupling.Linear(a=np.array([0.3]), b=np.array([0.0]))
    nsig = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    noise_inst = noise.Additive(nsig=nsig)
    integ = integrators.HeunStochastic(dt=DT, noise=noise_inst)

    stimulus = None
    stim_onset = CUT_TRANSIENT + 0.5 * (SIM_LEN - CUT_TRANSIENT)
    if with_stimulus:
        eqn_t = equations.PulseTrain()
        eqn_t.parameters["onset"] = np.array([stim_onset])
        eqn_t.parameters["tau"]   = np.array([STIM_DUR])
        eqn_t.parameters["T"]     = np.array([1e9])
        eqn_t.parameters["amp"]   = np.array([1.0])
        stim_weights = np.zeros((n_regions, 1))
        stim_weights[STIM_REGION, 0] = STIM_VAL
        stimulus = patterns.StimuliRegion(
            temporal=eqn_t, connectivity=conn, weight=stim_weights
        )
        model.stvar = np.array([0])

    mon = monitors.Raw()
    sim = simulator.Simulator(
        model=model, connectivity=conn, coupling=coupl,
        integrator=integ, monitors=(mon,),
        stimulus=stimulus,
    )
    sim.configure()

    ic = np.zeros((8, n_regions, 1))
    ic[5, :, 0] = 100.0
    sim.current_state[:] = ic
    sim.integrator.noise.random_stream.seed(seed)

    raw = sim.run(simulation_length=SIM_LEN)
    times = raw[0][0].flatten()
    fr = raw[0][1][:, 0, :, 0]
    return times, fr, stim_onset


def _worker(args):
    """Run one (condition, seed_block) and return (lzc, pci).

    Runs N_TRIALS simulations with different sub-seeds for PCI trial-averaging.
    LZc is computed from the first trial only (spontaneous activity).
    """
    cond_name, params, seed_block, debug_mode = args

    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    import sys, os

    for p in [PIPELINE_HUB, MARTIN_REPO, SACHA_TOOLS, MARTIN_MODEL_SRC,
              os.path.join(PIPELINE_HUB, "TVB")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from tvb.datatypes import connectivity
    from tvbsim.entropy.measures_functions import calculate_LempelZiv

    np.random.seed(seed_block)

    # -- Connectivity: Martin's DK68 --
    conn_path = os.path.join(MARTIN_CONN_DIR, MARTIN_CONN_FILE)
    conn = connectivity.Connectivity.from_file(conn_path)
    conn.configure()
    conn.weights = conn.weights / (np.sum(conn.weights, axis=0) + 1e-12)
    conn.speed = np.array([4.0])
    n_regions = conn.weights.shape[0]

    # In debug mode: only compute LZc (no PCI / stimulus)
    n_trials = 1 if debug_mode else N_TRIALS

    trial_windows = []
    lzc = None

    for trial_idx in range(n_trials):
        sub_seed = seed_block * N_TRIALS + trial_idx
        with_stimulus = not debug_mode
        times, fr, stim_onset = _run_one_sim(params, sub_seed, conn, n_regions,
                                              with_stimulus=with_stimulus)

        # LZc from first trial only (spontaneous, last 1s)
        if trial_idx == 0:
            fr_hz = fr * 1000.0
            mean_fr = fr_hz[-10000:, :].mean()
            print(f"    {cond_name} seed={seed_block}: mean FR = {mean_fr:.2f} Hz", flush=True)
            n_analysis = int(ANALYSIS_LAST_MS / DT)
            lzc = calculate_LempelZiv(fr[-n_analysis:, :])

        # PCI: extract +/-300ms window around stimulus for this trial
        if not debug_mode:
            dt_actual = times[1] - times[0]
            t_stim_bin = int(stim_onset / dt_actual)
            nbins_analysis = int(T_ANALYSIS / dt_actual)
            t0_bin = int(CUT_TRANSIENT / dt_actual)
            fr_post = fr[t0_bin:, :]
            t_stim_rel = t_stim_bin - t0_bin
            win_start = t_stim_rel - nbins_analysis
            win_end   = t_stim_rel + nbins_analysis

            if win_start < 0 or win_end > fr_post.shape[0]:
                print(f"    WARNING: PCI window OOB trial={trial_idx}", flush=True)
                continue

            sig_cut = fr_post[win_start:win_end, :].T  # (n_regions, 2*nbins_analysis)
            trial_windows.append(sig_cut)

    # -- PCI: binarise with trial-averaging (Sacha et al. method) --
    pci = 0.0
    if not debug_mode and len(trial_windows) >= 2:
        import nuu_tools_simulation_human as tools
        import pci_v2

        sig_3d = np.array(trial_windows)  # (n_trials, n_regions, 2*nbins_analysis)
        nbins_analysis = sig_3d.shape[2] // 2

        sig_binary = tools.binarise_signals(sig_3d, nbins_analysis,
                                            nshuffles=10, percentile=100)

        # Compute PCI for each trial, then average (matching Sacha)
        t_analysis_idx = int(T_ANALYSIS)  # 300, matching reference code
        pcis = []
        for t in range(sig_binary.shape[0]):
            binJ = sig_binary.astype(int)[t, :, t_analysis_idx:]
            binJs = pci_v2.sort_binJ(binJ)
            lz = pci_v2.lz_complexity_2D(binJs)
            norm = pci_v2.pci_norm_factor(binJs)
            pcis.append(lz / norm if norm > 0 else 0.0)
        pci = np.mean(pcis)

    if debug_mode:
        print(f"    done: {cond_name} seed={seed_block}  LZc={lzc:.5f}", flush=True)
    else:
        print(f"    done: {cond_name} seed={seed_block}  LZc={lzc:.5f}  PCI={pci:.4f}", flush=True)
    return lzc, pci


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, out_path):
    """Plot LZc + PCI bar charts for all 4 conditions."""
    cond_names = list(results.keys())
    colors     = [CONDITION_COLORS[c] for c in cond_names]
    x          = np.arange(len(cond_names))

    lzc_means = np.array([results[c]["lzc_mean"] for c in cond_names])
    lzc_sems  = np.array([results[c]["lzc_sem"]  for c in cond_names])
    lzc_all   = [results[c]["lzc_all"]            for c in cond_names]

    pci_means = np.array([results[c]["pci_mean"] for c in cond_names])
    pci_sems  = np.array([results[c]["pci_sem"]  for c in cond_names])
    pci_all   = [results[c]["pci_all"]            for c in cond_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Complexity Metrics Across Brain States  (n={N_SEEDS} seeds, mean +/- SEM)\n"
        "Martin Zerlaut_gK_gNa 2nd order - DK68 - coupling a=0.3 - column-sum weights\n"
        f"LZc: spontaneous (last 1s) - PCI: evoked (TMS stim, +/-300ms, {N_TRIALS} trials)",
        fontsize=10, fontweight="bold")

    def _panel(ax, means, sems, all_vals, ylabel, title, zoom=False):
        bars = ax.bar(x, means, color=colors, edgecolor="black",
                      linewidth=0.8, width=0.5, alpha=0.85, zorder=2)
        ax.errorbar(x, means, yerr=sems, fmt="none", color="black",
                    capsize=5, capthick=1.5, linewidth=1.5, zorder=3)
        rng = np.random.default_rng(0)
        for i, (vals, col) in enumerate(zip(all_vals, colors)):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(i + jitter, vals, color=col, edgecolors="black",
                       linewidths=0.4, s=18, alpha=0.7, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(cond_names, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if zoom:
            pad = max(max(sems) * 3, 0.005)
            lo  = (means - sems).min() - pad
            hi  = (means + sems).max() + pad
            ax.set_ylim(max(0.0, lo), hi)
        else:
            ax.set_ylim(0, max((means + sems).max() * 1.25, 0.01))

        for i, (m, s) in enumerate(zip(means, sems)):
            ax.text(i, m + s + (means.max() - means.min()) * 0.08 + 0.002,
                    f"{m:.4f}", ha="center", va="bottom", fontsize=8)

    _panel(axes[0], lzc_means, lzc_sems, lzc_all,
           "Normalised LZc",
           "Lempel-Ziv Complexity (spontaneous)",
           zoom=True)

    _panel(axes[1], pci_means, pci_sems, pci_all,
           "PCI",
           "Perturbational Complexity Index (evoked)")

    # Expected direction annotations
    axes[0].annotate("Expected: Awake > Propofol\nDOI >= Awake",
                     xy=(0.97, 0.97), xycoords="axes fraction",
                     ha="right", va="top", fontsize=7.5, color="gray", style="italic")
    axes[1].annotate("Expected: Awake PCI ~0.3-0.6\nPropofol PCI < Awake",
                     xy=(0.97, 0.97), xycoords="axes fraction",
                     ha="right", va="top", fontsize=7.5, color="gray", style="italic")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved -> {out_path}")


def plot_debug_awake_vs_doi(results, out_path):
    """Plot a single-panel LZc bar chart: Awake vs DOI only."""
    cond_names = list(results.keys())
    colors     = [CONDITION_COLORS[c] for c in cond_names]
    x          = np.arange(len(cond_names))

    lzc_means = np.array([results[c]["lzc_mean"] for c in cond_names])
    lzc_sems  = np.array([results[c]["lzc_sem"]  for c in cond_names])
    lzc_all   = [results[c]["lzc_all"]            for c in cond_names]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(
        f"Debug: Awake vs DOI LZc  (n={N_SEEDS} seeds, mean +/- SEM)\n"
        f"Martin Zerlaut_gK_gNa - DK68 - E_L_e baseline={E_L_E_BASE} mV\n"
        f"Expected diff ~0.003-0.004 (Martin et al. 2025)",
        fontsize=10, fontweight="bold")

    bars = ax.bar(x, lzc_means, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.4, alpha=0.85, zorder=2)
    ax.errorbar(x, lzc_means, yerr=lzc_sems, fmt="none", color="black",
                capsize=6, capthick=1.5, linewidth=1.5, zorder=3)

    rng = np.random.default_rng(0)
    for i, (vals, col) in enumerate(zip(lzc_all, colors)):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(i + jitter, vals, color=col, edgecolors="black",
                   linewidths=0.5, s=25, alpha=0.7, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, fontsize=12)
    ax.set_ylabel("Normalised LZc", fontsize=11)
    ax.set_title("Lempel-Ziv Complexity (spontaneous, last 1s)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Zoom to show difference clearly
    pad = max(max(lzc_sems) * 3, 0.005)
    lo  = (lzc_means - lzc_sems).min() - pad
    hi  = (lzc_means + lzc_sems).max() + pad
    ax.set_ylim(max(0.0, lo), hi)

    # Value labels
    for i, (m, s) in enumerate(zip(lzc_means, lzc_sems)):
        ax.text(i, m + s + 0.003, f"{m:.5f}", ha="center", va="bottom", fontsize=9)

    # Difference annotation
    diff = lzc_means[1] - lzc_means[0] if len(lzc_means) == 2 else 0
    ax.annotate(f"Diff = {diff:+.5f}\nExpected: ~+0.003 to +0.004",
                xy=(0.97, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="darkblue",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDebug figure saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TVB anesthesia + psychedelic complexity simulations")
    parser.add_argument("--debug", action="store_true",
                        help="Run only Awake + DOI (LZc only, no PCI) for quick validation")
    parser.add_argument("--propofol", action="store_true",
                        help="Run only Propofol + Propofol+DOI (LZc only, no PCI)")
    parser.add_argument("--all-lzc", action="store_true",
                        help="Run all 4 conditions, LZc only (no PCI)")
    args = parser.parse_args()

    t_total = time.time()

    if args.debug:
        active_conditions = CONDITIONS_DEBUG
        debug_mode = True
        out_fig = os.path.normpath(
            os.path.join(PROJECT_ROOT, "..", "figures", "debug_awake_vs_doi_LZc.png"))
        out_npz = os.path.normpath(
            os.path.join(PROJECT_ROOT, "..", "figures", "debug_awake_vs_doi_results.npz"))
    elif args.propofol:
        active_conditions = CONDITIONS_PROPOFOL
        debug_mode = True   # LZc only, no PCI
        out_fig = os.path.normpath(
            os.path.join(PROJECT_ROOT, "..", "figures", "debug_propofol_LZc.png"))
        out_npz = os.path.normpath(
            os.path.join(PROJECT_ROOT, "..", "figures", "debug_propofol_results.npz"))
    elif args.all_lzc:
        active_conditions = CONDITIONS
        debug_mode = True   # LZc only, no PCI (faster)
        out_fig = os.path.normpath(
            os.path.join(PROJECT_ROOT, "..", "figures", "complexity_LZC.png"))
        out_npz = os.path.normpath(
            os.path.join(PROJECT_ROOT, "..", "figures", "complexity_results.npz"))
    else:
        active_conditions = CONDITIONS
        debug_mode = False
        out_fig = os.path.normpath(
            os.path.join(PROJECT_ROOT, "..", "figures", "complexity_LZC_PCI.png"))
        out_npz = os.path.normpath(
            os.path.join(PROJECT_ROOT, "..", "figures", "complexity_results.npz"))

    seeds = list(range(N_SEEDS))
    jobs  = [(cond, params, s, debug_mode)
             for cond, params in active_conditions.items()
             for s in seeds]

    n_cores = min(mp.cpu_count(), len(jobs))
    mode_str = "[DEBUG]" if args.debug else "[PROPOFOL]" if args.propofol else "[ALL-LZC]" if args.all_lzc else ""
    print(f"\nRunning {len(jobs)} jobs ({len(active_conditions)} conditions x {N_SEEDS} seeds) "
          f"across {n_cores} cores {mode_str}...\n")

    with mp.Pool(processes=n_cores) as pool:
        raw_results = pool.map(_worker, jobs)

    results = {}
    idx = 0
    for cond in active_conditions:
        chunk = raw_results[idx:idx + N_SEEDS]
        idx  += N_SEEDS
        lzc_arr = np.array([r[0] for r in chunk])
        pci_arr = np.array([r[1] for r in chunk])
        results[cond] = {
            "lzc_all":  lzc_arr,
            "pci_all":  pci_arr,
            "lzc_mean": lzc_arr.mean(),
            "lzc_sem":  lzc_arr.std() / np.sqrt(N_SEEDS) if N_SEEDS > 1 else 0.0,
            "pci_mean": pci_arr.mean(),
            "pci_sem":  pci_arr.std() / np.sqrt(N_SEEDS) if N_SEEDS > 1 else 0.0,
        }

    print(f"\n{'='*70}")
    print(f"SUMMARY  (n={N_SEEDS} seeds) {'[DEBUG]' if args.debug else ''}")
    print(f"{'='*70}")
    if debug_mode:
        print(f"{'Condition':<20}  {'LZc':>12}")
        print("-" * 35)
        for cond, r in results.items():
            print(f"{cond:<20}  {r['lzc_mean']:>12.5f}")
        if len(results) == 2:
            conds = list(results.keys())
            diff = results[conds[1]]["lzc_mean"] - results[conds[0]]["lzc_mean"]
            print(f"\n{conds[1]} - {conds[0]} LZc diff = {diff:+.5f}")
            if args.debug:
                print(f"Expected: ~+0.003 to +0.004 (Martin et al. 2025)")
    else:
        print(f"{'Condition':<20}  {'LZc':>12}  {'PCI':>10}")
        print("-" * 50)
        for cond, r in results.items():
            print(f"{cond:<20}  {r['lzc_mean']:>12.5f}  {r['pci_mean']:>10.4f}")

        print(f"\nSacha et al. reference:")
        print(f"  wake:  PCI ~0.35-0.50  (stim=1e-3)")
        print(f"  gaba:  PCI ~0.20-0.35  (propofol, tau_i=7)")
        print(f"  nmda:  PCI ~0.15-0.30  (ketamine, tau_e=3.75)")
        print(f"  sleep: PCI ~0.10-0.25  (b_e=120)")

    print(f"\nTotal wall time: {(time.time()-t_total)/60:.1f} min")

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz, **{
        f"{cond.replace('+','_').replace(' ','_')}_{k}": v
        for cond, r in results.items()
        for k, v in r.items()
        if isinstance(v, np.ndarray)
    })
    print(f"Raw results saved -> {out_npz}")

    if args.debug:
        plot_debug_awake_vs_doi(results, out_fig)
    elif args.propofol:
        plot_debug_awake_vs_doi(results, out_fig)
    elif args.all_lzc:
        plot_results(results, out_fig)   # use full 4-condition plot (PCI will be zeros)
    else:
        plot_results(results, out_fig)


if __name__ == "__main__":
    main()
