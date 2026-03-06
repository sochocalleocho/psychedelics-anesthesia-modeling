#!/usr/bin/env python3
"""
tvb_quick_test.py — Quick TVB validation of alternative TF polynomials.

Runs N_SEEDS=1 (single seed) for all 4 drug conditions using different P_E polynomials,
comparing LZc against the CONFIG1 reference values.

This is a streamlined version of tvb_anesthesia_complexity.py for rapid testing.
Sequential execution (no multiprocessing) to avoid pickling issues with module swaps.
"""

import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup (same as production script)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
PIPELINE_HUB = os.path.join(PROJECT_ROOT, "paper_pipeline_hub")
MARTIN_REPO  = os.path.join(PROJECT_ROOT, "simulated_serotonergic_receptors_tvb")
SACHA_TOOLS  = os.path.join(PIPELINE_HUB, "TVB", "tvb_model_reference", "src")
MARTIN_MODEL_SRC = os.path.join(
    MARTIN_REPO, "tvbsim", "TVB", "tvb_model_reference", "src")
MARTIN_CONN_DIR  = os.path.join(MARTIN_REPO, "data", "connectivity", "DK68")
MARTIN_CONN_FILE = "connectivity_68_QL20120814.zip"

for p in [PIPELINE_HUB, MARTIN_REPO, SACHA_TOOLS, MARTIN_MODEL_SRC,
          os.path.join(PIPELINE_HUB, "TVB")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Biophysical constants (from production script)
# ---------------------------------------------------------------------------
E_NA   = 50.0
E_K    = -90.0
G_L    = 10.0
E_L_E_BASE = -64.0
E_L_I_BASE = -65.0

def _conversion(E_Na, E_K, E_L, g_L=None, g_Na=None):
    if g_L is not None:
        g_K = g_L * (E_L - E_Na) / (E_K - E_Na)
        g_Na_out = g_L - g_K
        return g_K, g_Na_out
    g_L_new = g_Na * (E_Na - E_K) / (E_L - E_K)
    g_K = g_L_new - g_Na
    return g_K, g_Na

G_K_E_BASE, G_NA_E = _conversion(E_NA, E_K, E_L_E_BASE, g_L=G_L)
G_K_I_BASE, G_NA_I = _conversion(E_NA, E_K, E_L_I_BASE, g_L=G_L)

E_L_E_PSI = -61.2
E_L_I_PSI = -64.4
G_K_E_PSI, _ = _conversion(E_NA, E_K, E_L_E_PSI, g_Na=G_NA_E)
G_K_I_PSI, _ = _conversion(E_NA, E_K, E_L_I_PSI, g_Na=G_NA_I)

# 5-HT2A map
density_file = os.path.join(MARTIN_REPO, "data", "receptors", "DK68", "5HT2a_reordered.txt")
_DENSITIES_68 = np.clip(np.loadtxt(density_file), 0, None)
d_max = _DENSITIES_68.max()
_GK_E_DOI = np.interp(_DENSITIES_68, [0, d_max], [G_K_E_BASE, G_K_E_PSI])
_GK_I_DOI = np.interp(_DENSITIES_68, [0, d_max], [G_K_I_BASE, G_K_I_PSI])

# ---------------------------------------------------------------------------
# Drug conditions
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Transfer Function Polynomials to test
# ---------------------------------------------------------------------------

# CONFIG1 reference (Sacha) — known working, P[2]=-0.008
P_E_SACHA = np.array([-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
                       0.00341614, -0.01156433, 0.00194753,  0.00274079, -0.01066769])
P_I_SACHA = np.array([-0.05184978, 0.00615930, -0.01403522, 0.00166511, -0.00205590,
                       0.00318432, -0.03112775, 0.00656668,  0.00171829, -0.04516385])

# Global optimum — P[2]=+0.004, CONFIRMED FAILED in TVB (no propofol collapse)
P_E_GLOBAL = np.array([-0.04886099,  0.0011248,   0.00381117, -0.00236892,  0.00099447,
                         0.01164049, -0.00175739, -0.00297966,  0.00041174, -0.01225042])

# --- Constrained fits from P[2] scan (tf_constrained_fit.py) ---
# These have P[2] fixed at various negative values, all other coeffs optimized via NM.
# Format: scan results from constrained_fit.log

# P[2]=-0.001: MSE=20.42 (near-optimal), barely negative
P_E_P2_001 = np.array([-0.04870354, 0.00136772, -0.00100000, 0.00047821, 0.00091095,
                         0.01004904, -0.01129805, -0.00197425, -0.00072667, -0.01847103])

# P[2]=-0.005: MSE=20.45
P_E_P2_005 = None  # Will be loaded from scan if available

# P[2]=-0.015: Basin-Hopping result, MSE=21.2
P_E_P2_015 = np.array([-0.04870238, 0.00223867, -0.01499234, 0.00502270, 0.00065347,
                         0.01250416, -0.03783639, 0.00066465, -0.00289020, -0.05296499])

# P[2]=-0.025: From scan, MSE=20.83
# (We don't have exact coefficients for this, using Di Volo as proxy for "very negative P[2]")
P_E_DIVOLO = np.array([-0.04983106, 0.005063550882777035, -0.023470121807314552,
                         0.0022951513725067503, -0.0004105302652029825, 0.010547051343547399,
                         -0.03659252821136933, 0.007437487505797858, 0.001265064721846073,
                         -0.04072161294490446])

POLYNOMIALS = {
    "CONFIG1":     (P_E_SACHA,   P_I_SACHA),   # P[2]=-0.008, reference
    "GLOBAL":      (P_E_GLOBAL,  P_I_SACHA),    # P[2]=+0.004, confirmed fail
    "P2=-0.001":   (P_E_P2_001,  P_I_SACHA),    # Near-zero P[2]
    "P2=-0.015":   (P_E_P2_015,  P_I_SACHA),    # More negative than CONFIG1
    "DiVolo":      (P_E_DIVOLO,  P_I_SACHA),    # P[2]=-0.024, Martin's polynomial
}

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SIM_LEN = 5000.0       # ms
DT = 0.1               # ms
ANALYSIS_LAST_MS = 1000.0

# ---------------------------------------------------------------------------
# Single simulation runner
# ---------------------------------------------------------------------------
def run_one_sim(params, P_E, P_I, seed, conn, n_regions):
    """Run a single TVB simulation and return (times, fr_kHz)."""
    from tvb.simulator import simulator, coupling, integrators, monitors, noise
    import Zerlaut_gK_gNa as custom_zerlaut

    model = custom_zerlaut.Zerlaut_adaptation_second_order(
        g_K_e  = params["g_K_e"],
        g_Na_e = np.array([G_NA_E]),
        g_K_i  = params["g_K_i"],
        g_Na_i = np.array([G_NA_I]),
        E_K_e  = np.array([E_K]),
        E_Na_e = np.array([E_NA]),
        E_K_i  = np.array([E_K]),
        E_Na_i = np.array([E_NA]),
        C_m    = np.array([200.0]),
        b_e    = np.array([params["b_e"]]),
        a_e    = np.array([0.0]),
        b_i    = np.array([0.0]),
        a_i    = np.array([0.0]),
        tau_w_e= np.array([500.0]),
        tau_w_i= np.array([1.0]),
        tau_e  = np.array([params["tau_e"]]),
        tau_i  = np.array([params["tau_i"]]),
        E_e    = np.array([0.0]),
        E_i    = np.array([-80.0]),
        Q_e    = np.array([1.5]),
        Q_i    = np.array([5.0]),
        N_tot       = np.array([10000]),
        p_connect_e = np.array([0.05]),
        p_connect_i = np.array([0.05]),
        g           = np.array([0.2]),
        T           = np.array([20.0]),
        K_ext_e = np.array([400]),
        K_ext_i = np.array([0]),
        external_input_ex_ex = np.array([0.315e-3]),
        external_input_ex_in = np.array([0.000]),
        external_input_in_ex = np.array([0.315e-3]),
        external_input_in_in = np.array([0.000]),
        tau_OU       = np.array([5.0]),
        weight_noise = np.array([1e-4]),
        P_e = P_E,
        P_i = P_I,
        inh_factor = np.array([1.0]),
    )

    coupl = coupling.Linear(a=np.array([0.3]), b=np.array([0.0]))
    nsig = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    noise_inst = noise.Additive(nsig=nsig)
    integ = integrators.HeunStochastic(dt=DT, noise=noise_inst)

    mon = monitors.Raw()
    sim = simulator.Simulator(
        model=model, connectivity=conn, coupling=coupl,
        integrator=integ, monitors=(mon,),
    )
    sim.configure()

    ic = np.zeros((8, n_regions, 1))
    ic[5, :, 0] = 100.0
    sim.current_state[:] = ic
    sim.integrator.noise.random_stream.seed(seed)

    raw = sim.run(simulation_length=SIM_LEN)
    times = raw[0][0].flatten()
    fr = raw[0][1][:, 0, :, 0]
    return times, fr


def compute_lzc(fr):
    """Compute LZc on last 1s of firing rate data."""
    from tvbsim.entropy.measures_functions import calculate_LempelZiv
    n_analysis = int(ANALYSIS_LAST_MS / DT)
    return calculate_LempelZiv(fr[-n_analysis:, :])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from tvb.datatypes import connectivity

    # Reference LZc values (CONFIG1, N=16 seeds)
    REF_LZC = {
        "Awake":        0.94445,
        "Propofol":     0.58588,
        "Propofol+DOI": 0.75657,
        "DOI only":     0.94465,
    }

    # Load connectivity
    conn_path = os.path.join(MARTIN_CONN_DIR, MARTIN_CONN_FILE)
    conn = connectivity.Connectivity.from_file(conn_path)
    conn.configure()
    conn.weights = conn.weights / (np.sum(conn.weights, axis=0) + 1e-12)
    conn.speed = np.array([4.0])
    n_regions = conn.weights.shape[0]

    seed = 0  # Single seed for quick test

    print("=" * 90)
    print("TVB QUICK TEST — Comparing TF Polynomials (N_SEEDS=1)")
    print("=" * 90)
    print(f"Simulation: {SIM_LEN}ms, dt={DT}ms, seed={seed}")
    print(f"LZc: last {ANALYSIS_LAST_MS}ms of spontaneous activity")
    print(f"Polynomials: {list(POLYNOMIALS.keys())}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print()

    # Results storage: {poly_name: {cond_name: (lzc, mean_fr)}}
    results = {}

    total_sims = len(POLYNOMIALS) * len(CONDITIONS)
    sim_idx = 0

    for poly_name, (P_E, P_I) in POLYNOMIALS.items():
        results[poly_name] = {}
        print(f"\n{'─' * 90}")
        print(f"POLYNOMIAL: {poly_name}")
        print(f"  P_E = [{', '.join(f'{x:.6f}' for x in P_E)}]")
        print(f"{'─' * 90}")

        for cond_name, params in CONDITIONS.items():
            sim_idx += 1
            t0 = time.time()
            print(f"\n  [{sim_idx}/{total_sims}] {poly_name} / {cond_name} ...", end='', flush=True)

            times, fr = run_one_sim(params, P_E, P_I, seed, conn, n_regions)

            fr_hz = fr * 1000.0
            mean_fr = fr_hz[-10000:, :].mean()
            lzc = compute_lzc(fr)
            elapsed = time.time() - t0

            results[poly_name][cond_name] = (lzc, mean_fr)
            print(f"  LZc={lzc:.5f}  FR={mean_fr:.2f} Hz  ({elapsed:.0f}s)", flush=True)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)

    # Header
    conds = list(CONDITIONS.keys())
    print(f"\n{'Polynomial':12s}", end='')
    for c in conds:
        print(f"  {c:>14s}", end='')
    print()
    print("-" * (12 + 16 * len(conds)))

    # Reference row
    print(f"{'REF (N=16)':12s}", end='')
    for c in conds:
        print(f"  {REF_LZC[c]:>14.5f}", end='')
    print()

    # Data rows
    for poly_name in POLYNOMIALS:
        print(f"{poly_name:12s}", end='')
        for c in conds:
            lzc, _ = results[poly_name][c]
            print(f"  {lzc:>14.5f}", end='')
        print()

    # Firing rate table
    print(f"\n{'Mean FR (Hz)':12s}", end='')
    for c in conds:
        print(f"  {c:>14s}", end='')
    print()
    print("-" * (12 + 16 * len(conds)))
    for poly_name in POLYNOMIALS:
        print(f"{poly_name:12s}", end='')
        for c in conds:
            _, fr = results[poly_name][c]
            print(f"  {fr:>14.2f}", end='')
        print()

    # Key ratios
    print(f"\n{'RATIOS':12s} {'Prop/Awake':>14s} {'P+DOI/Awake':>14s} {'DOI/Awake':>14s} {'DOI-Awake':>14s}")
    print("-" * 70)
    for poly_name in POLYNOMIALS:
        lzc_awake = results[poly_name]["Awake"][0]
        lzc_prop  = results[poly_name]["Propofol"][0]
        lzc_pdoi  = results[poly_name]["Propofol+DOI"][0]
        lzc_doi   = results[poly_name]["DOI only"][0]

        prop_ratio = lzc_prop / lzc_awake if lzc_awake > 0 else 0
        pdoi_ratio = lzc_pdoi / lzc_awake if lzc_awake > 0 else 0
        doi_ratio  = lzc_doi / lzc_awake if lzc_awake > 0 else 0
        doi_diff   = lzc_doi - lzc_awake

        print(f"{poly_name:12s} {prop_ratio:>14.4f} {pdoi_ratio:>14.4f} {doi_ratio:>14.4f} {doi_diff:>+14.5f}")

    # Reference ratios
    print(f"{'REF (N=16)':12s} "
          f"{REF_LZC['Propofol']/REF_LZC['Awake']:>14.4f} "
          f"{REF_LZC['Propofol+DOI']/REF_LZC['Awake']:>14.4f} "
          f"{REF_LZC['DOI only']/REF_LZC['Awake']:>14.4f} "
          f"{REF_LZC['DOI only']-REF_LZC['Awake']:>+14.5f}")

    # Assessment
    print("\n" + "=" * 90)
    print("ASSESSMENT")
    print("=" * 90)

    for poly_name in POLYNOMIALS:
        lzc_a = results[poly_name]["Awake"][0]
        lzc_p = results[poly_name]["Propofol"][0]
        lzc_pd = results[poly_name]["Propofol+DOI"][0]
        lzc_d = results[poly_name]["DOI only"][0]

        propofol_collapse = lzc_p < 0.7 * lzc_a
        doi_reversal = lzc_pd > lzc_p
        doi_above_awake = lzc_d > lzc_a
        correct_ordering = lzc_p < lzc_pd < lzc_a <= lzc_d

        print(f"\n{poly_name}:")
        print(f"  Propofol collapse (LZc_P < 0.7*LZc_A): {'YES' if propofol_collapse else 'NO'} "
              f"({lzc_p:.4f} vs {0.7*lzc_a:.4f})")
        print(f"  DOI reversal (LZc_P+DOI > LZc_P):      {'YES' if doi_reversal else 'NO'} "
              f"({lzc_pd:.4f} vs {lzc_p:.4f})")
        print(f"  DOI > Awake (LZc_DOI > LZc_A):         {'YES' if doi_above_awake else 'NO'} "
              f"({lzc_d:.4f} vs {lzc_a:.4f})")
        print(f"  Correct ordering (P < P+D < A <= D):    {'YES' if correct_ordering else 'NO'}")

    print(f"\nTotal wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    t_start = time.time()
    main()
