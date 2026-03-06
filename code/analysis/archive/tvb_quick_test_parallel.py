#!/usr/bin/env python3
"""
tvb_quick_test_parallel.py — Parallel TVB validation of TF polynomials.

Runs ONLY polynomials not yet tested. Uses multiprocessing to run all
(polynomial, condition) pairs in parallel, one per core.

Known results (DO NOT RE-RUN):
  CONFIG1:  Awake=0.9447, Propofol=0.5322, Prop+DOI=0.7304, DOI=0.9415
  GLOBAL:   Awake=0.9468, Propofol=0.9403, Prop+DOI=0.5477, DOI=0.9467  (FAILS)
  WEIGHTED: Awake=0.9119, Propofol=0.9514, Prop+DOI=0.9540, DOI=0.9123  (FAILS)
"""

import os
import sys
import time
import warnings
import numpy as np
import multiprocessing as mp

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
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

# ---------------------------------------------------------------------------
# Biophysical constants
# ---------------------------------------------------------------------------
E_NA = 50.0; E_K = -90.0; G_L = 10.0
E_L_E_BASE = -64.0; E_L_I_BASE = -65.0

def _conversion(E_Na, E_K, E_L, g_L=None, g_Na=None):
    if g_L is not None:
        g_K = g_L * (E_L - E_Na) / (E_K - E_Na)
        return g_K, g_L - g_K
    g_L_new = g_Na * (E_Na - E_K) / (E_L - E_K)
    return g_L_new - g_Na, g_Na

G_K_E_BASE, G_NA_E = _conversion(E_NA, E_K, E_L_E_BASE, g_L=G_L)
G_K_I_BASE, G_NA_I = _conversion(E_NA, E_K, E_L_I_BASE, g_L=G_L)
G_K_E_PSI, _ = _conversion(E_NA, E_K, -61.2, g_Na=G_NA_E)
G_K_I_PSI, _ = _conversion(E_NA, E_K, -64.4, g_Na=G_NA_I)

density_file = os.path.join(MARTIN_REPO, "data", "receptors", "DK68", "5HT2a_reordered.txt")
_DENSITIES_68 = np.clip(np.loadtxt(density_file), 0, None)
d_max = _DENSITIES_68.max()
_GK_E_DOI = np.interp(_DENSITIES_68, [0, d_max], [G_K_E_BASE, G_K_E_PSI])
_GK_I_DOI = np.interp(_DENSITIES_68, [0, d_max], [G_K_I_BASE, G_K_I_PSI])

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
CONDITIONS = {
    "Awake": {"b_e": 5.0, "tau_e": 5.0, "tau_i": 5.0,
              "g_K_e": np.array([G_K_E_BASE]), "g_K_i": np.array([G_K_I_BASE])},
    "Propofol": {"b_e": 30.0, "tau_e": 5.0, "tau_i": 7.0,
                 "g_K_e": np.array([G_K_E_BASE]), "g_K_i": np.array([G_K_I_BASE])},
    "Propofol+DOI": {"b_e": 30.0, "tau_e": 5.0, "tau_i": 7.0,
                     "g_K_e": _GK_E_DOI, "g_K_i": _GK_I_DOI},
    "DOI only": {"b_e": 5.0, "tau_e": 5.0, "tau_i": 5.0,
                 "g_K_e": _GK_E_DOI, "g_K_i": _GK_I_DOI},
}

# ---------------------------------------------------------------------------
# Polynomials — ONLY new ones not yet tested
# ---------------------------------------------------------------------------
P_I_SACHA = np.array([-0.05184978, 0.00615930, -0.01403522, 0.00166511, -0.00205590,
                       0.00318432, -0.03112775, 0.00656668,  0.00171829, -0.04516385])

# P[2]=-0.001: Near-zero P[2], MSE=20.42 (near-optimal)
P_E_P2_001 = np.array([-0.04870354, 0.00136772, -0.00100000, 0.00047821, 0.00091095,
                         0.01004904, -0.01129805, -0.00197425, -0.00072667, -0.01847103])

# P[2]=-0.015: Basin-Hopping result, MSE=21.2
P_E_P2_015 = np.array([-0.04870238, 0.00223867, -0.01499234, 0.00502270, 0.00065347,
                         0.01250416, -0.03783639, 0.00066465, -0.00289020, -0.05296499])

# Di Volo (Martin's polynomial): P[2]=-0.024
P_E_DIVOLO = np.array([-0.04983106, 0.005063550882777035, -0.023470121807314552,
                         0.0022951513725067503, -0.0004105302652029825, 0.010547051343547399,
                         -0.03659252821136933, 0.007437487505797858, 0.001265064721846073,
                         -0.04072161294490446])

POLYNOMIALS = {
    "P2=-0.001": (P_E_P2_001, P_I_SACHA),
    "P2=-0.015": (P_E_P2_015, P_I_SACHA),
    "DiVolo":    (P_E_DIVOLO, P_I_SACHA),
}

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SIM_LEN = 5000.0; DT = 0.1; ANALYSIS_LAST_MS = 1000.0; SEED = 0

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker(args):
    """Run one (polynomial, condition) and return (poly_name, cond_name, lzc, mean_fr)."""
    poly_name, cond_name, P_E, P_I, cond_params = args

    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, sys, os

    for p in [PIPELINE_HUB, MARTIN_REPO, SACHA_TOOLS, MARTIN_MODEL_SRC,
              os.path.join(PIPELINE_HUB, "TVB")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from tvb.datatypes import connectivity
    from tvb.simulator import simulator, coupling, integrators, monitors, noise
    from tvbsim.entropy.measures_functions import calculate_LempelZiv
    import Zerlaut_gK_gNa as custom_zerlaut

    # Connectivity
    conn = connectivity.Connectivity.from_file(
        os.path.join(MARTIN_CONN_DIR, MARTIN_CONN_FILE))
    conn.configure()
    conn.weights = conn.weights / (np.sum(conn.weights, axis=0) + 1e-12)
    conn.speed = np.array([4.0])
    n_regions = conn.weights.shape[0]

    # Model
    model = custom_zerlaut.Zerlaut_adaptation_second_order(
        g_K_e=cond_params["g_K_e"], g_Na_e=np.array([G_NA_E]),
        g_K_i=cond_params["g_K_i"], g_Na_i=np.array([G_NA_I]),
        E_K_e=np.array([E_K]), E_Na_e=np.array([E_NA]),
        E_K_i=np.array([E_K]), E_Na_i=np.array([E_NA]),
        C_m=np.array([200.0]),
        b_e=np.array([cond_params["b_e"]]), a_e=np.array([0.0]),
        b_i=np.array([0.0]), a_i=np.array([0.0]),
        tau_w_e=np.array([500.0]), tau_w_i=np.array([1.0]),
        tau_e=np.array([cond_params["tau_e"]]),
        tau_i=np.array([cond_params["tau_i"]]),
        E_e=np.array([0.0]), E_i=np.array([-80.0]),
        Q_e=np.array([1.5]), Q_i=np.array([5.0]),
        N_tot=np.array([10000]),
        p_connect_e=np.array([0.05]), p_connect_i=np.array([0.05]),
        g=np.array([0.2]), T=np.array([20.0]),
        K_ext_e=np.array([400]), K_ext_i=np.array([0]),
        external_input_ex_ex=np.array([0.315e-3]),
        external_input_ex_in=np.array([0.000]),
        external_input_in_ex=np.array([0.315e-3]),
        external_input_in_in=np.array([0.000]),
        tau_OU=np.array([5.0]), weight_noise=np.array([1e-4]),
        P_e=P_E, P_i=P_I, inh_factor=np.array([1.0]),
    )

    sim = simulator.Simulator(
        model=model, connectivity=conn,
        coupling=coupling.Linear(a=np.array([0.3]), b=np.array([0.0])),
        integrator=integrators.HeunStochastic(
            dt=DT,
            noise=noise.Additive(
                nsig=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))),
        monitors=(monitors.Raw(),),
    )
    sim.configure()

    ic = np.zeros((8, n_regions, 1))
    ic[5, :, 0] = 100.0
    sim.current_state[:] = ic
    sim.integrator.noise.random_stream.seed(SEED)

    raw = sim.run(simulation_length=SIM_LEN)
    fr = raw[0][1][:, 0, :, 0]

    fr_hz = fr * 1000.0
    mean_fr = fr_hz[-10000:, :].mean()

    n_analysis = int(ANALYSIS_LAST_MS / DT)
    lzc = calculate_LempelZiv(fr[-n_analysis:, :])

    print(f"  DONE: {poly_name:12s} / {cond_name:14s}  LZc={lzc:.5f}  FR={mean_fr:.2f} Hz",
          flush=True)
    return poly_name, cond_name, lzc, mean_fr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    # Known results (do not re-run)
    KNOWN = {
        "CONFIG1":  {"Awake": 0.94470, "Propofol": 0.53224,
                     "Propofol+DOI": 0.73039, "DOI only": 0.94150},
        "GLOBAL":   {"Awake": 0.94681, "Propofol": 0.94028,
                     "Propofol+DOI": 0.54765, "DOI only": 0.94672},
        "WEIGHTED": {"Awake": 0.91190, "Propofol": 0.95141,
                     "Propofol+DOI": 0.95404, "DOI only": 0.91232},
    }

    # Build job list (only new polynomials)
    jobs = []
    for poly_name, (P_E, P_I) in POLYNOMIALS.items():
        for cond_name, cond_params in CONDITIONS.items():
            jobs.append((poly_name, cond_name, P_E, P_I, cond_params))

    n_cores = min(mp.cpu_count(), len(jobs))
    print("=" * 80)
    print(f"TVB QUICK TEST — PARALLEL ({len(jobs)} new sims across {n_cores} cores)")
    print("=" * 80)
    print(f"New polynomials: {list(POLYNOMIALS.keys())}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Already known: {list(KNOWN.keys())} (not re-running)")
    print(f"Seed: {SEED}, SIM_LEN: {SIM_LEN}ms\n")

    with mp.Pool(processes=n_cores) as pool:
        raw_results = pool.map(_worker, jobs)

    # Collect results
    new_results = {}
    for poly_name, cond_name, lzc, mean_fr in raw_results:
        if poly_name not in new_results:
            new_results[poly_name] = {}
        new_results[poly_name][cond_name] = (lzc, mean_fr)

    # Merge with known results
    all_results = {}
    for name, lzc_dict in KNOWN.items():
        all_results[name] = {c: (v, None) for c, v in lzc_dict.items()}
    all_results.update(new_results)

    # Print summary
    conds = list(CONDITIONS.keys())
    print("\n" + "=" * 80)
    print("COMBINED RESULTS (known + new)")
    print("=" * 80)
    print(f"\n{'Polynomial':12s}", end='')
    for c in conds:
        print(f"  {c:>14s}", end='')
    print(f"  {'P[2]':>10s}")
    print("-" * (12 + 16 * len(conds) + 12))

    # P[2] values for display
    P_E_CONFIG1 = np.array([-0.05017034, 0.00451531, -0.00794377])
    P_E_GLOBAL_arr = np.array([-0.04886099, 0.0011248, 0.00381117])
    P_E_WEIGHTED_arr = np.array([-0.0480776, -0.00010327, 0.0092499])
    p2_vals = {"CONFIG1": -0.00794, "GLOBAL": 0.00381, "WEIGHTED": 0.00925,
               "P2=-0.001": -0.001, "P2=-0.015": -0.01499, "DiVolo": -0.02347}

    for poly_name in ["CONFIG1", "GLOBAL", "WEIGHTED", "P2=-0.001", "P2=-0.015", "DiVolo"]:
        if poly_name not in all_results:
            continue
        marker = " *" if poly_name in KNOWN else ""
        print(f"{poly_name:12s}", end='')
        for c in conds:
            lzc, _ = all_results[poly_name].get(c, (None, None))
            if lzc is not None:
                print(f"  {lzc:>14.5f}", end='')
            else:
                print(f"  {'???':>14s}", end='')
        p2 = p2_vals.get(poly_name, 0)
        print(f"  {p2:>10.5f}{marker}")

    print("\n* = previously known (not re-run)")

    # Assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT OF NEW POLYNOMIALS")
    print("=" * 80)

    for poly_name in POLYNOMIALS:
        if poly_name not in new_results:
            continue
        lzc_a = new_results[poly_name]["Awake"][0]
        lzc_p = new_results[poly_name]["Propofol"][0]
        lzc_pd = new_results[poly_name]["Propofol+DOI"][0]
        lzc_d = new_results[poly_name]["DOI only"][0]

        propofol_collapse = lzc_p < 0.7 * lzc_a
        doi_reversal = lzc_pd > lzc_p
        doi_above_awake = lzc_d > lzc_a
        correct_ordering = lzc_p < lzc_pd < lzc_a <= lzc_d

        print(f"\n{poly_name} (P[2]={p2_vals.get(poly_name, '?')}):")
        print(f"  Propofol collapse (LZc_P < 0.7*LZc_A): {'YES' if propofol_collapse else 'NO'} "
              f"({lzc_p:.4f} vs {0.7*lzc_a:.4f})")
        print(f"  DOI reversal (LZc_P+DOI > LZc_P):      {'YES' if doi_reversal else 'NO'} "
              f"({lzc_pd:.4f} vs {lzc_p:.4f})")
        print(f"  DOI > Awake (LZc_DOI > LZc_A):         {'YES' if doi_above_awake else 'NO'} "
              f"({lzc_d:.5f} vs {lzc_a:.5f})")
        print(f"  Correct ordering (P < P+D < A <= D):    {'YES' if correct_ordering else 'NO'}")
        if correct_ordering:
            print(f"  >>> CANDIDATE for full validation! DOI-Awake = {lzc_d-lzc_a:+.5f}")

    # Reference
    print(f"\nReference (CONFIG1 N=16): Awake=0.94445, Prop=0.58588, "
          f"P+DOI=0.75657, DOI=0.94465 (DOI-Awake=+0.00020)")

    print(f"\nTotal wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
