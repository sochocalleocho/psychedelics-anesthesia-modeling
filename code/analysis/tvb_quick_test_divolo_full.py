#!/usr/bin/env python3
"""
tvb_quick_test_divolo_full.py — Test Di Volo's FULL polynomial pair (P_E + P_I).

Previous tests used Di Volo's P_E with Sacha's P_I (mixed pair).
This tests the NATIVE pairing: Di Volo P_E + Di Volo P_I.

Also tests CONFIG1 P_E + Di Volo P_I to isolate the effect.
"""

import os, sys, time, warnings
import numpy as np
import multiprocessing as mp

warnings.filterwarnings("ignore")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
PIPELINE_HUB = os.path.join(PROJECT_ROOT, "paper_pipeline_hub")
MARTIN_REPO  = os.path.join(PROJECT_ROOT, "simulated_serotonergic_receptors_tvb")
SACHA_TOOLS  = os.path.join(PIPELINE_HUB, "TVB", "tvb_model_reference", "src")
MARTIN_MODEL_SRC = os.path.join(
    MARTIN_REPO, "tvbsim", "TVB", "tvb_model_reference", "src")
MARTIN_CONN_DIR  = os.path.join(MARTIN_REPO, "data", "connectivity", "DK68")
MARTIN_CONN_FILE = "connectivity_68_QL20120814.zip"

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
# Polynomials
# ---------------------------------------------------------------------------
# Sacha CONFIG1
P_E_SACHA = np.array([-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
                       0.00341614, -0.01156433, 0.00194753,  0.00274079, -0.01066769])
P_I_SACHA = np.array([-0.05184978, 0.00615930, -0.01403522, 0.00166511, -0.00205590,
                       0.00318432, -0.03112775, 0.00656668,  0.00171829, -0.04516385])

# Di Volo (from Martin's code — 10 coefficients, P[4] zeroed out from 11-param form)
P_E_DIVOLO = np.array([-0.04983106, 0.005063550882777035, -0.023470121807314552,
                         0.0022951513725067503, -0.0004105302652029825, 0.010547051343547399,
                         -0.03659252821136933, 0.007437487505797858, 0.001265064721846073,
                         -0.04072161294490446])
P_I_DIVOLO = np.array([-0.05149122024209484, 0.004003689190271077, -0.008352013668528155,
                         0.0002414237992765705, -0.0005070645080016026, 0.0014345394104282397,
                         -0.014686689498949967, 0.004502706285435741, 0.0028472190352532454,
                         -0.015357804594594548])

POLYNOMIALS = {
    # Known reference: CONFIG1 pair (Sacha P_E + Sacha P_I)
    # Awake=0.945, Prop=0.532, P+DOI=0.730, DOI=0.942  ← DO NOT RE-RUN

    # New test 1: Di Volo FULL pair
    "DiVolo_full": (P_E_DIVOLO, P_I_DIVOLO),

    # New test 2: CONFIG1 P_E + Di Volo P_I (test P_I effect)
    "C1_PE+DV_PI": (P_E_SACHA, P_I_DIVOLO),

    # New test 3: Di Volo P_E + Sacha P_I (already done, re-running for confirmation)
    # Known: Awake=0.936, Prop=0.921, P+DOI=0.928, DOI=0.942
    # Skip to save time — but include for completeness
    "DV_PE+C1_PI": (P_E_DIVOLO, P_I_SACHA),
}

SIM_LEN = 5000.0; DT = 0.1; ANALYSIS_LAST_MS = 1000.0; SEED = 0

def _worker(args):
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

    conn = connectivity.Connectivity.from_file(
        os.path.join(MARTIN_CONN_DIR, MARTIN_CONN_FILE))
    conn.configure()
    conn.weights = conn.weights / (np.sum(conn.weights, axis=0) + 1e-12)
    conn.speed = np.array([4.0])
    n_regions = conn.weights.shape[0]

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

    print(f"  DONE: {poly_name:16s} / {cond_name:14s}  LZc={lzc:.5f}  FR={mean_fr:.2f} Hz",
          flush=True)
    return poly_name, cond_name, lzc, mean_fr


def main():
    t_start = time.time()

    KNOWN = {
        "C1_PE+C1_PI":  {"Awake": (0.94470, 6.58), "Propofol": (0.53224, 1.68),
                          "Propofol+DOI": (0.73039, 2.70), "DOI only": (0.94150, 6.98)},
        "DV_PE+C1_PI":  {"Awake": (0.93617, None), "Propofol": (0.92089, None),
                          "Propofol+DOI": (0.92760, None), "DOI only": (0.94165, None)},
    }

    jobs = []
    for poly_name, (P_E, P_I) in POLYNOMIALS.items():
        for cond_name, cond_params in CONDITIONS.items():
            jobs.append((poly_name, cond_name, P_E, P_I, cond_params))

    n_cores = min(mp.cpu_count(), len(jobs))
    print("=" * 85)
    print(f"TVB QUICK TEST — Di Volo Full Pair ({len(jobs)} sims, {n_cores} cores)")
    print("=" * 85)
    print(f"Polynomials: {list(POLYNOMIALS.keys())}")
    print(f"Known: C1_PE+C1_PI (CONFIG1), DV_PE+C1_PI (mixed)")
    print()

    with mp.Pool(processes=n_cores) as pool:
        raw_results = pool.map(_worker, jobs)

    new_results = {}
    for poly_name, cond_name, lzc, mean_fr in raw_results:
        if poly_name not in new_results:
            new_results[poly_name] = {}
        new_results[poly_name][cond_name] = (lzc, mean_fr)

    conds = list(CONDITIONS.keys())
    print("\n" + "=" * 85)
    print("ALL RESULTS (known + new)")
    print("=" * 85)
    print(f"\n{'P_E + P_I':18s}", end='')
    for c in conds:
        print(f"  {c:>14s}", end='')
    print()
    print("-" * (18 + 16 * len(conds)))

    # Known results
    for name, data in KNOWN.items():
        print(f"{name:18s}", end='')
        for c in conds:
            lzc = data[c][0]
            print(f"  {lzc:>14.5f}", end='')
        print("  *")

    # New results
    for poly_name in POLYNOMIALS:
        if poly_name in new_results:
            print(f"{poly_name:18s}", end='')
            for c in conds:
                lzc, _ = new_results[poly_name][c]
                print(f"  {lzc:>14.5f}", end='')
            print()

    print("\n* = previously known")

    # Firing rates
    print(f"\n{'FR (Hz)':18s}", end='')
    for c in conds:
        print(f"  {c:>14s}", end='')
    print()
    print("-" * (18 + 16 * len(conds)))
    for poly_name in POLYNOMIALS:
        if poly_name in new_results:
            print(f"{poly_name:18s}", end='')
            for c in conds:
                _, fr = new_results[poly_name][c]
                print(f"  {fr:>14.2f}", end='')
            print()

    # Assessment
    print("\n" + "=" * 85)
    print("ASSESSMENT")
    print("=" * 85)
    for poly_name in POLYNOMIALS:
        if poly_name not in new_results:
            continue
        r = new_results[poly_name]
        lzc_a = r["Awake"][0]
        lzc_p = r["Propofol"][0]
        lzc_pd = r["Propofol+DOI"][0]
        lzc_d = r["DOI only"][0]

        print(f"\n{poly_name}:")
        print(f"  Prop collapse (P < 0.7*A): {'YES' if lzc_p < 0.7*lzc_a else 'NO'} "
              f"({lzc_p:.4f} vs {0.7*lzc_a:.4f})")
        print(f"  DOI reversal (P+D > P):    {'YES' if lzc_pd > lzc_p else 'NO'} "
              f"({lzc_pd:.4f} vs {lzc_p:.4f})")
        print(f"  DOI > Awake:               {'YES' if lzc_d > lzc_a else 'NO'} "
              f"({lzc_d:.5f} vs {lzc_a:.5f}, Δ={lzc_d-lzc_a:+.5f})")
        print(f"  Full ordering (P<P+D<A≤D): {'YES' if lzc_p < lzc_pd < lzc_a <= lzc_d else 'NO'}")

    print(f"\nWall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
