#!/usr/bin/env python3
"""
tvb_EL_sweep.py — Sweep E_L_e baseline with CONFIG1 polynomial.

Martin et al. 2025 uses E_L_e = -63.0 mV as baseline.
Sacha et al. 2025 uses E_L_e = -64.0 mV.
DOI always shifts E_L_e to -61.2 mV.

Hypothesis: Changing baseline E_L_e may amplify DOI sensitivity
while preserving CONFIG1's propofol collapse.
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

# Constants
E_NA = 50.0; E_K = -90.0; G_L = 10.0
E_L_I_BASE = -65.0  # inhibitory stays fixed
E_L_E_DOI  = -61.2  # DOI target (always the same)
E_L_I_DOI  = -64.4  # DOI target for inhibitory

# CONFIG1 polynomial (Sacha)
P_E_SACHA = np.array([-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
                       0.00341614, -0.01156433, 0.00194753,  0.00274079, -0.01066769])
P_I_SACHA = np.array([-0.05184978, 0.00615930, -0.01403522, 0.00166511, -0.00205590,
                       0.00318432, -0.03112775, 0.00656668,  0.00171829, -0.04516385])

def _conversion(E_Na, E_K, E_L, g_L=None, g_Na=None):
    """Convert between (g_L, E_L) and (g_K, g_Na) representations."""
    if g_L is not None:
        g_K = g_L * (E_L - E_Na) / (E_K - E_Na)
        return g_K, g_L - g_K
    g_L_new = g_Na * (E_Na - E_K) / (E_L - E_K)
    return g_L_new - g_Na, g_Na

# 5-HT2A receptor density for DOI regional variation
density_file = os.path.join(MARTIN_REPO, "data", "receptors", "DK68", "5HT2a_reordered.txt")
_DENSITIES_68 = np.clip(np.loadtxt(density_file), 0, None)
d_max = _DENSITIES_68.max()

# Sweep E_L_e baseline values
EL_E_VALUES = [-65.0, -64.5, -64.0, -63.5, -63.0, -62.5, -62.0]

def build_conditions(el_e_base):
    """Build the 4 drug conditions for a given E_L_e baseline."""
    # Baseline conductances
    g_K_e_base, g_Na_e = _conversion(E_NA, E_K, el_e_base, g_L=G_L)
    g_K_i_base, g_Na_i = _conversion(E_NA, E_K, E_L_I_BASE, g_L=G_L)

    # DOI conductances (g_Na stays fixed, only g_K changes via E_L shift)
    g_K_e_doi, _ = _conversion(E_NA, E_K, E_L_E_DOI, g_Na=g_Na_e)
    g_K_i_doi, _ = _conversion(E_NA, E_K, E_L_I_DOI, g_Na=g_Na_i)

    # Regional DOI variation
    gk_e_doi = np.interp(_DENSITIES_68, [0, d_max], [g_K_e_base, g_K_e_doi])
    gk_i_doi = np.interp(_DENSITIES_68, [0, d_max], [g_K_i_base, g_K_i_doi])

    conds = {
        "Awake": {"b_e": 5.0, "tau_e": 5.0, "tau_i": 5.0,
                  "g_K_e": np.array([g_K_e_base]), "g_K_i": np.array([g_K_i_base]),
                  "g_Na_e": g_Na_e, "g_Na_i": g_Na_i},
        "Propofol": {"b_e": 30.0, "tau_e": 5.0, "tau_i": 7.0,
                     "g_K_e": np.array([g_K_e_base]), "g_K_i": np.array([g_K_i_base]),
                     "g_Na_e": g_Na_e, "g_Na_i": g_Na_i},
        "Propofol+DOI": {"b_e": 30.0, "tau_e": 5.0, "tau_i": 7.0,
                         "g_K_e": gk_e_doi, "g_K_i": gk_i_doi,
                         "g_Na_e": g_Na_e, "g_Na_i": g_Na_i},
        "DOI only": {"b_e": 5.0, "tau_e": 5.0, "tau_i": 5.0,
                     "g_K_e": gk_e_doi, "g_K_i": gk_i_doi,
                     "g_Na_e": g_Na_e, "g_Na_i": g_Na_i},
    }
    return conds

SIM_LEN = 5000.0; DT = 0.1; ANALYSIS_LAST_MS = 1000.0; SEED = 0

def _worker(args):
    el_e_base, cond_name, P_E, P_I, cond_params = args
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
        g_K_e=cond_params["g_K_e"], g_Na_e=np.array([cond_params["g_Na_e"]]),
        g_K_i=cond_params["g_K_i"], g_Na_i=np.array([cond_params["g_Na_i"]]),
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

    label = f"EL={el_e_base:.1f}"
    print(f"  DONE: {label:10s} / {cond_name:14s}  LZc={lzc:.5f}  FR={mean_fr:.2f} Hz",
          flush=True)
    return el_e_base, cond_name, lzc, mean_fr


def main():
    t_start = time.time()

    # Build all jobs
    jobs = []
    for el_e in EL_E_VALUES:
        conds = build_conditions(el_e)
        for cond_name, cond_params in conds.items():
            jobs.append((el_e, cond_name, P_E_SACHA, P_I_SACHA, cond_params))

    n_cores = min(mp.cpu_count(), len(jobs))
    print("=" * 90)
    print(f"TVB E_L_e BASELINE SWEEP — CONFIG1 polynomial ({len(jobs)} sims, {n_cores} cores)")
    print("=" * 90)
    print(f"E_L_e values: {EL_E_VALUES}")
    print(f"E_L_e DOI target: {E_L_E_DOI} mV (fixed)")
    print(f"Polynomial: CONFIG1 (P_E_SACHA + P_I_SACHA)")
    print()

    # Print conductance table
    print(f"{'E_L_e':>8s}  {'g_K_e':>8s}  {'g_Na_e':>8s}  {'g_K_DOI':>8s}  {'ΔE_L':>8s}")
    print("-" * 50)
    for el_e in EL_E_VALUES:
        g_K_base, g_Na = _conversion(E_NA, E_K, el_e, g_L=G_L)
        g_K_doi, _ = _conversion(E_NA, E_K, E_L_E_DOI, g_Na=g_Na)
        delta_el = E_L_E_DOI - el_e
        print(f"{el_e:>8.1f}  {g_K_base:>8.4f}  {g_Na:>8.4f}  {g_K_doi:>8.4f}  {delta_el:>+8.1f}")
    print()

    with mp.Pool(processes=n_cores) as pool:
        raw_results = pool.map(_worker, jobs)

    # Organize results
    results = {}
    for el_e, cond_name, lzc, mean_fr in raw_results:
        if el_e not in results:
            results[el_e] = {}
        results[el_e][cond_name] = (lzc, mean_fr)

    conds = ["Awake", "Propofol", "Propofol+DOI", "DOI only"]

    # LZc table
    print("\n" + "=" * 90)
    print("LZc RESULTS")
    print("=" * 90)
    print(f"\n{'E_L_e':>8s}", end='')
    for c in conds:
        print(f"  {c:>14s}", end='')
    print(f"  {'DOI-Awake':>10s}  {'P collapse':>10s}  {'Ordering':>10s}")
    print("-" * (8 + 16*4 + 12*3))

    for el_e in EL_E_VALUES:
        if el_e not in results:
            continue
        r = results[el_e]
        lzc_a = r["Awake"][0]
        lzc_p = r["Propofol"][0]
        lzc_pd = r["Propofol+DOI"][0]
        lzc_d = r["DOI only"][0]

        collapse = "YES" if lzc_p < 0.7 * lzc_a else "no"
        ordering = "YES" if lzc_p < lzc_pd < lzc_a <= lzc_d else "no"
        doi_delta = lzc_d - lzc_a

        marker = " ◄" if el_e == -64.0 else (" ★" if el_e == -63.0 else "")

        print(f"{el_e:>8.1f}", end='')
        for c in conds:
            print(f"  {r[c][0]:>14.5f}", end='')
        print(f"  {doi_delta:>+10.5f}  {collapse:>10s}  {ordering:>10s}{marker}")

    # Firing rate table
    print(f"\n{'FR (Hz)':>8s}", end='')
    for c in conds:
        print(f"  {c:>14s}", end='')
    print()
    print("-" * (8 + 16*4))
    for el_e in EL_E_VALUES:
        if el_e not in results:
            continue
        r = results[el_e]
        print(f"{el_e:>8.1f}", end='')
        for c in conds:
            _, fr = r[c]
            print(f"  {fr:>14.2f}", end='')
        marker = " ◄ current" if el_e == -64.0 else (" ★ Martin" if el_e == -63.0 else "")
        print(marker)

    # Detailed assessment
    print("\n" + "=" * 90)
    print("ASSESSMENT")
    print("=" * 90)
    for el_e in EL_E_VALUES:
        if el_e not in results:
            continue
        r = results[el_e]
        lzc_a = r["Awake"][0]
        lzc_p = r["Propofol"][0]
        lzc_pd = r["Propofol+DOI"][0]
        lzc_d = r["DOI only"][0]

        print(f"\nE_L_e = {el_e:.1f} mV:")
        print(f"  Prop collapse (P < 0.7*A):  {'YES' if lzc_p < 0.7*lzc_a else 'NO'} "
              f"({lzc_p:.4f} vs {0.7*lzc_a:.4f})")
        print(f"  DOI reversal (P+D > P):     {'YES' if lzc_pd > lzc_p else 'NO'} "
              f"({lzc_pd:.4f} vs {lzc_p:.4f})")
        print(f"  DOI > Awake:                {'YES' if lzc_d > lzc_a else 'NO'} "
              f"({lzc_d:.5f} vs {lzc_a:.5f}, Δ={lzc_d-lzc_a:+.5f})")
        print(f"  Full ordering (P<P+D<A≤D):  {'YES' if lzc_p < lzc_pd < lzc_a <= lzc_d else 'NO'}")

    print(f"\nWall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
