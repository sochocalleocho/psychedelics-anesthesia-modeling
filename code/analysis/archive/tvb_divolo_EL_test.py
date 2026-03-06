#!/usr/bin/env python3
"""
tvb_divolo_EL_test.py — Test DiVolo full pair at E_L=-65 (Martin's ACTUAL setup).

CRITICAL FINDING: Martin's parameter file says E_L_e=-63 but has g_K_e=8.214
which corresponds to E_L=-65. The Zerlaut_gK_gNa model uses g_K directly,
so Martin's effective E_L is -65, not -63.

Known results:
  - CONFIG1 @ EL=-65: DOI-Awake = -0.004 (from cache)
  - CONFIG1 @ EL=-64: DOI-Awake = -0.003 (from cache)
  - CONFIG1 @ EL=-63: DOI-Awake = +0.001 (from cache)
  - DiVolo_full @ EL=-64: DOI-Awake = +0.008 (from cache)
  - DiVolo_full @ EL=-63: DOI-Awake = +0.001 (from cache)
  - DiVolo_full @ EL=-65: ??? (THIS TEST — Martin's actual config)

At EL=-65, DOI shift is larger (ΔE_L = 3.8 mV) → expect bigger DOI effect.
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
E_L_I_BASE = -65.0
E_L_E_DOI  = -61.2
E_L_I_DOI  = -64.4

P_E_DIVOLO = np.array([-0.04983106, 0.005063550882777035, -0.023470121807314552,
                         0.0022951513725067503, -0.0004105302652029825, 0.010547051343547399,
                         -0.03659252821136933, 0.007437487505797858, 0.001265064721846073,
                         -0.04072161294490446])
P_I_DIVOLO = np.array([-0.05149122024209484, 0.004003689190271077, -0.008352013668528155,
                         0.0002414237992765705, -0.0005070645080016026, 0.0014345394104282397,
                         -0.014686689498949967, 0.004502706285435741, 0.0028472190352532454,
                         -0.015357804594594548])

def _conversion(E_Na, E_K, E_L, g_L=None, g_Na=None):
    if g_L is not None:
        g_K = g_L * (E_L - E_Na) / (E_K - E_Na)
        return g_K, g_L - g_K
    g_L_new = g_Na * (E_Na - E_K) / (E_L - E_K)
    return g_L_new - g_Na, g_Na

density_file = os.path.join(MARTIN_REPO, "data", "receptors", "DK68", "5HT2a_reordered.txt")
_DENSITIES_68 = np.clip(np.loadtxt(density_file), 0, None)
d_max = _DENSITIES_68.max()

# Test E_L=-65 (Martin's actual effective E_L from g_K values)
EL_VALUES = [-65.0]

SIM_LEN = 5000.0; DT = 0.1; ANALYSIS_LAST_MS = 1000.0; SEED = 0

def build_conditions(el_e_base):
    g_K_e_base, g_Na_e = _conversion(E_NA, E_K, el_e_base, g_L=G_L)
    g_K_i_base, g_Na_i = _conversion(E_NA, E_K, E_L_I_BASE, g_L=G_L)
    g_K_e_doi, _ = _conversion(E_NA, E_K, E_L_E_DOI, g_Na=g_Na_e)
    g_K_i_doi, _ = _conversion(E_NA, E_K, E_L_I_DOI, g_Na=g_Na_i)
    gk_e_doi = np.interp(_DENSITIES_68, [0, d_max], [g_K_e_base, g_K_e_doi])
    gk_i_doi = np.interp(_DENSITIES_68, [0, d_max], [g_K_i_base, g_K_i_doi])
    return {
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

def _worker(args):
    label, cond_name, P_E, P_I, cond_params = args
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

    print(f"  DONE: {label:20s} / {cond_name:14s}  LZc={lzc:.5f}  FR={mean_fr:.2f} Hz",
          flush=True)
    return label, cond_name, lzc, mean_fr


def main():
    t_start = time.time()

    jobs = []
    for el_e in EL_VALUES:
        conds = build_conditions(el_e)
        for cond_name, cond_params in conds.items():
            jobs.append((f"DiVolo_EL={el_e}", cond_name,
                         P_E_DIVOLO, P_I_DIVOLO, cond_params))

    n_cores = min(mp.cpu_count(), len(jobs))
    print("=" * 85)
    print(f"DiVolo @ E_L=-65 TEST — Martin's actual config ({len(jobs)} sims, {n_cores} cores)")
    print("=" * 85)
    print()

    with mp.Pool(processes=n_cores) as pool:
        raw_results = pool.map(_worker, jobs)

    # Organize
    results = {}
    for label, cond_name, lzc, mean_fr in raw_results:
        if label not in results:
            results[label] = {}
        results[label][cond_name] = (lzc, mean_fr)

    conds = ["Awake", "Propofol", "Propofol+DOI", "DOI only"]

    # Comparison table: all known configs
    print("\n" + "=" * 85)
    print("COMPLETE COMPARISON: DOI SENSITIVITY BY POLYNOMIAL × E_L BASELINE")
    print("=" * 85)

    known = {
        "CONFIG1 @ EL=-65":  {"Awake": 0.94443, "Propofol": 0.50831,
                               "Propofol+DOI": 0.77352, "DOI only": 0.94058},
        "CONFIG1 @ EL=-64":  {"Awake": 0.94470, "Propofol": 0.53224,
                               "Propofol+DOI": 0.73039, "DOI only": 0.94150},
        "CONFIG1 @ EL=-63":  {"Awake": 0.94287, "Propofol": 0.90930,
                               "Propofol+DOI": 0.93930, "DOI only": 0.94430},
        "DiVolo  @ EL=-64":  {"Awake": 0.93585, "Propofol": 0.93632,
                               "Propofol+DOI": 0.94034, "DOI only": 0.94392},
        "DiVolo  @ EL=-63":  {"Awake": 0.94377, "Propofol": 0.93566,
                               "Propofol+DOI": 0.94024, "DOI only": 0.94498},
    }

    # Add new results
    for label, cond_data in results.items():
        row = {}
        for c in conds:
            row[c] = cond_data[c][0]
        known[f"DiVolo  @ EL=-65"] = row

    print(f"\n{'Config':25s}", end='')
    for c in conds:
        print(f"  {c:>14s}", end='')
    print(f"  {'DOI-Awake':>10s}  {'P collapse':>10s}")
    print("-" * (25 + 16*4 + 12*2))

    for name, data in known.items():
        doi_delta = data["DOI only"] - data["Awake"]
        collapse = "YES" if data["Propofol"] < 0.7 * data["Awake"] else "no"
        print(f"{name:25s}", end='')
        for c in conds:
            print(f"  {data[c]:>14.5f}", end='')
        print(f"  {doi_delta:>+10.5f}  {collapse:>10s}")

    print(f"\nWall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
