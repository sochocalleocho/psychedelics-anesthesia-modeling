#!/usr/bin/env python3
"""
tvb_b_e_sweep.py — Testing Option 2 (Biological Mechanism)

Tests CONFIG1 by running DOI as a combination of EL_e shift AND a reduction 
in excitatory adaptation (b_e). We vary new_b_e to see if dropping AHP 
sparks DOI > Awake.
"""

import os
import sys
import time
import warnings
import numpy as np

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

for p in [PIPELINE_HUB, MARTIN_REPO, SACHA_TOOLS, MARTIN_MODEL_SRC,
          os.path.join(PIPELINE_HUB, "TVB")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Biophysical constants
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

P_E_SACHA = np.array([-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
                       0.00341614, -0.01156433, 0.00194753,  0.00274079, -0.01066769])
P_I_SACHA = np.array([-0.05184978, 0.00615930, -0.01403522, 0.00166511, -0.00205590,
                       0.00318432, -0.03112775, 0.00656668,  0.00171829, -0.04516385])

SIM_LEN = 5000.0       # ms
DT = 0.1               # ms
ANALYSIS_LAST_MS = 1000.0
N_SEEDS = 4

# ---------------------------------------------------------------------------
# Single simulation runner
# ---------------------------------------------------------------------------
def run_one_sim(params, seed, conn, n_regions):
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
        tau_e  = np.array([5.0]),
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
        P_e = P_E_SACHA,
        P_i = P_I_SACHA,
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
    from tvbsim.entropy.measures_functions import calculate_LempelZiv
    n_analysis = int(ANALYSIS_LAST_MS / DT)
    return calculate_LempelZiv(fr[-n_analysis:, :])

def main():
    from tvb.datatypes import connectivity

    conn_path = os.path.join(MARTIN_CONN_DIR, MARTIN_CONN_FILE)
    conn = connectivity.Connectivity.from_file(conn_path)
    conn.configure()
    conn.weights = conn.weights / (np.sum(conn.weights, axis=0) + 1e-12)
    conn.speed = np.array([4.0])
    n_regions = conn.weights.shape[0]

    # Baseline DOI condition
    BASE_DOI_VALS = {
        "Awake": {
            "b_e": 5.0, "tau_i": 5.0,
            "g_K_e": np.array([G_K_E_BASE]), "g_K_i": np.array([G_K_I_BASE]),
        },
        "Propofol": {
            "b_e": 30.0, "tau_i": 7.0,
            "g_K_e": np.array([G_K_E_BASE]), "g_K_i": np.array([G_K_I_BASE]),
        }
    }
    
    # Sweep values for b_e during DOI
    B_E_VALS  = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    print("=" * 90)
    print("TVB QUICK TEST — Option 2: Decreased b_e for DOI (CONFIG1, N_SEEDS=4)")
    print("=" * 90)

    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as executor:
        print(f"\n{'─' * 40} AWAKE & PROPOFOL BASELINES {'─' * 40}")
        base_results = {}
        for cond_name, params in BASE_DOI_VALS.items():
            lzc_seeds = []
            fr_seeds = []
            futures = [executor.submit(run_one_sim, params, seed, conn, n_regions) for seed in range(N_SEEDS)]
            for future in concurrent.futures.as_completed(futures):
                times, fr = future.result()
                fr_hz = fr * 1000.0
                mean_fr = fr_hz[-10000:, :].mean()
                lzc = compute_lzc(fr)
                lzc_seeds.append(lzc)
                fr_seeds.append(mean_fr)
            base_results[cond_name] = np.mean(lzc_seeds)
            print(f"    {cond_name:14s}: LZc={np.mean(lzc_seeds):.4f}±{np.std(lzc_seeds):.4f}  FR={np.mean(fr_seeds):.2f} Hz", flush=True)
            
        print(f"\n{'─' * 40} DOI SWEEP (b_e drops) {'─' * 40}", flush=True)
        for b_val in B_E_VALS:
            print(f"\n  [DOI b_e = {b_val}]", flush=True)
            for drug_cond in ["Propofol+DOI", "DOI only"]:
                params = {
                    "b_e": b_val if drug_cond == "DOI only" else 30.0,  # Only drop in pure DOI? or does P+DOI keep high AHP?
                    "tau_i": 7.0 if "Propofol" in drug_cond else 5.0,
                    "g_K_e": _GK_E_DOI, "g_K_i": _GK_I_DOI,
                }
                
                # If doing P+DOI, assume b_e drops from 30 --> 25, 26 etc. For simplicity, just test pure DOI right now to see if we can beat awake.
                if drug_cond == "Propofol+DOI":
                    params["b_e"] = 30.0 - (5.0 - b_val)  # Drop proportionately
                
                lzc_seeds = []
                fr_seeds = []
                futures = [executor.submit(run_one_sim, params, seed, conn, n_regions) for seed in range(N_SEEDS)]
                for future in concurrent.futures.as_completed(futures):
                    times, fr = future.result()
                    fr_hz = fr * 1000.0
                    mean_fr = fr_hz[-10000:, :].mean()
                    lzc = compute_lzc(fr)
                    lzc_seeds.append(lzc)
                    fr_seeds.append(mean_fr)
                    print(f"      - Seed complete: LZc={lzc:.4f}  FR={mean_fr:.2f} Hz", flush=True)
                
                mean_lzc = np.mean(lzc_seeds)
                std_lzc = np.std(lzc_seeds)
                diff = mean_lzc - base_results["Awake"] if drug_cond == "DOI only" else mean_lzc - base_results["Propofol"]
                print(f"    {drug_cond:14s}: LZc={mean_lzc:.4f}±{std_lzc:.4f}  FR={np.mean(fr_seeds):.2f} Hz   (Diff from base = {diff:+.4f})", flush=True)

if __name__ == "__main__":
    t_start = time.time()
    main()
