#!/usr/bin/env python3
"""
tvb_coupling_sweep.py — Parameter sweep of global network coupling 'a'.

Tests the Di Volo polynomial against multiple global coupling parameters 
(a = [0.3, 0.6, 0.9, 1.2]) across all 4 drug conditions to assess its capacity 
for achieving Propofol collapse alongside the known DOI > Awake effect.
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
P_E_SACHA = np.array([-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
                       0.00341614, -0.01156433, 0.00194753,  0.00274079, -0.01066769])
P_I_SACHA = np.array([-0.05184978, 0.00615930, -0.01403522, 0.00166511, -0.00205590,
                       0.00318432, -0.03112775, 0.00656668,  0.00171829, -0.04516385])

P_E_DIVOLO = np.array([-0.04983106, 0.005063550882777035, -0.023470121807314552,
                         0.0022951513725067503, -0.0004105302652029825, 0.010547051343547399,
                         -0.03659252821136933, 0.007437487505797858, 0.001265064721846073,
                         -0.04072161294490446])

POLYNOMIALS = {
    "DiVolo": (P_E_DIVOLO, P_I_SACHA),
    "CONFIG1": (P_E_SACHA, P_I_SACHA),
}

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SIM_LEN = 5000.0       # ms
DT = 0.1               # ms
ANALYSIS_LAST_MS = 1000.0
# Only checking a few seeds for sweep to save compute
N_SEEDS = 4

# ---------------------------------------------------------------------------
# Single simulation runner
# ---------------------------------------------------------------------------
def run_one_sim(params, P_E, P_I, seed, conn, n_regions, coupling_a):
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

    coupl = coupling.Linear(a=np.array([coupling_a]), b=np.array([0.0]))
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

    C_VALS = [0.3, 0.6, 0.9, 1.2]
    
    print("=" * 90)
    print("TVB QUICK TEST — Sweeping coupling strength a (N_SEEDS=4)")
    print("=" * 90)

    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as executor:
        for poly_name, (P_E, P_I) in POLYNOMIALS.items():
            print(f"\n{'─' * 40} POLYNOMIAL: {poly_name} {'─' * 40}")
            for c_val in C_VALS:
                print(f"\n  [coupling_a = {c_val}]")
                for cond_name, params in CONDITIONS.items():
                    lzc_seeds = []
                    fr_seeds = []
                    t0 = time.time()
                    
                    futures = [executor.submit(run_one_sim, params, P_E, P_I, seed, conn, n_regions, c_val) for seed in range(N_SEEDS)]
                    
                    for future in concurrent.futures.as_completed(futures):
                        times, fr = future.result()
                        fr_hz = fr * 1000.0
                        mean_fr = fr_hz[-10000:, :].mean()
                        lzc = compute_lzc(fr)
                        
                        lzc_seeds.append(lzc)
                        fr_seeds.append(mean_fr)
                        print(f"      - Seed complete: LZc={lzc:.4f}  FR={mean_fr:.2f} Hz", flush=True)
                    
                    mean_lzc = np.mean(lzc_seeds)
                    mean_fr_cond = np.mean(fr_seeds)
                    std_lzc = np.std(lzc_seeds)
                    elapsed = time.time() - t0
                    print(f"    {cond_name:14s}: LZc={mean_lzc:.4f}±{std_lzc:.4f}  FR={mean_fr_cond:.2f} Hz  ({elapsed:.0f}s)", flush=True)

if __name__ == "__main__":
    t_start = time.time()
    main()
