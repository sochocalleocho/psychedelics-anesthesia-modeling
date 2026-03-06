#!/usr/bin/env python3
"""
Polynomial interpolation sweep: P_hybrid = α × P_E_SACHA + (1-α) × P_E_DIVOLO

Based on tvb_quick_test_divolo_full.py (VALIDATED against lzc_results_cache.json).
Uses Martin's Zerlaut_gK_gNa model, DK68 connectivity, all production parameters.

Tests α = 0.0, 0.1, 0.2, ..., 1.0 (11 values)
Each runs 4 drug conditions with seed=0 (quick single-seed screening).
"""

import os, sys, time, warnings
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

# ============================================================
# Constants (matching production & validated quick tests)
# ============================================================
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

# DOI conductances (EL_e → -61.2, EL_i → -64.4)
G_K_E_PSI, _ = _conversion(E_NA, E_K, -61.2, g_Na=G_NA_E)
G_K_I_PSI, _ = _conversion(E_NA, E_K, -64.4, g_Na=G_NA_I)

# 5-HT2A density map (region-specific DOI effect)
density_file = os.path.join(MARTIN_REPO, "data", "receptors", "DK68", "5HT2a_reordered.txt")
_DENSITIES_68 = np.clip(np.loadtxt(density_file), 0, None)
d_max = _DENSITIES_68.max()
_GK_E_DOI = np.interp(_DENSITIES_68, [0, d_max], [G_K_E_BASE, G_K_E_PSI])
_GK_I_DOI = np.interp(_DENSITIES_68, [0, d_max], [G_K_I_BASE, G_K_I_PSI])

CONDITIONS = {
    "Awake":        {"b_e": 5.0,  "tau_e": 5.0, "tau_i": 5.0,
                     "g_K_e": np.array([G_K_E_BASE]), "g_K_i": np.array([G_K_I_BASE])},
    "Propofol":     {"b_e": 30.0, "tau_e": 5.0, "tau_i": 7.0,
                     "g_K_e": np.array([G_K_E_BASE]), "g_K_i": np.array([G_K_I_BASE])},
    "Propofol+DOI": {"b_e": 30.0, "tau_e": 5.0, "tau_i": 7.0,
                     "g_K_e": _GK_E_DOI, "g_K_i": _GK_I_DOI},
    "DOI only":     {"b_e": 5.0,  "tau_e": 5.0, "tau_i": 5.0,
                     "g_K_e": _GK_E_DOI, "g_K_i": _GK_I_DOI},
}

SIM_LEN = 5000.0; DT = 0.1; ANALYSIS_LAST_MS = 1000.0; SEED = 0

# ============================================================
# Polynomials (MATCHING production scripts — validated values)
# ============================================================
P_E_SACHA = np.array([-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
                       0.00341614, -0.01156433, 0.00194753,  0.00274079, -0.01066769])
P_I_SACHA = np.array([-0.05184978, 0.00615930, -0.01403522, 0.00166511, -0.00205590,
                       0.00318432, -0.03112775, 0.00656668,  0.00171829, -0.04516385])

P_E_DIVOLO = np.array([-0.04983106, 0.005063550882777035, -0.023470121807314552,
                         0.0022951513725067503, -0.0004105302652029825, 0.010547051343547399,
                         -0.03659252821136933, 0.007437487505797858, 0.001265064721846073,
                         -0.04072161294490446])
P_I_DIVOLO = np.array([-0.05149122024209484, 0.004003689190271077, -0.008352013668528155,
                         0.0002414237992765705, -0.0005070645080016026, 0.0014345394104282397,
                         -0.014686689498949967, 0.004502706285435741, 0.0028472190352532454,
                         -0.015357804594594548])


def run_single(P_E, P_I, cond_name, cond_params):
    """Run single condition, return (lzc, mean_fr)."""
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
    ic[5, :, 0] = 100.0  # W_e
    sim.current_state[:] = ic
    sim.integrator.noise.random_stream.seed(SEED)

    raw = sim.run(simulation_length=SIM_LEN)
    fr = raw[0][1][:, 0, :, 0]
    fr_hz = fr * 1000.0
    mean_fr = fr_hz[-10000:, :].mean()
    n_analysis = int(ANALYSIS_LAST_MS / DT)
    lzc = calculate_LempelZiv(fr[-n_analysis:, :])

    return lzc, mean_fr


def main():
    alphas = np.arange(0, 1.05, 0.1)  # 0.0, 0.1, ..., 1.0

    print("=" * 80, flush=True)
    print("POLYNOMIAL INTERPOLATION SWEEP", flush=True)
    print("  P_hybrid = α × CONFIG1 + (1-α) × DiVolo", flush=True)
    print(f"  α values: {[f'{a:.1f}' for a in alphas]}", flush=True)
    print(f"  Model: Zerlaut_gK_gNa (Martin's variant)", flush=True)
    print(f"  Connectivity: DK68, seed = {SEED}", flush=True)
    print("=" * 80, flush=True)

    # Show P[2] interpolation
    print(f"\nP[2] values across α:", flush=True)
    for a in alphas:
        p2 = a * P_E_SACHA[2] + (1 - a) * P_E_DIVOLO[2]
        print(f"  α={a:.1f}: P[2]={p2:+.6f}", flush=True)
    print(flush=True)

    all_results = []

    for i, alpha in enumerate(alphas):
        P_e = alpha * P_E_SACHA + (1 - alpha) * P_E_DIVOLO
        P_i = alpha * P_I_SACHA + (1 - alpha) * P_I_DIVOLO

        p2 = P_e[2]
        label = f"α={alpha:.1f} (P[2]={p2:+.4f})"
        print(f"\n[{i+1}/{len(alphas)}] Running {label}...", flush=True)
        t0 = time.time()

        row = {'alpha': alpha, 'P2_e': p2, 'P6_e': P_e[6]}

        try:
            for cond_name, cond_params in CONDITIONS.items():
                lzc, mean_fr = run_single(P_e, P_i, cond_name, cond_params)
                row[f'{cond_name}_lzc'] = lzc
                row[f'{cond_name}_fr'] = mean_fr
                print(f"  {cond_name:14s}: LZc={lzc:.5f}  FR={mean_fr:.2f} Hz", flush=True)

            elapsed = time.time() - t0

            doi_minus_awake = row['DOI only_lzc'] - row['Awake_lzc']
            prop_collapse = row['Propofol_lzc'] < 0.7
            full_ordering = (row['Propofol_lzc'] < row['Propofol+DOI_lzc'] <
                           row['Awake_lzc'] < row['DOI only_lzc'])

            status = ""
            if prop_collapse:
                status += "COLLAPSE "
            if doi_minus_awake > 0.001:
                status += "DOI>AWAKE "
            if full_ordering:
                status += "FULL_ORDER "

            print(f"  DOI-Awake={doi_minus_awake:+.4f}  {status}  ({elapsed:.1f}s)", flush=True)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            row['error'] = str(e)

        all_results.append(row)

    # Summary table
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY TABLE", flush=True)
    print("=" * 80, flush=True)
    print(f"{'α':>5s} {'P[2]':>8s} {'Awake':>7s} {'Prop':>7s} {'P+DOI':>7s} "
          f"{'DOI':>7s} {'D-A':>7s} {'Collapse':>8s} {'DOI>A':>6s} {'Order':>6s}", flush=True)
    print("-" * 80, flush=True)

    for row in all_results:
        if 'error' in row:
            print(f"{row['alpha']:5.1f} {row['P2_e']:>+8.4f}  ERROR: {row['error'][:40]}", flush=True)
            continue

        doi_a = row['DOI only_lzc'] - row['Awake_lzc']
        collapse = "YES" if row['Propofol_lzc'] < 0.7 else "no"
        doi_above = "YES" if doi_a > 0.001 else "no"
        full_ord = "YES" if (row['Propofol_lzc'] < row['Propofol+DOI_lzc'] <
                             row['Awake_lzc'] < row['DOI only_lzc']) else "no"

        print(f"{row['alpha']:5.1f} {row['P2_e']:>+8.4f} "
              f"{row['Awake_lzc']:7.3f} {row['Propofol_lzc']:7.3f} "
              f"{row['Propofol+DOI_lzc']:7.3f} {row['DOI only_lzc']:7.3f} "
              f"{doi_a:>+7.4f} {collapse:>8s} {doi_above:>6s} {full_ord:>6s}", flush=True)

    # Find sweet spots
    print("\n" + "=" * 80, flush=True)
    print("SWEET SPOT ANALYSIS", flush=True)
    print("=" * 80, flush=True)

    valid = [r for r in all_results if 'error' not in r]
    if not valid:
        print("No valid results!", flush=True)
        return

    for row in valid:
        row['_collapse_score'] = max(0, 0.7 - row['Propofol_lzc'])
        row['_doi_score'] = max(0, row['DOI only_lzc'] - row['Awake_lzc'])
        row['_combined'] = row['_collapse_score'] * row['_doi_score']

    best = max(valid, key=lambda r: r['_combined'])
    print(f"\nBest combined (collapse × DOI>Awake):", flush=True)
    print(f"  α={best['alpha']:.1f}  P[2]={best['P2_e']:+.4f}", flush=True)
    print(f"  Propofol LZc={best['Propofol_lzc']:.3f}  "
          f"DOI-Awake={best['DOI only_lzc']-best['Awake_lzc']:+.4f}", flush=True)

    # Save results
    save_dict = {}
    for key in ['alpha', 'P2_e', 'Awake_lzc', 'Propofol_lzc',
                'Propofol+DOI_lzc', 'DOI only_lzc',
                'Awake_fr', 'Propofol_fr', 'Propofol+DOI_fr', 'DOI only_fr']:
        save_dict[key.replace('+', '_plus_').replace(' ', '_')] = \
            [r.get(key, np.nan) for r in all_results]
    np.savez('interpolation_sweep_results.npz', **save_dict)

    print("\nSaved: interpolation_sweep_results.npz", flush=True)
    print("\nDONE.", flush=True)


if __name__ == '__main__':
    main()
