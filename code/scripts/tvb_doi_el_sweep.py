"""
tvb_doi_el_sweep.py
-------------------
Sweep DOI E_L endpoint magnitude using the production script's infrastructure.

Hypothesis: Our DOI E_L endpoint (-61.2 mV) is too conservative. Martin's full
5-HT2A g_K reduction (8.21→5.37 nS) corresponds to effective E_L ≈ -54 mV
from our baseline (E_L_e=-64). We're applying only 12% of the intended g_K
reduction. This sweep tests whether a stronger DOI effect rescues DOI>Awake
while preserving propofol collapse.

Sweep: E_L_E_PSI ∈ [-61.2, -59, -57, -55, -54, -53]
       E_L_I_PSI scaled proportionally (same E/I ratio as current)
       CONFIG1 polynomial, Zerlaut_gK_gNa, heterogeneous 5-HT2A receptor map

For each E_L endpoint, runs all 4 conditions (Awake, Propofol, Propofol+DOI,
DOI only) with N_SEEDS seeds, LZc only (no PCI).

Usage:
  python tvb_doi_el_sweep.py                # full sweep (6 endpoints × 4 conditions)
  python tvb_doi_el_sweep.py --el -55       # single endpoint test
  python tvb_doi_el_sweep.py --seeds 4      # quick test with fewer seeds
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
# Path setup (same as production script)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
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
E_NA   = 50.0    # mV
E_K    = -90.0   # mV
G_L    = 10.0    # nS

E_L_E_BASE = -64.0   # mV -- excitatory baseline
E_L_I_BASE = -65.0   # mV -- inhibitory baseline

def _conversion(E_Na, E_K, E_L, g_L=None, g_Na=None):
    if g_L is not None:
        g_K = g_L * (E_L - E_Na) / (E_K - E_Na)
        g_Na_out = g_L - g_K
        return g_K, g_Na_out
    g_L_new = g_Na * (E_Na - E_K) / (E_L - E_K)
    g_K = g_L_new - g_Na
    return g_K, g_Na

# Baseline conductances
G_K_E_BASE, G_NA_E = _conversion(E_NA, E_K, E_L_E_BASE, g_L=G_L)
G_K_I_BASE, G_NA_I = _conversion(E_NA, E_K, E_L_I_BASE, g_L=G_L)

# Current DOI E/I shift ratio (for proportional I scaling)
_E_SHIFT_CURRENT = -61.2 - E_L_E_BASE   # +2.8
_I_SHIFT_CURRENT = -64.4 - E_L_I_BASE   # +0.6
_EI_RATIO = _I_SHIFT_CURRENT / _E_SHIFT_CURRENT  # ~0.214


def _load_5ht2a():
    density_file = os.path.join(
        MARTIN_REPO, "data", "receptors", "DK68", "5HT2a_reordered.txt")
    densities = np.loadtxt(density_file)
    return np.clip(densities, 0, None)


def _compute_gK_map(densities, E_L_E_PSI, E_L_I_PSI):
    """Compute per-region g_K arrays for a given DOI endpoint."""
    G_K_E_PSI, _ = _conversion(E_NA, E_K, E_L_E_PSI, g_Na=G_NA_E)
    G_K_I_PSI, _ = _conversion(E_NA, E_K, E_L_I_PSI, g_Na=G_NA_I)
    d_max = densities.max()
    n = len(densities)
    if d_max == 0:
        return np.full(n, G_K_E_BASE), np.full(n, G_K_I_BASE)
    g_K_e = np.interp(densities, [0, d_max], [G_K_E_BASE, G_K_E_PSI])
    g_K_i = np.interp(densities, [0, d_max], [G_K_I_BASE, G_K_I_PSI])
    return g_K_e, g_K_i


# ---------------------------------------------------------------------------
# Transfer Function Polynomial Coefficients (CONFIG1)
# ---------------------------------------------------------------------------
P_E = np.array([-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
                 0.00341614, -0.01156433, 0.00194753,  0.00274079, -0.01066769])
P_I = np.array([-0.05184978, 0.00615930, -0.01403522, 0.00166511, -0.00205590,
                 0.00318432, -0.03112775, 0.00656668,  0.00171829, -0.04516385])

# Simulation parameters
SIM_LEN  = 5000.0
DT       = 0.1
ANALYSIS_LAST_MS = 1000.0


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _run_one_sim(params, seed, conn, n_regions):
    import numpy as np
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
    fr = raw[0][1][:, 0, :, 0]
    return fr


def _worker(args):
    cond_name, params, seed, el_label = args
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    import sys, os

    for p in [PIPELINE_HUB, MARTIN_REPO, SACHA_TOOLS, MARTIN_MODEL_SRC,
              os.path.join(PIPELINE_HUB, "TVB")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from tvb.datatypes import connectivity
    from tvbsim.entropy.measures_functions import calculate_LempelZiv

    np.random.seed(seed)

    conn_path = os.path.join(MARTIN_CONN_DIR, MARTIN_CONN_FILE)
    conn = connectivity.Connectivity.from_file(conn_path)
    conn.configure()
    conn.weights = conn.weights / (np.sum(conn.weights, axis=0) + 1e-12)
    conn.speed = np.array([4.0])
    n_regions = conn.weights.shape[0]

    fr = _run_one_sim(params, seed, conn, n_regions)

    fr_hz = fr * 1000.0
    mean_fr = fr_hz[-10000:, :].mean()
    n_analysis = int(ANALYSIS_LAST_MS / DT)
    lzc = calculate_LempelZiv(fr[-n_analysis:, :])

    print(f"    {el_label} {cond_name} seed={seed}: FR={mean_fr:.1f}Hz LZc={lzc:.5f}",
          flush=True)
    return el_label, cond_name, seed, lzc, mean_fr


def main():
    parser = argparse.ArgumentParser(description="DOI E_L endpoint sweep")
    parser.add_argument("--el", type=float, default=None,
                        help="Single E_L_E_PSI value to test (default: sweep all)")
    parser.add_argument("--seeds", type=int, default=16,
                        help="Number of seeds (default: 16)")
    args = parser.parse_args()

    t_total = time.time()

    # Define sweep points
    if args.el is not None:
        el_e_values = [args.el]
    else:
        el_e_values = [-61.2, -59.0, -57.0, -55.0, -54.0, -53.0]

    n_seeds = args.seeds
    densities = _load_5ht2a()

    print(f"\n{'='*70}")
    print(f"DOI E_L Endpoint Sweep")
    print(f"{'='*70}")
    print(f"Baseline: E_L_e={E_L_E_BASE}, E_L_i={E_L_I_BASE}")
    print(f"Sweep E_L_E_PSI: {el_e_values}")
    print(f"Seeds: {n_seeds}, Polynomial: CONFIG1 (P_E_SACHA)")
    print(f"Model: Zerlaut_gK_gNa + heterogeneous 5-HT2A receptor map")
    print()

    # Print g_K info for each sweep point
    for el_e in el_e_values:
        e_shift = el_e - E_L_E_BASE
        i_shift = e_shift * _EI_RATIO
        el_i = E_L_I_BASE + i_shift
        g_K_e_psi, _ = _conversion(E_NA, E_K, el_e, g_Na=G_NA_E)
        g_L_doi = g_K_e_psi + G_NA_E
        pct_drop = (1 - g_K_e_psi / G_K_E_BASE) * 100
        print(f"  E_L_e={el_e:6.1f}: g_K_e={g_K_e_psi:.3f} ({pct_drop:.1f}% drop), "
              f"g_L={g_L_doi:.3f}, E_L_i={el_i:.2f}")
    print()

    # Build all jobs
    all_jobs = []
    for el_e in el_e_values:
        e_shift = el_e - E_L_E_BASE
        i_shift = e_shift * _EI_RATIO
        el_i = E_L_I_BASE + i_shift

        gK_e_doi, gK_i_doi = _compute_gK_map(densities, el_e, el_i)
        el_label = f"EL{el_e:.0f}"

        conditions = {
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
                "g_K_e": gK_e_doi,
                "g_K_i": gK_i_doi,
            },
            "DOI": {
                "b_e": 5.0, "tau_e": 5.0, "tau_i": 5.0,
                "g_K_e": gK_e_doi,
                "g_K_i": gK_i_doi,
            },
        }

        for cond_name, params in conditions.items():
            for s in range(n_seeds):
                all_jobs.append((cond_name, params, s, el_label))

    n_cores = min(mp.cpu_count(), len(all_jobs))
    print(f"Running {len(all_jobs)} jobs ({len(el_e_values)} endpoints × "
          f"4 conditions × {n_seeds} seeds) across {n_cores} cores...\n")

    with mp.Pool(processes=n_cores) as pool:
        raw_results = pool.map(_worker, all_jobs)

    # Aggregate results
    results = {}
    for el_label, cond_name, seed, lzc, mean_fr in raw_results:
        key = (el_label, cond_name)
        if key not in results:
            results[key] = {"lzc": [], "fr": []}
        results[key]["lzc"].append(lzc)
        results[key]["fr"].append(mean_fr)

    # Print summary table
    print(f"\n{'='*90}")
    print(f"RESULTS SUMMARY  (n={n_seeds} seeds each)")
    print(f"{'='*90}")
    print(f"{'E_L_e':>8}  {'Condition':<16}  {'LZc mean':>10}  {'LZc SEM':>10}  "
          f"{'FR mean':>10}  {'DOI-Awake':>10}")
    print("-" * 90)

    for el_e in el_e_values:
        el_label = f"EL{el_e:.0f}"
        awake_lzc = np.mean(results[(el_label, "Awake")]["lzc"])

        for cond_name in ["Awake", "Propofol", "Propofol+DOI", "DOI"]:
            key = (el_label, cond_name)
            lzc_arr = np.array(results[key]["lzc"])
            fr_arr = np.array(results[key]["fr"])
            lzc_mean = lzc_arr.mean()
            lzc_sem = lzc_arr.std() / np.sqrt(n_seeds) if n_seeds > 1 else 0
            fr_mean = fr_arr.mean()
            diff = lzc_mean - awake_lzc if cond_name != "Awake" else 0

            marker = ""
            if cond_name == "DOI" and diff > 0.001:
                marker = " ✓✓"
            elif cond_name == "DOI" and diff > 0:
                marker = " ✓"
            elif cond_name == "Propofol" and lzc_mean < 0.7:
                marker = " ✓✓"
            elif cond_name == "Propofol" and lzc_mean < 0.85:
                marker = " ✓"

            print(f"{el_e:>8.1f}  {cond_name:<16}  {lzc_mean:>10.5f}  {lzc_sem:>10.5f}  "
                  f"{fr_mean:>10.1f}  {diff:>+10.5f}{marker}")
        print()

    # Save raw results
    out_dir = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "figures"))
    os.makedirs(out_dir, exist_ok=True)

    save_dict = {}
    for (el_label, cond_name), data in results.items():
        prefix = f"{el_label}_{cond_name.replace('+','_').replace(' ','_')}"
        save_dict[f"{prefix}_lzc"] = np.array(data["lzc"])
        save_dict[f"{prefix}_fr"] = np.array(data["fr"])

    out_npz = os.path.join(out_dir, "doi_el_sweep_results.npz")
    np.savez(out_npz, **save_dict)
    print(f"Raw results saved -> {out_npz}")

    # Plot
    _plot_sweep(el_e_values, results, n_seeds, out_dir)

    print(f"\nTotal wall time: {(time.time()-t_total)/60:.1f} min")


def _plot_sweep(el_values, results, n_seeds, out_dir):
    """Generate sweep summary plot."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {"Awake": "#2ca02c", "Propofol": "#d62728",
              "Propofol+DOI": "#1f77b4", "DOI": "#9467bd"}

    # Panel 1: LZc vs E_L for all conditions
    ax = axes[0]
    for cond in ["Awake", "Propofol", "Propofol+DOI", "DOI"]:
        means = []
        sems = []
        for el_e in el_values:
            el_label = f"EL{el_e:.0f}"
            lzc_arr = np.array(results[(el_label, cond)]["lzc"])
            means.append(lzc_arr.mean())
            sems.append(lzc_arr.std() / np.sqrt(n_seeds) if n_seeds > 1 else 0)
        ax.errorbar(el_values, means, yerr=sems, marker='o', label=cond,
                    color=colors[cond], capsize=3, linewidth=2, markersize=6)
    ax.set_xlabel("DOI E_L endpoint (mV)")
    ax.set_ylabel("LZc")
    ax.set_title("LZc vs DOI Magnitude")
    ax.legend(fontsize=8)
    ax.axhline(y=means[0], color='gray', ls='--', alpha=0.3)  # rough Awake line

    # Panel 2: DOI - Awake difference
    ax = axes[1]
    doi_diffs = []
    doi_sems = []
    prop_doi_diffs = []
    for el_e in el_values:
        el_label = f"EL{el_e:.0f}"
        awake = np.array(results[(el_label, "Awake")]["lzc"])
        doi = np.array(results[(el_label, "DOI")]["lzc"])
        prop_doi = np.array(results[(el_label, "Propofol+DOI")]["lzc"])
        prop = np.array(results[(el_label, "Propofol")]["lzc"])
        doi_diffs.append((doi - awake).mean())
        doi_sems.append((doi - awake).std() / np.sqrt(n_seeds))
        prop_doi_diffs.append((prop_doi - prop).mean())

    ax.errorbar(el_values, doi_diffs, yerr=doi_sems, marker='s', color='#9467bd',
                label='DOI - Awake', capsize=3, linewidth=2, markersize=6)
    ax.plot(el_values, prop_doi_diffs, marker='^', color='#1f77b4',
            label='Prop+DOI - Prop', linewidth=2, markersize=6)
    ax.axhline(y=0, color='black', ls='-', alpha=0.3)
    ax.axhline(y=0.003, color='green', ls='--', alpha=0.5, label='Martin target (~0.003)')
    ax.set_xlabel("DOI E_L endpoint (mV)")
    ax.set_ylabel("ΔLZc")
    ax.set_title("DOI Effect Size")
    ax.legend(fontsize=8)

    # Panel 3: Propofol collapse check
    ax = axes[2]
    prop_means = []
    awake_means = []
    for el_e in el_values:
        el_label = f"EL{el_e:.0f}"
        prop_means.append(np.mean(results[(el_label, "Propofol")]["lzc"]))
        awake_means.append(np.mean(results[(el_label, "Awake")]["lzc"]))
    ax.plot(el_values, prop_means, marker='o', color='#d62728', label='Propofol',
            linewidth=2, markersize=6)
    ax.plot(el_values, awake_means, marker='o', color='#2ca02c', label='Awake',
            linewidth=2, markersize=6)
    ax.axhline(y=0.7, color='red', ls='--', alpha=0.5, label='Collapse threshold')
    ax.set_xlabel("DOI E_L endpoint (mV)")
    ax.set_ylabel("LZc")
    ax.set_title("Propofol Collapse Preserved?")
    ax.legend(fontsize=8)

    fig.suptitle("DOI E_L Endpoint Sweep — CONFIG1 + Zerlaut_gK_gNa + 5-HT2A map\n"
                 f"n={n_seeds} seeds per condition",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "doi_el_sweep.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sweep figure saved -> {out_path}")


if __name__ == "__main__":
    main()
