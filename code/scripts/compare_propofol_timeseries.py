#!/usr/bin/env python3
"""
compare_propofol_timeseries.py
------------------------------
Runs a SHORT TVB simulation (propofol condition only, 1 seed, no PCI) for
CONFIG1 vs 3-regime TF, saves and plots the E(t) time series.

This directly answers: what does propofol activity look like with each TF?
Expected:
  CONFIG1:   slow up-down oscillations → binarised signal = slow square wave → low LZc
  3-regime:  sustained moderate activity → noisy signal → high LZc

Usage:
  python compare_propofol_timeseries.py
"""

import os, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PIPELINE_HUB = os.path.join(PROJECT_ROOT, "paper_pipeline_hub")
MARTIN_REPO  = os.path.join(PROJECT_ROOT, "simulated_serotonergic_receptors_tvb")
MARTIN_MODEL_SRC = os.path.join(MARTIN_REPO, "tvbsim", "TVB", "tvb_model_reference", "src")
MARTIN_CONN_DIR  = os.path.join(MARTIN_REPO, "data", "connectivity", "DK68")

sys.path.insert(0, MARTIN_MODEL_SRC)
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Conductance setup (same as production script) ────────────────────────────
E_NA, E_K, G_L = 50.0, -90.0, 10.0

def _conversion(E_Na, E_K, E_L, g_L=None, g_Na=None):
    if g_L is not None:
        g_K = g_L * (E_L - E_Na) / (E_K - E_Na)
        return g_K, g_L - g_K
    g_L_new = g_Na * (E_Na - E_K) / (E_L - E_K)
    return g_L_new - g_Na, g_Na

G_K_E_BASE, G_NA_E = _conversion(E_NA, E_K, -64.0, g_L=G_L)
G_K_I_BASE, G_NA_I = _conversion(E_NA, E_K, -65.0, g_L=G_L)

# ─── TF coefficients ──────────────────────────────────────────────────────────
P_E_SACHA = np.array([-0.05017034,  0.00451531, -0.00794377, -0.00208418, -0.00054697,
                       0.00341614, -0.01156433,  0.00194753,  0.00274079, -0.01066769])
P_I_SACHA = np.array([-0.05184978,  0.00615930, -0.01403522,  0.00166511, -0.00205590,
                       0.00318432, -0.03112775,  0.00656668,  0.00171829, -0.04516385])

P_E_COMBINED = np.array([-0.04834604, -0.00046560,  0.00262377, -0.00344897,  0.00145982,
                           0.03122459,  0.02661126,  0.00293044, -0.00002222, -0.00270516])
P_I_COMBINED = np.array([-0.04853483, -0.00156604,  0.00440974,  0.01627825,  0.00051037,
                           0.03907078,  0.03995453,  0.00864973, -0.02007251, -0.00020276])

# ─── Simulation parameters ────────────────────────────────────────────────────
SIM_LEN   = 5000.0   # ms — total sim (2s transient + 3s useful)
DT        = 0.1      # ms
MON_DT    = 0.1      # ms
CUT_TRANS = 2000.0   # ms — discard as transient
SEED      = 42


def run_tvb_propofol(P_E, P_I, label=""):
    """Run TVB propofol condition (1 seed, no stimulus) and return E time series."""
    from tvb.simulator import simulator, coupling, integrators, monitors, noise
    import Zerlaut_gK_gNa as zg

    print(f"  Loading connectivity...")
    from tvb.datatypes.connectivity import Connectivity
    conn = Connectivity.from_file(
        os.path.join(MARTIN_CONN_DIR, "connectivity_68_QL20120814.zip"))
    conn.configure()
    conn.weights /= conn.weights.sum(axis=0, keepdims=True)  # column-sum normalise
    n_reg = conn.number_of_regions

    print(f"  Building TVB model ({label}, {n_reg} regions)...")
    model = zg.Zerlaut_adaptation_second_order(
        g_K_e  = np.array([G_K_E_BASE]),
        g_Na_e = np.array([G_NA_E]),
        g_K_i  = np.array([G_K_I_BASE]),
        g_Na_i = np.array([G_NA_I]),
        E_K_e  = np.array([E_K]),
        E_Na_e = np.array([E_NA]),
        E_K_i  = np.array([E_K]),
        E_Na_i = np.array([E_NA]),
        C_m    = np.array([200.0]),
        b_e    = np.array([30.0]),   # propofol
        a_e    = np.array([0.0]),
        b_i    = np.array([0.0]),
        a_i    = np.array([0.0]),
        tau_w_e= np.array([500.0]),
        tau_w_i= np.array([1.0]),
        tau_e  = np.array([5.0]),
        tau_i  = np.array([7.0]),    # propofol
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

    coupl  = coupling.Linear(a=np.array([0.3]), b=np.array([0.0]))
    nsig   = np.array([0., 0., 0., 0., 0., 0., 0., 1.0])
    integ  = integrators.HeunStochastic(dt=DT,
               noise=noise.Additive(nsig=nsig))
    mon    = monitors.Raw()
    sim    = simulator.Simulator(model=model, connectivity=conn, coupling=coupl,
                                  integrator=integ, monitors=(mon,))
    sim.configure()

    ic = np.zeros((8, n_reg, 1))
    ic[5, :, 0] = 100.0
    sim.current_state[:] = ic
    sim.integrator.noise.random_stream.seed(SEED)

    print(f"  Running {SIM_LEN:.0f} ms simulation...")
    t0 = time.time()
    raw = sim.run(simulation_length=SIM_LEN)
    print(f"  Done in {time.time()-t0:.1f} s")

    times = raw[0][0].flatten()                # ms
    E_kHz = raw[0][1][:, 0, :, 0]             # (n_steps, n_regions) in kHz

    # Discard transient
    mask  = times >= CUT_TRANS
    times = times[mask] - CUT_TRANS
    E_kHz = E_kHz[mask, :]

    return times, E_kHz * 1000.0   # → Hz


def lzc_estimate(E_Hz_arr, thresh_pct=50.0):
    """Quick LZc estimate: binarise each region at median, concatenate, LZ complexity."""
    try:
        from antropy import lziv_complexity
    except ImportError:
        return None

    E_bin = (E_Hz_arr > np.percentile(E_Hz_arr, thresh_pct, axis=0)).astype(int)
    flat  = E_bin.flatten().astype(str).tolist()
    seq   = ''.join(flat)
    return lziv_complexity(seq, normalize=True)


def main():
    print("=" * 60)
    print("PROPOFOL TVB TIME-SERIES COMPARISON")
    print("CONFIG1  (P[5]=0.0034)  vs  3-regime (P[5]=0.0312)")
    print("=" * 60)

    print("\n[1/2] CONFIG1 TF (Sacha's original)...")
    ts_s, E_s = run_tvb_propofol(P_E_SACHA, P_I_SACHA, label="CONFIG1")

    print("\n[2/2] 3-regime TF (combined fit)...")
    ts_c, E_c = run_tvb_propofol(P_E_COMBINED, P_I_COMBINED, label="3-regime")

    # ── Basic stats ──────────────────────────────────────────────────────────
    print("\n── Statistics ──────────────────────────────────────────────────")
    for lbl, E in [("CONFIG1", E_s), ("3-regime", E_c)]:
        mean_all = E.mean()
        std_all  = E.std()
        # Mean E per region (heterogeneity)
        reg_means = E.mean(axis=0)
        print(f"  {lbl}: global E = {mean_all:.2f} ± {std_all:.2f} Hz  |  "
              f"region mean range: [{reg_means.min():.2f}, {reg_means.max():.2f}] Hz")

    # ── LZc if antropy available ──────────────────────────────────────────────
    lzc_s = lzc_estimate(E_s)
    lzc_c = lzc_estimate(E_c)
    if lzc_s is not None:
        print(f"\n  Quick LZc: CONFIG1={lzc_s:.4f}  3-regime={lzc_c:.4f}")

    # ── Plotting ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Propofol TVB Time Series: CONFIG1 vs 3-regime TF", fontsize=12)

    t_s_arr = ts_s / 1000.0   # ms → s
    t_c_arr = ts_c / 1000.0

    # Row 1: Mean E(t) across all regions
    for col, (lbl, t_arr, E_arr, col_hex) in enumerate([
            ("CONFIG1", t_s_arr, E_s, '#d62728'),
            ("3-regime", t_c_arr, E_c, '#1f77b4')]):
        ax = axes[0, col]
        mean_E = E_arr.mean(axis=1)
        ax.plot(t_arr, mean_E, color=col_hex, lw=0.7, alpha=0.9)
        ax.set_title(f"{lbl}: mean E(t) across 68 regions", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("E (Hz)")

    # Row 2: Individual region traces (a few representative regions)
    rng = np.random.default_rng(0)
    sample_regs = rng.choice(E_s.shape[1], size=min(10, E_s.shape[1]), replace=False)
    for col, (lbl, t_arr, E_arr, col_hex) in enumerate([
            ("CONFIG1", t_s_arr, E_s, '#d62728'),
            ("3-regime", t_c_arr, E_c, '#1f77b4')]):
        ax = axes[1, col]
        for r in sample_regs:
            ax.plot(t_arr, E_arr[:, r], color=col_hex, lw=0.4, alpha=0.4)
        ax.plot(t_arr, E_arr.mean(axis=1), color='k', lw=1.2, label='mean')
        ax.set_title(f"{lbl}: 10 sampled regions + mean", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("E (Hz)")
        ax.legend(fontsize=8)

    # Row 3: E distribution + adaptation W_e (if available)
    for col, (lbl, E_arr, col_hex) in enumerate([
            ("CONFIG1", E_s, '#d62728'), ("3-regime", E_c, '#1f77b4')]):
        ax = axes[2, col]
        ax.hist(E_arr.flatten(), bins=80, color=col_hex, alpha=0.7, density=True)
        ax.set_xlabel("E (Hz)")
        ax.set_ylabel("Density")
        ax.set_title(f"{lbl}: E distribution (all regions × time)", fontsize=10)
        mn = E_arr.mean()
        ax.axvline(mn, color='k', ls='--', label=f"mean={mn:.1f} Hz")
        ax.legend(fontsize=8)

    plt.tight_layout()
    outpath = os.path.join(FIGURES_DIR, "propofol_timeseries_comparison.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {outpath}")

    # Save raw data for further analysis
    npy_path = os.path.join(FIGURES_DIR, "propofol_timeseries_data.npz")
    np.savez(npy_path, ts_s=ts_s, E_s=E_s, ts_c=ts_c, E_c=E_c)
    print(f"Data saved:   {npy_path}")


if __name__ == "__main__":
    main()
