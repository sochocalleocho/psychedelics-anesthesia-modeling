#!/usr/bin/env python3
"""
diagnose_bistability.py
-----------------------
Diagnoses why the 3-regime TF fails to produce propofol up-down oscillations
(bistability) in the Zerlaut mean-field model.

Two analyses:
  1. Time-domain ODE: integrate 1st-order mean-field for 10 s, compare time
     series of E(t) under CONFIG1 (P[5]=0.0034) vs 3-regime (P[5]=0.031) TF.
  2. Nullcline sweep: plot TF_e(E) - E along the E-nullcline (I set by TF_i=I)
     for multiple P[5] values to find the bistability threshold.

Bistability mechanism: propofol (b_e=30, tau_i=7) produces slow up-down
oscillations when the E nullcline has an S-curve (cubic shape) → TF_e(E)-E
crosses zero 3 times. Adaptation W_e drives slow switching between up/down
states. CONFIG1 has P[5]=0.0034 → S-curve; 3-regime P[5]=0.031 → no S-curve.

All functions are self-contained (no project imports needed).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import brentq
import os

# ─── Output directory ──────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── TF polynomial coefficients ────────────────────────────────────────────────
# Sacha's CONFIG1 — b_e=30, EL_e=-64, tau_i=5 (fits propofol single-neuron data)
# P[5] (sigma_V^2 coeff) = 0.0034 → produces propofol bistability
P_E_SACHA = np.array([-0.05017034,  0.00451531, -0.00794377, -0.00208418, -0.00054697,
                       0.00341614, -0.01156433,  0.00194753,  0.00274079, -0.01066769])
P_I_SACHA = np.array([-0.05184978,  0.00615930, -0.01403522,  0.00166511, -0.00205590,
                       0.00318432, -0.03112775,  0.00656668,  0.00171829, -0.04516385])

# 3-regime combined fit — awake + propofol + DOI-adj
# P[5] (sigma_V^2 coeff) = 0.0312 → NO propofol bistability
P_E_COMBINED = np.array([-0.04834604, -0.00046560,  0.00262377, -0.00344897,  0.00145982,
                           0.03122459,  0.02661126,  0.00293044, -0.00002222, -0.00270516])
P_I_COMBINED = np.array([-0.04853483, -0.00156604,  0.00440974,  0.01627825,  0.00051037,
                           0.03907078,  0.03995453,  0.00864973, -0.02007251, -0.00020276])

# ─── Network/cell parameters (propofol TVB operating point) ────────────────────
# Matches FS-RS_prop from cell_library.py + TVB propofol condition
# b_e, tau_i set at TVB mean-field level (NOT the single-neuron TF-fitting level)
PRMS = {
    # SI conversion applied internally in moments()
    'Q_e':   1.5,    # nS — excitatory quantal conductance
    'Q_i':   5.0,    # nS — inhibitory quantal conductance
    'tau_e': 5.0,    # ms — excitatory synaptic time constant
    'tau_i': 7.0,    # ms — inhibitory (PROPOFOL: 7 ms)
    'E_e':   0.0,    # mV — excitatory reversal
    'E_i': -80.0,    # mV — inhibitory reversal
    'Cm':  200.0,    # pF — membrane capacitance
    'tau_w': 500.0,  # ms — adaptation time constant
    'Gl':   10.0,    # nS — leak conductance
    'gei':   0.2,    # fraction inhibitory
    'Ntot': 10000,   # total neurons
    'p_con': 0.05,   # connection probability
    'a_e':   0.0,    # nS — subthreshold adaptation
    'b_e':  30.0,    # pA — spike-triggered adaptation (PROPOFOL: 30 pA)
    'EL_e': -64.0,   # mV — excitatory leak reversal
    'b_i':   0.0,    # pA — inhibitory adaptation (zero)
    'EL_i': -65.0,   # mV — inhibitory leak reversal
}

# TVB integration time constant T (same for all conditions, Sacha uses 5 ms)
T_TVB_S = 5e-3   # 5 ms in seconds


# ─── Core mean-field functions (all in SI, firing rates in Hz) ─────────────────

def moments(E_Hz, I_Hz, W_e_A, p, cell_type='RS'):
    """
    Compute mean-field moments (mu_V, sig_V, tau_V, tauN_V) for population.

    Inputs (SI):
        E_Hz, I_Hz : excitatory/inhibitory firing rates [Hz]
        W_e_A      : adaptation current [A]  (0 for inhibitory)
        p          : params dict (natural units: mV, pA, nS, ms, pF)
        cell_type  : 'RS' (excitatory) or 'FS' (inhibitory)

    Returns: mu_V [V], sig_V [V], tau_V [s], tauN_V [dimensionless]
    """
    Q_e   = p['Q_e']   * 1e-9    # nS → S
    Q_i   = p['Q_i']   * 1e-9
    tau_e = p['tau_e'] * 1e-3    # ms → s
    tau_i = p['tau_i'] * 1e-3
    E_e   = p['E_e']   * 1e-3    # mV → V
    E_i   = p['E_i']   * 1e-3
    Cm    = p['Cm']    * 1e-12   # pF → F
    Tw    = p['tau_w'] * 1e-3    # ms → s (not used if W explicit)
    g_L   = p['Gl']    * 1e-9    # nS → S
    gei   = p['gei']
    ntot  = p['Ntot']
    pcon  = p['p_con']

    if cell_type == 'RS':
        b    = p['b_e'] * 1e-12    # pA → A
        a    = p['a_e'] * 1e-9     # nS → S
        EL   = p['EL_e'] * 1e-3    # mV → V
    else:  # FS
        b    = p['b_i'] * 1e-12
        a    = p['a_i'] * 1e-9 if 'a_i' in p else 0.0
        EL   = p['EL_i'] * 1e-3
        W_e_A = 0.0

    fe = E_Hz * (1. - gei) * pcon * ntot   # [1/s]
    fi = I_Hz * gei * pcon * ntot

    mu_Ge = fe * tau_e * Q_e    # [S]
    mu_Gi = fi * tau_i * Q_i
    mu_G  = mu_Ge + mu_Gi + g_L
    tau_eff = Cm / mu_G         # [s]

    # Mean voltage (W in A, a_e=0 so skip subthreshold term)
    mu_V  = (mu_Ge * E_e + mu_Gi * E_i + g_L * EL - W_e_A) / mu_G   # [V]

    U_e = Q_e / mu_G * (E_e - mu_V)   # [V]
    U_i = Q_i / mu_G * (E_i - mu_V)

    sig_V2 = (fe * (U_e * tau_e)**2 / (2 * (tau_eff + tau_e)) +
              fi * (U_i * tau_i)**2 / (2 * (tau_eff + tau_i)))
    sig_V  = np.sqrt(np.maximum(sig_V2, 1e-30))

    numer  = fe * (U_e * tau_e)**2 + fi * (U_i * tau_i)**2
    denom  = (fe * (U_e * tau_e)**2 / (tau_eff + tau_e) +
              fi * (U_i * tau_i)**2 / (tau_eff + tau_i))
    tau_V  = numer / (denom + 1e-30)       # [s]
    tauN_V = tau_V * g_L / Cm              # dimensionless

    return mu_V, sig_V, tau_V, tauN_V


def V_eff_thr(mu_V, sig_V, tauN_V, P):
    """Zerlaut polynomial threshold [V]. Matches Sacha's threshold_func."""
    mu_0, mu_d   = -60e-3, 0.01
    sig_0, sig_d = 4e-3,   6e-3
    tau_0, tau_d = 0.5,    1.0
    V = (mu_V   - mu_0)  / mu_d
    S = (sig_V  - sig_0) / sig_d
    T = (tauN_V - tau_0) / tau_d
    return (P[0] + P[1]*V + P[2]*S + P[3]*T +
            P[4]*V**2 + P[5]*S**2 + P[6]*T**2 +
            P[7]*V*S  + P[8]*V*T  + P[9]*S*T)


def output_rate_Hz(P, E_Hz, I_Hz, W_e_A, p, cell_type='RS'):
    """TF output firing rate [Hz]."""
    mu_V, sig_V, tau_V, tauN_V = moments(E_Hz, I_Hz, W_e_A, p, cell_type)
    veff = V_eff_thr(mu_V, sig_V, tauN_V, P)
    # Clamp to avoid NaN: sig_V can be tiny at zero firing
    if np.ndim(sig_V) == 0:
        if sig_V < 1e-10:
            return 0.0
    else:
        sig_V = np.maximum(sig_V, 1e-10)
    rate = erfc((veff - mu_V) / (np.sqrt(2) * sig_V)) / (2 * tau_V)
    return np.maximum(rate, 0.0)


# ─── 1. Time-domain ODE integration ───────────────────────────────────────────

def integrate_mf(P_E, P_I, p, T_s=30.0, dt_s=5e-5, noise_amp=0.5,
                 E0=5.0, I0=5.0, W0_pA=0.0, seed=42, Fe_ext_Hz=0.0):
    """
    Integrate 1st-order Zerlaut mean-field ODE for a single region.

    ODE (with optional background coupling drive Fe_ext_Hz):
        T * dE/dt = F_e(E+Fe_ext, I, W_e) - E  + noise
        T * dI/dt = F_i(E+Fe_ext, I, 0)   - I  + noise
        dW_e/dt   = -W_e/tau_w + b_e * E

    Fe_ext_Hz: background long-range coupling drive [Hz]
               (~0.3 * E_avg from network, typically 2-5 Hz in propofol state)
    Units: E, I in Hz; W_e in A; time in s.
    """
    rng = np.random.default_rng(seed)
    tau_w  = p['tau_w'] * 1e-3    # ms → s
    b_e    = p['b_e']   * 1e-12   # pA → A
    T      = T_TVB_S               # 5 ms

    n_steps = int(T_s / dt_s)
    t_arr   = np.arange(n_steps) * dt_s
    E_arr   = np.zeros(n_steps)
    I_arr   = np.zeros(n_steps)
    W_arr   = np.zeros(n_steps)

    E, I, W = float(E0), float(I0), float(W0_pA) * 1e-12

    sqrt_dt = np.sqrt(dt_s)

    for k in range(n_steps):
        E_arr[k] = E
        I_arr[k] = I
        W_arr[k] = W * 1e12   # store in pA

        E_eff = max(E, 0.0) + Fe_ext_Hz
        Fe = output_rate_Hz(P_E, E_eff, max(I, 0.0), W, p, 'RS')
        Fi = output_rate_Hz(P_I, E_eff, max(I, 0.0), 0.0, p, 'FS')

        dE = (Fe - E) / T * dt_s + noise_amp * rng.standard_normal() * sqrt_dt
        dI = (Fi - I) / T * dt_s + noise_amp * rng.standard_normal() * sqrt_dt
        dW = (-W / tau_w + b_e * max(E, 0.0)) * dt_s

        E = max(E + dE, 0.0)
        I = max(I + dI, 0.0)
        W = W + dW

    return t_arr, E_arr, I_arr, W_arr


# ─── 2. Nullcline analysis ──────────────────────────────────────────────────────

def find_I_nullcline(E_Hz, W_e_pA, P_I, p, Fe_ext_Hz=0.0):
    """Find I satisfying TF_i(E_Hz+Fe_ext, I, 0) = I (inhibitory nullcline).

    Fe_ext_Hz: background coupling drive added to excitatory input [Hz].
    """
    E_eff = E_Hz + Fe_ext_Hz   # effective excitatory drive seen by FS cells

    def residual(I):
        return output_rate_Hz(P_I, E_eff, max(I, 0.1), 0.0, p, 'FS') - I

    # Scan for sign change across a fine grid to handle non-monotonic residuals
    I_grid = np.concatenate([np.linspace(0.1, 30, 60), np.linspace(30, 200, 40)])
    r_grid = np.array([residual(i) for i in I_grid])
    sign_changes = np.where(np.diff(np.sign(r_grid)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    # Return the FIRST root (lowest I) — the stable inhibitory fixed point
    idx = sign_changes[0]
    try:
        return brentq(residual, I_grid[idx], I_grid[idx+1], xtol=0.01, maxiter=100)
    except Exception:
        return None


def E_nullcline_gain(E_Hz, W_e_pA, P_E, P_I, p, Fe_ext_Hz=0.0):
    """Compute TF_e(E+Fe_ext, I_null(E), W_e) - E along the inhibitory nullcline."""
    I_null = find_I_nullcline(E_Hz, W_e_pA, P_I, p, Fe_ext_Hz)
    if I_null is None:
        return np.nan
    W = W_e_pA * 1e-12
    E_eff = E_Hz + Fe_ext_Hz
    Fe = output_rate_Hz(P_E, E_eff, I_null, W, p, 'RS')
    return Fe - E_Hz


# ─── 3. P[5] bistability sweep ─────────────────────────────────────────────────

def count_zero_crossings(E_arr, gain_arr):
    """Count sign changes in gain_arr (= number of E-nullcline zero crossings)."""
    valid = ~np.isnan(gain_arr)
    g = gain_arr[valid]
    crosses = np.sum(np.diff(np.sign(g)) != 0)
    return crosses


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("BISTABILITY DIAGNOSTIC")
    print("Propofol condition: b_e=30 pA, tau_i=7 ms, EL_e=-64 mV")
    print("KEY INSIGHT: bistability needs network coupling drive to sustain up-state")
    print("=" * 65)

    # ── Panel 1 & 2: Time-domain with/without coupling (30 s each) ──────────
    # TVB coupling_a=0.3; at E_avg~10 Hz in propofol: Fe_ext ≈ 0.3*10 = 3 Hz.
    # We sweep Fe_ext = 0, 3, 5 Hz to find where dynamics diverge.
    Fe_ext_vals = [0.0, 3.0, 5.0]
    print(f"\n[1/4] Running time-domain ODEs (30 s each, Fe_ext = {Fe_ext_vals} Hz)...")
    results = {}
    for Fe_ext in Fe_ext_vals:
        ts, Es, Is, Ws = integrate_mf(P_E_SACHA, P_I_SACHA, PRMS,
                                       T_s=30.0, noise_amp=0.5, seed=42,
                                       Fe_ext_Hz=Fe_ext)
        tc, Ec, Ic, Wc = integrate_mf(P_E_COMBINED, P_I_COMBINED, PRMS,
                                       T_s=30.0, noise_amp=0.5, seed=42,
                                       Fe_ext_Hz=Fe_ext)
        results[Fe_ext] = dict(ts=ts, Es=Es, tc=tc, Ec=Ec)
        frac_s  = len(ts) // 5  # discard first 20% as transient
        print(f"  Fe_ext={Fe_ext:.0f} Hz | "
              f"CONFIG1: E={Es[frac_s:].mean():.1f}±{Es[frac_s:].std():.1f} Hz  "
              f"3-regime: E={Ec[frac_s:].mean():.1f}±{Ec[frac_s:].std():.1f} Hz")

    # ── Panel 3: Nullclines with coupling (W_e swept for propofol up-state) ──
    print("\n[2/4] Computing E-nullclines with Fe_ext=3 Hz (coupling drive)...")
    E_scan = np.linspace(0.1, 60.0, 150)
    W_vals_pA = [0.0, 100.0, 200.0]
    Fe_ext_nc  = 3.0    # Hz — representative propofol coupling

    null_sacha = {w: np.array([E_nullcline_gain(e, w, P_E_SACHA, P_I_SACHA,
                                                  PRMS, Fe_ext_nc)
                                for e in E_scan]) for w in W_vals_pA}
    null_comb  = {w: np.array([E_nullcline_gain(e, w, P_E_COMBINED, P_I_COMBINED,
                                                  PRMS, Fe_ext_nc)
                                for e in E_scan]) for w in W_vals_pA}

    for w in W_vals_pA:
        zs = count_zero_crossings(E_scan, null_sacha[w])
        zc = count_zero_crossings(E_scan, null_comb[w])
        print(f"  W_e={w:5.0f} pA  CONFIG1: {zs} crossings  |  3-regime: {zc} crossings")

    # ── Panel 4: P[5] sweep at several coupling strengths ───────────────────
    print("\n[3/4] Sweeping P_E[5] at Fe_ext = 0, 3, 5 Hz...")
    P5_vals = np.linspace(0.001, 0.040, 25)
    sweep_results = {}  # {Fe_ext: [n_crossings at each p5]}

    for Fe_ext in [0.0, 3.0, 5.0]:
        ncross = []
        for p5 in P5_vals:
            P_mod = P_E_COMBINED.copy()
            P_mod[5] = p5
            g0 = np.array([E_nullcline_gain(e, 0.0, P_mod, P_I_COMBINED,
                                             PRMS, Fe_ext) for e in E_scan])
            ncross.append(count_zero_crossings(E_scan, g0))
        sweep_results[Fe_ext] = ncross
        bistable = np.array(ncross) >= 3
        thr_str = f"{P5_vals[bistable][-1]:.4f}" if bistable.any() else "none"
        print(f"  Fe_ext={Fe_ext:.0f} Hz: bistability (≥3 crossings) up to P[5] ≈ {thr_str}")

    print(f"\n  CONFIG1 P[5] = {P_E_SACHA[5]:.5f}")
    print(f"  3-regime P[5] = {P_E_COMBINED[5]:.5f}")

    # ── Panel 5: Coupling sweep — at what Fe_ext does propofol bistability appear? ──
    print("\n[4/4] Sweeping coupling drive Fe_ext (CONFIG1 vs 3-regime)...")
    Fe_sweep = np.linspace(0.0, 8.0, 30)
    cross_sacha = []
    cross_comb  = []
    for Fe in Fe_sweep:
        gs = np.array([E_nullcline_gain(e, 100.0, P_E_SACHA, P_I_SACHA, PRMS, Fe)
                       for e in E_scan])
        gc = np.array([E_nullcline_gain(e, 100.0, P_E_COMBINED, P_I_COMBINED, PRMS, Fe)
                       for e in E_scan])
        cross_sacha.append(count_zero_crossings(E_scan, gs))
        cross_comb.append(count_zero_crossings(E_scan, gc))

    s_arr = np.array(cross_sacha)
    c_arr = np.array(cross_comb)
    if (s_arr >= 3).any():
        print(f"  CONFIG1:   ≥3 crossings first at Fe_ext ≈ {Fe_sweep[(s_arr>=3)][0]:.2f} Hz")
    if (c_arr >= 3).any():
        print(f"  3-regime:  ≥3 crossings first at Fe_ext ≈ {Fe_sweep[(c_arr>=3)][0]:.2f} Hz")
    print("  → Gap shows the 'missing' coupling needed for 3-regime to reach bistability")

    # ── Plotting ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Bistability Diagnostic: Propofol (b_e=30 pA, τ_i=7 ms, coupling_a=0.3)",
                 fontsize=13)

    gs_layout = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # Row 1: Time series at Fe_ext=0, 3, 5 Hz for CONFIG1
    for col, Fe_ext in enumerate([0.0, 3.0, 5.0]):
        ax = fig.add_subplot(gs_layout[0, col])
        ts = results[Fe_ext]['ts']
        Es = results[Fe_ext]['Es']
        Ec = results[Fe_ext]['Ec']
        ax.plot(ts, Es, color='#d62728', lw=0.6, alpha=0.8, label='CONFIG1')
        ax.plot(ts, Ec, color='#1f77b4', lw=0.6, alpha=0.8, label='3-regime')
        ax.set_title(f"Time series  Fe_ext={Fe_ext:.0f} Hz", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("E (Hz)", fontsize=8)
        ax.set_ylim(-5, 60)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # Row 2: Nullclines with Fe_ext=3 Hz
    colors_w = ['#2ca02c', '#ff7f0e', '#9467bd']
    for col, (label, nulls) in enumerate([("CONFIG1", null_sacha), ("3-regime", null_comb)]):
        ax = fig.add_subplot(gs_layout[1, col])
        ax.axhline(0, color='k', lw=0.8, ls='--')
        for j, w in enumerate(W_vals_pA):
            g = nulls[w]
            ax.plot(E_scan, g, color=colors_w[j], label=f"W_e={w:.0f} pA", lw=1.5)
        ax.set_xlabel("E (Hz)", fontsize=8)
        ax.set_ylabel("TF_e − E (Hz)", fontsize=8)
        ax.set_title(f"{label}: E-nullcline (Fe_ext=3 Hz)", fontsize=9)
        ax.set_ylim(-20, 25)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Row 2 col 3: P[5] sweep at Fe_ext=3 Hz
    ax = fig.add_subplot(gs_layout[1, 2])
    colors_fe = ['#7f7f7f', '#ff7f0e', '#2ca02c']
    for i, (Fe_ext, col) in enumerate(zip([0.0, 3.0, 5.0], colors_fe)):
        ax.plot(P5_vals, sweep_results[Fe_ext], 'o-', color=col,
                label=f"Fe_ext={Fe_ext:.0f} Hz", ms=4)
    ax.axvline(P_E_SACHA[5],    color='#d62728', ls='--', lw=1.5,
               label=f"CONFIG1 ({P_E_SACHA[5]:.4f})")
    ax.axvline(P_E_COMBINED[5], color='#1f77b4', ls='--', lw=1.5,
               label=f"3-regime ({P_E_COMBINED[5]:.4f})")
    ax.axhline(3, color='gray', ls=':', label="3 = bistable")
    ax.set_xlabel("P_E[5]  (σ_V² coeff)", fontsize=8)
    ax.set_ylabel("# zero crossings", fontsize=8)
    ax.set_title("P[5] bistability threshold", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # Row 3: Coupling sweep
    ax = fig.add_subplot(gs_layout[2, 0])
    ax.plot(Fe_sweep, cross_sacha, 'o-', color='#d62728', label='CONFIG1')
    ax.plot(Fe_sweep, cross_comb,  's-', color='#1f77b4', label='3-regime')
    ax.axhline(3, color='gray', ls=':', label="3 = bistable")
    ax.set_xlabel("Coupling drive Fe_ext (Hz)", fontsize=8)
    ax.set_ylabel("# zero crossings  (W_e=100 pA)", fontsize=8)
    ax.set_title("Bistability vs coupling strength", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # Row 3: E-distribution histograms at Fe_ext=3 Hz
    ax = fig.add_subplot(gs_layout[2, 1])
    Fe_ext = 3.0
    Es3 = results[Fe_ext]['Es']
    Ec3 = results[Fe_ext]['Ec']
    skip = len(Es3)//5
    ax.hist(Es3[skip:], bins=60, color='#d62728', alpha=0.5,
            label=f"CONFIG1", density=True)
    ax.hist(Ec3[skip:], bins=60, color='#1f77b4', alpha=0.5,
            label=f"3-regime", density=True)
    ax.set_xlabel("E (Hz)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(f"E distribution  Fe_ext={Fe_ext:.0f} Hz", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)

    # Row 3 col 3: W_e time series at Fe_ext=3 Hz
    ax = fig.add_subplot(gs_layout[2, 2])
    tc3 = results[3.0]['tc']
    Ws_3 = np.array([])   # placeholder — redo quick run to get W
    # Re-run to get W (quick, no noise)
    ts_q, Es_q, Is_q, Ws_q = integrate_mf(P_E_SACHA, P_I_SACHA, PRMS,
                                            T_s=30.0, noise_amp=0.5, seed=42,
                                            Fe_ext_Hz=3.0)
    tc_q, Ec_q, Ic_q, Wc_q = integrate_mf(P_E_COMBINED, P_I_COMBINED, PRMS,
                                            T_s=30.0, noise_amp=0.5, seed=42,
                                            Fe_ext_Hz=3.0)
    ax.plot(ts_q, Ws_q, color='#d62728', lw=0.6, alpha=0.8, label='CONFIG1')
    ax.plot(tc_q, Wc_q, color='#1f77b4', lw=0.6, alpha=0.8, label='3-regime')
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("W_e (pA)", fontsize=8)
    ax.set_title("Adaptation W_e(t)  Fe_ext=3 Hz", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)

    outpath = os.path.join(FIGURES_DIR, "bistability_diagnostic.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {outpath}")
    print("\n── SUMMARY ──────────────────────────────────────────────────────────")
    print("If CONFIG1 reaches ≥3 crossings at lower Fe_ext than 3-regime:")
    print("  → 3-regime TF needs a HIGHER coupling drive to show bistability")
    print("  → Solution: constrain P[5] in fitting OR adjust coupling_a in TVB")
    print("If both reach ≥3 crossings at same Fe_ext:")
    print("  → Bistability is the same — the LZc difference comes from ELSEWHERE")
    print("    (e.g., firing rate in up-state, adaptation strength, noise)")
    print("─────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
