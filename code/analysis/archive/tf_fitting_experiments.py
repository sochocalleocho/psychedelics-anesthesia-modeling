#!/usr/bin/env python3
"""
TF Fitting Experiments A-D: Isolate what causes CONFIG1 vs Di Volo polynomial difference.

Experiments:
  A: Sacha's b_e=0 data (EL=-64) + Di Volo single-pass fit
  B: divolo_el65 data (EL=-65) + Di Volo single-pass fit
  C: divolo_el65 data (EL=-65) + Sacha loop_n=10 fit  [Option A already did this → P[2]=-0.0075]
  D: Sacha's b_e=0 data (EL=-64) + Sacha loop_n=10 fit  [should ≈ CONFIG1]

Also generates NEW Brian2 training data at EL=-65 (Di Volo cell) for comparison:
  E: Brian2 data (EL=-65) + Di Volo single-pass fit
  F: Brian2 data (EL=-65) + Sacha loop_n=10 fit

Plus statistical comparison of training data.
"""
import sys, os
import numpy as np
from scipy.optimize import minimize
from scipy.special import erfc, erfcinv

# === PATHS ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
TF_CALC_DIR = os.path.join(CODE_DIR, 'paper_pipeline_hub', 'Tf_calc')
DATA_DIR = os.path.join(TF_CALC_DIR, 'data')

# NOTE: We inline the 3 needed functions from theoretical_tools.py to avoid the
# import chain: theoretical_tools → functions → brian2. These are exact copies
# from code/paper_pipeline_hub/Tf_calc/theoretical_tools.py (lines 23-91).

def eff_thresh(mu_V, sig_V, tauN_V, params):
    """Compute effective threshold from 10-coefficient polynomial."""
    P_0, P_mu, P_sig, P_tau, P_mu2, P_sig2, P_tau2, P_mu_sig, P_mu_tau, P_sig_tau = params
    mu_0 = -60.0*1e-3; mu_d = 0.01
    sig_0 = 0.004; sig_d = 0.006
    tau_0 = 0.5; tau_d = 1.0
    V1 = (P_mu*(mu_V - mu_0) / mu_d +
          P_sig*(sig_V - sig_0) / sig_d +
          P_tau*(tauN_V - tau_0) / tau_d)
    V2 = (P_mu2*((mu_V - mu_0) / mu_d)**2 +
          P_sig2*((sig_V - sig_0) / sig_d)**2 +
          P_tau2*((tauN_V - tau_0) / tau_d)**2 +
          P_mu_sig*((mu_V - mu_0) / mu_d)*((sig_V - sig_0) / sig_d) +
          P_mu_tau*((mu_V - mu_0) / mu_d)*((tauN_V - tau_0) / tau_d) +
          P_sig_tau*((sig_V - sig_0) / sig_d)*((tauN_V - tau_0) / tau_d))
    return P_0 + V1 + V2

def mu_sig_tau_func(fexc, finh, fout, w_ad, params, cell_type, w_prec=False):
    """Compute subthreshold moments (mu_V, sig_V, tau_V, tauN_V)."""
    p = params
    Q_e = p['Q_e']*1e-9; Q_i = p['Q_i']*1e-9
    tau_e = p['tau_e']*1e-3; tau_i = p['tau_i']*1e-3
    E_e = p['E_e']*1e-3; E_i = p['E_i']*1e-3
    C_m = p['Cm']*1e-12; Tw = p['tau_w']*1e-3; g_L = p['Gl']*1e-9
    gei = p['gei']; ntot = p['Ntot']; pconnec = p['p_con']
    if cell_type == "RS":
        try:
            a, b, E_L = p['a_e']*1e-9, p['b_e']*1e-12, p['EL_e']*1e-3
        except KeyError:
            a, b, E_L = p['a']*1e-9, p['b']*1e-12, p['EL']*1e-3
    elif cell_type == "FS":
        try:
            a, b, E_L = p['a_i']*1e-9, p['b_i']*1e-12, p['EL_i']*1e-3
        except KeyError:
            a, b, E_L = p['a']*1e-9, p['b']*1e-12, p['EL']*1e-3
    f_e = fexc*(1.-gei)*pconnec*ntot
    f_i = finh*gei*pconnec*ntot
    mu_Ge = f_e*tau_e*Q_e
    mu_Gi = f_i*tau_i*Q_i
    mu_G = mu_Ge + mu_Gi + g_L
    tau_eff = C_m / mu_G
    if w_prec:
        mu_V = (mu_Ge*E_e + mu_Gi*E_i + g_L*E_L - w_ad) / mu_G
    else:
        mu_V = (mu_Ge*E_e + mu_Gi*E_i + g_L*E_L - fout*Tw*b + a*E_L) / mu_G
    U_e = Q_e / mu_G*(E_e - mu_V)
    U_i = Q_i / mu_G*(E_i - mu_V)
    sig_V = np.sqrt(f_e*(U_e*tau_e)**2 / (2*(tau_eff + tau_e)) +
                    f_i*(U_i*tau_i)**2 / (2*(tau_eff + tau_i)))
    tau_V = ((f_e*(U_e*tau_e)**2 + f_i*(U_i*tau_i)**2) /
             (f_e*(U_e*tau_e)**2 / (tau_eff + tau_e) +
              f_i*(U_i*tau_i)**2 / (tau_eff + tau_i)))
    tauN_V = tau_V*g_L / C_m
    return mu_V, sig_V, tau_V, tauN_V

def output_rate(params, mu_V, sig_V, tau_V, tauN_V):
    """Compute firing rate from polynomial threshold + erfc."""
    return erfc((eff_thresh(mu_V, sig_V, tauN_V, params) - mu_V) /
                (np.sqrt(2)*sig_V)) / (2*tau_V)

# === REFERENCE POLYNOMIALS ===
P_E_CONFIG1 = np.array([-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
                         0.00341614, -0.01156433, 0.00194753,  0.00274079, -0.01066769])
P_E_DIVOLO = np.array([-0.04983106, 0.005063550882777035, -0.023470121807314552,
                        0.0022951513725135524, -0.00041053262984498075, 0.010547051584498958,
                        -0.036592527674092555, 0.007437492568787784, 0.0012650625252625887,
                        -0.04072161028792653])

# === NORMALIZATION CONSTANTS (same in all codebases) ===
MU0 = -60e-3; DMU = 10e-3; SIG0 = 4e-3; DSIG = 6e-3; TAU0 = 0.5; DTAU = 1.0


def load_sacha_data(dataset_name, cell_type='RS'):
    """Load training data in Sacha's format."""
    suffix = f'50x50_{dataset_name}'

    exc_type = 'exc' if cell_type == 'RS' else 'inh'
    ff_file = os.path.join(DATA_DIR, f'ExpTF_{exc_type}_{suffix}.npy')
    adapt_file = os.path.join(DATA_DIR, f'ExpTF_Adapt_{suffix}.npy')
    params_file = os.path.join(DATA_DIR, f'params_range_{suffix}.npy')

    FF = np.load(ff_file).T  # Sacha transposes
    adapt = np.load(adapt_file).T
    ve, vi, params = np.load(params_file, allow_pickle=True)

    return FF, adapt, ve, vi, params


def compute_moments(FF, adapt, ve, vi, params, cell_type='RS'):
    """Compute subthreshold moments for all valid grid points."""
    n_fe, n_fi = FF.shape
    mu_V_all, sig_V_all, tau_V_all, tauN_V_all = [], [], [], []
    FF_valid, adapt_valid = [], []
    fe_all, fi_all = [], []

    for i in range(n_fe):
        for j in range(n_fi):
            fout = FF[i, j]
            if np.isnan(fout) or np.isinf(fout) or fout <= 0:
                continue
            try:
                mu_V, sig_V, tau_V, tauN_V = mu_sig_tau_func(
                    ve[i], vi[j], fout, adapt[i, j], params, cell_type)
                if np.isnan(mu_V) or np.isnan(sig_V) or sig_V <= 0 or np.isnan(tau_V):
                    continue
                mu_V_all.append(mu_V)
                sig_V_all.append(sig_V)
                tau_V_all.append(tau_V)
                tauN_V_all.append(tauN_V)
                FF_valid.append(fout)
                adapt_valid.append(adapt[i, j])
                fe_all.append(ve[i])
                fi_all.append(vi[j])
            except:
                continue

    return (np.array(mu_V_all), np.array(sig_V_all), np.array(tau_V_all),
            np.array(tauN_V_all), np.array(FF_valid), np.array(fe_all), np.array(fi_all))


def fit_divolo_singlepass(mu_V, sig_V, tau_V, tauN_V, Fout, filter_60hz=True):
    """
    Di Volo's single-pass fitting: SLSQP on threshold, then ONE Nelder-Mead on firing rate.
    Optionally applies Fout < 60 Hz filter (Di Volo's original approach).
    """
    # Apply Di Volo's data filter
    if filter_60hz:
        mask = (Fout > 0) & (Fout < 60)
    else:
        mask = Fout > 0

    mu_V_f = mu_V[mask]
    sig_V_f = sig_V[mask]
    tau_V_f = tau_V[mask]
    tauN_V_f = tauN_V[mask]
    Fout_f = Fout[mask]

    print(f"    Data points: {len(Fout_f)} (of {len(Fout)} total, filter_60hz={filter_60hz})")

    # Stage 1: Fit effective threshold via SLSQP
    # Compute target V_thre from data using inverse erfc
    arg = np.clip(Fout_f * 2 * tau_V_f, 1e-15, 2.0 - 1e-15)
    Veff_target = mu_V_f + np.sqrt(2) * sig_V_f * erfcinv(arg)

    # Remove any NaN/Inf from inverse erfc
    valid = np.isfinite(Veff_target)
    mu_V_f = mu_V_f[valid]
    sig_V_f = sig_V_f[valid]
    tau_V_f = tau_V_f[valid]
    tauN_V_f = tauN_V_f[valid]
    Fout_f = Fout_f[valid]
    Veff_target = Veff_target[valid]

    print(f"    After erfcinv cleanup: {len(Fout_f)} points")

    # Initial guess: baseline threshold
    P0 = np.array([-0.050, 0.005, -0.010, 0.001, -0.001, 0.005, -0.020, 0.005, 0.001, -0.020])

    def loss_thresh(P):
        pred = eff_thresh(mu_V_f, sig_V_f, tauN_V_f, P)
        return np.mean((pred - Veff_target)**2)

    res1 = minimize(loss_thresh, P0, method='SLSQP',
                    options={'ftol': 1e-15, 'maxiter': 40000, 'disp': False})
    P_stage1 = res1.x
    print(f"    Stage 1 (SLSQP): MSE={res1.fun:.6e}")

    # Stage 2: Fit firing rate via Nelder-Mead (SINGLE PASS — Di Volo's approach)
    def loss_rate(P):
        pred = output_rate(P, mu_V_f, sig_V_f, tau_V_f, tauN_V_f)
        return np.mean((pred - Fout_f)**2)

    res2 = minimize(loss_rate, P_stage1, method='nelder-mead',
                    options={'xatol': 1e-5, 'maxiter': 50000, 'disp': False})
    P_final = res2.x
    print(f"    Stage 2 (NM, single-pass): MSE={res2.fun:.6e}")

    return P_final


def fit_sacha_loop(mu_V, sig_V, tau_V, tauN_V, Fout, fe, fi, loop_n=10):
    """
    Sacha's iterative fitting: SLSQP on threshold, then loop_n Nelder-Mead passes
    with dynamic range adjustment.
    """
    # Sacha's filter: remove NaN/Inf only (no upper frequency cap)
    mask = np.isfinite(Fout) & (Fout > 0)
    mu_V_f = mu_V[mask]
    sig_V_f = sig_V[mask]
    tau_V_f = tau_V[mask]
    tauN_V_f = tauN_V[mask]
    Fout_f = Fout[mask]
    fe_f = fe[mask]
    fi_f = fi[mask]

    print(f"    Data points: {len(Fout_f)} (of {len(Fout)} total)")

    # Stage 1: SLSQP on threshold (same as Di Volo)
    arg = np.clip(Fout_f * 2 * tau_V_f, 1e-15, 2.0 - 1e-15)
    Veff_target = mu_V_f + np.sqrt(2) * sig_V_f * erfcinv(arg)
    valid = np.isfinite(Veff_target)

    mu_V_s1 = mu_V_f[valid]
    sig_V_s1 = sig_V_f[valid]
    tauN_V_s1 = tauN_V_f[valid]
    Veff_target = Veff_target[valid]

    P0 = np.array([-0.050, 0.005, -0.010, 0.001, -0.001, 0.005, -0.020, 0.005, 0.001, -0.020])

    def loss_thresh(P):
        pred = eff_thresh(mu_V_s1, sig_V_s1, tauN_V_s1, P)
        return np.mean((pred - Veff_target)**2)

    res1 = minimize(loss_thresh, P0, method='SLSQP',
                    options={'ftol': 1e-15, 'maxiter': 40000, 'disp': False})
    P_stage1 = res1.x
    print(f"    Stage 1 (SLSQP): MSE={res1.fun:.6e}")

    # Stage 2: Iterated Nelder-Mead with dynamic range adjustment
    params_all = []
    P_current = P_stage1.copy()

    # Use all valid data for fitting
    mu_fit = mu_V_f[valid]
    sig_fit = sig_V_f[valid]
    tau_fit = tau_V_f[valid]
    tauN_fit = tauN_V_f[valid]
    Fout_fit = Fout_f[valid]
    fe_fit = fe_f[valid]
    fi_fit = fi_f[valid]

    # Keep originals for re-weighting
    mu_orig = mu_V_f[valid].copy()
    sig_orig = sig_V_f[valid].copy()
    tau_orig = tau_V_f[valid].copy()
    tauN_orig = tauN_V_f[valid].copy()
    Fout_orig = Fout_f[valid].copy()

    for loop_i in range(loop_n):
        def loss_rate(P):
            pred = output_rate(P, mu_fit, sig_fit, tau_fit, tauN_fit)
            return np.mean((pred - Fout_fit)**2)

        res = minimize(loss_rate, P_current, method='nelder-mead',
                       options={'xatol': 1e-5, 'maxiter': 50000, 'disp': False})

        mean_err = res.fun
        params_all.append((res.x.copy(), mean_err))

        if loop_i < loop_n - 1:
            # Dynamic range adjustment: find region of max error on ORIGINAL data
            pred_orig = output_rate(res.x, mu_orig, sig_orig, tau_orig, tauN_orig)
            errors = (pred_orig - Fout_orig)**2

            # Focus on lower rates (where accuracy matters most)
            low_rate_mask = Fout_orig < np.percentile(Fout_orig, 50)
            if np.any(low_rate_mask):
                # Weight toward low-rate high-error region
                combined_score = errors.copy()
                combined_score[low_rate_mask] *= 2.0
                top_idx = np.argsort(combined_score)[-len(combined_score)//3:]

                # Re-weight: include top error points + all original points
                mu_fit = np.concatenate([mu_orig, mu_orig[top_idx]])
                sig_fit = np.concatenate([sig_orig, sig_orig[top_idx]])
                tau_fit = np.concatenate([tau_orig, tau_orig[top_idx]])
                tauN_fit = np.concatenate([tauN_orig, tauN_orig[top_idx]])
                Fout_fit = np.concatenate([Fout_orig, Fout_orig[top_idx]])

            P_current = res.x.copy()

    # Pick best
    errors = [p[1] for p in params_all]
    best_idx = np.argmin(errors)
    P_final = params_all[best_idx][0]
    print(f"    Stage 2 (NM, {loop_n} loops): best MSE={errors[best_idx]:.6e} (loop {best_idx})")

    return P_final


def generate_brian2_tf_data(cell_params, grid_size=50, sim_time_ms=10000, seed=0):
    """Generate TF training data using Brian2 conductance-based simulation."""
    from brian2 import (NeuronGroup, PoissonGroup, Synapses, StateMonitor,
                        SpikeMonitor, run, defaultclock, ms, mV, nS, pA, pF,
                        nA, Hz, second, seed as brian_seed, start_scope)

    start_scope()
    brian_seed(seed)
    defaultclock.dt = 0.1*ms

    # Cell parameters
    Cm = cell_params['Cm'] * pF
    gL = cell_params['gL'] * nS
    EL = cell_params['EL'] * mV
    VT = cell_params['VT'] * mV
    DeltaT = cell_params['DeltaT'] * mV
    Vreset = cell_params['Vreset'] * mV
    Trefrac = cell_params['Trefrac'] * ms
    a_sub = cell_params.get('a', 0) * nS
    b_spike = cell_params.get('b', 0) * pA
    tauw = cell_params.get('tauw', 500) * ms

    # Synapse parameters
    Qe = cell_params['Qe'] * nS
    Qi = cell_params['Qi'] * nS
    Ee = cell_params['Ee'] * mV
    Ei = cell_params['Ei'] * mV
    tau_e = cell_params.get('tau_e', 5) * ms
    tau_i = cell_params.get('tau_i', 5) * ms

    # Network parameters
    Ntot = cell_params.get('Ntot', 10000)
    gei = cell_params.get('gei', 0.2)
    pconn = cell_params.get('pconn', 0.05)
    N_exc = int(pconn * (1 - gei) * Ntot)
    N_inh = int(pconn * gei * Ntot)

    ve = np.linspace(0.1, 30, grid_size)
    vi = np.linspace(0.1, 30, grid_size)

    FR_out = np.zeros((grid_size, grid_size))
    Adapt_out = np.zeros((grid_size, grid_size))

    cut_frac = 0.2  # discard first 20%

    for i, fe in enumerate(ve):
        for j, fi in enumerate(vi):
            start_scope()
            brian_seed(seed)

            # Single neuron
            eqs = """
            dvm/dt = (gL*(EL-vm) + gL*DeltaT*exp(clip((vm-VT)/DeltaT, -100, 20))
                      - GsynE*(vm-Ee) - GsynI*(vm-Ei) - w) / Cm : volt (unless refractory)
            dw/dt = (a_sub*(vm-EL) - w) / tauw : amp
            dGsynE/dt = -GsynE/tau_e : siemens
            dGsynI/dt = -GsynI/tau_i : siemens
            """

            G = NeuronGroup(1, eqs, threshold='vm > VT + 5*DeltaT',
                           reset='vm = Vreset; w += b_spike',
                           refractory=Trefrac, method='heun')
            G.vm = EL
            G.w = 0*pA

            Pe = PoissonGroup(N_exc, rates=fe*Hz)
            Pi = PoissonGroup(N_inh, rates=fi*Hz)

            Se = Synapses(Pe, G, on_pre='GsynE_post += Qe')
            Se.connect()
            Si = Synapses(Pi, G, on_pre='GsynI_post += Qi')
            Si.connect()

            spk = SpikeMonitor(G)
            sm = StateMonitor(G, 'w', record=True, dt=1*ms)

            run(sim_time_ms * ms)

            # Compute firing rate from last 80%
            spike_times = spk.t / second
            cut_time = sim_time_ms * 1e-3 * cut_frac
            valid_spikes = spike_times[spike_times > cut_time]
            duration = sim_time_ms * 1e-3 - cut_time
            FR_out[i, j] = len(valid_spikes) / duration if duration > 0 else 0

            # Mean adaptation from last 80%
            w_trace = sm.w[0] / 1e-12  # back to pA
            t_trace = sm.t / second
            w_valid = w_trace[t_trace > cut_time]
            Adapt_out[i, j] = np.mean(w_valid) * 1e-12 if len(w_valid) > 0 else 0  # store in A

        if (i + 1) % 10 == 0:
            print(f"    Brian2 grid: row {i+1}/{grid_size} done")

    return FR_out, Adapt_out, ve, vi


def print_comparison(name, P, ref_config1=P_E_CONFIG1, ref_divolo=P_E_DIVOLO):
    """Print polynomial and compare to references."""
    labels = ['P0(const)', 'P1(muV)', 'P2(sigV)', 'P3(tauN)',
              'P4(muV2)', 'P5(sigV2)', 'P6(tauN2)', 'P7(mu*sig)', 'P8(mu*tau)', 'P9(sig*tau)']

    print(f"\n  {'Coeff':<12} {'Value':>12} {'CONFIG1':>12} {'DiVolo':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for k in range(10):
        print(f"  {labels[k]:<12} {P[k]:>12.6f} {ref_config1[k]:>12.6f} {ref_divolo[k]:>12.6f}")

    dist_c1 = np.sqrt(np.sum((P - ref_config1)**2))
    dist_dv = np.sqrt(np.sum((P - ref_divolo)**2))
    print(f"\n  L2 distance to CONFIG1: {dist_c1:.6f}")
    print(f"  L2 distance to DiVolo:  {dist_dv:.6f}")
    print(f"  Closer to: {'CONFIG1' if dist_c1 < dist_dv else 'DiVolo'}")
    print(f"  P[2] (critical): {P[2]:.6f}  (CONFIG1={ref_config1[2]:.6f}, DiVolo={ref_divolo[2]:.6f})")


def compare_training_data(FF1, name1, FF2, name2):
    """Statistical comparison of two training datasets."""
    mask1 = np.isfinite(FF1) & (FF1 > 0)
    mask2 = np.isfinite(FF2) & (FF2 > 0)

    f1 = FF1[mask1]
    f2 = FF2[mask2]

    print(f"\n  === Training Data Comparison: {name1} vs {name2} ===")
    print(f"  {'Metric':<25} {name1:>15} {name2:>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'Valid points':<25} {len(f1):>15d} {len(f2):>15d}")
    print(f"  {'Points < 60 Hz':<25} {np.sum(f1<60):>15d} {np.sum(f2<60):>15d}")
    print(f"  {'Mean FR (Hz)':<25} {np.mean(f1):>15.2f} {np.mean(f2):>15.2f}")
    print(f"  {'Median FR (Hz)':<25} {np.median(f1):>15.2f} {np.median(f2):>15.2f}")
    print(f"  {'Std FR (Hz)':<25} {np.std(f1):>15.2f} {np.std(f2):>15.2f}")
    print(f"  {'Max FR (Hz)':<25} {np.max(f1):>15.2f} {np.max(f2):>15.2f}")
    print(f"  {'% > 60 Hz':<25} {100*np.mean(f1>60):>14.1f}% {100*np.mean(f2>60):>14.1f}%")

    # Point-by-point comparison where both valid
    both_valid = mask1 & mask2
    if np.any(both_valid):
        diff = FF1[both_valid] - FF2[both_valid]
        print(f"  {'Mean diff (Hz)':<25} {np.mean(diff):>15.4f}")
        print(f"  {'Max |diff| (Hz)':<25} {np.max(np.abs(diff)):>15.4f}")
        print(f"  {'Correlation':<25} {np.corrcoef(FF1[both_valid], FF2[both_valid])[0,1]:>15.6f}")


# ===== MAIN =====
if __name__ == '__main__':
    print("=" * 70)
    print("TF FITTING EXPERIMENTS: Isolating CONFIG1 vs Di Volo differences")
    print("=" * 70)

    # ---- Load existing training data ----
    print("\n[1] Loading existing training data...")

    # Sacha b_e=0 (EL=-64) — CONFIG1 baseline
    FF_sacha, adapt_sacha, ve_s, vi_s, params_sacha = load_sacha_data('b_e_0')
    print(f"  Sacha b_e=0: shape={FF_sacha.shape}, FR range=[{np.nanmin(FF_sacha):.1f}, {np.nanmax(FF_sacha):.1f}] Hz")

    # Di Volo EL=-65 data (generated earlier with Sacha's NumPy sim)
    FF_dv, adapt_dv, ve_d, vi_d, params_dv = load_sacha_data('divolo_el65')
    print(f"  DiVolo EL=-65: shape={FF_dv.shape}, FR range=[{np.nanmin(FF_dv):.1f}, {np.nanmax(FF_dv):.1f}] Hz")

    # ---- Compare training data ----
    print("\n[2] Comparing training data (NumPy: EL=-64 vs EL=-65)...")
    compare_training_data(FF_sacha, "Sacha(EL-64)", FF_dv, "DiVolo(EL-65)")

    # ---- Compute moments for all datasets ----
    print("\n[3] Computing subthreshold moments...")

    print("  Sacha b_e=0 (EL=-64)...")
    mu_s, sig_s, tau_s, tauN_s, Fout_s, fe_s, fi_s = compute_moments(
        FF_sacha, adapt_sacha, ve_s, vi_s, params_sacha, 'RS')
    print(f"    Valid points: {len(Fout_s)}")

    print("  DiVolo EL=-65...")
    mu_d, sig_d, tau_d, tauN_d, Fout_d, fe_d, fi_d = compute_moments(
        FF_dv, adapt_dv, ve_d, vi_d, params_dv, 'RS')
    print(f"    Valid points: {len(Fout_d)}")

    # ---- EXPERIMENT A: Sacha data + Di Volo single-pass ----
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Sacha data (EL=-64) + Di Volo single-pass fit")
    print("=" * 70)
    P_A = fit_divolo_singlepass(mu_s, sig_s, tau_s, tauN_s, Fout_s, filter_60hz=True)
    print_comparison("Exp A", P_A)

    # Also without the 60Hz filter
    print("\n--- Exp A variant (no 60Hz filter): ---")
    P_A2 = fit_divolo_singlepass(mu_s, sig_s, tau_s, tauN_s, Fout_s, filter_60hz=False)
    print_comparison("Exp A (no filter)", P_A2)

    # ---- EXPERIMENT B: DiVolo data + Di Volo single-pass ----
    print("\n" + "=" * 70)
    print("EXPERIMENT B: DiVolo data (EL=-65, NumPy) + Di Volo single-pass fit")
    print("=" * 70)
    P_B = fit_divolo_singlepass(mu_d, sig_d, tau_d, tauN_d, Fout_d, filter_60hz=True)
    print_comparison("Exp B", P_B)

    print("\n--- Exp B variant (no 60Hz filter): ---")
    P_B2 = fit_divolo_singlepass(mu_d, sig_d, tau_d, tauN_d, Fout_d, filter_60hz=False)
    print_comparison("Exp B (no filter)", P_B2)

    # ---- EXPERIMENT C: DiVolo data + Sacha loop ----
    print("\n" + "=" * 70)
    print("EXPERIMENT C: DiVolo data (EL=-65, NumPy) + Sacha loop_n=10 fit")
    print("=" * 70)
    P_C = fit_sacha_loop(mu_d, sig_d, tau_d, tauN_d, Fout_d, fe_d, fi_d, loop_n=10)
    print_comparison("Exp C", P_C)

    # ---- EXPERIMENT D: Sacha data + Sacha loop ----
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Sacha data (EL=-64) + Sacha loop_n=10 fit")
    print("=" * 70)
    P_D = fit_sacha_loop(mu_s, sig_s, tau_s, tauN_s, Fout_s, fe_s, fi_s, loop_n=10)
    print_comparison("Exp D", P_D)

    # ---- EXPERIMENT E & F: Brian2 data ----
    print("\n" + "=" * 70)
    print("EXPERIMENT E/F: Generating Brian2 training data (EL=-65, Di Volo cell)")
    print("=" * 70)

    divolo_cell = {
        'Cm': 200, 'gL': 10, 'EL': -65, 'VT': -50, 'DeltaT': 2,
        'Vreset': -65, 'Trefrac': 5, 'a': 0, 'b': 0, 'tauw': 500,
        'Qe': 1.5, 'Qi': 5, 'Ee': 0, 'Ei': -80, 'tau_e': 5, 'tau_i': 5,
        'Ntot': 10000, 'gei': 0.2, 'pconn': 0.05
    }

    print("  Generating Brian2 data (50x50 grid, 10s per point)...")
    print("  This will take ~30-60 minutes...")
    FR_brian2, Adapt_brian2, ve_b, vi_b = generate_brian2_tf_data(
        divolo_cell, grid_size=50, sim_time_ms=10000, seed=0)
    print(f"  Brian2 data: FR range=[{np.nanmin(FR_brian2):.1f}, {np.nanmax(FR_brian2):.1f}] Hz")

    # Save Brian2 data
    np.save(os.path.join(DATA_DIR, 'ExpTF_exc_50x50_brian2_divolo_el65.npy'), FR_brian2)
    np.save(os.path.join(DATA_DIR, 'ExpTF_Adapt_50x50_brian2_divolo_el65.npy'), Adapt_brian2)

    # Compare Brian2 vs NumPy training data
    compare_training_data(FR_brian2.T, "Brian2(EL-65)", FF_dv, "NumPy(EL-65)")

    # Compute moments for Brian2 data
    # Need to create a params dict matching Di Volo's cell
    params_brian2 = params_dv.copy() if isinstance(params_dv, dict) else {
        'Cm': 200, 'gL': 10, 'EL_e': -65, 'EL_i': -65, 'VT': -50, 'DeltaT_e': 2,
        'Vreset': -65, 'Trefrac': 5, 'a_e': 0, 'b_e': 0, 'tauw_e': 500,
        'Qe': 1.5, 'Qi': 5, 'Ee': 0, 'Ei': -80, 'tau_e': 5, 'tau_i': 5,
        'Ntot': 10000, 'gei': 0.2, 'pconn': 0.05
    }

    print("\n  Computing moments for Brian2 data...")
    mu_b, sig_b, tau_b, tauN_b, Fout_b, fe_b, fi_b = compute_moments(
        FR_brian2.T, Adapt_brian2.T, ve_b, vi_b, params_brian2, 'RS')
    print(f"    Valid points: {len(Fout_b)}")

    # Experiment E: Brian2 data + Di Volo single-pass
    print("\n" + "=" * 70)
    print("EXPERIMENT E: Brian2 data (EL=-65) + Di Volo single-pass fit")
    print("=" * 70)
    P_E = fit_divolo_singlepass(mu_b, sig_b, tau_b, tauN_b, Fout_b, filter_60hz=True)
    print_comparison("Exp E", P_E)

    # Experiment F: Brian2 data + Sacha loop
    print("\n" + "=" * 70)
    print("EXPERIMENT F: Brian2 data (EL=-65) + Sacha loop_n=10 fit")
    print("=" * 70)
    P_F = fit_sacha_loop(mu_b, sig_b, tau_b, tauN_b, Fout_b, fe_b, fi_b, loop_n=10)
    print_comparison("Exp F", P_F)

    # ---- SUMMARY ----
    print("\n" + "=" * 70)
    print("SUMMARY: P[2] (sigma_V coefficient) across all experiments")
    print("=" * 70)
    print(f"  Reference CONFIG1:  P[2] = {P_E_CONFIG1[2]:.6f}")
    print(f"  Reference DiVolo:   P[2] = {P_E_DIVOLO[2]:.6f}")
    print(f"  ---")
    print(f"  A  (Sacha data,  DV fit):           P[2] = {P_A[2]:.6f}")
    print(f"  A' (Sacha data,  DV fit, no filter): P[2] = {P_A2[2]:.6f}")
    print(f"  B  (DV data,     DV fit):            P[2] = {P_B[2]:.6f}")
    print(f"  B' (DV data,     DV fit, no filter): P[2] = {P_B2[2]:.6f}")
    print(f"  C  (DV data,     Sacha loop):        P[2] = {P_C[2]:.6f}")
    print(f"  D  (Sacha data,  Sacha loop):        P[2] = {P_D[2]:.6f}")
    print(f"  E  (Brian2 data, DV fit):            P[2] = {P_E[2]:.6f}")
    print(f"  F  (Brian2 data, Sacha loop):        P[2] = {P_F[2]:.6f}")
    print(f"\n  If E ≈ DiVolo (-0.024): Brian2 data is the key difference")
    print(f"  If A ≈ DiVolo (-0.024): Fitting method is the key difference")
    print(f"  If neither: Multiple factors interact")
