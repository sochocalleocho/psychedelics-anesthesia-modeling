#!/usr/bin/env python3
"""
fit_option_a.py — Fit TF polynomial from Option A training data (E_L=-65).

FULLY STANDALONE: copies needed functions, no external imports beyond numpy/scipy.
"""
import sys, os
import numpy as np
from scipy.optimize import minimize
from scipy.special import erfc, erfcinv
from scipy.ndimage import maximum_filter

# ---------- Functions copied from theoretical_tools.py ----------

def eff_thresh(mu_V, sig_V, tauN_V, params):
    P_0, P_mu, P_sig, P_tau, P_mu2, P_sig2, P_tau2, P_mu_sig, P_mu_tau, P_sig_tau = params
    mu_0 = -60.0*1e-3; mu_d = 0.01; sig_0 = 0.004; sig_d = 0.006; tau_0 = 0.5; tau_d = 1.0
    V1 = (P_mu*(mu_V - mu_0)/mu_d + P_sig*(sig_V - sig_0)/sig_d + P_tau*(tauN_V - tau_0)/tau_d)
    V2 = (P_mu2*((mu_V - mu_0)/mu_d)**2 + P_sig2*((sig_V - sig_0)/sig_d)**2 +
          P_tau2*((tauN_V - tau_0)/tau_d)**2 + P_mu_sig*((mu_V - mu_0)/mu_d)*((sig_V - sig_0)/sig_d) +
          P_mu_tau*((mu_V - mu_0)/mu_d)*((tauN_V - tau_0)/tau_d) +
          P_sig_tau*((sig_V - sig_0)/sig_d)*((tauN_V - tau_0)/tau_d))
    return P_0 + V1 + V2

def mu_sig_tau_func(fexc, finh, fout, w_ad, params, cell_type, w_prec=False):
    p = params
    Q_e, Q_i = p['Q_e']*1e-9, p['Q_i']*1e-9
    tau_e, tau_i = p['tau_e']*1e-3, p['tau_i']*1e-3
    E_e, E_i = p['E_e']*1e-3, p['E_i']*1e-3
    C_m, Tw, g_L = p['Cm']*1e-12, p['tau_w']*1e-3, p['Gl']*1e-9
    gei, ntot, pconnec = p['gei'], p['Ntot'], p['p_con']
    if cell_type == "RS":
        try: a, b, E_L = p['a_e']*1e-9, p['b_e']*1e-12, p['EL_e']*1e-3
        except KeyError: a, b, E_L = p['a']*1e-9, p['b']*1e-12, p['EL']*1e-3
    elif cell_type == "FS":
        try: a, b, E_L = p['a_i']*1e-9, p['b_i']*1e-12, p['EL_i']*1e-3
        except KeyError: a, b, E_L = p['a']*1e-9, p['b']*1e-12, p['EL']*1e-3
    f_e = fexc*(1.-gei)*pconnec*ntot
    f_i = finh*gei*pconnec*ntot
    mu_Ge = f_e*tau_e*Q_e; mu_Gi = f_i*tau_i*Q_i
    mu_G = mu_Ge + mu_Gi + g_L
    tau_eff = C_m / mu_G
    if w_prec:
        mu_V = (mu_Ge*E_e + mu_Gi*E_i + g_L*E_L - w_ad) / mu_G
    else:
        mu_V = (mu_Ge*E_e + mu_Gi*E_i + g_L*E_L - fout*Tw*b + a*E_L) / mu_G
    U_e = Q_e / mu_G*(E_e - mu_V)
    U_i = Q_i / mu_G*(E_i - mu_V)
    sig_V = np.sqrt(f_e*(U_e*tau_e)**2/(2*(tau_eff+tau_e)) + f_i*(U_i*tau_i)**2/(2*(tau_eff+tau_i)))
    tau_V = ((f_e*(U_e*tau_e)**2 + f_i*(U_i*tau_i)**2) /
             (f_e*(U_e*tau_e)**2/(tau_eff+tau_e) + f_i*(U_i*tau_i)**2/(tau_eff+tau_i)))
    tauN_V = tau_V*g_L / C_m
    return mu_V, sig_V, tau_V, tauN_V

def output_rate(params, mu_V, sig_V, tau_V, tauN_V):
    return erfc((eff_thresh(mu_V, sig_V, tauN_V, params) - mu_V) / (np.sqrt(2)*sig_V)) / (2*tau_V)

def eff_thresh_estimate(ydata, mu_V, sig_V, tau_V):
    return mu_V + np.sqrt(2)*sig_V*erfcinv(ydata*2*tau_V)

def get_rid_of_nans(vve, vvi, adapt, FF, params, cell_type, w_prec=False):
    ve2 = vve.flatten(); vi2 = vvi.flatten(); FF2 = FF.flatten(); adapt2 = adapt.flatten()
    muV2, sV2, Tv2, TNv2 = mu_sig_tau_func(ve2, vi2, FF2, adapt2, params, cell_type, w_prec=w_prec)
    Veff = eff_thresh_estimate(FF2, muV2, sV2, Tv2)
    nanindex = np.where(np.isnan(Veff)); infindex = np.where(np.isinf(Veff))
    bigindex = np.concatenate([nanindex, infindex], axis=1)
    print(f"  {len(ve2)} total points, {len(bigindex[0])} NaN/Inf removed → {len(ve2)-len(bigindex[0])} valid")
    ve2 = np.delete(ve2, bigindex); vi2 = np.delete(vi2, bigindex)
    FF2 = np.delete(FF2, bigindex); adapt2 = np.delete(adapt2, bigindex)
    return ve2, vi2, FF2, adapt2

def find_max_error(out_rate, fit_rate, ve, vi, window=12, thresh_pc=0.9):
    error = np.sqrt((out_rate - fit_rate)**2).T
    print(f'  mean error = {np.mean(error):.6f}')
    if window > len(ve)/3: window = int(len(ve)/3)
    local_maxima = maximum_filter(error, size=window)
    max_indices = np.argwhere(local_maxima == error)
    all_errors = []
    for i, j in max_indices:
        mean_error = np.nanmean(error[max(0,i-window):min(error.shape[0],i+window+1),
                                      max(0,j-window):min(error.shape[1],j+window+1)])
        rect = (max(0,i-window), max(0,j-window), min(error.shape[0],i+window+1), min(error.shape[1],j+window+1))
        all_errors.append([mean_error, sum(rect), rect])
    all_errors = np.array(all_errors, dtype='object')
    y = np.argsort(all_errors[:,1], kind='mergesort'); all_errors = all_errors[y]
    thresh = np.max(all_errors[:,0]) * thresh_pc
    for i in range(all_errors.shape[0]):
        if all_errors[i,0] > thresh:
            max_mean_error_rect = all_errors[i,2]; break
    x_start, y_start, x_end, y_end = max_mean_error_rect
    return (ve[y_start], ve[y_end-1]), (vi[x_start], vi[x_end-1])

def adjust_ranges(ve, vi, FF, adapt, params, cell_type, range_inh, range_exc, w_prec=False):
    vve, vvi = np.meshgrid(ve, vi)
    if range_inh:
        ds, de = range_inh; s, e = np.argmin(np.abs(vi-ds)), np.argmin(np.abs(vi-de))
        rid = list([0,1,3,5,-3,-5,-1,-2]) + list(range(s,e))
    if range_exc:
        ds, de = range_exc; s, e = np.argmin(np.abs(ve-ds)), np.argmin(np.abs(ve-de))
        red = list([0,1,3,5,-3,-5,-1,-2]) + list(range(s,e))
    vve2 = vve[np.ix_(rid, red)]; vvi2 = vvi[np.ix_(rid, red)]
    FF2 = FF[np.ix_(rid, red)]; adapt2 = adapt[np.ix_(rid, red)]
    ve2, vi2, FF3, adapt3 = get_rid_of_nans(vve2, vvi2, adapt2, FF2, params, cell_type, w_prec=w_prec)
    mu_V, sig_V, tau_V, tauN_V = mu_sig_tau_func(ve2, vi2, FF3, adapt3, params, cell_type, w_prec=w_prec)
    return mu_V, sig_V, tau_V, tauN_V, FF3

def make_fit(DATA, cell_type, params_file, adapt_file, range_exc=None, range_inh=None, loop_n=1, seed=10):
    """Standalone make_fit_from_data without import chain."""
    FF = np.load(DATA).T
    adapt = np.load(adapt_file).T
    ve, vi, params = np.load(params_file, allow_pickle=True)
    vve, vvi = np.meshgrid(ve, vi)

    ve2, vi2, FF2, adapt2 = get_rid_of_nans(vve, vvi, adapt, FF, params, cell_type)
    mu_V, sig_V, tau_V, tauN_V = mu_sig_tau_func(ve2, vi2, FF2, adapt2, params, cell_type)
    Veff_thresh = eff_thresh_estimate(FF2, mu_V, sig_V, tau_V)

    print("  Fitting Veff threshold (SLSQP)...")
    params_init = np.ones(10)*1e-3
    def res_func(p): return np.mean((Veff_thresh - eff_thresh(mu_V, sig_V, tauN_V, p))**2)
    fit = minimize(res_func, params_init, method='SLSQP', tol=1e-17,
                   options={'disp': False, 'maxiter': 30000})
    print(f"  Veff fit MSE: {fit.fun:.6e}")

    print("  Fitting Transfer Function (nelder-mead)...")
    params_init2 = fit['x']
    params_all = []
    for i in range(loop_n):
        if range_inh or range_exc:
            mu_V_r, sig_V_r, tau_V_r, tauN_V_r, FF2_r = adjust_ranges(
                ve, vi, FF, adapt, params, cell_type, range_inh, range_exc)
        else:
            mu_V_r, sig_V_r, tau_V_r, tauN_V_r, FF2_r = mu_V, sig_V, tau_V, tauN_V, FF2
        def res2_func(p): return np.mean((output_rate(p, mu_V_r, sig_V_r, tau_V_r, tauN_V_r) - FF2_r)**2)
        fit2 = minimize(res2_func, params_init2, method='nelder-mead', tol=1e-17,
                        options={'disp': False, 'maxiter': 30000})
        P = fit2['x']
        muV, sigV, tauV, tauNV = mu_sig_tau_func(vve, vvi, FF, adapt, params, cell_type)
        fit_rate = output_rate(P, muV, sigV, tauV, tauNV)
        mean_error = np.mean(np.sqrt((FF - fit_rate)**2))
        if loop_n > 1:
            range_exc, range_inh = find_max_error(FF, fit_rate, ve, vi)
            params_init2 = P
        params_all.append([P, mean_error])
        seed += 10
    params_all = np.array(params_all, dtype='object')
    P = params_all[np.argmin(params_all[:,1])][0]
    print(f"  Final TF MSE: {params_all[np.argmin(params_all[:,1])][1]:.6e}")
    return P

# ---------- Main ----------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'paper_pipeline_hub', 'Tf_calc', 'data')
grid = '50x50'; name = 'divolo_el65'

exc_file = os.path.join(DATA_DIR, f'ExpTF_exc_{grid}_{name}.npy')
inh_file = os.path.join(DATA_DIR, f'ExpTF_inh_{grid}_{name}.npy')
params_file = os.path.join(DATA_DIR, f'params_range_{grid}_{name}.npy')
adapt_file = os.path.join(DATA_DIR, f'ExpTF_Adapt_{grid}_{name}.npy')

assert os.path.exists(exc_file), f"Missing: {exc_file}"

print("=" * 70)
print("OPTION A: Fitting polynomial from Di Volo params (E_L=-65)")
print("=" * 70)

print("\n--- Excitatory (RS) ---")
P_E = make_fit(exc_file, 'RS', params_file, adapt_file, range_exc=(0,5), range_inh=(0,5), loop_n=1)
save_e = os.path.join(DATA_DIR, 'P_E_optionA.npy'); np.save(save_e, P_E)
print(f"  Saved: {save_e}")

print("\n--- Inhibitory (FS) ---")
P_I = make_fit(inh_file, 'FS', params_file, adapt_file, range_exc=(0,5), range_inh=(0,5), loop_n=1)
save_i = os.path.join(DATA_DIR, 'P_I_optionA.npy'); np.save(save_i, P_I)
print(f"  Saved: {save_i}")

# Compare
P_E_DIVOLO = np.array([-0.04983106, 0.005063550882777035, -0.023470121807314552,
                        0.0022951513725067503, -0.0004105302652029825, 0.010547051343547399,
                        -0.03659252821136933, 0.007437487505797858, 0.001265064721846073,
                        -0.04072161294490446])
P_I_DIVOLO = np.array([-0.05149122024209484, 0.004003689190271077, -0.008352013668528155,
                        0.0002414237992765705, -0.0005070645080016026, 0.0014345394104282397,
                        -0.014686689498949967, 0.004502706285435741, 0.0028472190352532454,
                        -0.015357804594594548])
P_E_SACHA = np.array([-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
                       0.00341614, -0.01156433, 0.00194753, 0.00274079, -0.01066769])

labels = ['P[0] const', 'P[1] muV', 'P[2] sigV', 'P[3] TvN',
          'P[4] muV^2', 'P[5] sigV^2', 'P[6] TvN^2',
          'P[7] muV*sigV', 'P[8] muV*TvN', 'P[9] sigV*TvN']

print("\n" + "=" * 70)
print("EXCITATORY P_E COMPARISON")
print("=" * 70)
print(f"  {'Coeff':15s}  {'OptionA':>11s}  {'DiVolo':>11s}  {'CONFIG1':>11s}  {'A-DV':>10s}")
print("-" * 70)
for i, l in enumerate(labels):
    print(f"  {l:15s}  {P_E[i]:>+11.6f}  {P_E_DIVOLO[i]:>+11.6f}  {P_E_SACHA[i]:>+11.6f}  {P_E[i]-P_E_DIVOLO[i]:>+10.6f}")
mse_dv = np.mean((P_E - P_E_DIVOLO)**2)
mse_c1 = np.mean((P_E - P_E_SACHA)**2)
print(f"\n  MSE vs DiVolo: {mse_dv:.2e}  |  MSE vs CONFIG1: {mse_c1:.2e}")
print(f"  *** P[2]: OptionA={P_E[2]:+.6f}  DiVolo={P_E_DIVOLO[2]:+.6f}  CONFIG1={P_E_SACHA[2]:+.6f}")

print("\n" + "=" * 70)
print("INHIBITORY P_I COMPARISON")
print("=" * 70)
print(f"  {'Coeff':15s}  {'OptionA':>11s}  {'DiVolo':>11s}  {'A-DV':>10s}")
print("-" * 55)
for i, l in enumerate(labels):
    print(f"  {l:15s}  {P_I[i]:>+11.6f}  {P_I_DIVOLO[i]:>+11.6f}  {P_I[i]-P_I_DIVOLO[i]:>+10.6f}")
mse_i = np.mean((P_I - P_I_DIVOLO)**2)
print(f"\n  MSE vs DiVolo P_I: {mse_i:.2e}")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
if mse_dv < 1e-6:
    print("  CLOSELY REPRODUCES Di Volo's polynomial!")
elif np.sign(P_E[2]) == np.sign(P_E_DIVOLO[2]):
    print(f"  Same P[2] SIGN as Di Volo (both negative)")
    print(f"  Quantitative MSE={mse_dv:.2e}")
else:
    print(f"  DIFFERENT P[2] sign! OptionA={P_E[2]:+.6f} vs DiVolo={P_E_DIVOLO[2]:+.6f}")
    print(f"  → Need to investigate Di Volo's original fitting code")
