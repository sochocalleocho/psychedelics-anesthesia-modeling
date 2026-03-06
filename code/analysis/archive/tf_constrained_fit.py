#!/usr/bin/env python3
"""
tf_constrained_fit.py — Constrained TF polynomial fitting.

The key insight from TVB validation: P[2] (sigma_V coefficient) MUST be negative
for propofol up-down oscillation dynamics to emerge. The global MSE optimum has
P[2] > 0 and fails in TVB.

This script explores:
1. P[2] constrained to be negative (but free to optimize magnitude)
2. P[2] fixed at CONFIG1's value (-0.008) while optimizing all others
3. P[2] scanned across a range to find the bifurcation boundary
4. All coefficients constrained to CONFIG1's signs

Goal: Find a polynomial with better MSE than CONFIG1 that PRESERVES propofol dynamics.
"""

import os
import sys
import time
import warnings
import numpy as np
from scipy.special import erfc, erfcinv
from scipy.optimize import minimize, differential_evolution

warnings.filterwarnings('ignore', category=RuntimeWarning)

PROJECT = '/Users/soichi/Desktop/Research/Psychedelics & Anesthesia Modeling Study'
DATA_DIR = os.path.join(PROJECT, 'code/original_repos/sacha_et_al_2025/Tf_calc/data/')

# ============================================================
# Load and preprocess training data (same as tf_global_fit.py)
# ============================================================
FF_raw = np.load(os.path.join(DATA_DIR, 'ExpTF_50x50_b_e_0_RS.npy')).T
adapt_raw = np.load(os.path.join(DATA_DIR, '50x50_b_e_0_RS_adapt.npy')).T
ve, vi, params = np.load(os.path.join(DATA_DIR, '50x50_b_e_0_RS_params.npy'), allow_pickle=True)

def mu_sig_tau_func(fexc, finh, fout, w_ad, p, cell_type='RS', w_prec=False):
    Q_e = p['Q_e'] * 1e-9; Q_i = p['Q_i'] * 1e-9
    tau_e = p['tau_e'] * 1e-3; tau_i = p['tau_i'] * 1e-3
    E_e = p['E_e'] * 1e-3; E_i = p['E_i'] * 1e-3
    C_m = p['Cm'] * 1e-12; Tw = p['tau_w'] * 1e-3
    g_L = p['Gl'] * 1e-9; gei = p['gei']; ntot = p['Ntot']; pconnec = p['p_con']
    if cell_type == "RS":
        try: a = p['a_e'] * 1e-9; b = p['b_e'] * 1e-12; E_L = p['EL_e'] * 1e-3
        except KeyError: a = p['a'] * 1e-9; b = p['b'] * 1e-12; E_L = p['EL'] * 1e-3
    else:
        try: a = p['a_i'] * 1e-9; b = p['b_i'] * 1e-12; E_L = p['EL_i'] * 1e-3
        except KeyError: a = p['a'] * 1e-9; b = p['b'] * 1e-12; E_L = p['EL'] * 1e-3
    f_e = fexc * (1. - gei) * pconnec * ntot
    f_i = finh * gei * pconnec * ntot
    mu_Ge = f_e * tau_e * Q_e; mu_Gi = f_i * tau_i * Q_i; mu_G = mu_Ge + mu_Gi + g_L
    tau_eff = C_m / mu_G
    if w_prec: mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - w_ad) / mu_G
    else: mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - fout * Tw * b + a * E_L) / mu_G
    U_e = Q_e / mu_G * (E_e - mu_V); U_i = Q_i / mu_G * (E_i - mu_V)
    sig_V = np.sqrt(f_e * (U_e * tau_e)**2 / (2 * (tau_eff + tau_e)) +
                    f_i * (U_i * tau_i)**2 / (2 * (tau_eff + tau_i)))
    tau_V_num = f_e * (U_e * tau_e)**2 + f_i * (U_i * tau_i)**2
    tau_V_den = (f_e * (U_e * tau_e)**2 / (tau_eff + tau_e) +
                 f_i * (U_i * tau_i)**2 / (tau_eff + tau_i))
    tau_V = tau_V_num / tau_V_den
    tauN_V = tau_V * g_L / C_m
    return mu_V, sig_V, tau_V, tauN_V

def eff_thresh(mu_V, sig_V, tauN_V, P):
    mu_0, mu_d = -60.0e-3, 0.01; sig_0, sig_d = 0.004, 0.006; tau_0, tau_d = 0.5, 1.0
    V = (mu_V - mu_0) / mu_d; S = (sig_V - sig_0) / sig_d; T = (tauN_V - tau_0) / tau_d
    return (P[0] + P[1]*V + P[2]*S + P[3]*T + P[4]*V**2 + P[5]*S**2 +
            P[6]*T**2 + P[7]*V*S + P[8]*V*T + P[9]*S*T)

def output_rate(P, mu_V, sig_V, tau_V, tauN_V):
    return erfc((eff_thresh(mu_V, sig_V, tauN_V, P) - mu_V) / (np.sqrt(2) * sig_V)) / (2 * tau_V)

def eff_thresh_estimate(ydata, mu_V, sig_V, tau_V):
    return mu_V + np.sqrt(2) * sig_V * erfcinv(ydata * 2 * tau_V)

# Preprocess
vve, vvi = np.meshgrid(ve, vi)
ve2 = vve.flatten(); vi2 = vvi.flatten(); FF2 = FF_raw.flatten(); adapt2 = adapt_raw.flatten()
mu_V_all, sig_V_all, tau_V_all, tauN_V_all = mu_sig_tau_func(ve2, vi2, FF2, adapt2, params, 'RS')
Veff_all = eff_thresh_estimate(FF2, mu_V_all, sig_V_all, tau_V_all)
bad_idx = np.union1d(np.where(np.isnan(Veff_all))[0], np.where(np.isinf(Veff_all))[0])
ve2 = np.delete(ve2, bad_idx); vi2 = np.delete(vi2, bad_idx)
FF2 = np.delete(FF2, bad_idx); adapt2 = np.delete(adapt2, bad_idx)
mu_V, sig_V, tau_V, tauN_V = mu_sig_tau_func(ve2, vi2, FF2, adapt2, params, 'RS')
Veff_thresh = eff_thresh_estimate(FF2, mu_V, sig_V, tau_V)

print(f"Valid training points: {len(FF2)}")

# Reference polynomials
P_CONFIG1 = np.load(os.path.join(DATA_DIR, 'RS-cell0_CONFIG1_fit.npy'))
P_GLOBAL = np.load(os.path.join(PROJECT, 'code/analysis/P_E_global_opt.npy'))

def res_vthr(P): return np.mean((Veff_thresh - eff_thresh(mu_V, sig_V, tauN_V, P))**2)
def res_tf(P): return np.mean((output_rate(P, mu_V, sig_V, tau_V, tauN_V) - FF2)**2)

# ============================================================
# DOI sensitivity evaluation
# ============================================================
drug_states = {
    'Awake':    {'b_e': 5,  'tau_i': 5.0, 'EL_e': -64.0},
    'Propofol': {'b_e': 30, 'tau_i': 7.0, 'EL_e': -64.0},
    'Prop+DOI': {'b_e': 30, 'tau_i': 7.0, 'EL_e': -61.2},
    'DOI':      {'b_e': 5,  'tau_i': 5.0, 'EL_e': -61.2},
}

def eval_drug_sensitivity(P):
    """Return dict of firing rates for each drug state."""
    frs = {}
    for state_name, drug in drug_states.items():
        p = dict(params)
        p['b_e'] = drug['b_e']; p['tau_i'] = drug['tau_i']; p['EL_e'] = drug['EL_e']
        Fe, Fi = 2.5, 6.0
        W_e_ss = drug['b_e'] * 1e-12 * Fe * p['tau_w'] * 1e-3
        mu_V_s, sig_V_s, tau_V_s, tauN_V_s = mu_sig_tau_func(
            np.array([Fe]), np.array([Fi]), np.array([Fe]), np.array([W_e_ss]),
            p, 'RS', w_prec=True)
        frs[state_name] = output_rate(P, mu_V_s, sig_V_s, tau_V_s, tauN_V_s)[0]
    return frs


# ============================================================
# STRATEGY 1: P[2] constrained negative via bounds
# ============================================================
print("\n" + "=" * 80)
print("STRATEGY 1: Differential Evolution with P[2] < 0")
print("=" * 80)

bounds_neg_p2 = [
    (-0.1, 0.0),        # P[0]
    (-0.01, 0.02),       # P[1]
    (-0.05, -1e-6),      # P[2] MUST be negative
    (-0.02, 0.02),       # P[3]
    (-0.005, 0.005),     # P[4]
    (-0.005, 0.03),      # P[5]
    (-0.06, 0.02),       # P[6]
    (-0.01, 0.02),       # P[7]
    (-0.005, 0.01),      # P[8]
    (-0.06, 0.02),       # P[9]
]

t0 = time.time()
result1 = differential_evolution(
    res_tf, bounds_neg_p2, seed=42, maxiter=5000, tol=1e-20,
    polish=True, mutation=(0.5, 1.5), recombination=0.9, popsize=30, workers=1
)
P_NEG_P2 = result1.x
t1 = time.time() - t0
print(f"  MSE: {result1.fun:.6e} (CONFIG1: {res_tf(P_CONFIG1):.6e})")
print(f"  P[2]: {P_NEG_P2[2]:.8f} (CONFIG1: {P_CONFIG1[2]:.8f})")
print(f"  Time: {t1:.1f}s")

frs1 = eval_drug_sensitivity(P_NEG_P2)
print(f"  Awake FR: {frs1['Awake']:.4f} Hz")
print(f"  Prop FR: {frs1['Propofol']:.4f} Hz (ratio: {frs1['Propofol']/frs1['Awake']:.4f})")
print(f"  DOI FR: {frs1['DOI']:.4f} Hz (ratio: {frs1['DOI']/frs1['Awake']:.4f})")
print(f"  DOI-Awake: {frs1['DOI']-frs1['Awake']:+.6f} Hz")


# ============================================================
# STRATEGY 2: P[2] scanned across negative values
# ============================================================
print("\n" + "=" * 80)
print("STRATEGY 2: Scan P[2] from -0.001 to -0.05")
print("=" * 80)

# SLSQP initial guess (same for all)
params_init = np.ones(10) * 1e-3
slsqp_result = minimize(res_vthr, params_init, method='SLSQP',
                         options={'ftol': 1e-15, 'maxiter': 30000})
x0 = slsqp_result.x.copy()

p2_values = [-0.001, -0.003, -0.005, -0.008, -0.010, -0.012, -0.015, -0.020, -0.025, -0.030, -0.040, -0.050]

print(f"\n{'P[2]':>8s} {'MSE':>12s} {'Prop/Awake':>12s} {'DOI/Awake':>12s} {'DOI-Awake':>12s}")
print("-" * 60)

scan_results = {}
for p2_val in p2_values:
    # Fix P[2], optimize rest with Nelder-Mead
    def res_tf_fixed_p2(P_free):
        P_full = np.insert(P_free, 2, p2_val)
        return res_tf(P_full)

    x0_free = np.delete(x0, 2)
    fit = minimize(res_tf_fixed_p2, x0_free, method='nelder-mead',
                   options={'xatol': 1e-17, 'maxiter': 100000})
    P_scan = np.insert(fit.x, 2, p2_val)

    mse = res_tf(P_scan)
    frs = eval_drug_sensitivity(P_scan)

    prop_ratio = frs['Propofol'] / frs['Awake'] if frs['Awake'] > 1e-10 else float('inf')
    doi_ratio = frs['DOI'] / frs['Awake'] if frs['Awake'] > 1e-10 else float('inf')
    doi_diff = frs['DOI'] - frs['Awake']

    scan_results[p2_val] = {'P': P_scan, 'MSE': mse, 'frs': frs,
                             'prop_ratio': prop_ratio, 'doi_ratio': doi_ratio}

    print(f"{p2_val:8.3f} {mse:12.6e} {prop_ratio:12.4f} {doi_ratio:12.4f} {doi_diff:>+12.6f}")


# ============================================================
# STRATEGY 3: Constrain ALL signs to match CONFIG1
# ============================================================
print("\n" + "=" * 80)
print("STRATEGY 3: All coefficient signs constrained to match CONFIG1")
print("=" * 80)

# CONFIG1 signs: [-, +, -, -, -, +, -, +, +, -]
config1_signs = np.sign(P_CONFIG1)
bounds_signed = []
for i in range(10):
    if config1_signs[i] > 0:
        bounds_signed.append((1e-8, 0.05))
    else:
        bounds_signed.append((-0.05, -1e-8))

t0 = time.time()
result3 = differential_evolution(
    res_tf, bounds_signed, seed=42, maxiter=5000, tol=1e-20,
    polish=True, mutation=(0.5, 1.5), recombination=0.9, popsize=30, workers=1
)
P_SIGNED = result3.x
t3 = time.time() - t0
print(f"  MSE: {result3.fun:.6e} (CONFIG1: {res_tf(P_CONFIG1):.6e})")
print(f"  Time: {t3:.1f}s")
print(f"  Converged: {result3.success}")

frs3 = eval_drug_sensitivity(P_SIGNED)
print(f"  Awake FR: {frs3['Awake']:.4f} Hz")
print(f"  Prop FR: {frs3['Propofol']:.4f} Hz (ratio: {frs3['Propofol']/frs3['Awake']:.4f})")
print(f"  DOI FR: {frs3['DOI']:.4f} Hz (ratio: {frs3['DOI']/frs3['Awake']:.4f})")
print(f"  DOI-Awake: {frs3['DOI']-frs3['Awake']:+.6f} Hz")


# ============================================================
# STRATEGY 4: Warm-start from CONFIG1, bounded to stay negative P[2]
# ============================================================
print("\n" + "=" * 80)
print("STRATEGY 4: Basin-hopping from CONFIG1, P[2] < 0 constraint")
print("=" * 80)

from scipy.optimize import basinhopping

class NegP2Bounds:
    """Reject steps where P[2] > 0."""
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        return x[2] < 0

t0 = time.time()
result4 = basinhopping(
    res_tf, P_CONFIG1, niter=300, T=1e-6,
    minimizer_kwargs={'method': 'nelder-mead', 'options': {'maxiter': 100000}},
    seed=42, accept_test=NegP2Bounds()
)
P_BH_NEG = result4.x
t4 = time.time() - t0
print(f"  MSE: {result4.fun:.6e} (CONFIG1: {res_tf(P_CONFIG1):.6e})")
print(f"  P[2]: {P_BH_NEG[2]:.8f}")
print(f"  Time: {t4:.1f}s")

frs4 = eval_drug_sensitivity(P_BH_NEG)
print(f"  Awake FR: {frs4['Awake']:.4f} Hz")
print(f"  Prop FR: {frs4['Propofol']:.4f} Hz (ratio: {frs4['Propofol']/frs4['Awake']:.4f})")
print(f"  DOI FR: {frs4['DOI']:.4f} Hz (ratio: {frs4['DOI']/frs4['Awake']:.4f})")
print(f"  DOI-Awake: {frs4['DOI']-frs4['Awake']:+.6f} Hz")


# ============================================================
# COMPREHENSIVE COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE COMPARISON")
print("=" * 80)

all_results = {
    "CONFIG1":          P_CONFIG1,
    "Global (uncnstr)": P_GLOBAL,
    "DE (P[2]<0)":      P_NEG_P2,
    "Signed":           P_SIGNED,
    "BH (P[2]<0)":      P_BH_NEG,
}

# Add best scan results
best_scan_p2 = min(scan_results.keys(), key=lambda k: scan_results[k]['MSE'])
all_results[f"Scan P2={best_scan_p2}"] = scan_results[best_scan_p2]['P']

print(f"\n{'Name':22s} {'MSE':>12s} {'P[2]':>10s} {'Prop/Awake':>12s} {'DOI/Awake':>12s} {'DOI-Awake':>12s}")
print("-" * 82)

for name, P in all_results.items():
    mse = res_tf(P)
    frs = eval_drug_sensitivity(P)
    prop_ratio = frs['Propofol'] / frs['Awake'] if frs['Awake'] > 1e-10 else float('inf')
    doi_ratio = frs['DOI'] / frs['Awake'] if frs['Awake'] > 1e-10 else float('inf')
    doi_diff = frs['DOI'] - frs['Awake']
    print(f"{name:22s} {mse:12.6e} {P[2]:10.6f} {prop_ratio:12.4f} {doi_ratio:12.4f} {doi_diff:>+12.6f}")


# Coefficient table for top candidates
print("\n" + "=" * 80)
print("COEFFICIENT TABLE — Top candidates (P[2] negative)")
print("=" * 80)

labels = ['P0 (const)', 'P1 (muV)', 'P2 (sigV)', 'P3 (tauV)',
          'P4 (muV^2)', 'P5 (sigV^2)', 'P6 (tauV^2)',
          'P7 (muV*sigV)', 'P8 (muV*tauV)', 'P9 (sigV*tauV)']

top_polys = {k: v for k, v in all_results.items() if v[2] < 0}
print(f"\n{'Label':20s}", end='')
for name in top_polys:
    print(f" {name[:14]:>14s}", end='')
print()
print("-" * (20 + 15 * len(top_polys)))

for i, label in enumerate(labels):
    print(f"{label:20s}", end='')
    for P in top_polys.values():
        print(f" {P[i]:14.8f}", end='')
    print()


# ============================================================
# Save best constrained polynomial
# ============================================================
# Find best polynomial with P[2] < 0
best_constrained = None
best_mse = float('inf')
best_name = None
for name, P in all_results.items():
    if P[2] < 0 and name != "CONFIG1" and name != "Global (uncnstr)":
        mse = res_tf(P)
        if mse < best_mse:
            best_mse = mse
            best_constrained = P
            best_name = name

if best_constrained is not None:
    outpath = os.path.join(PROJECT, 'code/analysis/P_E_constrained_best.npy')
    np.save(outpath, best_constrained)
    print(f"\nBest constrained polynomial ({best_name}): MSE={best_mse:.6e}")
    print(f"  Improvement over CONFIG1: {(1 - best_mse/res_tf(P_CONFIG1))*100:.2f}%")
    print(f"  Saved to: {outpath}")
    print(f"  Coefficients: [{', '.join(f'{x:.8f}' for x in best_constrained)}]")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
