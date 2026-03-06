#!/usr/bin/env python3
"""
tf_global_fit.py — Comprehensive TF polynomial fitting using multiple optimizers.

Loads Sacha's b_e=0 training data (the exact data CONFIG1 was fit to),
fits using Nelder-Mead, Differential Evolution, Basin-Hopping, and L-BFGS-B,
then compares all results with CONFIG1 and Di Volo's polynomial.

Also tests DOI sensitivity for each polynomial at representative operating points.
"""

import os
import sys
import time
import warnings
import numpy as np
from scipy.special import erfc, erfcinv
from scipy.optimize import minimize, differential_evolution, basinhopping

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================
# PATHS
# ============================================================
PROJECT = '/Users/soichi/Desktop/Research/Psychedelics & Anesthesia Modeling Study'
DATA_DIR = os.path.join(PROJECT, 'code/original_repos/sacha_et_al_2025/Tf_calc/data/')

# ============================================================
# 1. LOAD SACHA'S TRAINING DATA
# ============================================================
print("=" * 80)
print("LOADING SACHA'S TRAINING DATA (b_e=0, RS cell)")
print("=" * 80)

FF_raw = np.load(os.path.join(DATA_DIR, 'ExpTF_50x50_b_e_0_RS.npy')).T
adapt_raw = np.load(os.path.join(DATA_DIR, '50x50_b_e_0_RS_adapt.npy')).T
ve, vi, params = np.load(os.path.join(DATA_DIR, '50x50_b_e_0_RS_params.npy'), allow_pickle=True)

print(f"FF_raw shape: {FF_raw.shape}")
print(f"adapt_raw shape: {adapt_raw.shape}")
print(f"ve range: [{ve[0]:.2f}, {ve[-1]:.2f}] Hz, {len(ve)} points")
print(f"vi range: [{vi[0]:.2f}, {vi[-1]:.2f}] Hz, {len(vi)} points")
print(f"FF range: [{np.nanmin(FF_raw):.2f}, {np.nanmax(FF_raw):.2f}] Hz")
print(f"\nParams: {params}")

# ============================================================
# 2. MOMENT EQUATIONS (exactly as Sacha's theoretical_tools.py, SI units)
# ============================================================
def mu_sig_tau_func(fexc, finh, fout, w_ad, p, cell_type='RS', w_prec=False):
    """Compute subthreshold moments (mu_V, sig_V, tau_V, tauN_V) in SI units."""
    Q_e = p['Q_e'] * 1e-9
    Q_i = p['Q_i'] * 1e-9
    tau_e = p['tau_e'] * 1e-3
    tau_i = p['tau_i'] * 1e-3
    E_e = p['E_e'] * 1e-3
    E_i = p['E_i'] * 1e-3
    C_m = p['Cm'] * 1e-12
    Tw = p['tau_w'] * 1e-3
    g_L = p['Gl'] * 1e-9
    gei = p['gei']
    ntot = p['Ntot']
    pconnec = p['p_con']

    if cell_type == "RS":
        try:
            a = p['a_e'] * 1e-9
            b = p['b_e'] * 1e-12
            E_L = p['EL_e'] * 1e-3
        except KeyError:
            a = p['a'] * 1e-9
            b = p['b'] * 1e-12
            E_L = p['EL'] * 1e-3
    elif cell_type == "FS":
        try:
            a = p['a_i'] * 1e-9
            b = p['b_i'] * 1e-12
            E_L = p['EL_i'] * 1e-3
        except KeyError:
            a = p['a'] * 1e-9
            b = p['b'] * 1e-12
            E_L = p['EL'] * 1e-3

    f_e = fexc * (1. - gei) * pconnec * ntot
    f_i = finh * gei * pconnec * ntot

    mu_Ge = f_e * tau_e * Q_e
    mu_Gi = f_i * tau_i * Q_i
    mu_G = mu_Ge + mu_Gi + g_L
    tau_eff = C_m / mu_G

    if w_prec:
        mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - w_ad) / mu_G
    else:
        mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - fout * Tw * b + a * E_L) / mu_G

    U_e = Q_e / mu_G * (E_e - mu_V)
    U_i = Q_i / mu_G * (E_i - mu_V)

    sig_V = np.sqrt(
        f_e * (U_e * tau_e) ** 2 / (2 * (tau_eff + tau_e)) +
        f_i * (U_i * tau_i) ** 2 / (2 * (tau_eff + tau_i))
    )

    tau_V_num = f_e * (U_e * tau_e) ** 2 + f_i * (U_i * tau_i) ** 2
    tau_V_den = (f_e * (U_e * tau_e) ** 2 / (tau_eff + tau_e) +
                 f_i * (U_i * tau_i) ** 2 / (tau_eff + tau_i))
    tau_V = tau_V_num / tau_V_den

    tauN_V = tau_V * g_L / C_m

    return mu_V, sig_V, tau_V, tauN_V


def eff_thresh(mu_V, sig_V, tauN_V, P):
    """Effective threshold polynomial (second-order, 10 coefficients)."""
    mu_0, mu_d = -60.0e-3, 0.01
    sig_0, sig_d = 0.004, 0.006
    tau_0, tau_d = 0.5, 1.0

    V = (mu_V - mu_0) / mu_d
    S = (sig_V - sig_0) / sig_d
    T = (tauN_V - tau_0) / tau_d

    return (P[0] +
            P[1] * V + P[2] * S + P[3] * T +
            P[4] * V ** 2 + P[5] * S ** 2 + P[6] * T ** 2 +
            P[7] * V * S + P[8] * V * T + P[9] * S * T)


def output_rate(P, mu_V, sig_V, tau_V, tauN_V):
    """TF output firing rate from polynomial P."""
    return erfc((eff_thresh(mu_V, sig_V, tauN_V, P) - mu_V) / (np.sqrt(2) * sig_V)) / (2 * tau_V)


def eff_thresh_estimate(ydata, mu_V, sig_V, tau_V):
    """Inverse: compute effective threshold from known firing rate."""
    return mu_V + np.sqrt(2) * sig_V * erfcinv(ydata * 2 * tau_V)


# ============================================================
# 3. DATA PREPROCESSING — Remove NaNs (same logic as Sacha's get_rid_of_nans)
# ============================================================
print("\n" + "=" * 80)
print("PREPROCESSING: Computing moments and removing NaN/Inf points")
print("=" * 80)

vve, vvi = np.meshgrid(ve, vi)

# Flatten
ve2 = vve.flatten()
vi2 = vvi.flatten()
FF2 = FF_raw.flatten()
adapt2 = adapt_raw.flatten()

print(f"Total grid points: {len(FF2)}")

# Compute moments and Veff for filtering (same as Sacha's code)
mu_V_all, sig_V_all, tau_V_all, tauN_V_all = mu_sig_tau_func(
    ve2, vi2, FF2, adapt2, params, 'RS', w_prec=False)
Veff_all = eff_thresh_estimate(FF2, mu_V_all, sig_V_all, tau_V_all)

# Remove NaN/Inf in Veff (Sacha's exact filtering)
nan_idx = np.where(np.isnan(Veff_all))[0]
inf_idx = np.where(np.isinf(Veff_all))[0]
bad_idx = np.union1d(nan_idx, inf_idx)

print(f"NaN Veff points: {len(nan_idx)}")
print(f"Inf Veff points: {len(inf_idx)}")
print(f"Total removed: {len(bad_idx)}")

ve2 = np.delete(ve2, bad_idx)
vi2 = np.delete(vi2, bad_idx)
FF2 = np.delete(FF2, bad_idx)
adapt2 = np.delete(adapt2, bad_idx)

print(f"Valid points: {len(FF2)} / {vve.size}")

# Recompute moments on clean data
mu_V, sig_V, tau_V, tauN_V = mu_sig_tau_func(
    ve2, vi2, FF2, adapt2, params, 'RS', w_prec=False)

# Veff for threshold fitting
Veff_thresh = eff_thresh_estimate(FF2, mu_V, sig_V, tau_V)

print(f"\nMoment ranges:")
print(f"  mu_V:  [{mu_V.min()*1e3:.2f}, {mu_V.max()*1e3:.2f}] mV")
print(f"  sig_V: [{sig_V.min()*1e3:.3f}, {sig_V.max()*1e3:.3f}] mV")
print(f"  tauN_V: [{tauN_V.min():.4f}, {tauN_V.max():.4f}]")
print(f"  FF2:   [{FF2.min():.4f}, {FF2.max():.4f}] Hz")

# ============================================================
# 4. LOAD REFERENCE POLYNOMIALS
# ============================================================
print("\n" + "=" * 80)
print("REFERENCE POLYNOMIALS")
print("=" * 80)

P_CONFIG1 = np.load(os.path.join(DATA_DIR, 'RS-cell0_CONFIG1_fit.npy'))
P_DIVOLO = np.array([
    -0.04983106, 0.005063550882777035, -0.023470121807314552,
    0.0022951513725067503, -0.0004105302652029825, 0.010547051343547399,
    -0.03659252821136933, 0.007437487505797858, 0.001265064721846073,
    -0.04072161294490446
])

print("\nCONFIG1 coefficients:")
for i, v in enumerate(P_CONFIG1):
    print(f"  P[{i}] = {v:.10f}")

print("\nDi Volo coefficients:")
for i, v in enumerate(P_DIVOLO):
    print(f"  P[{i}] = {v:.10f}")

# Evaluate references
for name, P in [("CONFIG1", P_CONFIG1), ("DIVOLO", P_DIVOLO)]:
    pred = output_rate(P, mu_V, sig_V, tau_V, tauN_V)
    mse = np.mean((pred - FF2) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - FF2))
    max_err = np.max(np.abs(pred - FF2))
    print(f"\n{name}: MSE={mse:.6e}, RMSE={rmse:.4f} Hz, MAE={mae:.4f} Hz, MaxErr={max_err:.2f} Hz")

# ============================================================
# 5. FITTING WITH MULTIPLE OPTIMIZERS
# ============================================================

# --- Objective functions ---
def res_vthr(P):
    """Vthr fit: MSE between polynomial Vthr and estimated Vthr."""
    return np.mean((Veff_thresh - eff_thresh(mu_V, sig_V, tauN_V, P)) ** 2)


def res_tf(P):
    """TF fit: MSE between polynomial output rate and simulated firing rate."""
    pred = output_rate(P, mu_V, sig_V, tau_V, tauN_V)
    return np.mean((pred - FF2) ** 2)


# =========================
# 5a. Nelder-Mead (Sacha's approach)
# =========================
print("\n" + "=" * 80)
print("OPTIMIZER 1: Nelder-Mead (Sacha's 2-step approach)")
print("=" * 80)

t0 = time.time()

# Step 1: SLSQP for Vthr
params_init = np.ones(10) * 1e-3
fit1 = minimize(res_vthr, params_init, method='SLSQP',
                options={'ftol': 1e-15, 'maxiter': 30000, 'disp': False})
print(f"  Step 1 (SLSQP Vthr): converged={fit1.success}, fun={fit1.fun:.6e}")

# Step 2: Nelder-Mead for TF
fit2 = minimize(res_tf, fit1.x, method='nelder-mead',
                options={'xatol': 1e-17, 'maxiter': 50000, 'disp': False})
P_NM = fit2.x
t_nm = time.time() - t0
print(f"  Step 2 (NM TF):      converged={fit2.success}, niter={fit2.nit}, fun={fit2.fun:.6e}")
print(f"  Total time: {t_nm:.1f}s")

# =========================
# 5b. Differential Evolution (global optimizer)
# =========================
print("\n" + "=" * 80)
print("OPTIMIZER 2: Differential Evolution (global)")
print("=" * 80)

# Bounds based on CONFIG1 and DIVOLO ranges, expanded
bounds = [
    (-0.1, 0.0),       # P[0] (const)
    (-0.01, 0.02),      # P[1] (muV)
    (-0.05, 0.01),      # P[2] (sigV)
    (-0.01, 0.01),      # P[3] (tauV)
    (-0.005, 0.005),    # P[4] (muV^2)
    (-0.005, 0.02),     # P[5] (sigV^2)
    (-0.06, 0.01),      # P[6] (tauV^2)
    (-0.01, 0.02),      # P[7] (muV*sigV)
    (-0.005, 0.01),     # P[8] (muV*tauV)
    (-0.06, 0.01),      # P[9] (sigV*tauV)
]

t0 = time.time()
result_de = differential_evolution(
    res_tf, bounds, seed=42, maxiter=5000, tol=1e-20,
    polish=True, mutation=(0.5, 1.5), recombination=0.9,
    popsize=30, workers=1
)
P_DE = result_de.x
t_de = time.time() - t0
print(f"  Converged: {result_de.success}")
print(f"  Iterations: {result_de.nit}")
print(f"  Fun: {result_de.fun:.6e}")
print(f"  Time: {t_de:.1f}s")

# =========================
# 5c. Basin-Hopping from CONFIG1
# =========================
print("\n" + "=" * 80)
print("OPTIMIZER 3: Basin-Hopping (starting from CONFIG1)")
print("=" * 80)

t0 = time.time()
result_bh = basinhopping(
    res_tf, P_CONFIG1, niter=200, T=1e-6,
    minimizer_kwargs={'method': 'nelder-mead', 'options': {'maxiter': 50000, 'disp': False}},
    seed=42
)
P_BH = result_bh.x
t_bh = time.time() - t0
print(f"  Fun: {result_bh.fun:.6e}")
print(f"  Message: {result_bh.message[0]}")
print(f"  Time: {t_bh:.1f}s")

# =========================
# 5d. L-BFGS-B (gradient-based with bounds)
# =========================
print("\n" + "=" * 80)
print("OPTIMIZER 4: L-BFGS-B (gradient-based, bounded)")
print("=" * 80)

t0 = time.time()
result_lb = minimize(
    res_tf, fit1.x, method='L-BFGS-B', bounds=bounds,
    options={'maxiter': 50000, 'ftol': 1e-20, 'disp': False}
)
P_LB = result_lb.x
t_lb = time.time() - t0
print(f"  Converged: {result_lb.success}")
print(f"  Fun: {result_lb.fun:.6e}")
print(f"  Message: {result_lb.message}")
print(f"  Time: {t_lb:.1f}s")

# =========================
# 5e. Powell (derivative-free, no bounds)
# =========================
print("\n" + "=" * 80)
print("OPTIMIZER 5: Powell (derivative-free)")
print("=" * 80)

t0 = time.time()
result_pw = minimize(
    res_tf, fit1.x, method='Powell',
    options={'maxiter': 50000, 'ftol': 1e-20, 'disp': False}
)
P_PW = result_pw.x
t_pw = time.time() - t0
print(f"  Converged: {result_pw.success}")
print(f"  Fun: {result_pw.fun:.6e}")
print(f"  Time: {t_pw:.1f}s")

# =========================
# 5f. Differential Evolution seeded from CONFIG1 (warm-start)
# =========================
print("\n" + "=" * 80)
print("OPTIMIZER 6: Differential Evolution (warm-start from CONFIG1 neighborhood)")
print("=" * 80)

# Create initial population around CONFIG1
np.random.seed(42)
n_pop = 30 * 10  # popsize * ndim
init_pop = np.tile(P_CONFIG1, (n_pop, 1))
# Add small perturbations (5% scale of each coefficient)
for i in range(n_pop):
    perturbation = np.random.randn(10) * np.abs(P_CONFIG1) * 0.05
    init_pop[i] += perturbation
    # Clip to bounds
    for j in range(10):
        init_pop[i, j] = np.clip(init_pop[i, j], bounds[j][0], bounds[j][1])

t0 = time.time()
result_de2 = differential_evolution(
    res_tf, bounds, seed=42, maxiter=5000, tol=1e-20,
    polish=True, init=init_pop
)
P_DE2 = result_de2.x
t_de2 = time.time() - t0
print(f"  Converged: {result_de2.success}")
print(f"  Iterations: {result_de2.nit}")
print(f"  Fun: {result_de2.fun:.6e}")
print(f"  Time: {t_de2:.1f}s")

# ============================================================
# 6. COMPREHENSIVE COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("POLYNOMIAL COMPARISON — FIT QUALITY ON TRAINING DATA")
print("=" * 80)

all_P = {
    "CONFIG1 (ref)": P_CONFIG1,
    "Di Volo (ref)": P_DIVOLO,
    "Nelder-Mead": P_NM,
    "DiffEvolution": P_DE,
    "BasinHopping": P_BH,
    "L-BFGS-B": P_LB,
    "Powell": P_PW,
    "DE-warmstart": P_DE2,
}

times = {
    "CONFIG1 (ref)": 0, "Di Volo (ref)": 0,
    "Nelder-Mead": t_nm, "DiffEvolution": t_de,
    "BasinHopping": t_bh, "L-BFGS-B": t_lb,
    "Powell": t_pw, "DE-warmstart": t_de2,
}

print(f"\n{'Name':22s} {'MSE':>12s} {'RMSE(Hz)':>10s} {'MAE(Hz)':>10s} {'MaxErr(Hz)':>12s} {'Time(s)':>8s}")
print("-" * 80)
for name, P in all_P.items():
    pred = output_rate(P, mu_V, sig_V, tau_V, tauN_V)
    mse = np.mean((pred - FF2) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - FF2))
    max_err = np.max(np.abs(pred - FF2))
    t = times[name]
    print(f"{name:22s} {mse:12.6e} {rmse:10.4f} {mae:10.4f} {max_err:12.2f} {t:8.1f}")

# ============================================================
# 7. COEFFICIENT TABLE
# ============================================================
print("\n" + "=" * 80)
print("COEFFICIENT TABLE")
print("=" * 80)

labels = [
    'P0 (const)', 'P1 (muV)', 'P2 (sigV)', 'P3 (tauV)',
    'P4 (muV^2)', 'P5 (sigV^2)', 'P6 (tauV^2)',
    'P7 (muV*sigV)', 'P8 (muV*tauV)', 'P9 (sigV*tauV)'
]

# Print header
print(f"\n{'Label':20s}", end='')
for name in all_P:
    short = name[:14]
    print(f" {short:>14s}", end='')
print()
print("-" * (20 + 15 * len(all_P)))

for i, label in enumerate(labels):
    print(f"{label:20s}", end='')
    for name, P in all_P.items():
        print(f" {P[i]:14.8f}", end='')
    print()

# ============================================================
# 8. COEFFICIENT DISTANCE FROM CONFIG1
# ============================================================
print("\n" + "=" * 80)
print("DISTANCE FROM CONFIG1")
print("=" * 80)

print(f"\n{'Name':22s} {'L2 dist':>12s} {'Max|diff|':>12s} {'Rel L2 (%)':>12s}")
print("-" * 62)
c1_norm = np.linalg.norm(P_CONFIG1)
for name, P in all_P.items():
    if name == "CONFIG1 (ref)":
        continue
    diff = P - P_CONFIG1
    l2 = np.linalg.norm(diff)
    max_d = np.max(np.abs(diff))
    rel = 100 * l2 / c1_norm if c1_norm > 0 else 0
    print(f"{name:22s} {l2:12.6e} {max_d:12.6e} {rel:12.4f}")

# ============================================================
# 9. DOI SENSITIVITY TEST
# ============================================================
print("\n" + "=" * 80)
print("DOI SENSITIVITY TEST")
print("  (Fe=2.5 Hz, Fi=6.0 Hz, steady-state W_e)")
print("=" * 80)

drug_states = {
    'Awake':    {'b_e': 5,  'tau_i': 5.0, 'EL_e': -64.0},
    'Propofol': {'b_e': 30, 'tau_i': 7.0, 'EL_e': -64.0},
    'Prop+DOI': {'b_e': 30, 'tau_i': 7.0, 'EL_e': -61.2},
    'DOI':      {'b_e': 5,  'tau_i': 5.0, 'EL_e': -61.2},
}

for state_name, drug in drug_states.items():
    # Modify params for this state
    p = dict(params)  # make a copy
    p['b_e'] = drug['b_e']
    p['tau_i'] = drug['tau_i']
    p['EL_e'] = drug['EL_e']

    Fe, Fi = 2.5, 6.0
    # Steady-state adaptation: W_e = b * Fe * tau_w
    W_e_ss = drug['b_e'] * 1e-12 * Fe * p['tau_w'] * 1e-3  # in Amperes

    mu_V_s, sig_V_s, tau_V_s, tauN_V_s = mu_sig_tau_func(
        np.array([Fe]), np.array([Fi]), np.array([Fe]), np.array([W_e_ss]),
        p, 'RS', w_prec=True
    )

    print(f"\n{state_name}: muV={mu_V_s[0]*1e3:.2f} mV, sigV={sig_V_s[0]*1e3:.3f} mV, "
          f"tauN_V={tauN_V_s[0]:.4f}")

    for name, P in all_P.items():
        fr = output_rate(P, mu_V_s, sig_V_s, tau_V_s, tauN_V_s)
        print(f"  {name:22s}: FR = {fr[0]:.6f} Hz")

# ============================================================
# 10. DOI SENSITIVITY SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("DOI SENSITIVITY SUMMARY — Key ratios")
print("=" * 80)

print(f"\n{'Name':22s} {'Prop/Awake':>12s} {'P+DOI/Prop':>12s} {'DOI/Awake':>12s} {'DOI-Awake':>12s}")
print("-" * 74)

for name, P in all_P.items():
    frs = {}
    for state_name, drug in drug_states.items():
        p = dict(params)
        p['b_e'] = drug['b_e']
        p['tau_i'] = drug['tau_i']
        p['EL_e'] = drug['EL_e']
        Fe, Fi = 2.5, 6.0
        W_e_ss = drug['b_e'] * 1e-12 * Fe * p['tau_w'] * 1e-3
        mu_V_s, sig_V_s, tau_V_s, tauN_V_s = mu_sig_tau_func(
            np.array([Fe]), np.array([Fi]), np.array([Fe]), np.array([W_e_ss]),
            p, 'RS', w_prec=True
        )
        frs[state_name] = output_rate(P, mu_V_s, sig_V_s, tau_V_s, tauN_V_s)[0]

    prop_awake = frs['Propofol'] / frs['Awake'] if frs['Awake'] > 1e-10 else float('inf')
    pdoi_prop = frs['Prop+DOI'] / frs['Propofol'] if frs['Propofol'] > 1e-10 else float('inf')
    doi_awake = frs['DOI'] / frs['Awake'] if frs['Awake'] > 1e-10 else float('inf')
    doi_minus_awake = frs['DOI'] - frs['Awake']

    print(f"{name:22s} {prop_awake:12.4f} {pdoi_prop:12.4f} {doi_awake:12.4f} {doi_minus_awake:12.6f}")

# ============================================================
# 11. BEST POLYNOMIAL IDENTIFICATION
# ============================================================
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

# Find best MSE among new fits
new_fits = {k: v for k, v in all_P.items() if "(ref)" not in k}
best_name = None
best_mse = float('inf')
for name, P in new_fits.items():
    pred = output_rate(P, mu_V, sig_V, tau_V, tauN_V)
    mse = np.mean((pred - FF2) ** 2)
    if mse < best_mse:
        best_mse = mse
        best_name = name

# CONFIG1 MSE
pred_c1 = output_rate(P_CONFIG1, mu_V, sig_V, tau_V, tauN_V)
mse_c1 = np.mean((pred_c1 - FF2) ** 2)

print(f"\nBest new fit: {best_name} (MSE={best_mse:.6e})")
print(f"CONFIG1 MSE:  {mse_c1:.6e}")
print(f"Ratio (CONFIG1/best): {mse_c1/best_mse:.4f}")

if best_mse < mse_c1:
    improvement = (1 - best_mse / mse_c1) * 100
    print(f"\n>>> {best_name} is BETTER than CONFIG1 by {improvement:.2f}%")
elif best_mse > mse_c1:
    degradation = (best_mse / mse_c1 - 1) * 100
    print(f"\n>>> CONFIG1 is still BETTER than {best_name} by {degradation:.2f}%")
else:
    print(f"\n>>> CONFIG1 and {best_name} have identical MSE")

print("\nNOTE: A lower MSE on training data does NOT mean the polynomial will work")
print("better in TVB. CONFIG1's specific coefficient structure (especially P[2], P[5],")
print("P[6]) is critical for correct propofol collapse and DOI reversal dynamics.")
print("See CLAUDE.md and MEMORY.md for why refitting is inadvisable.")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
