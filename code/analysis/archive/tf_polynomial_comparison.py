#!/usr/bin/env python3
"""
TF Polynomial Comparison: Sacha CONFIG1 vs Di Volo/Martin
=========================================================
Compares two sets of transfer-function polynomials across 4 drug states
by computing self-consistent firing rates, moment decompositions,
sensitivity analyses, and threshold surface comparisons.
"""

import numpy as np
from scipy.special import erfc
from scipy.optimize import fsolve
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: Define polynomials
# ============================================================
BASE_DIR = "/Users/soichi/Desktop/Research/Psychedelics & Anesthesia Modeling Study"

P_E_SACHA = np.load(os.path.join(BASE_DIR,
    "code/original_repos/sacha_et_al_2025/Tf_calc/data/RS-cell0_CONFIG1_fit.npy"))
P_I_SACHA = np.load(os.path.join(BASE_DIR,
    "code/original_repos/sacha_et_al_2025/Tf_calc/data/FS-cell_CONFIG1_fit.npy"))

P_E_DIVOLO = np.array([
    -0.04983106, 0.005063550882777035, -0.023470121807314552,
    0.0022951513725067503, -0.0004105302652029825, 0.010547051343547399,
    -0.03659252821136933, 0.007437487505797858, 0.001265064721846073,
    -0.04072161294490446
])

P_I_DIVOLO = np.array([
    -0.05149122024209484, 0.004003689190271077, -0.008352013668528155,
    0.0002414237992765705, -0.0005070645080016026, 0.0014345394104282397,
    -0.014686689498949967, 0.004502706285435741, 0.0028472190352532454,
    -0.015357804594594548
])

print("=" * 80)
print("STEP 1: Polynomial Coefficients")
print("=" * 80)
print(f"\nP_E_SACHA  = {P_E_SACHA}")
print(f"P_I_SACHA  = {P_I_SACHA}")
print(f"P_E_DIVOLO = {P_E_DIVOLO}")
print(f"P_I_DIVOLO = {P_I_DIVOLO}")

print("\n--- Coefficient-by-coefficient comparison (Excitatory) ---")
labels = ['P[0] const', 'P[1] V', 'P[2] S', 'P[3] T',
          'P[4] V^2', 'P[5] S^2', 'P[6] T^2',
          'P[7] V*S', 'P[8] V*T', 'P[9] S*T']
print(f"{'Term':<12} {'SACHA':>12} {'DIVOLO':>12} {'Ratio':>10} {'Diff':>12}")
for i, lbl in enumerate(labels):
    ratio = P_E_SACHA[i] / P_E_DIVOLO[i] if abs(P_E_DIVOLO[i]) > 1e-15 else float('inf')
    diff = P_E_SACHA[i] - P_E_DIVOLO[i]
    print(f"{lbl:<12} {P_E_SACHA[i]:>12.6f} {P_E_DIVOLO[i]:>12.6f} {ratio:>10.3f} {diff:>12.6f}")

print("\n--- Coefficient-by-coefficient comparison (Inhibitory) ---")
print(f"{'Term':<12} {'SACHA':>12} {'DIVOLO':>12} {'Ratio':>10} {'Diff':>12}")
for i, lbl in enumerate(labels):
    ratio = P_I_SACHA[i] / P_I_DIVOLO[i] if abs(P_I_DIVOLO[i]) > 1e-15 else float('inf')
    diff = P_I_SACHA[i] - P_I_DIVOLO[i]
    print(f"{lbl:<12} {P_I_SACHA[i]:>12.6f} {P_I_DIVOLO[i]:>12.6f} {ratio:>10.3f} {diff:>12.6f}")


# ============================================================
# STEP 2: Moment equations (EXACT copy from TVB Zerlaut.py)
# ============================================================
def get_fluct_regime_vars(Fe, Fi, Fe_ext, Fi_ext, W,
                          Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                          g_L, C_m, E_L, N_tot,
                          p_connect_e, p_connect_i, g,
                          K_ext_e, K_ext_i):
    """Compute mu_V, sigma_V, T_V from mean-field equations."""
    fe = (Fe + 1.0e-6) * (1.0 - g) * p_connect_e * N_tot + Fe_ext * K_ext_e
    fi = (Fi + 1.0e-6) * g * p_connect_i * N_tot + Fi_ext * K_ext_i

    mu_Ge = Q_e * tau_e * fe
    mu_Gi = Q_i * tau_i * fi
    mu_G = g_L + mu_Ge + mu_Gi

    T_m = C_m / mu_G
    mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - W) / mu_G

    U_e = Q_e / mu_G * (E_e - mu_V)
    U_i = Q_i / mu_G * (E_i - mu_V)

    sigma_V = np.sqrt(
        fe * (U_e * tau_e)**2 / (2.0 * (tau_e + T_m)) +
        fi * (U_i * tau_i)**2 / (2.0 * (tau_i + T_m))
    )

    T_V_num = fe * (U_e * tau_e)**2 + fi * (U_i * tau_i)**2
    T_V_den = (fe * (U_e * tau_e)**2 / (tau_e + T_m) +
               fi * (U_i * tau_i)**2 / (tau_i + T_m))

    if np.isscalar(T_V_den):
        T_V = T_V_num / T_V_den if abs(T_V_den) > 1e-30 else 0.5
    else:
        T_V = np.where(np.abs(T_V_den) > 1e-30, T_V_num / T_V_den, 0.5)

    return mu_V, sigma_V, T_V


# ============================================================
# STEP 3: Threshold function and TF (EXACT copy from TVB)
# ============================================================
def threshold_func(muV, sigmaV, TvN, P):
    """10-coefficient polynomial threshold."""
    muV0, DmuV0 = -60.0, 10.0
    sV0, DsV0 = 4.0, 6.0
    TvN0, DTvN0 = 0.5, 1.0

    V = (muV - muV0) / DmuV0
    S = (sigmaV - sV0) / DsV0
    T = (TvN - TvN0) / DTvN0

    return (P[0] + P[1]*V + P[2]*S + P[3]*T +
            P[4]*V**2 + P[5]*S**2 + P[6]*T**2 +
            P[7]*V*S + P[8]*V*T + P[9]*S*T)


def threshold_func_terms(muV, sigmaV, TvN, P):
    """Return individual polynomial term contributions."""
    muV0, DmuV0 = -60.0, 10.0
    sV0, DsV0 = 4.0, 6.0
    TvN0, DTvN0 = 0.5, 1.0

    V = (muV - muV0) / DmuV0
    S = (sigmaV - sV0) / DsV0
    T = (TvN - TvN0) / DTvN0

    terms = {
        'P[0]': P[0],
        'P[1]*V': P[1]*V,
        'P[2]*S': P[2]*S,
        'P[3]*T': P[3]*T,
        'P[4]*V^2': P[4]*V**2,
        'P[5]*S^2': P[5]*S**2,
        'P[6]*T^2': P[6]*T**2,
        'P[7]*V*S': P[7]*V*S,
        'P[8]*V*T': P[8]*V*T,
        'P[9]*S*T': P[9]*S*T,
    }
    return terms, V, S, T


def TF(Fe, Fi, Fe_ext, Fi_ext, W, P, E_L, params):
    """Transfer function: firing rate output given inputs."""
    p = params.copy()
    p['E_L'] = E_L
    mu_V, sigma_V, T_V = get_fluct_regime_vars(Fe, Fi, Fe_ext, Fi_ext, W, **p)
    TvN = T_V * p['g_L'] / p['C_m']
    V_thre = threshold_func(mu_V, sigma_V, TvN, P)
    V_thre *= 1e3  # V to mV conversion (line 458 in Zerlaut.py)

    f_out = erfc((V_thre - mu_V) / (np.sqrt(2) * sigma_V)) / (2.0 * T_V)
    return f_out


def TF_with_moments(Fe, Fi, Fe_ext, Fi_ext, W, P, E_L, params):
    """TF that also returns all intermediate values for analysis."""
    p = params.copy()
    p['E_L'] = E_L
    mu_V, sigma_V, T_V = get_fluct_regime_vars(Fe, Fi, Fe_ext, Fi_ext, W, **p)
    TvN = T_V * p['g_L'] / p['C_m']
    V_thre_raw = threshold_func(mu_V, sigma_V, TvN, P)
    V_thre = V_thre_raw * 1e3  # V to mV
    terms, V, S, T = threshold_func_terms(mu_V, sigma_V, TvN, P)
    f_out = erfc((V_thre - mu_V) / (np.sqrt(2) * sigma_V)) / (2.0 * T_V)
    return {
        'f_out': f_out,
        'mu_V': mu_V,
        'sigma_V': sigma_V,
        'T_V': T_V,
        'TvN': TvN,
        'V_thre_raw': V_thre_raw,
        'V_thre_mV': V_thre,
        'V': V, 'S': S, 'T_norm': T,
        'terms': terms,
    }


# ============================================================
# STEP 4: Define drug states and base parameters
# ============================================================
base_params = dict(
    Q_e=1.5, tau_e=5.0, E_e=0.0,
    Q_i=5.0, tau_i=5.0, E_i=-80.0,
    g_L=10.0, C_m=200.0,
    N_tot=10000, p_connect_e=0.05, p_connect_i=0.05,
    g=0.2, K_ext_e=400, K_ext_i=0,
)

drug_states = {
    'Awake':     {'b_e': 5,  'tau_e': 5.0, 'tau_i': 5.0, 'E_L_e': -64.0},
    'Propofol':  {'b_e': 30, 'tau_e': 5.0, 'tau_i': 7.0, 'E_L_e': -64.0},
    'Prop+DOI':  {'b_e': 30, 'tau_e': 5.0, 'tau_i': 7.0, 'E_L_e': -61.2},
    'DOI':       {'b_e': 5,  'tau_e': 5.0, 'tau_i': 5.0, 'E_L_e': -61.2},
}

E_L_i = -65.0
tau_w = 500.0


# ============================================================
# STEP 5: Find self-consistent fixed points
# ============================================================
def make_params_for_state(state):
    """Build params dict with state-specific tau_i and tau_e."""
    p = base_params.copy()
    p['tau_e'] = state['tau_e']
    p['tau_i'] = state['tau_i']
    return p


def fixed_point_equations(x, P_E, P_I, state):
    """System of equations: TF_e(Fe,Fi) - Fe = 0, TF_i(Fe,Fi) - Fi = 0."""
    Fe, Fi = x
    if Fe < 0 or Fi < 0:
        return [100.0, 100.0]

    b_e = state['b_e']
    E_L_e = state['E_L_e']
    params = make_params_for_state(state)

    W_e = b_e * Fe * tau_w
    W_i = 0.0

    Fe_out = TF(Fe, Fi, 0.0, 0.0, W_e, P_E, E_L_e, params)
    Fi_out = TF(Fe, Fi, 0.0, 0.0, W_i, P_I, E_L_i, params)

    return [Fe_out - Fe, Fi_out - Fi]


def find_fixed_points(P_E, P_I, state, n_grid=50):
    """Grid search + fsolve to find all stable fixed points."""
    Fe_range = np.linspace(0.05, 15.0, n_grid)
    Fi_range = np.linspace(0.05, 15.0, n_grid)

    solutions = []
    for Fe0 in Fe_range:
        for Fi0 in Fi_range:
            try:
                sol = fsolve(fixed_point_equations, [Fe0, Fi0],
                             args=(P_E, P_I, state),
                             full_output=True)
                x, info, ier, msg = sol
                if ier == 1 and x[0] > 0.01 and x[1] > 0.01 and x[0] < 50 and x[1] < 50:
                    is_new = True
                    for prev in solutions:
                        if abs(prev[0] - x[0]) < 0.05 and abs(prev[1] - x[1]) < 0.05:
                            is_new = False
                            break
                    if is_new:
                        residual = fixed_point_equations(x, P_E, P_I, state)
                        if abs(residual[0]) < 1e-6 and abs(residual[1]) < 1e-6:
                            solutions.append(x)
            except Exception:
                pass

    return solutions


print("\n" + "=" * 80)
print("STEP 5: Self-Consistent Fixed Points")
print("=" * 80)

poly_sets = {
    'SACHA':  (P_E_SACHA, P_I_SACHA),
    'DIVOLO': (P_E_DIVOLO, P_I_DIVOLO),
}

all_results = {}

for poly_name, (P_E, P_I) in poly_sets.items():
    print(f"\n{'---' * 14}")
    print(f"  Polynomial: {poly_name}")
    print(f"{'---' * 14}")

    for state_name, state in drug_states.items():
        fps = find_fixed_points(P_E, P_I, state)
        print(f"\n  {state_name} (b_e={state['b_e']}, tau_i={state['tau_i']}, "
              f"E_L_e={state['E_L_e']}):")

        if not fps:
            print(f"    ** No fixed point found **")
            all_results[(poly_name, state_name)] = None
            continue

        for j, fp in enumerate(fps):
            Fe, Fi = fp
            print(f"    Fixed point #{j+1}: Fe*={Fe:.4f} Hz, Fi*={Fi:.4f} Hz")

        Fe_star, Fi_star = fps[0]
        all_results[(poly_name, state_name)] = (Fe_star, Fi_star)

        params = make_params_for_state(state)
        W_e = state['b_e'] * Fe_star * tau_w

        info_e = TF_with_moments(Fe_star, Fi_star, 0.0, 0.0, W_e,
                                  P_E, state['E_L_e'], params)
        info_i = TF_with_moments(Fe_star, Fi_star, 0.0, 0.0, 0.0,
                                  P_I, E_L_i, params)

        print(f"\n    --- Excitatory cell at fixed point ---")
        print(f"    W_e = {W_e:.2f} pA")
        print(f"    mu_V  = {info_e['mu_V']:.4f} mV")
        print(f"    sigma_V = {info_e['sigma_V']:.4f} mV")
        print(f"    T_V   = {info_e['T_V']:.4f} ms")
        print(f"    TvN   = {info_e['TvN']:.6f}")
        print(f"    V_thre (raw poly output) = {info_e['V_thre_raw']:.6f}")
        print(f"    V_thre (x1000, mV) = {info_e['V_thre_mV']:.4f} mV")
        print(f"    Normalized inputs: V={info_e['V']:.4f}, "
              f"S={info_e['S']:.4f}, T={info_e['T_norm']:.4f}")
        print(f"    TF_e output = {info_e['f_out']:.6f} Hz")

        print(f"\n    Polynomial term contributions (Excitatory):")
        for term_name, term_val in info_e['terms'].items():
            print(f"      {term_name:<10} = {term_val:>12.8f}")
        print(f"      {'SUM':<10} = {sum(info_e['terms'].values()):>12.8f}")

        print(f"\n    --- Inhibitory cell at fixed point ---")
        print(f"    mu_V  = {info_i['mu_V']:.4f} mV")
        print(f"    sigma_V = {info_i['sigma_V']:.4f} mV")
        print(f"    T_V   = {info_i['T_V']:.4f} ms")
        print(f"    TvN   = {info_i['TvN']:.6f}")
        print(f"    V_thre (raw poly output) = {info_i['V_thre_raw']:.6f}")
        print(f"    V_thre (x1000, mV) = {info_i['V_thre_mV']:.4f} mV")
        print(f"    Normalized inputs: V={info_i['V']:.4f}, "
              f"S={info_i['S']:.4f}, T={info_i['T_norm']:.4f}")
        print(f"    TF_i output = {info_i['f_out']:.6f} Hz")

        print(f"\n    Polynomial term contributions (Inhibitory):")
        for term_name, term_val in info_i['terms'].items():
            print(f"      {term_name:<10} = {term_val:>12.8f}")
        print(f"      {'SUM':<10} = {sum(info_i['terms'].values()):>12.8f}")


# ============================================================
# STEP 6: Summary comparison table
# ============================================================
print("\n" + "=" * 80)
print("STEP 6: Summary Comparison Table")
print("=" * 80)
print(f"\n{'State':<12} {'SACHA Fe*':>10} {'SACHA Fi*':>10} "
      f"{'DIVOLO Fe*':>11} {'DIVOLO Fi*':>11} "
      f"{'Fe ratio':>10} {'Fi ratio':>10}")
print("-" * 76)
for state_name in drug_states:
    s = all_results.get(('SACHA', state_name))
    d = all_results.get(('DIVOLO', state_name))
    s_fe = f"{s[0]:.4f}" if s else "N/A"
    s_fi = f"{s[1]:.4f}" if s else "N/A"
    d_fe = f"{d[0]:.4f}" if d else "N/A"
    d_fi = f"{d[1]:.4f}" if d else "N/A"
    if s and d:
        fe_r = f"{s[0]/d[0]:.3f}" if d[0] > 0 else "N/A"
        fi_r = f"{s[1]/d[1]:.3f}" if d[1] > 0 else "N/A"
    else:
        fe_r, fi_r = "N/A", "N/A"
    print(f"{state_name:<12} {s_fe:>10} {s_fi:>10} "
          f"{d_fe:>11} {d_fi:>11} "
          f"{fe_r:>10} {fi_r:>10}")


# ============================================================
# STEP 7: Sensitivity analysis
# ============================================================
print("\n" + "=" * 80)
print("STEP 7: Sensitivity Analysis")
print("=" * 80)

eps_EL = 0.1
eps_b = 0.1
eps_tau = 0.01

for poly_name, (P_E, P_I) in poly_sets.items():
    print(f"\n{'---' * 14}")
    print(f"  Polynomial: {poly_name}")
    print(f"{'---' * 14}")
    print(f"\n  {'State':<12} {'dTF_e/dEL_e':>14} {'dTF_e/db_e':>14} {'dTF_e/dtau_i':>14}")
    print(f"  {'-'*56}")

    for state_name, state in drug_states.items():
        result = all_results.get((poly_name, state_name))
        if result is None:
            print(f"  {state_name:<12} {'N/A':>14} {'N/A':>14} {'N/A':>14}")
            continue

        Fe, Fi = result
        params = make_params_for_state(state)
        b_e = state['b_e']
        E_L_e = state['E_L_e']
        W_e = b_e * Fe * tau_w

        f_base = TF(Fe, Fi, 0.0, 0.0, W_e, P_E, E_L_e, params)

        f_plus = TF(Fe, Fi, 0.0, 0.0, W_e, P_E, E_L_e + eps_EL, params)
        dTF_dEL = (f_plus - f_base) / eps_EL

        W_e_plus = (b_e + eps_b) * Fe * tau_w
        f_plus_b = TF(Fe, Fi, 0.0, 0.0, W_e_plus, P_E, E_L_e, params)
        dTF_db = (f_plus_b - f_base) / eps_b

        params_plus = params.copy()
        params_plus['tau_i'] = state['tau_i'] + eps_tau
        f_plus_tau = TF(Fe, Fi, 0.0, 0.0, W_e, P_E, E_L_e, params_plus)
        dTF_dtau = (f_plus_tau - f_base) / eps_tau

        print(f"  {state_name:<12} {dTF_dEL:>14.6f} {dTF_db:>14.6f} {dTF_dtau:>14.6f}")

    print(f"\n  Cross-sensitivity: DOI EL shift effect on Propofol state:")
    result_prop = all_results.get((poly_name, 'Propofol'))
    if result_prop:
        Fe, Fi = result_prop
        params = make_params_for_state(drug_states['Propofol'])
        b_e = drug_states['Propofol']['b_e']
        W_e = b_e * Fe * tau_w
        E_L_base = -64.0
        E_L_doi = -61.2

        f_base = TF(Fe, Fi, 0.0, 0.0, W_e, P_E, E_L_base, params)
        f_doi = TF(Fe, Fi, 0.0, 0.0, W_e, P_E, E_L_doi, params)
        print(f"    TF_e at E_L=-64.0: {f_base:.6f} Hz")
        print(f"    TF_e at E_L=-61.2: {f_doi:.6f} Hz")
        print(f"    Delta TF_e (DOI shift): {f_doi - f_base:.6f} Hz "
              f"({(f_doi - f_base)/f_base*100:.1f}%)")


# ============================================================
# STEP 8: Polynomial surface comparison
# ============================================================
print("\n" + "=" * 80)
print("STEP 8: Polynomial Surface Comparison")
print("=" * 80)

print("\n--- Threshold along mu_V axis (sigma_V=4.0, TvN=0.5) ---")
print(f"{'mu_V':>8} {'V':>8} {'Vthre_SACHA':>14} {'Vthre_DIVOLO':>14} {'Diff':>10}")
print("-" * 58)
for mu_V in np.arange(-70, -49, 2.0):
    V = (mu_V - (-60.0)) / 10.0
    vthre_s = P_E_SACHA[0] + P_E_SACHA[1]*V + P_E_SACHA[4]*V**2
    vthre_d = P_E_DIVOLO[0] + P_E_DIVOLO[1]*V + P_E_DIVOLO[4]*V**2
    print(f"{mu_V:>8.1f} {V:>8.3f} {vthre_s:>14.8f} {vthre_d:>14.8f} {vthre_s-vthre_d:>10.8f}")

print("\n--- Threshold along sigma_V axis (mu_V=-60.0, TvN=0.5) ---")
print(f"{'sigma_V':>8} {'S':>8} {'Vthre_SACHA':>14} {'Vthre_DIVOLO':>14} {'Diff':>10}")
print("-" * 58)
for sigma_V in np.arange(1.0, 11.0, 1.0):
    S = (sigma_V - 4.0) / 6.0
    vthre_s = P_E_SACHA[0] + P_E_SACHA[2]*S + P_E_SACHA[5]*S**2
    vthre_d = P_E_DIVOLO[0] + P_E_DIVOLO[2]*S + P_E_DIVOLO[5]*S**2
    print(f"{sigma_V:>8.1f} {S:>8.3f} {vthre_s:>14.8f} {vthre_d:>14.8f} {vthre_s-vthre_d:>10.8f}")

print("\n--- Threshold along TvN axis (mu_V=-60.0, sigma_V=4.0) ---")
print(f"{'TvN':>8} {'T':>8} {'Vthre_SACHA':>14} {'Vthre_DIVOLO':>14} {'Diff':>10}")
print("-" * 58)
for TvN in np.arange(0.0, 1.05, 0.1):
    T = (TvN - 0.5) / 1.0
    vthre_s = P_E_SACHA[0] + P_E_SACHA[3]*T + P_E_SACHA[6]*T**2
    vthre_d = P_E_DIVOLO[0] + P_E_DIVOLO[3]*T + P_E_DIVOLO[6]*T**2
    print(f"{TvN:>8.2f} {T:>8.3f} {vthre_s:>14.8f} {vthre_d:>14.8f} {vthre_s-vthre_d:>10.8f}")

print("\n--- Full threshold at actual fixed-point moments ---")
print(f"{'State':<12} {'Poly':<8} {'mu_V':>8} {'sig_V':>8} {'TvN':>8} "
      f"{'Vthre_raw':>12} {'Vthre_mV':>10} {'f_out':>8}")
print("-" * 80)
for state_name, state in drug_states.items():
    for poly_name, (P_E, P_I) in poly_sets.items():
        result = all_results.get((poly_name, state_name))
        if result is None:
            print(f"{state_name:<12} {poly_name:<8} {'--- no fixed point ---'}")
            continue
        Fe, Fi = result
        params = make_params_for_state(state)
        W_e = state['b_e'] * Fe * tau_w
        info = TF_with_moments(Fe, Fi, 0.0, 0.0, W_e, P_E, state['E_L_e'], params)
        print(f"{state_name:<12} {poly_name:<8} "
              f"{info['mu_V']:>8.3f} {info['sigma_V']:>8.3f} {info['TvN']:>8.5f} "
              f"{info['V_thre_raw']:>12.8f} {info['V_thre_mV']:>10.4f} "
              f"{info['f_out']:>8.4f}")

# ============================================================
# BONUS: Nullcline visualization data
# ============================================================
print("\n" + "=" * 80)
print("BONUS: Nullcline Data (Fe-nullcline and Fi-nullcline)")
print("=" * 80)

for state_name in ['Awake', 'Propofol']:
    state = drug_states[state_name]
    print(f"\n--- {state_name} ---")
    print(f"{'Fe_in':>8} | {'SACHA TF_e':>11} {'SACHA TF_i':>11} | "
          f"{'DIVOLO TF_e':>12} {'DIVOLO TF_i':>12}")
    print("-" * 65)

    for Fe_in in [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        Fi_in = Fe_in
        s_e, s_i, d_e, d_i = 0, 0, 0, 0
        for pn, (PE, PI) in poly_sets.items():
            params = make_params_for_state(state)
            W_e = state['b_e'] * Fe_in * tau_w
            try:
                tf_e = TF(Fe_in, Fi_in, 0.0, 0.0, W_e, PE, state['E_L_e'], params)
                tf_i = TF(Fe_in, Fi_in, 0.0, 0.0, 0.0, PI, E_L_i, params)
            except Exception:
                tf_e, tf_i = float('nan'), float('nan')

            if pn == 'SACHA':
                s_e, s_i = tf_e, tf_i
            else:
                d_e, d_i = tf_e, tf_i

        print(f"{Fe_in:>8.1f} | {s_e:>11.6f} {s_i:>11.6f} | "
              f"{d_e:>12.6f} {d_i:>12.6f}")


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
