#!/usr/bin/env python3
"""
Option B: Reproduce Di Volo's polynomial using Zerlaut's TF generation approach.

Key differences from Option A (which used Sacha's tf_simulation_fast):
  1. Grid:  inverse moment-space (μ_V, σ_V, τ_V/τ_m) → compute required inputs
  2. Solver: Euler integration (not Heun)
  3. Noise:  Poisson shot noise w/ exponential ISIs (not binomial per timestep)
  4. Three fitting methods compared:
       - Zerlaut: minimize(NM default) on Veff, then leastsq(LM) on Fout
       - DiVolo:  SLSQP on Veff, then NM on Fout (Fout<60 filter)
       - Sacha:   SLSQP on Veff, then NM on Fout (all nonzero)

Cell: Di Volo RS-cellbis (E_L=-65, a=0, b=0, delta_v=2, V_reset=-65)

Usage:
  python option_b_zerlaut_tf.py [sampling]
  sampling = low | medium | high  (default: medium)
"""

import numpy as np
import numba
import scipy.special as sp_spec
from scipy.optimize import minimize, leastsq
import time
import sys

# ============================================================
# Cell parameters — Di Volo RS-cellbis (SI units)
# ============================================================
CELL = {
    'El': -65e-3,       # V
    'Gl': 10e-9,        # S
    'Cm': 200e-12,      # F
    'Vthre': -50e-3,    # V
    'Vreset': -65e-3,   # V
    'delta_v': 2e-3,    # V
    'a': 0.,            # S  (no subthreshold adaptation)
    'b': 0.,            # A  (TF always fit at b=0)
    'tauw': 0.5,        # s  (500 ms)
    'Trefrac': 5e-3,    # s
}

# ============================================================
# Reference polynomials (NORMALIZED 10-coefficient form)
# ============================================================
P_E_DIVOLO = np.array([
    -0.04983106,  0.005063550882777035, -0.023470121807314552,
     0.0022951513725067503, -0.0004105302652029825,  0.010547051343547399,
    -0.03659252821136933,  0.007437487505797858,  0.001265064721846073,
    -0.04072161294490446
])

P_E_SACHA = np.array([
    -0.06373626,  0.00786373, -0.00791921,  0.00170594,
    -0.00049024,  0.00329717, -0.01731714,  0.00435419,
     0.00054003, -0.02073425
])

# Normalization constants (Zerlaut 2018 standard — used by ALL codebases)
muV0, DmuV0 = -60e-3, 10e-3   # V
sV0,  DsV0  =   4e-3,  6e-3   # V
TvN0, DTvN0 =    0.5,  1.0    # dimensionless

# ============================================================
# 1. SHOT NOISE GENERATION  (from Zerlaut)
# ============================================================
def generate_conductance_shotnoise(freq, t, N, Q, Tsyn, g0=0, seed=0):
    """Poisson shot noise with exponential decay. Returns conductance trace."""
    if freq == 0:
        freq = 1e-9
    n_events = max(int(3 * freq * t[-1] * N), 1)
    np.random.seed(seed=seed)
    spike_events = np.cumsum(np.random.exponential(1. / (N * freq), n_events))
    spike_events = np.concatenate([spike_events, [t[-1] + 1.]])
    g = np.ones(t.size) * g0
    dt = t[1] - t[0]
    decay = np.exp(-dt / Tsyn)
    t0 = t[0]
    event = 0
    for i in range(1, t.size):
        g[i] = g[i - 1] * decay
        while spike_events[event] <= (t[i] - t0):
            g[i] += Q
            event += 1
    return g


# ============================================================
# 2. NUMBA EULER SOLVER  (shunt conductance input — Zerlaut approach)
# ============================================================
@numba.jit(nopython=True)
def adexp_euler_shunt(t, I, Gs, muV_target,
                      El, Gl, Cm, Vthre, Vreset,
                      Trefrac, delta_v, a, b, tauw):
    """Euler AdExp with DC + noise current and shunt conductance."""
    one_over_dv = 0. if delta_v == 0 else 1. / delta_v
    last_spike = -1e10
    V0, V1 = Vreset, Vreset
    nspikes = 0
    dt = t[1] - t[0]
    w = 0.

    for i in range(len(t) - 1):
        V0 = V1
        w = w + dt / tauw * (a * (V0 - El) - w)
        i_exp = Gl * delta_v * np.exp((V0 - Vthre) * one_over_dv)
        if (t[i] - last_spike) > Trefrac:
            V1 = V0 + dt / Cm * (I[i] + i_exp - w
                                  + Gl * (El - V0)
                                  + Gs * (muV_target - V0))
        if V1 >= Vthre + 5. * delta_v:
            V1 = Vreset
            w = w + b
            last_spike = t[i + 1]
            nspikes += 1

    return nspikes / t[-1]


# ============================================================
# 3. INVERSE MOMENT-SPACE → INPUT PARAMETERS  (from Zerlaut)
# ============================================================
def moments_to_inputs(muGn, muV, sV, Ts_ratio, Gl, Cm, El, DV):
    """Convert target (muV, sV, TvN) to (I0, Gs, f, Q, Ts)."""
    Tm0 = Cm / Gl
    Ts  = Ts_ratio * Tm0
    muG = muGn * Gl
    Gs  = muG - Gl                          # shunt conductance
    Tv  = Ts + Tm0 / muGn                   # membrane time constant
    I0  = Gl * (muV - El)                   # DC current for target mean
    f   = 2000. + 0 * I0                    # Hz — fixed Poisson rate
    Q   = muG * sV * np.sqrt(Tv / f) / Ts / DV   # quantal size → target σ_V
    return I0, Gs, f, Q, Ts


# ============================================================
# 4. TF DATA GENERATION — Zerlaut inverse moment-space grid
# ============================================================
def generate_tf_data(params, DmuV=8, DsV=10, DTvN=6,
                     DV=20e-3, sampling='medium'):
    """Generate transfer function training data using Zerlaut's approach."""

    # Sampling parameters
    if sampling == 'low':
        SEED = np.arange(2) + 1;  dt = 1e-4;  tstop = 2.
    elif sampling == 'medium':
        SEED = np.arange(2) + 1;  dt = 5e-5;  tstop = 5.
    else:  # high (paper quality)
        SEED = np.arange(3) + 1;  dt = 1e-5;  tstop = 10.

    El, Gl, Cm = params['El'], params['Gl'], params['Cm']

    # ---- Grid ranges (Zerlaut 2018 defaults) ----
    muV_min, muV_max = -70e-3, -55e-3
    sV_min1, sV_max1 =   5e-3,   9e-3   # σ_V range at most negative μ_V
    sV_min2, sV_max2 =   1e-3,   5e-3   # σ_V range at most positive μ_V
    Ts_ratio = 0.25                       # τ_syn / τ_m
    muGn_min, muGn_max = 1.15, 8.

    t = np.arange(int(tstop / dt)) * dt
    n_steps = len(t)

    # Build meshgrid
    muV_arr  = np.linspace(muV_min, muV_max, DmuV, endpoint=True)
    DmuG     = DTvN
    Tv_ratio = np.linspace(1. / muGn_max + Ts_ratio,
                           1. / muGn_min + Ts_ratio, DmuG, endpoint=True)
    muGn_arr = 1. / (Tv_ratio - Ts_ratio)

    muV, sV, muGn = np.meshgrid(muV_arr, np.zeros(DsV), muGn_arr)

    # Trapezoidal σ_V range (varies with μ_V)
    for i in range(DmuV):
        sv1 = sV_min1 + i * (sV_min2 - sV_min1) / (DmuV - 1)
        sv2 = sV_max1 + i * (sV_max2 - sV_max1) / (DmuV - 1)
        for l in range(DmuG):
            sV[:, i, l] = np.linspace(sv1, sv2, DsV, endpoint=True)

    # Compute required inputs for each grid point
    I0, Gs, f, Q, Ts = moments_to_inputs(
        muGn, muV, sV, Ts_ratio * np.ones(muGn.shape), Gl, Cm, El, DV)

    Fout = np.zeros((DsV, DmuV, DmuG, len(SEED)))
    total = DmuV * DsV * DmuG * len(SEED)
    done = 0
    t0 = time.time()

    # Warm up Numba JIT
    print("  Warming up Numba...")
    _dummy = adexp_euler_shunt(
        np.arange(100) * 1e-4, np.zeros(100), 0., -65e-3,
        El, Gl, Cm, params['Vthre'], params['Vreset'],
        params['Trefrac'], params['delta_v'], params['a'], params['b'], params['tauw'])

    print(f"  Running {total} simulations ({n_steps} timesteps each)...")

    for i_muV in range(DmuV):
        print(f'    muV = {1e3 * muV[0, i_muV, 0]:.1f} mV')
        for i_sV in range(DsV):
            for ig in range(DmuG):
                for i_s in range(len(SEED)):
                    sv = int(SEED[i_s] + i_muV + i_sV + ig)

                    Ge = generate_conductance_shotnoise(
                        f[i_sV, i_muV, ig], t, 1.,
                        Q[i_sV, i_muV, ig], Ts[i_sV, i_muV, ig],
                        g0=0, seed=sv)
                    Gi = generate_conductance_shotnoise(
                        f[i_sV, i_muV, ig], t, 1.,
                        Q[i_sV, i_muV, ig], Ts[i_sV, i_muV, ig],
                        g0=0, seed=sv ** 2 + 1)

                    I_arr = np.ones(n_steps) * I0[i_sV, i_muV, ig] + (Ge - Gi) * DV

                    Fout[i_sV, i_muV, ig, i_s] = adexp_euler_shunt(
                        t, I_arr,
                        Gs[i_sV, i_muV, ig],
                        muV[i_sV, i_muV, ig],
                        El, Gl, Cm,
                        params['Vthre'], params['Vreset'],
                        params['Trefrac'], params['delta_v'],
                        params['a'], params['b'], params['tauw'])

                    done += 1
                    if done % 100 == 0:
                        elapsed = time.time() - t0
                        eta = elapsed / done * (total - done)
                        print(f'      {done}/{total}  '
                              f'({elapsed:.0f}s elapsed, ETA {eta:.0f}s)')

    TvN = Ts_ratio + 1. / muGn

    return {
        'muV':   muV.flatten(),               # V (SI)
        'sV':    sV.flatten(),                 # V (SI)
        'TvN':   TvN.flatten(),                # dimensionless
        'muGn':  muGn.flatten(),
        'Fout':  Fout.mean(axis=-1).flatten(), # Hz
        's_Fout': Fout.std(axis=-1).flatten(),
    }


# ============================================================
# 5. THRESHOLD & RATE FUNCTIONS  (normalized form)
# ============================================================
def threshold_norm(P, muV, sV, TvN):
    """10-coefficient normalized threshold — same form as ALL codebases."""
    V = (muV - muV0) / DmuV0
    S = (sV  - sV0)  / DsV0
    T = (TvN - TvN0) / DTvN0
    return (P[0] + P[1]*V + P[2]*S + P[3]*T
            + P[4]*V**2 + P[5]*S**2 + P[6]*T**2
            + P[7]*V*S + P[8]*V*T + P[9]*S*T)


def erfc_rate(muV, sV, TvN, Vthre, Gl, Cm):
    return 0.5 / TvN * Gl / Cm * sp_spec.erfc((Vthre - muV) / np.sqrt(2) / sV)


def inv_Vthre(Y, muV, sV, TvN, Gl, Cm):
    """Effective threshold from observed firing rate (inverse erfc)."""
    return muV + np.sqrt(2) * sV * sp_spec.erfcinv(Y * 2. * TvN * Cm / Gl)


# ============================================================
# 6. FITTING METHODS
# ============================================================

def fit_zerlaut(Fout, muV, sV, TvN, muGn, Gl, Cm):
    """Zerlaut 2018: minimize(NM) on Veff, then scipy.optimize.leastsq on Fout."""
    ok = Fout > 0
    m, s, t, g, f = muV[ok], sV[ok], TvN[ok], muGn[ok], Fout[ok]
    veff = inv_Vthre(f, m, s, t, Gl, Cm)

    P0 = np.zeros(10);  P0[0] = -45e-3

    def res_thresh(p):
        return np.mean((veff - threshold_norm(p, m, s, t)) ** 2) / len(veff)

    r1 = minimize(res_thresh, P0, tol=1e-18, options={'maxiter': int(1e5)})
    print(f"    Stage 1 (NM→Veff):  success={r1.success}  fun={r1.fun:.3e}")

    def res_fout(p, muV, sV, TvN, muGn, Fout):
        return Fout - erfc_rate(muV, sV, TvN, threshold_norm(p, muV, sV, TvN), Gl, Cm)

    P = leastsq(res_fout, r1.x, args=(m, s, t, g, f))[0]
    return P


def fit_divolo(Fout, muV, sV, TvN, muGn, Gl, Cm):
    """Di Volo 2019: SLSQP on Veff, then NM on Fout (Fout<60 filter)."""
    ok = (Fout > 0) & (Fout < 60.)
    m, s, t, g, f = muV[ok], sV[ok], TvN[ok], muGn[ok], Fout[ok]
    veff = inv_Vthre(f, m, s, t, Gl, Cm)

    P0 = np.zeros(10)
    P0[0] = veff.mean();  P0[1:4] = 1e-3

    def res_thresh(p):
        return np.mean((veff - threshold_norm(p, m, s, t)) ** 2)

    r1 = minimize(res_thresh, P0, method='SLSQP',
                  options={'ftol': 1e-15, 'maxiter': 40000})
    print(f"    Stage 1 (SLSQP→Veff): success={r1.success}  fun={r1.fun:.3e}")

    def res_fout(p):
        return np.mean((f - erfc_rate(m, s, t,
                        threshold_norm(p, m, s, t), Gl, Cm)) ** 2)

    r2 = minimize(res_fout, r1.x, method='nelder-mead',
                  options={'xatol': 1e-5, 'maxiter': 50000})
    print(f"    Stage 2 (NM→Fout):   success={r2.success}  fun={r2.fun:.3e}")
    return r2.x


def fit_sacha(Fout, muV, sV, TvN, muGn, Gl, Cm):
    """Sacha 2025: SLSQP on Veff, then NM on Fout (all nonzero data)."""
    ok = Fout > 0
    m, s, t, g, f = muV[ok], sV[ok], TvN[ok], muGn[ok], Fout[ok]
    veff = inv_Vthre(f, m, s, t, Gl, Cm)

    P0 = np.zeros(10);  P0[0] = -0.050

    def res_thresh(p):
        return np.mean((veff - threshold_norm(p, m, s, t)) ** 2)

    r1 = minimize(res_thresh, P0, method='SLSQP', tol=1e-20,
                  options={'maxiter': int(1e6), 'ftol': 1e-20})
    print(f"    Stage 1 (SLSQP→Veff): success={r1.success}  fun={r1.fun:.3e}")

    def res_fout(p):
        return np.mean((f - erfc_rate(m, s, t,
                        threshold_norm(p, m, s, t), Gl, Cm)) ** 2)

    r2 = minimize(res_fout, r1.x, method='Nelder-Mead',
                  options={'maxiter': int(1e6), 'xatol': 1e-15, 'fatol': 1e-15})
    print(f"    Stage 2 (NM→Fout):   success={r2.success}  fun={r2.fun:.3e}")
    return r2.x


# ============================================================
# 7. MAIN
# ============================================================
if __name__ == '__main__':
    sampling = sys.argv[1] if len(sys.argv) > 1 else 'medium'

    # Grid sizes
    grids = {'low': (6, 8, 6), 'medium': (8, 10, 6), 'high': (10, 12, 8)}
    DmuV, DsV, DTvN = grids.get(sampling, grids['medium'])

    print("=" * 70)
    print("OPTION B: Reproduce Di Volo polynomial")
    print("  Method:  Zerlaut inverse moment-space grid")
    print("  Solver:  Euler + Poisson shot noise")
    print(f"  Cell:    RS-cellbis  E_L={1e3*CELL['El']:.0f} mV, "
          f"a={CELL['a']}, b={CELL['b']}, δ_v={1e3*CELL['delta_v']:.0f} mV")
    print(f"  Grid:    {DmuV}×{DsV}×{DTvN}  sampling={sampling}")
    print("=" * 70)

    # ---- Step 1: Generate TF data ----
    print(f"\nStep 1: Generate TF training data")
    t0_all = time.time()
    data = generate_tf_data(CELL, DmuV=DmuV, DsV=DsV, DTvN=DTvN,
                            DV=20e-3, sampling=sampling)
    t_gen = time.time() - t0_all
    print(f"\n  Generation: {t_gen:.1f}s")
    print(f"  Grid points: {len(data['Fout'])}")
    print(f"  Non-zero:    {np.count_nonzero(data['Fout'])}")
    print(f"  Fout range:  [{data['Fout'].min():.1f}, {data['Fout'].max():.1f}] Hz")

    np.savez('option_b_tf_data.npz', **data)
    print("  Saved: option_b_tf_data.npz")

    muV  = data['muV']
    sV   = data['sV']
    TvN  = data['TvN']
    muGn = data['muGn']
    Fout = data['Fout']
    Gl, Cm = CELL['Gl'], CELL['Cm']

    # ---- Step 2: Fit with three methods ----
    methods = [
        ("Zerlaut (minimize+leastsq)", fit_zerlaut),
        ("DiVolo  (SLSQP+NM, F<60)",   fit_divolo),
        ("Sacha   (SLSQP+NM, all)",    fit_sacha),
    ]
    results = {}

    for name, func in methods:
        print(f"\n{'='*70}")
        print(f"  FIT: {name}")
        print(f"{'='*70}")
        t0_fit = time.time()
        P = func(Fout, muV, sV, TvN, muGn, Gl, Cm)
        t_fit = time.time() - t0_fit
        results[name] = P
        print(f"  Time: {t_fit:.1f}s")

    # ---- Step 3: Compare ----
    labels = ['P[0] const', 'P[1] muV', 'P[2] σV', 'P[3] TvN',
              'P[4] muV²', 'P[5] σV²', 'P[6] TvN²',
              'P[7] muV·σV', 'P[8] muV·TvN', 'P[9] σV·TvN']

    print(f"\n{'='*70}")
    print("COEFFICIENT COMPARISON (all normalized)")
    print(f"{'='*70}")

    header = f"{'':15s} {'DiVolo_ref':>11s}"
    for name in results:
        short = name.split('(')[0].strip()
        header += f" {short:>12s}"
    header += f" {'CONFIG1':>11s}"
    print(header)
    print("-" * (15 + 11 + 12 * len(results) + 11 + 4))

    for i, lab in enumerate(labels):
        row = f"{lab:15s} {P_E_DIVOLO[i]:>+11.6f}"
        for name in results:
            row += f" {results[name][i]:>+12.6f}"
        row += f" {P_E_SACHA[i]:>+11.6f}"
        print(row)

    # MSE
    print(f"\nMSE vs Di Volo reference:")
    for name, P in results.items():
        short = name.split('(')[0].strip()
        print(f"  {short:12s} {np.mean((P - P_E_DIVOLO)**2):.3e}")

    print(f"\nMSE vs CONFIG1 reference:")
    for name, P in results.items():
        short = name.split('(')[0].strip()
        print(f"  {short:12s} {np.mean((P - P_E_SACHA)**2):.3e}")

    # CRITICAL: P[2]
    print(f"\n{'='*70}")
    print("CRITICAL:  P[2] (σ_V coefficient — determines propofol dynamics)")
    print(f"{'='*70}")
    print(f"  Di Volo ref:  {P_E_DIVOLO[2]:+.6f}")
    print(f"  CONFIG1 ref:  {P_E_SACHA[2]:+.6f}")
    for name, P in results.items():
        short = name.split('(')[0].strip()
        print(f"  {short:12s}  {P[2]:+.6f}")
    print(f"  Option A:     -0.007493  (Sacha generation, E_L=-65)")

    # Fit quality: RMSE of predicted vs actual firing rates
    print(f"\n{'='*70}")
    print("FIT QUALITY:  RMSE on firing rate (Hz)")
    print(f"{'='*70}")
    ok = Fout > 0
    for name, P in results.items():
        Fout_pred = erfc_rate(muV[ok], sV[ok], TvN[ok],
                              threshold_norm(P, muV[ok], sV[ok], TvN[ok]),
                              Gl, Cm)
        rmse = np.sqrt(np.mean((Fout[ok] - Fout_pred) ** 2))
        r2 = 1 - np.sum((Fout[ok] - Fout_pred)**2) / np.sum((Fout[ok] - Fout[ok].mean())**2)
        short = name.split('(')[0].strip()
        print(f"  {short:12s}  RMSE={rmse:.3f} Hz   R²={r2:.6f}")

    # Also test Di Volo and CONFIG1 references on this data
    for name, P in [("DiVolo_ref", P_E_DIVOLO), ("CONFIG1_ref", P_E_SACHA)]:
        Fout_pred = erfc_rate(muV[ok], sV[ok], TvN[ok],
                              threshold_norm(P, muV[ok], sV[ok], TvN[ok]),
                              Gl, Cm)
        rmse = np.sqrt(np.mean((Fout[ok] - Fout_pred) ** 2))
        r2 = 1 - np.sum((Fout[ok] - Fout_pred)**2) / np.sum((Fout[ok] - Fout[ok].mean())**2)
        print(f"  {name:12s}  RMSE={rmse:.3f} Hz   R²={r2:.6f}")

    # Save
    for name, P in results.items():
        tag = name.split('(')[0].strip().lower()
        np.save(f'P_E_optB_{tag}.npy', P)
    print(f"\nSaved polynomial files.")

    total_time = time.time() - t0_all
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("\nDONE.")
