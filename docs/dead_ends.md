# Dead Ends — Closed Investigations

<!-- CHECK THIS FILE before attempting ANY new polynomial fitting, interpolation,
     or TF reproduction approach. Every entry here represents hours of compute
     and analysis that definitively ruled out an approach. -->

## 1. Refitting CONFIG1 Polynomial (2026-03)
**Approach:** Re-run Sacha's exact fitting pipeline with his training data + parameters.
**Result:** FAILS — modern scipy ignores `seed=10` kwarg; Nelder-Mead hits maxiter without
converging to CONFIG1's minimum. All modern optimizers converge to a *different* global minimum
with P[2] = +0.004 (wrong sign), which fails TVB validation (no propofol collapse).
**Why it matters:** CONFIG1 is a *lucky local minimum* that only Sacha's older scipy could find.
**Script:** `code/analysis/tf_global_fit.py`

## 2. Global Optimization of TF Polynomial (2026-03-04)
**Approach:** Find the true MSE-optimal polynomial using DE, Basin-Hopping, L-BFGS-B, Powell, NM.
**Result:** Global optimum MSE=20.4 (7x better than CONFIG1's 147.7) but FAILS TVB validation.
P[2] = +0.004 → no propofol up-down oscillations → LZc stays at 0.94.
**Lesson:** Training data MSE is NOT a reliable proxy for TVB simulation quality.
**Script:** `code/analysis/tf_global_fit.py`
**Saved polynomial:** `code/analysis/P_E_global_opt.npy`

## 3. Weighted/Constrained TF Fitting (2026-03-04)
**Approach:** Weight low-rate region more heavily; constrain P[2] to be negative.
**Result:** Weighted fit → P[2] = +0.009 (FAILS). Constrained fits change ALL coefficients,
not just P[2], so no intermediate solution preserves CONFIG1's qualitative behavior.
**Script:** `code/analysis/tf_constrained_fit.py`
**Saved polynomial:** `code/analysis/P_E_weighted.npy`, `P_E_constrained_best.npy`

## 4. Option A — Refit with Di Volo Cell Parameters (2026-03-04)
**Approach:** Generate TF training data using Di Volo's cell config (E_L=-65, a=0, b=0)
via Sacha's `tf_simulation_fast.py`, then fit polynomial.
**Result:** P[2] = -0.0075 — nearly identical to CONFIG1 (-0.0079), NOT Di Volo (-0.0235).
**Lesson:** Sacha's tf_simulation_fast produces CONFIG1-like polynomials regardless of E_L.
The difference between CONFIG1 and Di Volo is NOT the cell parameters — it's the fitting code.
**Script:** `code/analysis/fit_option_a.py`

## 5. Option B — Zerlaut Inverse Moment-Space Approach (2026-03-04)
**Approach:** Reproduce Di Volo's polynomial using Zerlaut 2016's original inverse method
(compute moments analytically, then fit threshold function).
**Result:** All fitting methods converge to P[2] = -0.001 (10x too small).
Di Volo reference polynomial on this data: RMSE=35.8 Hz, R²=-545 (catastrophic).
**Root cause:** Zerlaut's inverse approach uses current injection (ADDITIVE noise → Gaussian
→ tiny polynomial corrections). Di Volo/Sacha use conductance injection (MULTIPLICATIVE noise
→ non-Gaussian → large |P[2]|). The approaches produce fundamentally different data.
**Script:** `code/analysis/option_b_zerlaut_tf.py`

## 6. Polynomial Interpolation Sweep (2026-03-05)
**Approach:** Blend CONFIG1 and Di Volo: P_hybrid = alpha * CONFIG1 + (1-alpha) * DiVolo.
Test alpha = 0.0 to 1.0 in 0.1 steps (44 TVB simulations on Great Lakes).
**Result:** NO sweet spot exists. The two effects are mutually exclusive:
- Propofol collapse requires alpha >= 0.8 (CONFIG1-dominant)
- DOI > Awake requires alpha <= 0.5 (DiVolo-dominant)
- Sharp nonlinear bifurcation at alpha ~0.8 (not gradual)
**Lesson:** Propofol collapse requires MULTIPLE polynomial coefficients to align simultaneously.
Linear blending cannot preserve CONFIG1's specific coefficient pattern while incorporating
Di Volo's stronger sigma_V response. The tradeoff is absolute in interpolation space.
**Script:** `code/analysis/tvb_interpolation_sweep.py`

## 7. Global Coupling Sweep with Di Volo Polynomial (2026-03-05, Gemini)
**Approach:** Increase global coupling a ∈ [0.3, 0.6, 0.9, 1.2] to boost network drive (Fe_ext)
for Di Volo polynomial. Hypothesis: more drive → bistability → propofol collapse.
**Result:** FAILS. LZc stays ~0.93 across all coupling values. At a=1.2, Propofol LZc=0.96
(even higher than Awake=0.96). Di Volo's S-curve is too flat for UP-DOWN transitions regardless
of network drive level.
**Script:** `code/scripts/archive/tvb_coupling_sweep.py` (Gemini)

## 8. DOI b_e Drop with CONFIG1 Polynomial (2026-03-05, Gemini)
**Approach:** Model DOI as b_e reduction (adaptation drop) + E_L shift, instead of E_L shift alone.
b_e ∈ [0, 1, 2, 3, 4, 5] for DOI condition, CONFIG1 polynomial.
**Result:** NUMERICALLY WORKS — DOI LZc > Awake LZc at b_e ≤ 2.0 while propofol collapse preserved.
At b_e=0: DOI LZc=0.9449 > Awake=0.9428. At b_e=1: DOI LZc=0.9446 > Awake=0.9428.
**BUT: PHYSICALLY UNJUSTIFIED** — 5-HT2A agonism has no known mechanism for reducing
spike-triggered adaptation. The LZc increase comes from a nonphysical parameter change.
Effect size is also tiny (~0.002). Not a viable solution for the paper.
**Script:** `code/scripts/archive/tvb_b_e_sweep.py` (Gemini)

---

## Summary: What We Know About the Polynomial Landscape

The core tradeoff: **P[2] controls both propofol collapse AND DOI sensitivity, but in opposite directions.**

- P[2] ~ -0.008 (CONFIG1): Propofol collapses (LZc=0.53), but DOI ~ Awake
- P[2] ~ -0.024 (DiVolo): DOI > Awake (+0.020), but propofol stays high (0.93)
- No intermediate P[2] achieves both (proven by interpolation sweep)

### Still Open

**NEW DISCOVERY (2026-03-05):** The 60Hz filter is the dominant factor controlling P[2].
Experiments A-D showed that Di Volo's `Fout<60` filter vs Sacha's NaN-only filter explains
most of the CONFIG1-vs-DiVolo polynomial difference. See `docs/results_tables.md`.

**LEADING HYPOTHESIS (2026-03-05): DOI is massively underdosed.**
Our E_L_E_PSI=-61.2 gives only 12% g_K reduction. Martin's full 5-HT2A effect is 35% g_K
reduction (E_L≈-54 from our baseline). The production script already uses Zerlaut_gK_gNa +
heterogeneous receptor map — the model is correct, only the endpoint magnitude is wrong.
Sweep RUNNING: `code/scripts/tvb_doi_el_sweep.py` with E_L_E_PSI ∈ [-61.2, -59, -57, -55, -54, -53].

Remaining strategies if E_L sweep fails:
1. **Noise amplitude per condition**: 5-HT2A increases neural variability — sweep σ_noise.
2. **P_I manipulation**: Sweep P_I coefficients while keeping CONFIG1 P_E.
3. **Condition-specific coupling**: Different coupling_a for DOI.
4. **Variable frequency filter** (80-120Hz cutoffs) — intermediate P[2] via polynomial refit.
5. **Weight function on firing rate**: Soft weighting instead of hard cutoff.

Closed strategies (see numbered dead ends above):
- Coupling sweep a=0.3→1.2 with DiVolo: CLOSED (dead end #7)
- b_e drop for DOI with CONFIG1: numerically works but physically unjustified (dead end #8)
- Brian2 vs NumPy data: COMPLETED — Brian2 does NOT reproduce DiVolo (see results_tables.md)
