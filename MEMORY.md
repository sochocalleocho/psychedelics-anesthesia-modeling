# Persistent Memory — Key Findings

<!-- This file captures accumulated knowledge that a new Claude session needs.
     Keep it factual and concise. No speculation. -->

## Project Status (2026-03-05)

**Where we are:** CONFIG1 polynomial produces propofol collapse (LZc 0.94→0.53) but DOI≈Awake.
Di Volo polynomial produces DOI>Awake (+0.020) but no propofol collapse. No single polynomial
achieves both. The core tradeoff is in P[2] (σ_V coefficient). See `docs/dead_ends.md` "Still Open"
for untried strategies.

**Closed since last update:**
- Coupling sweep (a=0.3→1.2) with Di Volo: FAILED — S-curve too flat (dead end #7)
- DOI b_e drop with CONFIG1: numerically works but physically unjustified (dead end #8)

**What to try next** (prioritized):
1. **RUNNING: DOI E_L endpoint sweep** — E_L_E_PSI ∈ [-61.2, -59, -57, -55, -54, -53].
   Script: `code/scripts/tvb_doi_el_sweep.py`, Slurm: `run_doi_el_sweep.sh`
2. If sweep finds a working E_L: **update production script** with correct endpoint
3. If sweep fails: try noise amplitude, P_I manipulation, condition-specific coupling
4. Variable frequency filter (80-120Hz cutoffs) — polynomial approach, lower priority

## Two P_E_SACHA Values

Production scripts use different coefficients than the .npy file. Both produce propofol collapse.

**Production scripts** (validated, used in ALL cached results):
```
[-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
  0.00341614, -0.01156433, 0.00194753,  0.00274079, -0.01066769]
```

**.npy file** (`RS-cell0_CONFIG1_fit.npy`):
```
[-0.06373626, 0.00786373, -0.00791921, 0.00170594, -0.00049024,
  0.00329717, -0.01731714, 0.00435419, 0.00054003, -0.02073425]
```

Discrepancy origin unknown. Both have P[2] ≈ -0.008. Use production values.

## Key Findings

### P[2] Is the Critical Coefficient
- P[2] (σ_V sensitivity) controls propofol dynamics: negative → collapse, positive → fails
- P[2] ~ -0.008 (CONFIG1): propofol LZc=0.53 ✓, DOI-Awake=-0.003 ✗
- P[2] ~ -0.024 (DiVolo): propofol LZc=0.93 ✗, DOI-Awake=+0.020 ✓
- Linear interpolation proved no sweet spot exists (dead end #6)

### 60Hz Filter Is the Dominant Factor in P[2]
- Di Volo filters training data to Fout<60Hz (~460 of ~2050 points)
- Sacha only removes NaN/Inf (keeps all ~2050 points)
- With 60Hz filter: P[2] always small (±0.006) regardless of data source
- Without filter: P[2] sign unpredictable, depends on data source
- Mechanism: below 60Hz neurons fire from noise fluctuations (σ_V matters → large |P[2]|).
  Above 60Hz, mean drive dominates (σ_V irrelevant → pulls P[2] toward zero)

### Brian2 Data Does NOT Reproduce Di Volo
- Brian2 conductance-based sim (EL=-65, Di Volo cell) + 60Hz filter → P[2]=+0.004 (POSITIVE)
- Brian2 + no filter → P[2]=+0.037 (even more positive)
- Only NumPy data at EL=-65 without filter gave large negative P[2] (-0.032)
- Optimizer landscape is extremely rough: P[2] ranges -0.032 to +0.037 across experiments
- Di Volo's polynomial likely requires undocumented steps or specific scipy version

### CONFIG1 Is a Lucky Local Minimum
- Modern scipy cannot reproduce CONFIG1 — Nelder-Mead `seed` kwarg silently ignored
- All modern optimizers converge to P[2]=+0.004 (wrong sign), which fails TVB validation
- Global MSE optimum (MSE=20.4) is 7× better fit but FAILS TVB (no propofol collapse)
- CONFIG1 (MSE=147.7) works despite worse fit — training MSE ≠ simulation quality

### P_E Determines Propofol, P_I Modulates DOI
- Swapping P_E is what controls propofol collapse
- Swapping P_I modulates DOI sensitivity
- Di Volo P_I removes propofol collapse even from CONFIG1 P_E

### Global Coupling Cannot Rescue Di Volo (Gemini, 2026-03-05)
- Sweeping coupling a from 0.3 to 1.2 with Di Volo polynomial: propofol never collapses
- Di Volo's S-curve is too flat for bistability at any network drive level
- Confirms: the polynomial IS the bottleneck for propofol, not the network parameters

### b_e Drop Works Numerically but Not Physically (Gemini, 2026-03-05)
- CONFIG1 + DOI modeled as b_e reduction (0-2) + E_L shift → DOI LZc > Awake
- Effect size tiny (~0.002), and 5-HT2A has no known mechanism for adaptation reduction
- Lesson: non-polynomial parameters CAN modulate DOI-Awake gap, just need physical justification

### DOI Is Massively Underdosed (2026-03-05) ← CRITICAL
- Martin's full 5-HT2A effect: g_K 8.214→5.37 (34.6% drop), effective E_L -65→-55 (10 mV shift)
- Our script: E_L_E_PSI=-61.2 from E_L_E_BASE=-64 → g_K 8.143→7.171 (11.9% drop, only 2.8 mV)
- **We're applying ~1/3 of the intended g_K reduction**
- The -61.2 endpoint was calibrated for Martin's E_L_start=-63, NOT our E_L_start=-64
- Correct endpoint for our baseline: E_L_E_PSI ≈ -54.0 (matching Martin's g_K=5.37 target)
- Sweep running on Great Lakes: E_L_E_PSI ∈ [-61.2, -59, -57, -55, -54, -53]
- NOTE: production script already uses Zerlaut_gK_gNa + heterogeneous 5-HT2A receptor map,
  so the model infrastructure is correct — only the endpoint magnitude was wrong
