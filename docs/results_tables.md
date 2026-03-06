# Complete Results Tables

<!-- Reference data from all TVB validation runs. Source of truth: figures/lzc_results_cache.json -->

## Single-Seed LZc Results (N=1, seed=0)

### By Polynomial
| Polynomial | P[2]   | Awake | Propofol | P+DOI | DOI   | Prop collapse? |
|------------|--------|-------|----------|-------|-------|----------------|
| CONFIG1    | -0.008 | 0.945 | 0.532    | 0.730 | 0.942 | YES            |
| GLOBAL     | +0.004 | 0.947 | 0.940    | 0.548 | 0.947 | NO (FAILS)     |
| WEIGHTED   | +0.009 | 0.912 | 0.951    | 0.954 | 0.912 | NO (FAILS)     |
| P2=-0.001  | -0.001 | 0.943 | 0.941    | 0.886 | 0.946 | NO             |
| P2=-0.015  | -0.015 | 0.935 | 0.802    | 0.784 | 0.950 | Partial only   |
| DiVolo     | -0.024 | 0.936 | 0.921    | 0.928 | 0.942 | NO             |

### Polynomial x E_L_e
| Config              | Awake | Propofol | P+DOI | DOI   | DOI-A   | Collapse |
|---------------------|-------|----------|-------|-------|---------|----------|
| CONFIG1 @ EL=-65    | 0.944 | 0.508    | 0.774 | 0.941 | -0.004  | YES      |
| CONFIG1 @ EL=-64    | 0.945 | 0.532    | 0.730 | 0.942 | -0.003  | YES      |
| CONFIG1 @ EL=-63.5  | 0.943 | 0.747    | 0.938 | 0.943 | +0.0004 | partial  |
| CONFIG1 @ EL=-63    | 0.943 | 0.909    | 0.939 | 0.944 | +0.001  | NO       |
| DiVolo  @ EL=-65    | 0.927 | 0.925    | 0.935 | 0.947 | +0.020  | NO       |
| DiVolo  @ EL=-64    | 0.936 | 0.936    | 0.940 | 0.944 | +0.008  | NO       |
| DiVolo  @ EL=-63    | 0.944 | 0.936    | 0.940 | 0.945 | +0.001  | NO       |

### Di Volo Polynomial Pairs
| P_E     | P_I     | Awake | Propofol | P+DOI | DOI   | DOI-A  |
|---------|---------|-------|----------|-------|-------|--------|
| DV      | DV      | 0.936 | 0.936    | 0.940 | 0.944 | +0.008 |
| CONFIG1 | DV      | ~0.95 | ~0.95    | ~0.95 | ~0.95 | ~0     |
| DV      | CONFIG1 | 0.941 | 0.935    | 0.942 | 0.946 | +0.005 |
| CONFIG1 | CONFIG1 | 0.945 | 0.532    | 0.730 | 0.942 | -0.003 |

> P_E determines propofol collapse. P_I modulates DOI sensitivity.

### Interpolation Sweep (alpha * CONFIG1 + (1-alpha) * DiVolo)
| alpha | P[2]    | Awake | Propofol | P+DOI | DOI   | DOI-A   | Collapse? |
|-------|---------|-------|----------|-------|-------|---------|-----------|
| 0.0   | -0.0235 | 0.936 | 0.936    | 0.940 | 0.944 | +0.008  | NO        |
| 0.1   | -0.0219 | 0.936 | 0.936    | 0.942 | 0.944 | +0.008  | NO        |
| 0.2   | -0.0204 | 0.938 | 0.936    | 0.942 | 0.945 | +0.007  | NO        |
| 0.3   | -0.0188 | 0.943 | 0.935    | 0.944 | 0.947 | +0.004  | NO        |
| 0.4   | -0.0173 | 0.942 | 0.937    | 0.944 | 0.946 | +0.005  | NO        |
| 0.5   | -0.0157 | 0.944 | 0.940    | 0.945 | 0.945 | +0.001  | NO        |
| 0.6   | -0.0142 | 0.943 | 0.939    | 0.944 | 0.944 | +0.001  | NO        |
| 0.7   | -0.0126 | 0.945 | 0.937    | 0.944 | 0.942 | -0.003  | NO        |
| 0.8   | -0.0110 | 0.944 | 0.833    | 0.936 | 0.941 | -0.003  | NO        |
| 0.9   | -0.0095 | 0.943 | 0.790    | 0.928 | 0.942 | -0.001  | NO        |
| 1.0   | -0.0079 | 0.945 | 0.532    | 0.730 | 0.941 | -0.003  | YES       |

## Production Simulation (N=16 seeds, CONFIG1 @ EL=-64)
| Condition  | Mean LZc | Std    | vs Awake p-value |
|------------|----------|--------|------------------|
| Awake      | 0.94445  | ~0.003 | —                |
| Propofol   | 0.58588  | ~0.03  | < 1e-15          |
| Prop+DOI   | 0.75657  | ~0.02  | < 1e-8           |
| DOI        | 0.94465  | ~0.003 | 0.731 (NS)       |

DOI > Awake: delta = +0.0002 — NOT statistically significant.

## TF Fitting Experiments A-D (2026-03-05)

Isolating what causes CONFIG1 vs Di Volo polynomial difference.
All use Sacha's NumPy simulator + identical fitting code (SLSQP → Nelder-Mead).

| Exp | Data (EL)    | Fit Method      | 60Hz Filter | P[2]    | L2→C1  | L2→DV  | Closest |
|-----|-------------|-----------------|-------------|---------|--------|--------|---------|
| A   | Sacha(-64)  | DV single-pass  | YES         | -0.0058 | 0.037  | 0.046  | CONFIG1 |
| A'  | Sacha(-64)  | DV single-pass  | NO          | +0.0072 | 0.097  | 0.107  | CONFIG1 |
| B   | DV(-65)     | DV single-pass  | YES         | +0.0047 | 0.156  | 0.191  | CONFIG1 |
| B'  | DV(-65)     | DV single-pass  | NO          | -0.0321 | 0.320  | 0.305  | DiVolo  |
| C   | DV(-65)     | Sacha loop      | NO          | -0.0321 | 0.320  | 0.305  | DiVolo  |
| D   | Sacha(-64)  | Sacha loop      | NO          | +0.0072 | 0.097  | 0.107  | CONFIG1 |

Training data comparison (Sacha EL=-64 vs DV EL=-65): correlation=0.9984, mean diff=0.48 Hz.

**Key finding**: The 60Hz filter is the dominant factor controlling P[2].
- With filter (Fout<60): ~460 of ~2050 points → small |P[2]| (CONFIG1-like)
- Without filter: all data → large |P[2]| (DiVolo-like)
- Sacha loop = single-pass when loop doesn't improve (C=B', D=A')
- EL=-64 vs -65 has minor effect compared to filter
- E/F (Brian2 data) completed — see below

## TF Fitting Experiments E-F (2026-03-05, Brian2 data)

Brian2 conductance-based single-neuron sim (EL=-65, Di Volo cell, 50x50 grid, 10s/point).
Brian2 vs NumPy data: correlation=0.9995, but mean diff=3.43 Hz (7x larger than EL-64 vs EL-65 diff).

| Exp | Data (EL)     | Fit Method      | 60Hz Filter | P[2]    | L2→C1  | L2→DV  | Closest |
|-----|--------------|-----------------|-------------|---------|--------|--------|---------|
| E   | Brian2(-65)  | DV single-pass  | YES         | +0.0035 | 0.045  | 0.081  | CONFIG1 |
| E'  | Brian2(-65)  | DV single-pass  | NO          | +0.0373 | 0.284  | 0.318  | CONFIG1 |
| F   | Brian2(-65)  | Sacha loop      | NO          | +0.0373 | 0.284  | 0.318  | CONFIG1 |

**Critical finding**: Brian2 does NOT reproduce DiVolo's polynomial!
- E (Brian2 + 60Hz filter) = P[2]=+0.004 → POSITIVE (DiVolo is -0.024)
- E' (Brian2, no filter) = P[2]=+0.037 → even MORE positive
- The ONLY experiment that produced large negative P[2] was B'/C (NumPy EL=-65, no filter): -0.032

## Complete P[2] Summary (All Experiments)

| Exp | Data          | Filter | P[2]    | Notes |
|-----|---------------|--------|---------|-------|
| A   | NumPy EL=-64  | 60Hz   | -0.006  | Closest to CONFIG1 |
| A'  | NumPy EL=-64  | None   | +0.007  | |
| B   | NumPy EL=-65  | 60Hz   | +0.005  | |
| B'  | NumPy EL=-65  | None   | -0.032  | Closest to DiVolo |
| C   | NumPy EL=-65  | None   | -0.032  | = B' (loop doesn't help) |
| D   | NumPy EL=-64  | None   | +0.007  | = A' (loop doesn't help) |
| E   | Brian2 EL=-65 | 60Hz   | +0.003  | Brian2 ≠ DiVolo |
| E'  | Brian2 EL=-65 | None   | +0.037  | |
| F   | Brian2 EL=-65 | None   | +0.037  | = E' (loop doesn't help) |

Pattern: With 60Hz filter, P[2] is always small (±0.006). Without filter, sign depends
on data source in unpredictable ways. Optimizer landscape is very rough.

## Two Distinct P_E_SACHA Values

Production scripts use different values than the .npy file:

**Production scripts** (validated, used in all cached results):
```
[-0.05017034, 0.00451531, -0.00794377, -0.00208418, -0.00054697,
  0.00341614, -0.01156433, 0.00194753,  0.00274079, -0.01066769]
```

**.npy file** (`RS-cell0_CONFIG1_fit.npy`):
```
[-0.06373626, 0.00786373, -0.00791921, 0.00170594, -0.00049024,
  0.00329717, -0.01731714, 0.00435419, 0.00054003, -0.02073425]
```

Both produce propofol collapse. Discrepancy origin unknown — investigate which is the "real" CONFIG1.
