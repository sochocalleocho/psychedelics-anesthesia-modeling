# Analysis Directory

Active TVB validation tools. All scripts are standalone (run on Great Lakes via Slurm).

## Active Tools (3 files)
- `tvb_quick_test_divolo_full.py` — **PRIMARY** production validation. Tests P_E/P_I pairs with Martin's Zerlaut_gK_gNa model.
- `tvb_quick_test.py` — Backup: single-seed validation using Sacha's Zerlaut model.
- `tvb_EL_sweep.py` — Parametric sweep of E_L_e from -65 to -62 with any polynomial.

## archive/
Completed investigations — see `docs/dead_ends.md` for why each was closed.
**IMPORTANT: Do not re-run archived scripts or retry their approaches.**
Includes: TF fitting experiments A-F (2026-03-05), interpolation sweep, global optimization, etc.
