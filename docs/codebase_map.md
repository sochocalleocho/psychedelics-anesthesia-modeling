# Codebase Map

<!-- Purpose-to-file mapping. For Claude Code navigation via Glob/Grep. -->

## Production Code (actively used)

| Purpose | File | Notes |
|---------|------|-------|
| **Main TVB simulation** | `code/scripts/tvb_anesthesia_complexity.py` | N_SEEDS=16, all 4 drug conditions, LZc+PCI |
| **Propofol timeseries** | `code/scripts/compare_propofol_timeseries.py` | Quick CONFIG1-vs-alternative TF comparison |
| **Bistability diagnostic** | `code/scripts/diagnose_bistability.py` | Nullcline/ODE analysis for propofol collapse |
| **TVB Zerlaut model** | `code/paper_pipeline_hub/TVB/tvb_model_reference/src/Zerlaut.py` | MUST use (fixes covariance bug) |
| **TF simulator** | `code/paper_pipeline_hub/Tf_calc/tf_simulation_fast.py` | NumPy AdEx, sweeps (f_exc, f_inh) grid |
| **Moment formulas** | `code/paper_pipeline_hub/Tf_calc/theoretical_tools.py` | DO NOT MODIFY — make_fit_from_data here |
| **Cell configs** | `code/paper_pipeline_hub/Tf_calc/cell_library.py` | FS-RS_0 (Sacha), FS-RS_divolo (Di Volo) |
| **TVB sim runner** | `code/paper_pipeline_hub/TVB/tvb_model_reference/src/nuu_tools_simulation_human.py` | High-level TVB runner |

## Active Analysis Tools

| Purpose | File | Notes |
|---------|------|-------|
| **Quick TVB validation** | `code/analysis/tvb_quick_test.py` | Single-seed test of any polynomial (Sacha model) |
| **Di Volo pair tests** | `code/analysis/tvb_quick_test_divolo_full.py` | PRIMARY — Martin's Zerlaut_gK_gNa model |
| **E_L sweep** | `code/analysis/tvb_EL_sweep.py` | Parametric sweep of baseline E_L |

## Archived Code

All archived scripts are in `archive/` subdirectories. See `docs/dead_ends.md` for why each was closed.

**`code/analysis/archive/`** — Closed TF fitting investigations:
- `tf_global_fit.py`, `tf_constrained_fit.py`, `tf_polynomial_comparison.py`
- `fit_option_a.py`, `option_b_zerlaut_tf.py`
- `tvb_interpolation_sweep.py`, `tvb_divolo_EL_test.py`
- `tf_fitting_experiments.py`, `tf_experiments_ef.py` (A-F experiments — completed)
- `run_*.sh` (Slurm scripts for above)

**`code/scripts/archive/`** — One-off exploration scripts:
- `tvb_anesthesia_complexity_repro.py` (legacy backup of production script)
- `tvb_anesthesia_simulation.py` (early TVB prototype)
- `tvb_anesthesia_doi_firing_rates.py` (NIH proposal analysis)
- `reproduce_fig3*.py`, `mean_field_simulation.py`, `generate_*.py` (early exploration)
- `adex_simulation.py`, `crop_image.py`, `extract_pdf_info.py` (utilities)

## Reference Repos (READ ONLY — external published code)

All under `code/original_repos/`.

| Repo | Purpose | Key files |
|------|---------|-----------|
| `code/original_repos/sacha_et_al_2025/` | CONFIG1 source | `Tf_calc/data/RS-cell0_CONFIG1_fit.npy` |
| `code/original_repos/martin_et_al_2025/` | Martin's TVB | `tvbsim/TVB/tvb_model_reference/src/Zerlaut_gK_gNa.py`, `data/connectivity/DK68/` |
| `code/original_repos/di_volo_2019/` | Di Volo polynomial | `theoretical_tools.py`, `cell_library.py` |
| `code/original_repos/zerlaut_neural_network_dynamics/` | Original Zerlaut 2016 | Historical reference |
| `code/original_repos/zerlaut_2018_modeldb/` | Zerlaut 2018 | Historical reference |
| `code/original_repos/cakan_neurolib_2020/` | Independent MF impl. | Alternative reference |

## Data Locations

| Data | Location | Notes |
|------|----------|-------|
| CONFIG1 polynomials | `code/original_repos/sacha_et_al_2025/Tf_calc/data/` | READ ONLY — never overwrite |
| TF training grids | `code/paper_pipeline_hub/Tf_calc/data/*.npy` | ~117 files, various configs |
| DK68 connectivity | `code/original_repos/martin_et_al_2025/data/connectivity/DK68/` | 68-region Desikan-Killiany |
| Failed polynomials | `code/analysis/archive/P_E_*.npy` | Global opt, weighted, etc. |
| Results cache | `figures/lzc_results_cache.json` | CHECK BEFORE ANY NEW SIMS |
| Data staging | `data/` | `raw/`, `generated/`, `processed/` subdirs |

## Documentation

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project context for Claude Code (deps, rules, params) |
| `MEMORY.md` | Persistent memory (key findings, polynomial values) |
| `docs/dead_ends.md` | CLOSED approaches — check before new ideas |
| `docs/results_tables.md` | All LZc results (single-seed, production, experiments) |
| `docs/sharp_edges.md` | Known Claude failure modes on this project |
| `docs/codebase_map.md` | THIS FILE |
