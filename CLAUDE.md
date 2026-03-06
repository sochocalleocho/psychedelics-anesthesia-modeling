# Psychedelics & Anesthesia Modeling — Claude Context

<!-- MAINTAINING THIS FILE
  Test: "Would removing this line cause Claude to make a mistake?" If not, cut it.
  Target: < 150 lines. Use IMPORTANT/MUST for critical rules.
  Detailed results → docs/results_tables.md. Dead ends → docs/dead_ends.md.
  File map → docs/codebase_map.md. -->

## Goal
Find ONE TVB polynomial that produces BOTH propofol LZc collapse AND DOI > Awake.
**IMPORTANT: Before trying any polynomial fitting or TF reproduction approach, read `docs/dead_ends.md`.**

## Current Status
**Read `MEMORY.md`** for project status, key findings, and what to try next.
Core tradeoff: P[2]≈-0.008 gives propofol collapse but no DOI effect; P[2]≈-0.024 gives DOI>Awake but no collapse. No intermediate works. See `docs/dead_ends.md` "Still Open" for untried strategies.

## Drug Conditions
| Condition    | b_e | tau_e | tau_i | EL_e  |
|--------------|-----|-------|-------|-------|
| Wake         | 5   | 5.0   | 5.0   | -64   |
| Propofol     | 30  | 5.0   | 7.0   | -64   |
| Ketamine     | 30  | 3.75  | 5.0   | -64   |
| DOI (add-on) | +0  | +0    | +0    | -61.2 |

DOI adds EL_e=-61.2 on top of any condition.

## Key Files
```
code/scripts/tvb_anesthesia_complexity.py   ← PRODUCTION entry point
code/paper_pipeline_hub/
  TVB/.../src/Zerlaut.py                    ← MUST use (NOT TVB built-in)
  Tf_calc/tf_simulation_fast.py             ← TF simulator
  Tf_calc/theoretical_tools.py              ← moment formulas — DO NOT MODIFY
  Tf_calc/cell_library.py                   ← neuron configs
code/analysis/                              ← active validation tools
code/analysis/archive/                      ← CLOSED investigations (do not retry)
code/original_repos/                        ← READ ONLY reference code
figures/lzc_results_cache.json              ← CHECK BEFORE ANY NEW SIMS
docs/dead_ends.md                           ← CLOSED approaches (check before new ideas)
docs/results_tables.md                      ← all LZc results
docs/codebase_map.md                        ← full file index
```

## TVB Parameters (CONFIG1 production)
```python
E_L_e=-64.0, E_L_i=-65.0, coupling_a=0.3, weight_noise=1e-4
weight normalization: column-sum (axis=0)
noise nsig: [0,0,0,0,0,0,0,1.0]  # ou_drift only
init: E=0, I=0, W_e=100 (index 5 in 8-var model), others=0
P_e=P_E_SACHA, P_i=P_I_SACHA (CONFIG1 polynomial)
```

## IMPORTANT: Martin's Config != What His Parameter File Says
- `E_L_e=-63` in file but `g_K_e=8.214` → **effective E_L=-65** (E_L_e is vestigial)
- Martin uses **Di Volo polynomial** (NOT CONFIG1) with `Zerlaut_gK_gNa` model

## Transfer Function — CRITICAL Rules
1. **ALWAYS fit TF at b_e=0, a_e=0, tau_i=5** (zero adaptation). Any other params contaminate the fit.
2. **MUST use P_E_SACHA** for production. DO NOT refit (platform non-reproducible — see MEMORY.md).
3. P[2] sign determines propofol dynamics: negative → collapse, positive → fails.

## Fixed Bugs — Do Not Reintroduce
- **TVB Zerlaut built-in**: wrong covariance → MUST use `Zerlaut.py` from paper_pipeline_hub
- **tf_simulation_fast.py line 129**: adaptation frozen during refractory — FIXED
- **PCI slicing**: slice at `t_analysis=300` (ms index), NOT `nbins_analysis`

## Great Lakes HPC
**IMPORTANT: Run ALL TVB simulations on Great Lakes, not locally.**
```
ssh alias:    greatlakes
Account:      lsa1 (standard) / lsa3 (large jobs)
Scratch:      /scratch/lsa_root/lsa1/soichi/
Conda env:    tvb_sim (module: python3.11-anaconda/2024.02)
```

**Before simulation:** `rsync -av --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' --exclude='.venv' code/ greatlakes:/scratch/lsa_root/lsa1/soichi/code/`

**Slurm template:**
```bash
#!/bin/bash
#SBATCH --account=lsa1 --partition=standard
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16 --mem=64G --time=01:00:00
#SBATCH --output=/scratch/lsa_root/lsa1/soichi/logs/%x_%j.out
module load python3.11-anaconda/2024.02 && source activate tvb_sim
cd /scratch/lsa_root/lsa1/soichi/code/scripts
python3 tvb_anesthesia_complexity.py
```

## NotebookLM — USE for All Literature Questions
**IMPORTANT:** Query NotebookLM for drug mechanisms, paper results, neuroscience theory.
Tool: `mcp__notebooklm__ask_question` with `notebook_id="psychedelics-anesthesia-resear"`
If tool fails: call `mcp__notebooklm__re_auth` silently, then retry.

## Experiment Lifecycle — IMPORTANT
After EVERY experiment or analysis run:
1. **Record results** in `docs/results_tables.md`
2. **Record insights** in MEMORY.md (Key Findings section)
3. **If approach failed**, add to `docs/dead_ends.md` with WHY it failed
4. **If approach opened new strategies**, update "Still Open" in `docs/dead_ends.md`
5. **Before starting new work**, re-read the relevant sections to avoid repeating mistakes

## Dependencies
**Great Lakes conda env `tvb_sim`:**
```
module load python3.11-anaconda/2024.02 && source activate tvb_sim
```
Key packages: `tvb-library`, `tvb-data`, `brian2` (requires `numpy<2.0`), `scipy`, `matplotlib`.
**IMPORTANT:** Never upgrade numpy>=2.0 (breaks brian2 `ndarray.ptp`). After ANY numpy change, verify: `python -c "import tvb.simulator"`.

## Folder Structure
```
CLAUDE.md                          ← THIS FILE (project context for Claude)
Makefile                           ← Pipeline orchestrator (make help)
data/                              ← All data (raw/, generated/, processed/)
docs/                              ← dead_ends.md, results_tables.md, sharp_edges.md, codebase_map.md
figures/                           ← Plots + lzc_results_cache.json
code/
  scripts/                         ← Entry points (production + archived exploration)
    tvb_anesthesia_complexity.py   ← PRODUCTION
    archive/                       ← Old one-off scripts
  analysis/                        ← TVB validation tools
    archive/                       ← CLOSED investigations
  paper_pipeline_hub/              ← Forked pipeline code (TF sim, Zerlaut, cell configs)
  original_repos/                  ← READ ONLY upstream repos
```

## Sharp Edges
**Read `docs/sharp_edges.md`** for known Claude failure modes on this project (optimizer non-reproducibility, brian2 import chain, normalization conventions, etc.).

## User Preferences
- **IMPORTANT: Always ask before modeling decisions** (parameters, atlas, connectome, operating points)
- Working dir: `/Users/soichi/Desktop/Research/Psychedelics & Anesthesia Modeling Study`
