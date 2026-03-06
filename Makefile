# =============================================================================
# Psychedelics & Anesthesia Modeling Study — Pipeline Orchestrator
# =============================================================================
# Usage: make <target>
# Run `make help` to see all available targets.
#
# PIPELINE DAG (data lineage):
#   cell_library.py → tf_simulation_fast.py → .npy grids
#     → theoretical_tools.py (make_fit_from_data) → polynomial coefficients
#       → tvb_anesthesia_complexity.py (Slurm) → LZc/PCI results
#         → figures/ (plots + lzc_results_cache.json)
#
# NOTE: TVB production sims MUST run on Great Lakes (not local). Use `make sync`
# to push code and `make pull-results` to retrieve outputs.

PROJECT_ROOT := $(shell pwd)
TF_DIR       := $(PROJECT_ROOT)/code/paper_pipeline_hub/Tf_calc
SCRIPTS_DIR  := $(PROJECT_ROOT)/code/scripts
ANALYSIS_DIR := $(PROJECT_ROOT)/code/analysis
FIGURES_DIR  := $(PROJECT_ROOT)/figures
DATA_DIR     := $(PROJECT_ROOT)/data

# Great Lakes paths
GL_HOST     := greatlakes
GL_SCRATCH  := /scratch/lsa_root/lsa1/soichi
GL_CODE     := $(GL_SCRATCH)/code
GL_LOGS     := $(GL_SCRATCH)/logs

# Grid size for TF simulations (NxN). 50x50 ≈ 2 min with fast sim.
GRID := 0.1,30,50
# Simulation duration per TF point (ms). Sacha uses 4000 for CONFIG1.
TF_TIME := 5000

# =============================================================================
# HELP
# =============================================================================
.PHONY: help
help:
	@echo ""
	@echo "  Psychedelics & Anesthesia Modeling Study — Pipeline Orchestrator"
	@echo "  ================================================================"
	@echo ""
	@echo "  HPC SYNC (Great Lakes)"
	@echo "  ----------------------"
	@echo "  make sync              Push code/ to Great Lakes scratch"
	@echo "  make pull-results      Pull figures/ and logs from Great Lakes"
	@echo "  make gl-status         Check running Slurm jobs"
	@echo ""
	@echo "  TRANSFER FUNCTION (TF) PIPELINE"
	@echo "  --------------------------------"
	@echo "  make tf-awake          Run fast TF sim for awake config (FS-RS_5, b_e=5)"
	@echo "  make tf-propofol       Run fast TF sim for propofol config (b_e=30, tau_i=7)"
	@echo "  make tf-all            Run both TF sims sequentially"
	@echo "  make tf-awake-brian    Run awake TF using Sacha's Brian2 sim (slow, correct)"
	@echo "  make fit-awake         Fit polynomial TF to awake data"
	@echo "  make fit-propofol      Fit polynomial TF to propofol data"
	@echo "  make fit-combined      Fit combined TF across awake + propofol data"
	@echo ""
	@echo "  TVB PRODUCTION SIMULATIONS (local — use HPC for real runs)"
	@echo "  -----------------------------------------------------------"
	@echo "  make run               Run production simulation (background)"
	@echo "  make run-debug         Run debug simulation (Awake + DOI only)"
	@echo "  make run-fg            Run production simulation in foreground"
	@echo ""
	@echo "  VALIDATION & PLOTTING"
	@echo "  ---------------------"
	@echo "  make validate          Show cached LZc results summary"
	@echo "  make plot-lzc          Generate LZc bar chart from cached results"
	@echo "  make plot-timeseries   Generate propofol E(t) timeseries comparison"
	@echo ""
	@echo "  MONITORING"
	@echo "  ----------"
	@echo "  make log               Tail the production run log (Ctrl-C to stop)"
	@echo "  make status            Check if any TVB simulation is currently running"
	@echo "  make last-results      Print last LZc/PCI results from log"
	@echo ""
	@echo "  MAINTENANCE"
	@echo "  -----------"
	@echo "  make check-data        List all TF .npy data files with sizes and dates"
	@echo "  make check-deps        Verify key Python imports work"
	@echo "  make clean-logs        Remove old log files from figures/"
	@echo ""

# =============================================================================
# HPC SYNC (GREAT LAKES)
# =============================================================================

.PHONY: sync
sync:
	@echo ">>> Syncing code/ to Great Lakes..."
	rsync -av --exclude='*.pyc' --exclude='__pycache__' \
		--exclude='.git' --exclude='.venv' --exclude='original_repos' \
		$(PROJECT_ROOT)/code/ $(GL_HOST):$(GL_CODE)/
	@echo ">>> Done. Code synced to $(GL_HOST):$(GL_CODE)/"

.PHONY: pull-results
pull-results:
	@echo ">>> Pulling results from Great Lakes..."
	rsync -av $(GL_HOST):$(GL_SCRATCH)/figures/ $(FIGURES_DIR)/
	rsync -av $(GL_HOST):$(GL_LOGS)/ $(FIGURES_DIR)/slurm_logs/
	@echo ">>> Done. Check figures/ for updated results."

.PHONY: gl-status
gl-status:
	@echo ">>> Checking Great Lakes Slurm jobs..."
	ssh $(GL_HOST) 'squeue -u soichi --format="%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null' || echo "  (could not connect)"

# =============================================================================
# TRANSFER FUNCTION SIMULATIONS
# =============================================================================

# Awake: FS-RS_5 config (b_e=5 pA, EL_e=-64 mV)
.PHONY: tf-awake
tf-awake:
	@echo ">>> Running fast TF sim: awake (b_e=5, EL_e=-64)"
	@echo "    Grid: $(GRID), Duration: $(TF_TIME) ms"
	cd $(TF_DIR) && python3 tf_simulation_fast.py \
		--cells FS-RS_5 \
		--range_exc $(GRID) \
		--range_inh $(GRID) \
		--time $(TF_TIME) \
		--save_name b_e_5
	@echo ">>> Done. Output: $(TF_DIR)/data/*b_e_5*.npy"

# Propofol: FS-RS_prop config (b_e=30 pA, tau_i=7 ms, EL_e=-64 mV)
.PHONY: tf-propofol
tf-propofol:
	@echo ">>> Running fast TF sim: propofol (FS-RS_prop: b_e=30, tau_i=7, EL_e=-64)"
	@echo "    Grid: $(GRID), Duration: $(TF_TIME) ms"
	cd $(TF_DIR) && python3 tf_simulation_fast.py \
		--cells FS-RS_prop \
		--range_exc $(GRID) \
		--range_inh $(GRID) \
		--time $(TF_TIME) \
		--save_name prop_b30_ti7
	@echo ">>> Done. Output: $(TF_DIR)/data/*prop_b30_ti7*.npy"

# DOI-adjacent: FS-RS_5_63 (b_e=5, EL_e=-63 mV)
.PHONY: tf-doi-adj
tf-doi-adj:
	@echo ">>> Running fast TF sim: DOI-adjacent (FS-RS_5_63: b_e=5, EL_e=-63)"
	@echo "    Grid: $(GRID), Duration: $(TF_TIME) ms"
	cd $(TF_DIR) && python3 tf_simulation_fast.py \
		--cells FS-RS_5_63 \
		--range_exc $(GRID) \
		--range_inh $(GRID) \
		--time $(TF_TIME) \
		--save_name b5_EL63
	@echo ">>> Done. Output: $(TF_DIR)/data/*b5_EL63*.npy"

# Run both core TF sims (awake + propofol)
.PHONY: tf-all
tf-all: tf-awake tf-propofol
	@echo ">>> Both TF sims complete."

# Run all three TF sims (awake + propofol + DOI-adjacent)
.PHONY: tf-all3
tf-all3: tf-awake tf-propofol tf-doi-adj
	@echo ">>> All three TF sims complete."

# Awake using Sacha's Brian2 sim (slow but correct physics, use as ground-truth check)
.PHONY: tf-awake-brian
tf-awake-brian:
	@echo ">>> Running Brian2 TF sim (SLOW ~67 min): awake"
	cd $(TF_DIR) && python3 tf_simulation.py \
		--cells FS-RS_5 \
		--range_exc $(GRID) \
		--range_inh $(GRID) \
		--time $(TF_TIME)
	@echo ">>> Done."

# =============================================================================
# TF FITTING
# =============================================================================

.PHONY: fit-awake
fit-awake:
	@echo ">>> Fitting TF polynomial to awake data (b_e=5)"
	cd $(TF_DIR) && python3 fit_b5_EL63.py
	@echo ">>> Done."

.PHONY: fit-propofol
fit-propofol:
	@echo ">>> Fitting TF polynomial to propofol data (b_e=30, tau_i=7)"
	cd $(TF_DIR) && python3 fit_prop_b30_ti7.py
	@echo ">>> Done."

.PHONY: fit-combined
fit-combined:
	@echo ">>> Fitting combined TF across awake + propofol data"
	cd $(TF_DIR) && python3 fit_combined.py
	@echo ">>> Done."

# =============================================================================
# TVB PRODUCTION SIMULATIONS
# =============================================================================

# Full production run in background — log goes to figures/run.log
.PHONY: run
run:
	@echo ">>> Launching production TVB simulation in background..."
	@echo "    Log: $(FIGURES_DIR)/run.log"
	@echo "    Monitor with: make log"
	nohup python3 $(SCRIPTS_DIR)/tvb_anesthesia_complexity.py \
		> $(FIGURES_DIR)/run.log 2>&1 &
	@echo "    PID: $$!"
	@echo "    Started. Run 'make log' to follow progress."

# Debug run: Awake + DOI only, no PCI, fast
.PHONY: run-debug
run-debug:
	@echo ">>> Launching DEBUG simulation (Awake + DOI only) in background..."
	nohup python3 $(SCRIPTS_DIR)/tvb_anesthesia_complexity.py --debug \
		> $(FIGURES_DIR)/run_debug.log 2>&1 &
	@echo "    PID: $$!  |  Log: $(FIGURES_DIR)/run_debug.log"

# Foreground run (verbose, blocks terminal, good for debugging)
.PHONY: run-fg
run-fg:
	@echo ">>> Running production simulation in foreground (verbose)..."
	python3 $(SCRIPTS_DIR)/tvb_anesthesia_complexity.py

# =============================================================================
# VALIDATION & PLOTTING
# =============================================================================

.PHONY: validate
validate:
	@echo "=== Cached LZc Results ==="
	@test -f "$(FIGURES_DIR)/lzc_results_cache.json" && \
		python3 -c "import json; d=json.load(open('$(FIGURES_DIR)/lzc_results_cache.json')); \
		[print(f'  {k}: {v}') for k,v in d.items()]" \
		|| echo "  No cached results found. Run a simulation first."

.PHONY: plot-lzc
plot-lzc:
	@echo ">>> Generating LZc bar chart from cached results..."
	@python3 "$(PROJECT_ROOT)/code/scripts/plot_lzc_bar.py" "$(FIGURES_DIR)"

.PHONY: plot-timeseries
plot-timeseries:
	@echo ">>> Generating propofol timeseries comparison..."
	python3 $(SCRIPTS_DIR)/compare_propofol_timeseries.py
	@echo ">>> Done. Check figures/ for timeseries plots."

# =============================================================================
# MONITORING
# =============================================================================

.PHONY: log
log:
	@test -f $(FIGURES_DIR)/run.log || (echo "No run.log found. Start a run first with: make run"; exit 1)
	tail -f $(FIGURES_DIR)/run.log

.PHONY: status
status:
	@echo "=== Running TVB / Python simulation processes ==="
	@ps aux | grep -E "tvb_anesthesia|tf_simulation|python3" | grep -v grep || echo "  (none found)"

.PHONY: last-results
last-results:
	@echo "=== Last LZc/PCI results in run.log ==="
	@test -f $(FIGURES_DIR)/run.log && grep -E "LZc|PCI|SUMMARY|Awake|Propofol|DOI" $(FIGURES_DIR)/run.log | tail -30 || echo "No run.log found."

# =============================================================================
# MAINTENANCE
# =============================================================================

.PHONY: check-data
check-data:
	@echo "=== TF .npy data files ==="
	@ls -lh $(TF_DIR)/data/*.npy 2>/dev/null || echo "  (no .npy files found)"

.PHONY: check-deps
check-deps:
	@echo "=== Checking Python dependencies ==="
	@python3 -c "import tvb.simulator; print('  tvb.simulator: OK')" 2>/dev/null || echo "  tvb.simulator: MISSING"
	@python3 -c "import numpy; print(f'  numpy {numpy.__version__}: OK')" 2>/dev/null || echo "  numpy: MISSING"
	@python3 -c "import scipy; print(f'  scipy {scipy.__version__}: OK')" 2>/dev/null || echo "  scipy: MISSING"
	@python3 -c "import matplotlib; print(f'  matplotlib {matplotlib.__version__}: OK')" 2>/dev/null || echo "  matplotlib: MISSING"
	@python3 -c "import brian2; print(f'  brian2 {brian2.__version__}: OK')" 2>/dev/null || echo "  brian2: MISSING (only needed for TF ground-truth)"
	@python3 -c "import numpy; v=tuple(int(x) for x in numpy.__version__.split('.')[:2]); \
		print('  numpy<2.0 (brian2 compat): OK' if v < (2,0) else '  WARNING: numpy>=2.0 breaks brian2!')"

.PHONY: clean-logs
clean-logs:
	@echo "Removing old .log files from figures/..."
	@ls $(FIGURES_DIR)/*.log 2>/dev/null && rm $(FIGURES_DIR)/*.log && echo "Done." || echo "  (no logs found)"
