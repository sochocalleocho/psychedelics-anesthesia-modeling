#!/bin/bash
#SBATCH --job-name=interp_sweep
#SBATCH --account=lsa1
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8 --mem=32G --time=02:00:00
#SBATCH --output=/scratch/lsa_root/lsa1/soichi/logs/interp_sweep_%j.out

# Polynomial interpolation sweep: P_hybrid = α × CONFIG1 + (1-α) × DiVolo
# Tests α = 0.0, 0.1, ..., 1.0 (11 values × 4 conditions = 44 sims)
# Expected runtime: ~90 min (each sim ~2 min)

module load python3.11-anaconda/2024.02 && source activate tvb_sim

cd /scratch/lsa_root/lsa1/soichi/code/analysis

echo "============================================================"
echo "POLYNOMIAL INTERPOLATION SWEEP"
echo "  P_hybrid = α × CONFIG1 + (1-α) × DiVolo"
echo "  11 α values × 4 drug conditions = 44 TVB simulations"
echo "============================================================"

python3 -u tvb_interpolation_sweep.py

echo ""
echo "Interpolation sweep complete."
