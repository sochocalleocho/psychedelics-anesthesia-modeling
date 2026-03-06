#!/bin/bash
#SBATCH --job-name=doi_el_sweep
#SBATCH --account=lsa1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/lsa_root/lsa1/soichi/logs/doi_el_sweep_%j.out

# DOI E_L Endpoint Sweep
# 6 endpoints × 4 conditions × 16 seeds = 384 TVB simulations
# Estimated: ~2-3 hours with 16 cores

module load python3.11-anaconda/2024.02
source activate tvb_sim

cd /scratch/lsa_root/lsa1/soichi/code/scripts

echo "=== DOI E_L Endpoint Sweep ==="
echo "Start: $(date)"
echo "Host: $(hostname)"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo ""

python3 tvb_doi_el_sweep.py --seeds 16

echo ""
echo "End: $(date)"
