#!/bin/bash
#SBATCH --job-name=divolo_el65
#SBATCH --account=lsa1
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8 --mem=32G --time=00:30:00
#SBATCH --output=/scratch/lsa_root/lsa1/soichi/logs/divolo_el65_%j.out

# DiVolo polynomial at E_L=-65 (Martin's ACTUAL config)
# Expected: larger DOI effect than at E_L=-64 (+0.008)

module load python3.11-anaconda/2024.02 && source activate tvb_sim

cd /scratch/lsa_root/lsa1/soichi/code/analysis
python3 tvb_divolo_EL_test.py
