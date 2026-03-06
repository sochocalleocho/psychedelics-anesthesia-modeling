#!/bin/bash
#SBATCH --job-name=optionB_zerlaut_tf
#SBATCH --account=lsa1
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8 --mem=32G --time=01:00:00
#SBATCH --output=/scratch/lsa_root/lsa1/soichi/logs/optionB_%j.out

# Option B: Reproduce Di Volo's polynomial using Zerlaut's TF generation
# - Inverse moment-space grid (muV, sigV, TvN)
# - Euler integration + Poisson shot noise
# - Three fitting methods compared
# Expected: P[2] closer to -0.024 (DiVolo) than -0.008 (CONFIG1)

module load python3.11-anaconda/2024.02 && source activate tvb_sim

cd /scratch/lsa_root/lsa1/soichi/code/analysis

echo "============================================================"
echo "OPTION B: Zerlaut TF generation — medium sampling"
echo "============================================================"

python3 option_b_zerlaut_tf.py medium

echo ""
echo "Option B complete."
