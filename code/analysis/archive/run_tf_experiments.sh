#!/bin/bash
#SBATCH --job-name=tf_experiments
#SBATCH --account=lsa1
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4 --mem=16G --time=03:00:00
#SBATCH --output=/scratch/lsa_root/lsa1/soichi/logs/%x_%j.out

module load python3.11-anaconda/2024.02
source activate tvb_sim

cd /scratch/lsa_root/lsa1/soichi/code/analysis
python3 -u tf_fitting_experiments.py
