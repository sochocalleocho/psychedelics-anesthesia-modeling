#!/bin/bash
#SBATCH --account=lsa1 --partition=standard
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16 --mem=64G --time=04:00:00
#SBATCH --output=/scratch/lsa_root/lsa1/soichi/logs/b_e_sweep_%j.out

module load python3.11-anaconda/2024.02 && source activate tvb_sim
cd /scratch/lsa_root/lsa1/soichi/code/scripts
python3 -u tvb_b_e_sweep.py
