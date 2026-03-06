#!/bin/bash
#SBATCH --job-name=optionA_tf_divolo
#SBATCH --account=lsa1
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16 --mem=64G --time=02:00:00
#SBATCH --output=/scratch/lsa_root/lsa1/soichi/logs/optionA_tf_%j.out

# Option A: Generate TF training data at Di Volo's E_L=-65, fit polynomial,
# compare to Di Volo's stored polynomial.

module load python3.11-anaconda/2024.02 && source activate tvb_sim

cd /scratch/lsa_root/lsa1/soichi/code/paper_pipeline_hub/Tf_calc

echo "============================================================"
echo "OPTION A: TF Generation at E_L=-65 (Di Volo params)"
echo "============================================================"
echo ""
echo "Step 1: Generate excitatory TF data (FS-RS_divolo config)"
echo "  EL_e=-65, a_e=0, b_e=0, tau_i=5"
echo ""

# Step 1: Generate TF training data for excitatory cell
python3 tf_simulation_fast.py \
    --cells FS-RS_divolo \
    --range_exc 0.1,30,50 \
    --range_inh 0.1,30,50 \
    --time 5000 \
    --save_name divolo_el65 \
    --seed 10

echo ""
echo "Step 2: Fit polynomial from generated data"
echo ""

# Step 2: Fit polynomial using theoretical_tools
python3 -c "
import sys, os, numpy as np
sys.path.insert(0, '.')
from theoretical_tools import make_fit_from_data

data_dir = './data'
grid = '50x50'
name = 'divolo_el65'

# Check output files exist
exc_file = os.path.join(data_dir, f'ExpTF_exc_{grid}_{name}.npy')
params_file = os.path.join(data_dir, f'params_range_{grid}_{name}.npy')
adapt_file = os.path.join(data_dir, f'ExpTF_Adapt_{grid}_{name}.npy')

if not os.path.exists(exc_file):
    print(f'ERROR: {exc_file} not found!')
    sys.exit(1)

print(f'Fitting excitatory TF from {exc_file}')
print(f'Using loop_n=1, range=(0,5)')
print()

P_E = make_fit_from_data(
    exc_file, 'RS', params_file, adapt_file,
    range_exc=(0, 5), range_inh=(0, 5),
    loop_n=1, seed=10
)

print()
print('=' * 70)
print('FITTED EXCITATORY POLYNOMIAL:')
print('=' * 70)
print(f'P_E = {P_E}')
print()

# Compare to Di Volo's polynomial
P_E_DIVOLO = np.array([-0.04983106, 0.005063550882777035, -0.023470121807314552,
                        0.0022951513725067503, -0.0004105302652029825, 0.010547051343547399,
                        -0.03659252821136933, 0.007437487505797858, 0.001265064721846073,
                        -0.04072161294490446])

print('COMPARISON TO DI VOLO:')
labels = ['P[0] const', 'P[1] muV', 'P[2] sigV', 'P[3] TvN',
          'P[4] muV^2', 'P[5] sigV^2', 'P[6] TvN^2',
          'P[7] muV*sigV', 'P[8] muV*TvN', 'P[9] sigV*TvN']
for i, (ours, dv, label) in enumerate(zip(P_E, P_E_DIVOLO, labels)):
    diff = ours - dv
    print(f'  {label:15s}: ours={ours:+.6f}  DV={dv:+.6f}  Δ={diff:+.6f}')

mse = np.mean((P_E - P_E_DIVOLO)**2)
print(f'\nMSE between polynomials: {mse:.2e}')
print(f'Max abs difference: {np.max(np.abs(P_E - P_E_DIVOLO)):.6f}')

# Save
np.save(os.path.join(data_dir, f'P_E_optionA.npy'), P_E)
print(f'\nSaved to {data_dir}/P_E_optionA.npy')

# Also fit inhibitory
inh_file = os.path.join(data_dir, f'ExpTF_inh_{grid}_{name}.npy')
if os.path.exists(inh_file):
    print()
    print('Fitting inhibitory TF...')
    P_I = make_fit_from_data(
        inh_file, 'FS', params_file, adapt_file,
        range_exc=(0, 5), range_inh=(0, 5),
        loop_n=1, seed=10
    )
    print()
    print('FITTED INHIBITORY POLYNOMIAL:')
    print(f'P_I = {P_I}')

    P_I_DIVOLO = np.array([-0.05149122024209484, 0.004003689190271077, -0.008352013668528155,
                            0.0002414237992765705, -0.0005070645080016026, 0.0014345394104282397,
                            -0.014686689498949967, 0.004502706285435741, 0.0028472190352532454,
                            -0.015357804594594548])

    mse_i = np.mean((P_I - P_I_DIVOLO)**2)
    print(f'MSE vs Di Volo P_I: {mse_i:.2e}')
    np.save(os.path.join(data_dir, f'P_I_optionA.npy'), P_I)
    print(f'Saved to {data_dir}/P_I_optionA.npy')

print()
print('DONE. Check if polynomials match Di Volo.')
"

echo ""
echo "Option A complete."
