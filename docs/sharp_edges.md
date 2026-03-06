# Sharp Edges — What Claude Gets Wrong on This Project

<!-- Track things Claude consistently struggles with.
     Review periodically to see if capabilities improve. -->

## Numerical Optimization
- **Nelder-Mead is highly non-reproducible**: Same algorithm on 99.8%-correlated data gives
  P[2] ranging from -0.032 to +0.037. Claude tends to assume optimization is deterministic.
- **Training MSE ≠ simulation quality**: Claude will propose "minimize MSE" as the objective,
  but the global MSE optimum (MSE=20.4) FAILS TVB validation while CONFIG1 (MSE=147.7) works.
  Always validate polynomials in TVB, never trust MSE alone.
- **scipy version sensitivity**: Nelder-Mead `seed` kwarg is silently ignored in modern scipy.
  Claude doesn't know this and may suggest seed-based reproducibility.

## Transfer Function Physics
- **Conductance vs current noise**: Claude may confuse these. ALL our codebases use
  conductance-based synapses (multiplicative noise). Current injection (additive noise)
  produces fundamentally different TF shapes — see dead_end #5.
- **Polynomial normalization**: The 10 coefficients are NOT raw voltages — they're normalized
  by (muV0, DmuV0, sV0, DsV0, TvN0, DTvN0). Claude sometimes forgets the normalization.
- **Di Volo vs Sacha indexing**: Di Volo uses 11 coefficients (P[4]=dead), Sacha uses 10.
  Claude must check which format when comparing polynomials.

## Code Dependencies
- **theoretical_tools.py → functions.py → brian2**: This accidental import chain means
  importing ANY function from theoretical_tools.py requires brian2. Claude should inline
  needed functions rather than importing from this file.
- **brian2 + numpy compatibility**: brian2 2.9.0 requires numpy <2.0 (uses deprecated ptp).
  Claude may suggest numpy upgrades that break brian2.
- **Di Volo's theoretical_tools.py is Python 2**: Has bare `print` statements. Claude must
  NOT add Di Volo's directory to sys.path (conflicts with Sacha's same-named file).

## TVB Simulation
- **Initial conditions matter**: W_e must be initialized to 100 (index 5 in 8-var model).
  Default (all zeros) gives wrong transient behavior. Claude may forget this.
- **Must use custom Zerlaut.py**: TVB's built-in ZerlautAdaptationSecondOrder has wrong
  covariance equations. Claude may import from tvb.simulator instead of paper_pipeline_hub.

## HPC / Great Lakes
- **Login node vs compute node**: pip installs on login node may not be visible in Slurm jobs
  if conda activation differs. Claude should always use `module load && source activate`.
- **numpy downgrade risk**: Downgrading numpy for brian2 might break TVB. Always test
  `import tvb.simulator` after any numpy version change.

## General Patterns
- Claude tends to be overconfident about optimization convergence
- Claude may declare a result "reproduced" when values are in the same order of magnitude
  but differ by 3-4x (e.g., P[2]=-0.006 vs -0.024)
- Claude's summaries sometimes simplify findings beyond what the data supports
  (e.g., "60Hz filter is THE factor" when it's one of several interacting factors)
