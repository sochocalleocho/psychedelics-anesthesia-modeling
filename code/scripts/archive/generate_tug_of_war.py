import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import matplotlib.font_manager as fm

# === Parameters ===
# Standard AdEx Mean Field parameters (from mean_field_simulation.py)
# V_thr = -50 mV
# mu_base = -55 mV
V_thr = -50.0
mu_base = -55.0
sigma_base = 5.0 # Noise scaling

# === Transfer Function ===
def transfer_function(nu_in, tau_V, w_input, mu_offset, gain_boost=1.0):
    """
    Calculates Output Rate (nu_out) given Input Rate (nu_in).
    
    Parameters:
    - nu_in: Input rate (Hz)
    - tau_V: Effective membrane time constant (ms). Higher = Anesthesia.
    - w_input: Input gain (mV/Hz). Higher = Psychedelic (due to decreased g_L).
    - mu_offset: Baseline drive (mV).
    - gain_boost: Multiplier for w_input to simulate g_L decrease.
    """
    
    # Inputs to voltage moments
    # mu_V = mu_base + mu_ext + w * nu_in
    # We incorporate gain_boost into the w term effective modulation
    
    w_effective = w_input * gain_boost
    mu_V = mu_base + mu_offset + w_effective * nu_in
    
    # Sigma (State-dependent noise)
    # sigma_V = sigma_base * sqrt(nu_in/scale + 1)
    # Using scale=10.0 as per simulation
    sigma_V = sigma_base * np.sqrt(nu_in / 10.0 + 1.0)
    
    # Calculate F
    # F = 1/(2tau) * erfc( (V_thr - mu) / (sqrt(2)*sigma) )
    arg = (V_thr - mu_V) / (np.sqrt(2) * sigma_V)
    
    # Convert tau from ms to seconds for Hz calculation?
    # No, in code: (1.0 / (2.0 * tau)) * erfc(...) * 1000.0
    F = (1.0 / (2.0 * tau_V)) * erfc(arg) * 1000.0
    
    return F

# === Setup ===
nu_in = np.linspace(0, 50, 500)  # Input Rate 0-50 Hz

# Refined Parameters for Smoother Sigmoidal Shape (More realistic)
# 1. Anesthesia: Suppressed State
# High effective time constant (tau) slows response.
# Low/Negative drive keeps it sub-threshold mostly.
params_anes = {
    'tau_V': 30.0,       # Max rate ~33 Hz (1000/30)
    'w_input': 0.3,      # Shallow slope
    'mu_offset': -8.0,   # Sub-threshold bias
    'gain_boost': 1.0
}

# 2. Psychedelic: Restored Active State
# Lower effective time constant (or just better coupling).
# Higher gain shifts curve left/up.
params_psych = {
    'tau_V': 20.0,       # Max rate 50 Hz
    'w_input': 0.3,      # Same slope, just shifted
    'mu_offset': 2.0,    # Boosted drive
    'gain_boost': 1.2    # Slight steepening
}

# Override sigma to be larger for smoothness
sigma_base = 8.0 # Smooth transition

# Compute Curves
F_anes = transfer_function(nu_in, **params_anes)
F_psych = transfer_function(nu_in, **params_psych)

# Find Fixed Points (Intersections with Identity)
idx_anes = np.argwhere(np.diff(np.sign(F_anes - nu_in))).flatten()
idx_psych = np.argwhere(np.diff(np.sign(F_psych - nu_in))).flatten()

fixed_points_anes = []
for idx in idx_anes:
    x1, x2 = nu_in[idx], nu_in[idx+1]
    y1, y2 = F_anes[idx] - nu_in[idx], F_anes[idx+1] - nu_in[idx+1]
    x_root = x1 - y1 * (x2 - x1) / (y2 - y1)
    y_root = x_root
    fixed_points_anes.append((x_root, y_root))

fixed_points_psych = []
for idx in idx_psych:
    x1, x2 = nu_in[idx], nu_in[idx+1]
    y1, y2 = F_psych[idx] - nu_in[idx], F_psych[idx+1] - nu_in[idx+1]
    x_root = x1 - y1 * (x2 - x1) / (y2 - y1)
    y_root = x_root
    fixed_points_psych.append((x_root, y_root))


# === Plotting ===
plt.style.use('default')
fig, ax = plt.subplots(figsize=(7, 6))

# Identity Line
ax.plot(nu_in, nu_in, color='#94a3b8', linestyle='--', linewidth=2, label='Identity Line')

# Anesthesia Curve
ax.plot(nu_in, F_anes, color='#005088', linewidth=3, label='Anesthesia')

# Psychedelic Curve
ax.plot(nu_in, F_psych, color='#e11d48', linewidth=3, label='Psychedelic')

# Plot Fixed Points
if fixed_points_anes:
    fp = fixed_points_anes[-1] # Highest stable point usually
    ax.plot(fp[0], fp[1], 'o', color='#005088', markersize=8, zorder=10)

if fixed_points_psych:
    fp = fixed_points_psych[-1]
    ax.plot(fp[0], fp[1], 'o', color='#e11d48', markersize=10, markeredgecolor='white', markeredgewidth=2, zorder=10)
    ax.annotate(r'$\leftarrow$ Restored State', xy=fp, xytext=(fp[0]+3, fp[1]), 
                fontsize=12, color='#e11d48', fontweight='bold', va='center')

# Styling
ax.set_xlabel('Input Rate (Hz)', fontsize=12)
ax.set_ylabel('Output Rate (Hz)', fontsize=12)
ax.set_title('Dynamics: The "Tug-of-War"', fontsize=14, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.text(35, 10, 'Anesthesia', color='#005088', fontsize=12, fontweight='bold', ha='center')
ax.text(35, 42, 'Psychedelic', color='#e11d48', fontsize=12, fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '../../figures/tug_of_war_plot.png'), dpi=300)
print(f"Plot saved to tug_of_war_plot.png")
print(f"Fixed Points Anesthesia: {fixed_points_anes}")
print(f"Fixed Points Psychedelic: {fixed_points_psych}")
