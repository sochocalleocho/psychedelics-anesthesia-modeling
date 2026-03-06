import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# === Parameters (Aligned with Presentation JS & Tuned for Oscillations) ===
# Time constants
T_E = 20.0        # ms (Standard JS tau_V approx)
T_I = 20.0        # ms

# Adaptation (The "Brake")
tau_w = 500.0     # ms
b = 40.0          # pA (Sufficient to kill the UP state)
a = 0.0           # nS
g_L = 10.0        # nS (For converting w current to voltage)

# Transfer Function Parameters (From JS)
# JS: mu_base = -55e-3 V (-55 mV)
# JS: V_thr = -50e-3 V (-50 mV)
# JS: tau_V = 0.02 s (20 ms)
mu_base = -55.0   # mV
V_thr = -50.0     # mV
tau_V = 20.0      # ms

# Connectivity (Phenomenological mV shift per Hz)
# JS used wE=0.001 (1mV). 
# To get self-sustained UP state, we need mu_V > V_thr when E is high.
# If E=50Hz -> w*E = 50mV. -55 + 50 = -5mV > -50. So 1mV is strong enough.
# Tuned for oscillation "Tug of War":
w_EE = 1.2        # mV/Hz (Strong enough to hold UP)
w_EI = 2.0        # mV/Hz (Inhibition)
w_IE = 1.2        # mV/Hz
w_II = 1.0        # mV/Hz

# External Drive (The "Gas Pedal")
# Needs to be just right: High enough to potentially start UP, 
# but low enough that Adaptation can kill it.
# Noise usually helps kick it start.
mu_ext_E = 3.0    # mV (Baseline drive)
mu_ext_I = 0.0

# === Transfer Function (JS Replica) ===
def transfer_function(mu, sigma, tau):
    # F = 1/(2tau) * erfc( (V_thr - mu) / (sqrt(2)*sigma) )
    if sigma <= 0: return 0
    arg = (V_thr - mu) / (np.sqrt(2) * sigma)
    F = (1.0 / (2.0 * tau)) * erfc(arg) * 1000.0 # Hz (1000 for ms conversion)
    return F

# === Initial State ===
dt = 0.1
t = np.arange(0, 3000, dt) # 3 seconds
nu_E = np.zeros_like(t)
nu_I = np.zeros_like(t)
W = np.zeros_like(t)

nu_E[0] = 0.1 # Start Quiet (DOWN state)
nu_I[0] = 0.1
W[0] = 0.0    

# === Simulation ===
# JS used sigma_base = 0.005 (5mV)
sigma_base = 5.0 # mV

# Noise trace for stochastic kicking
np.random.seed(42)
noise_process = np.random.normal(0, 1.5, len(t)) # 1.5mV std deviation noise

for i in range(len(t) - 1):
    E = nu_E[i]
    I = nu_I[i]
    w_curr = W[i]
    
    # Voltages (Mean Potentials)
    # Note: JS formula was mu_base + wE*nu_E ... 
    # Current W term in mV: w_curr / g_L
    mu_E = mu_base + mu_ext_E + w_EE*E - w_EI*I - (w_curr / g_L) + noise_process[i]
    mu_I = mu_base + mu_ext_I + w_IE*E - w_II*I 
    
    # State dependent noise (from JS: sigma_base * sqrt(Rates))
    # JS: sigma_V = sigma_base * Math.sqrt(nu_E + nu_I + 0.1);
    sigma_E = sigma_base * np.sqrt((E + I)/10.0 + 1.0) # Scaling Hz appropriately? JS range was 100 Hz.
    sigma_I = sigma_base * np.sqrt((E + I)/10.0 + 1.0)
    
    # Target Rates
    F_E = transfer_function(mu_E, sigma_E, T_E)
    F_I = transfer_function(mu_I, sigma_I, T_I)
    
    # Dynamics (Rate)
    dE = (F_E - E) / T_E
    dI = (F_I - I) / T_I
    
    # Adaptation (AdEx style)
    # dW/dt = -W/tau_w + b * E + a...
    dW = -w_curr / tau_w + b * E
    
    nu_E[i+1] = np.clip(E + dE * dt, 0, 500)
    nu_I[i+1] = np.clip(I + dI * dt, 0, 500)
    W[i+1] = np.clip(w_curr + dW * dt, 0, 10000)

# === Plotting ===
plt.figure(figsize=(6, 5))

# Top: Rates
plt.subplot(2, 1, 1)
plt.plot(t, nu_E, color='#e11d48', label=r'Exc Rate $\nu_E$', linewidth=1.5)
plt.plot(t, nu_I, color='#3b82f6', linestyle='--', label=r'Inh Rate $\nu_I$', linewidth=1.5)
plt.ylabel('Rate (Hz)')
plt.title('Mean Field Dynamics: Slow Oscillations (UP/DOWN)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(0, 3000)
plt.ylim(0, 60) # Typical range

# Bottom: Adaptation
plt.subplot(2, 1, 2)
plt.plot(t, W, color='#005088', label=r'Adaptation $W$', linewidth=1.5)
plt.ylabel('Adaptation (pA)')
plt.xlabel('Time (ms)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.xlim(0, 3000)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '../../figures/mean_field_plot.png'), dpi=150)
print(f"Done. Final E: {nu_E[-1]:.2f} Hz. Max E: {np.max(nu_E):.2f} Hz.")
