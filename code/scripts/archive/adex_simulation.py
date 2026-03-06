import os

import numpy as np
import matplotlib.pyplot as plt

def run_adex(params, label):
    # Unpack parameters
    C_m = params['C_m']
    g_L = params['g_L']
    E_L = params['E_L']
    V_T = params['V_T']
    Delta_T = params['Delta_T']
    a = params['a']
    tau_w = params['tau_w']
    b = params['b']
    V_reset = params['V_reset']
    V_peak = params['V_peak']

    # Time settings
    dt = 0.1
    T = 400.0
    t = np.arange(0, T, dt)

    # Input current
    I_stim = np.zeros_like(t)
    I_start = 50.0
    I_end = 350.0
    I_amp = 700.0 if params['type'] == 'RS' else 500.0 # FS often needs less current or responds stronger
    I_stim[(t >= I_start) & (t <= I_end)] = I_amp

    # Variables
    v = np.zeros_like(t)
    w = np.zeros_like(t)
    v[0] = E_L
    w[0] = 0.0

    # Simulation Loop
    for i in range(len(t) - 1):
        exp_term = g_L * Delta_T * np.exp((v[i] - V_T) / Delta_T)
        if exp_term > 1e6: exp_term = 1e6 
        
        dv_dt = (-g_L * (v[i] - E_L) + exp_term - w[i] + I_stim[i]) / C_m
        dw_dt = (a * (v[i] - E_L) - w[i]) / tau_w
        
        v[i+1] = v[i] + dv_dt * dt
        w[i+1] = w[i] + dw_dt * dt
        
        if v[i+1] >= V_peak:
            v[i+1] = V_reset
            w[i+1] += b
            v[i] = V_peak 
            
    return t, v, w

# Parameters
# RS: Regular Spiking (Excitatory) - Adapting
rs_params = {
    'type': 'RS', 'C_m': 281.0, 'g_L': 30.0, 'E_L': -70.6, 'V_T': -50.4,
    'Delta_T': 2.0, 'a': 4.0, 'tau_w': 144.0, 'b': 80.5, 'V_reset': -70.6, 'V_peak': 20.0
}

# FS: Fast Spiking (Inhibitory) - Non-adapting
fs_params = {
    'type': 'FS', 'C_m': 200.0, 'g_L': 10.0, 'E_L': -65.0, 'V_T': -50.0,
    'Delta_T': 2.0, 'a': 0.0, 'tau_w': 144.0, 'b': 0.0, 'V_reset': -65.0, 'V_peak': 20.0
}

# Run simulations
t, v_rs, w_rs = run_adex(rs_params, "RS (Excitatory)")
t, v_fs, w_fs = run_adex(fs_params, "FS (Inhibitory)")

# Plotting 2x2 Grid
# Sharex=False to show time axis on all plots as requested
fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=False)

# Top Left: RS Voltage
axs[0, 0].plot(t, v_rs, color='#e11d48', linewidth=1.5)
axs[0, 0].set_ylabel('Voltage (mV)', fontsize=11)
axs[0, 0].set_xlabel('Time (ms)', fontsize=10) # Added explicit label
axs[0, 0].set_title('RS Cell (Excitatory)\nAdapting', fontsize=12, fontweight='bold')
axs[0, 0].grid(True, linestyle='--', alpha=0.5)
axs[0, 0].set_ylim([-80, 40])

# Top Right: FS Voltage
axs[0, 1].plot(t, v_fs, color='#1e3a8a', linewidth=1.5) # Blue for inhibitory
axs[0, 1].set_xlabel('Time (ms)', fontsize=10) # Added explicit label
axs[0, 1].set_title('FS Cell (Inhibitory)\nNon-Adapting', fontsize=12, fontweight='bold')
axs[0, 1].grid(True, linestyle='--', alpha=0.5)
axs[0, 1].set_ylim([-80, 40])

# Bottom Left: RS Adaptation
axs[1, 0].plot(t, w_rs, color='#005088', linewidth=1.5)
axs[1, 0].set_ylabel('Adaptation w (pA)', fontsize=11)
axs[1, 0].set_xlabel('Time (ms)', fontsize=11)
axs[1, 0].grid(True, linestyle='--', alpha=0.5)
axs[1, 0].set_ylim([-50, 400])

# Bottom Right: FS Adaptation
axs[1, 1].plot(t, w_fs, color='#1e3a8a', linewidth=1.5) # Blue
axs[1, 1].set_xlabel('Time (ms)', fontsize=11)
axs[1, 1].grid(True, linestyle='--', alpha=0.5)
axs[1, 1].set_ylim([-50, 400])

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '../../figures/adex_plot.png'), dpi=150)
print("Simulation complete. Generated RS vs FS grid with time axes.")

