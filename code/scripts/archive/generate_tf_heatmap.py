import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# === Biophysical Parameters (from Martin et al. 2025 / Zerlaut 2018) ===
# Found in text:
# Q_e = 1.5 nS, Q_i = 5.0 nS
# g_L = 10.0 nS
# C_m = 200.0 pF
# E_L = -65.0 mV (Inh), -64 (Exc) -> Using -65 average base
# N = 10000, p = 0.05
# Exc/Inh Ratio = 80/20

g_L = 10.0       # nS
C_m = 200.0      # pF
E_L = -65.0      # mV
E_e = 0.0        # mV
E_i = -80.0      # mV (Standard GABA_A)
V_thr = -50.0    # mV
tau_m = C_m / g_L # 20 ms
tau_syn_e = 5.0  # ms
tau_syn_i = 5.0  # ms

# Connectivity
N_tot = 10000
p_conn = 0.05
N_E = int(0.8 * N_tot) # 8000
N_I = int(0.2 * N_tot) # 2000

K_E = N_E * p_conn # 400
K_I = N_I * p_conn # 100

Q_e = 1.5 # nS
Q_i = 5.0 # nS

# Time scale for Transfer Function
T_TF = 5.0 # ms (response time, not membrane time)

# === Helper Functions for Mean Field Moments ===
def calculate_moments(nu_E, nu_I):
    # Rates in Hz -> convert to kHz for ms units if needed, 
    # but let's stick to SI (Hz, Volts, Seconds) then convert to mV/ms
    
    # Unit conversions:
    # nu: Hz (1/s)
    # Q: nS (1e-9 S)
    # g: nS
    # C: pF (1e-12 F)
    # V: mV (1e-3 V)
    
    # But it's easier to work in "Neuro" units:
    # Time: ms, Rate: kHz (or Hz * 1e-3)
    # Cond: nS, Cap: pF, Volt: mV
    # Current: pA
    
    nu_E_kz = nu_E * 1e-3
    nu_I_kz = nu_I * 1e-3
    
    # Mean Conductances (nS)
    mu_Ge = nu_E_kz * K_E * tau_syn_e * Q_e
    mu_Gi = nu_I_kz * K_I * tau_syn_i * Q_i
    
    g_tot = g_L + mu_Ge + mu_Gi
    
    # Effective Membrane Potential (Mean)
    # mu_V = (g_L E_L + g_e E_e + g_i E_i) / g_tot
    numerator = g_L * E_L + mu_Ge * E_e + mu_Gi * E_i
    mu_V = numerator / g_tot
    
    # Effective Membrane Time Constant
    tau_eff = C_m / g_tot # ms
    
    # Noise (Sigma_V)
    # Zerlaut 2018 approximation for sigma_V
    # sigma^2 = sum K_s * nu_s * (U_s)^2 * tau_eff / 2
    # U_s (PSP size) approx = Q_s/g_tot * (E_s - mu_V)
    
    U_e = (Q_e / g_tot) * (E_e - mu_V)
    U_i = (Q_i / g_tot) * (E_i - mu_V)
    
    # Variance
    sigma_sq = (K_E * nu_E_kz * U_e**2 * tau_eff / 2.0) + \
               (K_I * nu_I_kz * U_i**2 * tau_eff / 2.0)
               
    sigma_V = np.sqrt(sigma_sq)
    
    return mu_V, sigma_V, tau_eff

def transfer_function(mu, sigma, tau):
    # Zerlaut Link Function
    # F = 1/(2*tau) * erfc( (V_thr - mu) / (sqrt(2)*sigma) )
    
    # Avoid division by zero
    if sigma < 1e-6: sigma = 1e-6
    
    arg = (V_thr - mu) / (np.sqrt(2) * sigma)
    F = (1.0 / (2.0 * tau)) * erfc(arg) * 1000.0 # Output in Hz
    return F

# === Grid Generation Function ===
# Define grid globally or pass it
resolution = 100
nu_vals = np.linspace(0, 100, resolution)
nu_E_grid, nu_I_grid = np.meshgrid(nu_vals, nu_vals)

def get_TF_grid(cell_type='Exc'):
    # Parameters based on cell type (Martin et al. 2025)
    if cell_type == 'Exc':
        E_L_val = -64.0 # mV
        Delta_T = 2.0   # mV (Slope factor, affects effective threshold)
    else:
        E_L_val = -65.0 # mV
        Delta_T = 0.5   # mV
        
    # Common parameters
    # Note: V_thr is "effective" threshold. In AdEx, V_T is the exponential threshold.
    # The erfc approximation usually uses an "effective threshold" V_eff approx V_T + ...
    # We will use the standard V_thr = -50 for both as a base, but E_L difference matters.
    
    F_grid = np.zeros_like(nu_E_grid)
    
    for i in range(resolution):
        for j in range(resolution):
            E_in = nu_E_grid[i, j]
            I_in = nu_I_grid[i, j]
            
            # Recalculate mean voltage with specific E_L
            nu_E_kz = E_in * 1e-3
            nu_I_kz = I_in * 1e-3
            
            mu_Ge = nu_E_kz * K_E * tau_syn_e * Q_e
            mu_Gi = nu_I_kz * K_I * tau_syn_i * Q_i
            g_tot = g_L + mu_Ge + mu_Gi
            
            numerator = g_L * E_L_val + mu_Ge * E_e + mu_Gi * E_i
            mu_V_val = numerator / g_tot
            
            tau_eff_val = C_m / g_tot
            
            # Noise (same formula)
            U_e = (Q_e / g_tot) * (E_e - mu_V_val)
            U_i = (Q_i / g_tot) * (E_i - mu_V_val)
            sigma_sq = (K_E * nu_E_kz * U_e**2 * tau_eff_val / 2.0) + \
                       (K_I * nu_I_kz * U_i**2 * tau_eff_val / 2.0)
            sigma_V_val = np.sqrt(sigma_sq)
            sigma_tot = np.sqrt(sigma_V_val**2 + 2.0**2) # Intrinsic base noise
            
            # Calculate F
            F_grid[i, j] = transfer_function(mu_V_val, sigma_tot, tau_eff_val)
            
    return F_grid

# === Calculate Grids ===
F_E_grid = get_TF_grid('Exc')
F_I_grid = get_TF_grid('Inh')

# === Plotting ===
plt.figure(figsize=(10, 4.5))

# Plot Excitatory
plt.subplot(1, 2, 1)
plt.imshow(F_E_grid, origin='lower', extent=[0, 100, 0, 100], 
           cmap='OrRd', aspect='auto', interpolation='bicubic')
plt.colorbar(label='Output Rate (Hz)')
plt.title(r'Excitatory ($F_E$)', fontsize=14)
plt.xlabel(r'$\nu_E$ (Hz)')
plt.ylabel(r'$\nu_I$ (Hz)')

# Plot Inhibitory
plt.subplot(1, 2, 2)
plt.imshow(F_I_grid, origin='lower', extent=[0, 100, 0, 100], 
           cmap='Blues', aspect='auto', interpolation='bicubic')
plt.colorbar(label='Output Rate (Hz)')
plt.title(r'Inhibitory ($F_I$)', fontsize=14)
plt.xlabel(r'$\nu_E$ (Hz)')
plt.yticks([]) # Share Y axis label conceptually

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '../../figures/transfer_functions_combined.png'), dpi=150)
print("Combined heatmap generated: transfer_functions_combined.png")
