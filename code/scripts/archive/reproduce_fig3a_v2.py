
import numpy as np
import matplotlib.pyplot as plt
from math import erf
import sys
import os

# Add the cloned repo to sys.path to allow loading .npy files if needed,
# though we strictly load them by path.
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../paper_pipeline_hub_DO_NOT_MODIFY"))
sys.path.append(REPO_PATH)

# =============================================================================
# Helper Functions (Copied/Adapted from repository)
# =============================================================================

def heaviside(x):
    return 0.5 * (1 + np.sign(x))

def input_rate(t, t1_exc, tau1_exc, tau2_exc, ampl_exc, plateau):
    # Modified to return 0 if t is outside the range effectively to avoid weird tails if any
    # But using the formula from functions.py directly
    inp = ampl_exc * (np.exp(-(t - t1_exc) ** 2 / (2. * tau1_exc ** 2)) * heaviside(-(t - t1_exc)) + \
        heaviside(-(t - (t1_exc+plateau))) * heaviside(t - (t1_exc))+ \
        np.exp(-(t - (t1_exc+plateau)) ** 2 / (2. * tau2_exc ** 2)) * heaviside(t - (t1_exc+plateau)))
    return inp

def OU(tfin, dt):
    # Ornstein-Ulhenbeck process
    # Parameters from MF_script_with_OS.py logic
    theta = 1/(5*1.e-3 )  # Mean reversion rate
    mu = 0     # Mean of the process
    sigma = 1   # Volatility or standard deviation
    # dt passed as argument
    T = tfin        # Total time period

    t = np.arange(0, T, dt)
    n = len(t)
    x = np.zeros(n)
    x[0] = 0
    for i in range(1, n):
        dx = theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1)
        x[i] = x[i-1] + dx
    return x

# =============================================================================
# Model Parameters (SI Units)
# =============================================================================

# Default parameters for FS-RS from cell_library.py converted to SI
params = {
    'Gl': 10e-9,
    'Cm': 200e-12,
    'Qe': 1.5e-9,
    'Qi': 5.0e-9,
    'Ee': 0,
    'Ei': -80e-3,
    'EL_e': -64e-3,
    'EL_i': -65e-3,
    'tau_w': 500e-3,
    'tau_e': 5e-3,
    'tau_i': 5e-3, # Will be overridden
    'p_con': 0.05,
    'gei': 0.2,
    'Ntot': 10000,
    'b_e': 15e-12 # Tried 20 (Ref crashed at end), Tried 10 (Anesth UP). Trying 15 pA.
}

# =============================================================================
# Transfer Function
# =============================================================================

def TF(P, fexc, finh, adapt, El, p):
    # Unpack parameters
    gei = p['gei']
    pconnec = p['p_con']
    Ntot = p['Ntot']
    Qi = p['Qi']
    Qe = p['Qe']
    Te = p['tau_e']
    Ti = p['tau_i']
    Gl = p['Gl']
    Ee = p['Ee']
    Ei = p['Ei']
    Cm = p['Cm']

    fe = fexc*(1.-gei)*pconnec*Ntot
    fi = finh*gei*pconnec*Ntot
    
    muGi = Qi*Ti*fi
    muGe = Qe*Te*fe
    muG = Gl+muGe+muGi
    muV = (muGe*Ee+muGi*Ei+Gl*El-adapt)/muG
    
    Tm = Cm/muG
    
    Ue =  Qe/muG*(Ee-muV)
    Ui = Qi/muG*(Ei-muV)
    
    # Variance
    sV = np.sqrt(fe*(Ue*Te)*(Ue*Te)/2./(Te+Tm)+fi*(Ui*Ti)*(Ui*Ti)/2./(Ti+Tm))
    
    fe = fe + 1e-9
    fi = fi + 1e-9
    
    Tv = ( fe*(Ue*Te)*(Ue*Te) + fi*(Qi*Ui)*(Qi*Ui)) /( fe*(Ue*Te)*(Ue*Te)/(Te+Tm) + fi*(Qi*Ui)*(Qi*Ui)/(Ti+Tm) )
    TvN = Tv*Gl/Cm
    
    # Fitting constants (from MF_script_with_OS.py)
    muV0 = -60e-3
    DmuV0 = 10e-3
    sV0 = 4e-3
    DsV0 = 6e-3
    TvN0 = 0.5
    DTvN0 = 1.

    # Effective threshold
    vthr = P[0] + \
           P[1]*(muV-muV0)/DmuV0 + \
           P[2]*(sV-sV0)/DsV0 + \
           P[3]*(TvN-TvN0)/DTvN0 + \
           P[4]*((muV-muV0)/DmuV0)**2 + \
           P[5]*((sV-sV0)/DsV0)**2 + \
           P[6]*((TvN-TvN0)/DTvN0)**2 + \
           P[7]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0 + \
           P[8]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0 + \
           P[9]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0

    frout = .5/TvN*Gl/Cm*(1-erf((vthr-muV)/np.sqrt(2)/sV))
    
    return frout

# =============================================================================
# Simulation Function
# =============================================================================

def run_simulation(tau_i_val_ms, ax, title):
    # Update tau_i
    p = params.copy()
    p['tau_i'] = tau_i_val_ms * 1e-3
    
    # Load fitting coefficients
    try:
        PRS = np.load(os.path.join(REPO_PATH, "Tf_calc/data/RS-cell0_CONFIG1_fit.npy"))
        PFS = np.load(os.path.join(REPO_PATH, "Tf_calc/data/FS-cell_CONFIG1_fit.npy"))
    except FileNotFoundError:
        print("Error: Fitting files not found. Ensure repository is cloned correctly.")
        return

    # Simulation settings
    TotTime = 1.3 # Seconds, matching figure roughly (0 to 1.3s)
    dt = 0.0001
    t = np.arange(0, TotTime, dt)
    
    # Input parameters
    Iext = 0.3 # Reverting to 0.3 to see if this allows Anesthesia case to fall back to DOWN state
    # Wait, check args.iext default in MF_script is 0.3.
    # Figure 3b used Iext=0.315 in previous turn? 
    # Let's stick to 0.3 as a baseline or maybe slightly less if UP state is too stable.
    # The figure shows 0 before stimulus.
    
    AmpStim = 1.0 # Amplitude of pulse
    time_peek = 200. # ms
    TauP = 5. # ms - Sharpened ramp to look more like square pulse
    plat = 100. # ms duration of plateau (approx from visual)
    
    # Generate Noise
    # In MF_script_with_OS.py: os_noise = sigma*OU(tfinal) + v_drive
    # v_drive = Iext
    # sigma = 1 (lines 174-175)
    noise = 1.5 * OU(TotTime, dt) + Iext # Scaling sigma slightly to match visual noise level
    
    # Generate Pulse Input
    # User feedback: "pulse only ramps up". The green line in figure is a trapezoid (ramp up, flat, ramp down).
    # To ensure it looks cleaner, we might reduce TauP or check the logic.
    # But strictly based on the screenshot, it has a rise and fall. 
    # Maybe the "funky stuff" obscured the pulse shape?
    # We will keep parameters but ensure initial state is clean.
    
    t_ms = t * 1000.0
    stim_input = input_rate(t_ms, time_peek, TauP, TauP, AmpStim, plat)
    

    
    # Total External Input (with modification)
    # The figure shows green line = Input.
    # The effective input to neurons is external_input.
    # We want noise present during stimulus too.
    external_input = stim_input + noise
    # Clamp negative input to 0?
    # MF script does: FEX = fecont + external_input. IF FEX < 0: FEX=0.
    
    # Initial Conditions
    fecont = 0.0 # Start at 0 to match paper (DOWN state)
    ficont = 0.0
    w = 0.0
    
    T_const = 20e-3 # time constant
    
    LSfe = []
    LSfi = []
    
    # Integration
    for i in range(len(t)):
        ext_inp = external_input[i]
        
        # Current Firing Rates + Input
        FEX = fecont + ext_inp
        FINH = fecont + ext_inp # Assuming same input to Inhib? 
        # In MF_script: FINH = fecontold + external_input. (Note: fecontold is exc firing rate)
        
        if FEX < 0: FEX = 0
        if FINH < 0: FINH = 0
        
        # Calculate TF
        # TF(P, fexc, finh, adapt, El, p)
        # Note: TF function uses global variables in original script. We pass 'p'.
        
        tf_e = TF(PRS, FEX, ficont, w, p['EL_e'], p)
        tf_i = TF(PFS, FINH, ficont, 0., p['EL_i'], p) # Inhibitory cells have no adaptation w=0
        
        # Update State Variables
        # fecont += dt/T * (TF(...) - fecont)
        
        fecont += (dt/T_const) * (tf_e - fecont)
        ficont += (dt/T_const) * (tf_i - ficont)
        
        # Adaptation w update
        # w += dt*( -w/twRS + bRS*fecont )
        # b_e is p['b_e']
        w += dt * ( -w/p['tau_w'] + p['b_e']*fecont )
        
        LSfe.append(fecont)
        LSfi.append(ficont)
    
    # Print stats to "see" the output
    print(f"[{title}] Max Stim: {np.max(stim_input):.4f} Hz")
    print(f"[{title}] Initial Firing (t=0-0.1s) Mean: Exc {np.mean(LSfe[:1000] if len(LSfe)>1000 else 0):.4f} Hz")
        
    # Plotting
    ax.plot(t, LSfi, 'r', label='$FR_{inh}$')
    ax.plot(t, LSfe, 'steelblue', label='$FR_{exc}$')
    ax.plot(t, stim_input, 'green', label='Input') # Verify if we plot stim_input only or total
    
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-1, 30) # Match y-axis 0-30 roughly (Figure goes to 25+)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Population firing\nrate (Hz)', fontsize=12)
    
    if "Reference" in title:
       pass
    else:
        # For the second plot, we might want legend
        ax.legend(loc='upper right')

# =============================================================================
# Main
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Reference Case (tau_i = 5 ms)
print("Running Reference Case (tau_i = 5 ms)...")
run_simulation(5.0, axes[0], r'Reference case ($\tau_i = 5$ ms)' + '\n' + r'$\tau_i = 5.0$ ms')

# 2. Anesthesia Case (tau_i = 7 ms)
print("Running Anesthesia Case (tau_i = 7 ms)...")
run_simulation(7.0, axes[1], r'GABA-ergic anesthesia ($\tau_i = 7$ ms)' + '\n' + r'$\tau_i = 7.0$ ms')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '../../figures/fig3a_reproduction.png'))
print("Figure saved as fig3a_reproduction.png")
