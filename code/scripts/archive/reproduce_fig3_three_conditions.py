
import numpy as np
import matplotlib.pyplot as plt
from math import erf
import sys
import os

# Add the cloned repo to sys.path
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../paper_pipeline_hub_DO_NOT_MODIFY"))
sys.path.append(REPO_PATH)

# =============================================================================
# Helper Functions
# =============================================================================

def heaviside(x):
    return 0.5 * (1 + np.sign(x))

def input_rate(t, t1_exc, tau1_exc, tau2_exc, ampl_exc, plateau):
    # Rise
    if tau1_exc > 1e-9:
        rise = np.exp(-(t - t1_exc) ** 2 / (2. * tau1_exc ** 2)) * heaviside(-(t - t1_exc))
    else:
        rise = 0. # Should not happen based on request, but safe
    
    # Plateau
    plat = heaviside(-(t - (t1_exc+plateau))) * heaviside(t - (t1_exc))
    
    # Fall
    if tau2_exc > 1e-9:
        fall = np.exp(-(t - (t1_exc+plateau)) ** 2 / (2. * tau2_exc ** 2)) * heaviside(t - (t1_exc+plateau))
    else:
        fall = 0.0 # Instant fall
        
    return ampl_exc * (rise + plat + fall)

def OU(tfin, dt):
    theta = 1/(5*1.e-3 )
    mu = 0
    sigma = 1
    t = np.arange(0, tfin, dt)
    n = len(t)
    x = np.zeros(n)
    for i in range(1, n):
        dx = theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1)
        x[i] = x[i-1] + dx
    return x

# =============================================================================
# Default Parameters (SI Units)
# =============================================================================
# Based on User Request "Constant Parameters (All Curves)"
default_params = {
    'Cm': 200e-12,       # 200 pF
    'Qe': 1.5e-9,        # 1.5 nS
    'Qi': 5.0e-9,        # 5.0 nS
    'Ee': 0.0,           # 0 mV
    'Ei': -80e-3,        # -80 mV
    'tau_w': 500e-3,     # 500 ms
    'p_con': 0.05,       # 0.05
    'gei': 0.2,          # 0.2
    'Ntot': 10000,       # 10000
    'tau_e': 5.0e-3,     # 5.0 ms
    'b_e': 5.0e-12,      # 5.0 pA (User specified!)
    # Curve Specific will be overridden
    'tau_i': 5.0e-3,     # Default
    'Gl': 10e-9,         # Default
    'EL_e': -65e-3,      # Default
    'EL_i': -65e-3       # Default
}

# =============================================================================
# Transfer Function
# =============================================================================
def TF(P, fexc, finh, adapt, El, p):
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
    
    sV = np.sqrt(fe*(Ue*Te)*(Ue*Te)/2./(Te+Tm)+fi*(Ui*Ti)*(Ui*Ti)/2./(Ti+Tm))
    
    fe = fe + 1e-9
    fi = fi + 1e-9
    
    Tv = ( fe*(Ue*Te)*(Ue*Te) + fi*(Qi*Ui)*(Qi*Ui)) /( fe*(Ue*Te)*(Ue*Te)/(Te+Tm) + fi*(Qi*Ui)*(Qi*Ui)/(Ti+Tm) )
    TvN = Tv*Gl/Cm
    
    muV0=-60e-3; DmuV0 = 10e-3; sV0 =4e-3; DsV0= 6e-3; TvN0=0.5; DTvN0 = 1.
    
    vthr=P[0]+P[1]*(muV-muV0)/DmuV0+P[2]*(sV-sV0)/DsV0+P[3]*(TvN-TvN0)/DTvN0+\
         P[4]*((muV-muV0)/DmuV0)**2+P[5]*((sV-sV0)/DsV0)**2+P[6]*((TvN-TvN0)/DTvN0)**2+\
         P[7]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+P[8]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+\
         P[9]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0

    frout=.5/TvN*Gl/Cm*(1-erf((vthr-muV)/np.sqrt(2)/sV))
    return frout

# =============================================================================
# Run Sim Function
# =============================================================================
def run_simulation(cond_name, params_override, ax):
    p = default_params.copy()
    p.update(params_override)
    
    # Load fitting
    try:
        PRS = np.load(os.path.join(REPO_PATH, "Tf_calc/data/RS-cell0_CONFIG1_fit.npy"))
        PFS = np.load(os.path.join(REPO_PATH, "Tf_calc/data/FS-cell_CONFIG1_fit.npy"))
    except:
        print("Fit files not found")
        return

    TotTime = 1.3
    dt = 0.00001 # 0.01 ms as requested
    t = np.arange(0, TotTime, dt)
    
    # Stim Parameters
    # Using previous tuned values
    AmpStim = 1.0 # Hz
    time_peek = 200e-3 # s for input? input_rate function takes what?
    # Our previous script passed t*1000 so t was ms.
    # input_rate args: t, t1_exc, tau1_exc ...
    # Display Parameters on Plot
    # Parameters that distinguish the conditions: tau_i, Gl, EL_e, EL_i, b_e
    
    # helper for formatting
    def fmt(val, unit, scale=1.0):
        # formatted value
        return f"{val*scale:.3g} {unit}"

    param_str = (
        f"$\\tau_i$: {p['tau_i']*1000:.1f} ms\n"
        f"$g_L$: {p['Gl']*1e9:.2f} nS\n"
        f"$E_L^E$: {p['EL_e']*1000:.1f} mV\n"
        f"$E_L^I$: {p['EL_i']*1000:.1f} mV\n"
        f"$b_e$: {p['b_e']*1e12:.0f} pA\n"
        f"$\\sigma$: 0"
    )
    
    Tp_rise = 5.0 # ms (Keep rise 5ms)
    Tp_fall = 0.0 # ms (Make fall 0ms)
    Plat = 100.0 # ms
    
    Iext = 0.3 # User Request: Turn Iext back on (Active Fixed Point Hypothesis)
    
    # Noise
    # User requested: "remove the noise"
    noise = 0.0 * OU(TotTime, dt) + Iext
    
    # Pulse
    stim = input_rate(t*1000.0, 200.0, Tp_rise, Tp_fall, AmpStim, Plat)
    
    ext_inp = stim + noise
    
    fe = 0.0
    fi = 0.0
    w = 0.0
    T_const = 20e-3
    
    LSfe = []
    LSfi = []
    
    for i in range(len(t)):
        drive = ext_inp[i]
        FEX = fe + drive
        FINH = fe + drive
        if FEX < 0: FEX=0
        if FINH < 0: FINH=0
        
        # TF
        tfe = TF(PRS, FEX, fi, w, p['EL_e'], p)
        tfi = TF(PFS, FINH, fi, 0., p['EL_i'], p)
        
        fe += (dt/T_const)*(tfe - fe)
        fi += (dt/T_const)*(tfi - fi)
        w += dt*(-w/p['tau_w'] + p['b_e']*fe)
        
        LSfe.append(fe)
        LSfi.append(fi)
        
    ax.plot(t, LSfi, 'r', label='Inh')
    ax.plot(t, LSfe, 'steelblue', label='Exc')
    ax.plot(t, stim, 'g', alpha=0.5, label='Input')
    ax.set_ylim(-1, 30)
    ax.set_title(cond_name)
    # User Request: "merge the 2 legends into one"
    # We do this by putting the params as the legend title.
    ax.legend(loc='upper right', title=param_str, title_fontsize=8, fontsize=8)
    
    # Calculate pre-stimulus mean (t < 0.2)
    pre_stim_idx = np.where(t < 0.2)[0]
    pre_stim_mean = np.mean(np.array(LSfe)[pre_stim_idx])
    
    print(f"[{cond_name}] Pre-Stim Exc Rate: {pre_stim_mean:.6f} Hz")
    print(f"[{cond_name}] Final Exc Rate: {fe:.4f} Hz")
    return t, np.array(LSfe)

# =============================================================================
# Main
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Case 1: Reference
# tau_i: 5.0 ms, gL: 10 nS, EL: -65 mV
p1 = {
    'tau_i': 5.0e-3,
    'Gl': 10e-9,
    'EL_e': -65e-3,
    'EL_i': -65e-3
}
run_simulation("Reference (Awake)", p1, axes[0])

# Case 2: Propofol
# tau_i: 7.0 ms, gL: 10 nS, EL: -65 mV
p2 = {
    'tau_i': 7.0e-3,
    'Gl': 10e-9,
    'EL_e': -65e-3,
    'tau_i': 7.0e-3,
    'Gl': 10e-9,
    'EL_e': -65e-3,
    'EL_i': -65e-3,
    'b_e': 30e-12 # User Request: High adaptation for Anesthesia
}
run_simulation("Propofol", p2, axes[1])

# Case 3: Propofol + DOI
# tau_i: 7.0 ms, gL: 7.16 nS, EL: -55 mV
p3 = {
    'tau_i': 7.0e-3,
    'Gl': 7.16e-9,
    'EL_e': -55e-3, # User Request: Set both to -55mV
    'tau_i': 7.0e-3,
    'Gl': 7.16e-9,
    'EL_e': -55e-3, # User Request: Set both to -55mV
    'EL_i': -55e-3, # User Request: Set both to -55mV
    'b_e': 30e-12 # User Request: High adaptation for Anesthesia
}
run_simulation("Propofol + DOI", p3, axes[2])

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../../figures/fig3_three_conditions.png"))
print("Saved fig3_three_conditions.png")
