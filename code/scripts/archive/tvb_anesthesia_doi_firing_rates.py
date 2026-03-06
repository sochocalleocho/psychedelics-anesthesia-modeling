import os
"""
TVB Simulation: Population Firing Rates for NIH Proposal
Conditions: Awake, Propofol, Propofol + DOI
Model: AdEx Mean-Field (Zerlaut Adaptation First Order)

Parameters from Sacha et al. 2025 (verified via user image):

Constant Parameters:
- C_m = 200 pF
- Q_e = 1.5 nS (Excitatory Quantal Conductance)
- Q_i = 5.0 nS (Inhibitory Quantal Conductance)
- E_e = 0 mV
- E_i = -80 mV
- tau_w = 500 ms (Adaptation Time Constant)
- p_connect = 0.05
- g_ei = 0.2 (Inhibitory Fraction - implicit in N_tot/connectivity usually, or handled by TVB)
- N_tot = 10000
- tau_e = 5.0 ms
- b_e = 5.0 pA (Spike-triggered adaptation)

Curve-Specific Parameters:
- Awake:          tau_i=5ms, g_L=10nS,   E_L=-65mV
- Propofol:       tau_i=7ms, g_L=10nS,   E_L=-65mV
- Propofol + DOI: tau_i=7ms, g_L=7.16nS, E_L=-55mV
"""

import numpy as np
import matplotlib.pyplot as plt
from tvb.simulator import simulator, models, coupling, integrators, monitors, noise
from tvb.datatypes import connectivity

# =============================================================================
# 1. PARAMETER DEFINITIONS
# =============================================================================

CONDITIONS = {
    'Awake': {
        'tau_i': 5.0,   
        'g_L': 10.0,    
        'E_L': -65.0,   
        'color': '#22c55e'
    },
    'Propofol': {
        'tau_i': 7.0,   
        'g_L': 10.0,    
        'E_L': -65.0,   
        'color': '#ef4444'
    },
    'Propofol + DOI': {
        'tau_i': 7.0,   
        'g_L': 7.16,    
        'E_L': -55.0,   
        'color': '#3b82f6'
    }
}

# =============================================================================
# 2. SIMULATION FUNCTION
# =============================================================================

def run_condition(cond_name, params, sim_len=2000.0):
    print(f"Running condition: {cond_name}...")
    
    # 1. Connectivity
    conn = connectivity.Connectivity.from_file()
    conn.configure()
    conn.weights = conn.weights / np.max(conn.weights) * 2.0 
    conn.speed = np.array([4.0])

    # 2. Model: Zerlaut Adaptation (First Order)
    # Mapping image parameters to TVB Zerlaut model fields (Corrected parameter names)
    model = models.ZerlautAdaptationFirstOrder(
        # -- Dynamic (Condition-specific) --
        g_L=np.array([params['g_L']]),          # Leak conductance (nS)
        E_L_e=np.array([params['E_L']]),        # Leak reversal E (mV) - Note the underscore E_L_e
        E_L_i=np.array([params['E_L']]),        # Leak reversal I (mV)
        tau_i=np.array([params['tau_i']]),      # Inhibitory decay (ms)
        
        # -- Constant (from Image) --
        C_m=np.array([200.0]),                  # Capacitance (pF)
        b_e=np.array([5.0]),                    # Adaptation b_e (pA)
        b_i=np.array([0.0]),                    # Adaptation b_i (usually 0)
        tau_w_e=np.array([500.0]),              # Adaptation time constant E (ms)
        tau_w_i=np.array([500.0]),              # Adaptation time constant I (ms)
        tau_e=np.array([5.0]),                  # Excitatory decay (ms)
        
        # Reversal potentials
        E_e=np.array([0.0]),                    # Excitatory reversal (mV)
        E_i=np.array([-80.0]),                  # Inhibitory reversal (mV)
        
        # Quantal Conductances (Q_e, Q_i)
        Q_e=np.array([1.5]),                    # nS
        Q_i=np.array([5.0]),                    # nS
        
        # External drive to kickstart (baseline activity)
        # Using P_e (external Poisson rate)
        P_e=np.array([0.005])                   # kHz (5 Hz) background
    )
    
    # 3. Coupling
    coupl = coupling.Linear(a=np.array([0.02]))

    # 4. Integrator & Noise
    # Additive noise to drive transitions
    noise_inst = noise.Additive(nsig=np.array([1e-4, 1e-4, 0, 0, 0, 0, 0])) 
    integ = integrators.HeunStochastic(dt=0.1, noise=noise_inst)

    # 5. Monitors
    mon = monitors.TemporalAverage(period=1.0) # 1ms resolution

    # 6. Simulator
    sim = simulator.Simulator(
        model=model,
        connectivity=conn,
        coupling=coupl,
        integrator=integ,
        monitors=(mon,),
        simulation_length=sim_len
    )
    
    sim.configure()
    
    # Run
    (time, data), = sim.run()
    
    # Extract Excitatory Firing Rate (nu_e) - typically index 0 in Zerlaut
    # Output is usually in kHz, convert to Hz for plotting
    firing_rate_e = data[:, 0, :, 0] * 1000.0 
    
    return time, firing_rate_e

# =============================================================================
# 3. MAIN EXECUTION & PLOTTING
# =============================================================================

if __name__ == "__main__":
    
    results = {}
    
    # Run simulations
    for name, p in CONDITIONS.items():
        try:
            t, fr = run_condition(name, p)
            # Global mean
            mean_fr = np.mean(fr, axis=1)
            
            results[name] = {
                'time': t,
                'mean_fr': mean_fr
            }
        except Exception as e:
            print(f"Failed to run {name}: {e}")

    # --- Plot 1: Time Series ---
    plt.figure(figsize=(10, 6))
    
    for name, res in results.items():
        # Plot last 1500ms
        t_segment = res['time'][-1500:]
        fr_segment = res['mean_fr'][-1500:]
        
        plt.plot(t_segment, fr_segment, 
                 label=name, 
                 color=CONDITIONS[name]['color'], 
                 linewidth=2.5)
        
    plt.title("Population Firing Rates (AdEx Mean Field)", fontsize=14)
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Firing Rate (Hz)", fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "../../figures/NIH_FiringRates_TimeSeries.png"), dpi=300)
    print("Saved Time Series Plot")
    
    # --- Plot 2: Bar Chart ---
    plt.figure(figsize=(8, 6))
    
    means = []
    labels = []
    colors = []
    
    for name, res in results.items():
        # Average over stable period
        avg_rate = np.mean(res['mean_fr'][-1000:]) 
        means.append(avg_rate)
        labels.append(name)
        colors.append(CONDITIONS[name]['color'])
        
    bars = plt.bar(labels, means, color=colors, alpha=0.8, edgecolor='black', width=0.6)
    plt.ylabel("Mean Firing Rate (Hz)", fontsize=12)
    plt.title("State Comparison: Firing Rate Restoration", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f} Hz',
                 ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "../../figures/NIH_FiringRates_BarChart.png"), dpi=300)
    print("Saved Bar Chart")
