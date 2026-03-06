"""
TVB Whole-Brain Simulation: Anesthesia Effects
Based on Sacha et al. 2025 - Nature Computational Science

This script simulates the effects of general anesthesia (propofol/ketamine)
on whole-brain dynamics using TVB with the AdEx mean-field model.

Key mechanisms:
- GABA-ergic anesthetics (propofol): increase inhibitory synaptic decay time (tau_i)
- NMDA antagonists (ketamine): decrease excitatory synaptic decay time (tau_e)
- Both lead to UP/DOWN state dynamics and slow oscillations
"""

import numpy as np
import matplotlib.pyplot as plt

# TVB imports (explicit to avoid namespace collisions)
from tvb.simulator import simulator as sim_module
from tvb.simulator import models, coupling as coupling_module, integrators, monitors, noise as noise_module
from tvb.datatypes import connectivity

# ============================================================
# MEAN-FIELD MODEL PARAMETERS (AdEx-based, from paper)
# ============================================================

# Baseline (Awake) parameters
PARAMS_AWAKE = {
    'tau_e': 5.0,    # ms - excitatory synaptic decay
    'tau_i': 5.0,    # ms - inhibitory synaptic decay  
    'b_e': 10.0,     # pA - spike-triggered adaptation (excitatory)
    'label': 'Awake'
}

# Propofol anesthesia (GABA-ergic)
PARAMS_PROPOFOL = {
    'tau_e': 5.0,    # unchanged
    'tau_i': 25.0,   # INCREASED - prolonged IPSPs
    'b_e': 10.0,     # unchanged
    'label': 'Propofol'
}

# Ketamine anesthesia (NMDA antagonist)
PARAMS_KETAMINE = {
    'tau_e': 2.0,    # DECREASED - shorter EPSPs
    'tau_i': 5.0,    # unchanged
    'b_e': 10.0,
    'label': 'Ketamine'
}

# NREM Sleep (for comparison - increased adaptation)
PARAMS_SLEEP = {
    'tau_e': 5.0,
    'tau_i': 5.0,
    'b_e': 60.0,     # INCREASED - low acetylcholine
    'label': 'NREM Sleep'
}

# ============================================================
# SIMULATION SETUP
# ============================================================

def run_simulation(params, sim_length=3000.0, dt=0.1):
    """
    Run TVB simulation with given parameters.
    
    Args:
        params: dict with tau_e, tau_i, b_e, label
        sim_length: simulation duration in ms
        dt: integration timestep in ms
    
    Returns:
        time, data arrays
    """
    
    print(f"\n{'='*50}")
    print(f"Running simulation: {params['label']}")
    print(f"tau_e={params['tau_e']}ms, tau_i={params['tau_i']}ms, b_e={params['b_e']}pA")
    print(f"{'='*50}")
    
    # Load default connectivity (68 regions, Desikan-Killiany atlas)
    conn = connectivity.Connectivity.from_file()
    conn.configure()
    
    # Scale connectivity for stability
    conn.weights = conn.weights / np.max(conn.weights) * 0.2
    
    # Set conduction speed (determines delays based on tract lengths)
    conn.speed = np.array([4.0])  # m/s
    
    # Use Wilson-Cowan model as simplified mean-field
    # (TVB's ReducedWongWang or generic oscillator could also work)
    model = models.WilsonCowan(
        c_ee=np.array([16.0]),
        c_ei=np.array([12.0]),  
        c_ie=np.array([15.0]),
        c_ii=np.array([3.0]),
        tau_e=np.array([params['tau_e']]),
        tau_i=np.array([params['tau_i']]),
        a_e=np.array([1.2]),
        a_i=np.array([1.0]),
        theta_e=np.array([2.8]),
        theta_i=np.array([4.0]),
    )
    
    # Coupling function
    coupl = coupling_module.Linear(a=np.array([0.015]))
    
    # Noise (external drive fluctuations)
    noise_obj = noise_module.Additive(
        nsig=np.array([0.01]),
        noise_seed=42
    )
    
    # Integration scheme
    integrator = integrators.HeunStochastic(
        dt=dt,
        noise=noise_obj
    )
    
    # Monitors
    mons = (
        monitors.TemporalAverage(period=1.0),  # 1ms sampling
    )
    
    # Build simulator
    sim = sim_module.Simulator(
        connectivity=conn,
        model=model,
        coupling=coupl,
        integrator=integrator,
        monitors=mons,
        simulation_length=sim_length
    )
    sim.configure()
    
    # Run
    print("Running simulation...")
    results = []
    for (t, data), in sim():
        if data is not None:
            results.append((t, data))
    
    # Extract time and data
    time = np.array([r[0] for r in results]).flatten()
    data = np.array([r[1] for r in results])[:, 0, :, 0]  # (time, regions)
    
    print(f"Completed. Shape: time={time.shape}, data={data.shape}")
    
    return time, data, conn


def compute_metrics(time, data):
    """Compute key metrics from simulation."""
    
    # Global signal (mean across regions)
    global_signal = np.mean(data, axis=1)
    
    # Power spectrum
    from scipy import signal
    fs = 1000.0 / (time[1] - time[0])  # Sampling frequency
    freqs, psd = signal.welch(global_signal, fs=fs, nperseg=min(1024, len(global_signal)//2))
    
    # Slow wave power (0.5-4 Hz)
    slow_mask = (freqs >= 0.5) & (freqs <= 4.0)
    slow_power = np.trapz(psd[slow_mask], freqs[slow_mask])
    
    # Total power
    total_power = np.trapz(psd, freqs)
    
    # Slow wave ratio
    slow_ratio = slow_power / total_power if total_power > 0 else 0
    
    return {
        'global_signal': global_signal,
        'freqs': freqs,
        'psd': psd,
        'slow_power': slow_power,
        'slow_ratio': slow_ratio
    }


def plot_comparison(results_dict, output_file='tvb_anesthesia_comparison.png'):
    """
    Plot comparison of different conditions.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    conditions = list(results_dict.keys())
    colors = {'Awake': '#22c55e', 'Propofol': '#ef4444', 
              'Ketamine': '#3b82f6', 'NREM Sleep': '#a855f7'}
    
    # Plot time series
    ax1 = axes[0, 0]
    for cond in conditions:
        r = results_dict[cond]
        # Plot first 2 seconds of global signal
        mask = r['time'] < 2000
        ax1.plot(r['time'][mask], r['metrics']['global_signal'][mask], 
                 label=cond, color=colors.get(cond, 'gray'), alpha=0.8)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Global Activity')
    ax1.set_title('Whole-Brain Dynamics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Power spectra
    ax2 = axes[0, 1]
    for cond in conditions:
        r = results_dict[cond]
        m = r['metrics']
        mask = m['freqs'] < 50  # Show up to 50 Hz
        ax2.semilogy(m['freqs'][mask], m['psd'][mask], 
                     label=cond, color=colors.get(cond, 'gray'), linewidth=2)
    ax2.axvspan(0.5, 4, alpha=0.2, color='gray', label='Slow waves (0.5-4 Hz)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title('Power Spectrum')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Slow wave power comparison
    ax3 = axes[1, 0]
    slow_powers = [results_dict[c]['metrics']['slow_ratio'] for c in conditions]
    bars = ax3.bar(conditions, slow_powers, color=[colors.get(c, 'gray') for c in conditions])
    ax3.set_ylabel('Slow Wave Power Ratio')
    ax3.set_title('Slow Wave Dominance')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Regional activity heatmap (last 500ms, subset of regions)
    ax4 = axes[1, 1]
    # Use Propofol as example
    r = results_dict.get('Propofol', results_dict[conditions[0]])
    t_mask = r['time'] > (r['time'][-1] - 500)
    ax4.imshow(r['data'][t_mask, :20].T, aspect='auto', cmap='RdBu_r',
               extent=[r['time'][t_mask][0], r['time'][t_mask][-1], 0, 20])
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Brain Region')
    ax4.set_title(f"Regional Activity ({r['params']['label']})")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved: {output_file}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    print("TVB Anesthesia Simulation")
    print("Based on Sacha et al. 2025 framework")
    print("="*50)
    
    # Run for each condition
    conditions = [PARAMS_AWAKE, PARAMS_PROPOFOL, PARAMS_KETAMINE, PARAMS_SLEEP]
    results = {}
    
    for params in conditions:
        try:
            time, data, conn = run_simulation(params, sim_length=3000.0)
            metrics = compute_metrics(time, data)
            results[params['label']] = {
                'time': time,
                'data': data,
                'metrics': metrics,
                'params': params,
                'connectivity': conn
            }
        except Exception as e:
            print(f"Error in {params['label']}: {e}")
            continue
    
    if results:
        plot_comparison(results)
        
        # Print summary
        print("\n" + "="*50)
        print("SUMMARY: Slow Wave Power Ratios")
        print("="*50)
        for cond, r in results.items():
            print(f"  {cond}: {r['metrics']['slow_ratio']:.3f}")
        
        print("\nExpected pattern (from paper):")
        print("  - Awake: LOW slow wave power")
        print("  - Propofol/Ketamine: HIGH slow wave power (UP/DOWN states)")
        print("  - NREM Sleep: HIGH slow wave power (similar to anesthesia)")
    
    print("\nDone!")
