#!/usr/bin/env python3
"""
Quadriplegic Bed Optimization Comparison
=========================================
Compares different bed intervention strategies to find the optimal
configuration that minimizes pressure ulcer risk with NO nurse intervention.

Strategies tested:
1. Lateral rotation period (how often the bed completes a full tilt cycle)
2. Lateral rotation angle (how far the bed tilts)
3. Alternating pressure cycle period
4. Zone control aggressiveness
5. Combined optimal strategies

Author: Biomedical Simulation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pressure_ulcer_simulation_advanced import (
    AdvancedPressureUlcerSimulation,
    QUADRIPLEGIC_FACTORS,
    MOISTURE_MANAGEMENT
)
import time

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Base configuration for quadriplegic patient
BASE_CONFIG = {
    'body_mass': 80,
    'cell_size': 5,
    'simulation_time': 3600,  # 1 hour
    'timestep': 30,
    'waist_hinge_angle': 15,  # Slight incline
    'quadriplegic_mode': True,
}

# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

# Strategy 1: Lateral Rotation Period Comparison
ROTATION_PERIOD_STRATEGIES = {
    'No Rotation': {
        'enable_lateral_rotation': False,
        'enable_zone_control': True,
        'enable_alternating': True,
    },
    'Slow (20 min cycle)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 1200,
        'lateral_rotation_angle': 25,
        'enable_zone_control': True,
        'enable_alternating': True,
    },
    'Medium (10 min cycle)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 600,
        'lateral_rotation_angle': 25,
        'enable_zone_control': True,
        'enable_alternating': True,
    },
    'Fast (5 min cycle)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 25,
        'enable_zone_control': True,
        'enable_alternating': True,
    },
    'Very Fast (2 min cycle)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 120,
        'lateral_rotation_angle': 25,
        'enable_zone_control': True,
        'enable_alternating': True,
    },
}

# Strategy 2: Lateral Rotation Angle Comparison
ROTATION_ANGLE_STRATEGIES = {
    'No Rotation (0°)': {
        'enable_lateral_rotation': False,
        'enable_zone_control': True,
        'enable_alternating': True,
    },
    'Gentle (15°)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 15,
        'enable_zone_control': True,
        'enable_alternating': True,
    },
    'Moderate (25°)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 25,
        'enable_zone_control': True,
        'enable_alternating': True,
    },
    'Aggressive (35°)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 35,
        'enable_zone_control': True,
        'enable_alternating': True,
    },
    'Maximum (45°)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 45,
        'enable_zone_control': True,
        'enable_alternating': True,
    },
}

# Strategy 3: Alternating Pressure Cycle Comparison
ALTERNATING_STRATEGIES = {
    'No Alternating': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 25,
        'enable_zone_control': True,
        'enable_alternating': False,
    },
    'Slow (10 min cycle)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 25,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 600,
    },
    'Medium (5 min cycle)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 25,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 300,
    },
    'Fast (2 min cycle)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 25,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 120,
    },
    'Very Fast (1 min cycle)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 25,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 60,
    },
}

# Strategy 4: Combined Interventions
COMBINED_STRATEGIES = {
    'Baseline (No Interventions)': {
        'enable_lateral_rotation': False,
        'enable_zone_control': False,
        'enable_alternating': False,
    },
    'Zone Control Only': {
        'enable_lateral_rotation': False,
        'enable_zone_control': True,
        'enable_alternating': False,
    },
    'Alternating Only': {
        'enable_lateral_rotation': False,
        'enable_zone_control': False,
        'enable_alternating': True,
        'pressure_cycle_period': 300,
    },
    'Rotation Only': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 30,
        'enable_zone_control': False,
        'enable_alternating': False,
    },
    'Zone + Alternating': {
        'enable_lateral_rotation': False,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 300,
    },
    'Zone + Rotation': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 30,
        'enable_zone_control': True,
        'enable_alternating': False,
    },
    'Alternating + Rotation': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 30,
        'enable_zone_control': False,
        'enable_alternating': True,
        'pressure_cycle_period': 300,
    },
    'ALL INTERVENTIONS': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 30,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 300,
    },
}

# Strategy 5: Optimized Combinations (based on expected best parameters)
OPTIMIZED_STRATEGIES = {
    'Conservative': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 600,  # 10 min
        'lateral_rotation_angle': 20,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 300,  # 5 min
    },
    'Moderate': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,  # 5 min
        'lateral_rotation_angle': 30,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 180,  # 3 min
    },
    'Aggressive': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 180,  # 3 min
        'lateral_rotation_angle': 35,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 120,  # 2 min
    },
    'Maximum Protection': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 120,  # 2 min
        'lateral_rotation_angle': 40,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 60,  # 1 min
    },
}

# Strategy 6: Moisture Management Comparison
# Uses moderate bed settings, varies only moisture control
MOISTURE_STRATEGIES = {
    'No Moisture Control': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 30,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 300,
        'moisture_management': 'none',
    },
    'Basic (Absorbent Pads)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 30,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 300,
        'moisture_management': 'basic',
    },
    'Moderate (Barrier Cream)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 30,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 300,
        'moisture_management': 'moderate',
    },
    'Comprehensive (Catheter+)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 30,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 300,
        'moisture_management': 'comprehensive',
    },
    'Optimal (Full Protocol)': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 300,
        'lateral_rotation_angle': 30,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 300,
        'moisture_management': 'optimal',
    },
}

# Strategy 7: Optimized with Moisture Control (best bed settings + moisture control)
OPTIMIZED_WITH_MOISTURE = {
    'Aggressive Bed + No Moisture': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 180,
        'lateral_rotation_angle': 35,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 120,
        'moisture_management': 'none',
    },
    'Aggressive Bed + Basic Moisture': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 180,
        'lateral_rotation_angle': 35,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 120,
        'moisture_management': 'basic',
    },
    'Aggressive Bed + Comprehensive': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 180,
        'lateral_rotation_angle': 35,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 120,
        'moisture_management': 'comprehensive',
    },
    'Max Bed + Optimal Moisture': {
        'enable_lateral_rotation': True,
        'lateral_rotation_period': 120,
        'lateral_rotation_angle': 40,
        'enable_zone_control': True,
        'enable_alternating': True,
        'pressure_cycle_period': 60,
        'moisture_management': 'optimal',
    },
}


def run_strategy_comparison(strategies, comparison_name, verbose=True):
    """
    Run simulation for each strategy and collect results.
    """
    results = {}

    print(f"\n{'='*70}")
    print(f"COMPARISON: {comparison_name}")
    print(f"{'='*70}")

    for name, config in strategies.items():
        print(f"\n  Running: {name}...")
        start_time = time.time()

        # Merge with base config
        full_config = {**BASE_CONFIG, **config}

        # Run simulation
        sim = AdvancedPressureUlcerSimulation(**full_config)
        sim_results = sim.run(verbose=False)

        elapsed = time.time() - start_time

        # Extract key metrics
        stats = sim_results['stats_history']
        risk_cats = sim_results['risk_categories']

        results[name] = {
            # Final state
            'peak_pressure': sim_results['final_pressure'].max(),
            'mean_pressure': sim_results['final_pressure'][sim_results['final_pressure'] > 0].mean(),
            'peak_shear': sim_results['final_shear'].max() / 1000,
            'max_damage': sim_results['final_risk'].max(),
            'mean_damage': sim_results['final_risk'][sim_results['final_risk'] > 0].mean() if (sim_results['final_risk'] > 0).any() else 0,

            # Risk cell counts
            'critical_cells': int(risk_cats['critical'].sum()),
            'high_cells': int(risk_cats['high'].sum()),
            'moderate_cells': int(risk_cats['moderate'].sum()),
            'low_cells': int(risk_cats['low'].sum()),
            'minimal_cells': int(risk_cats['minimal'].sum()),

            # Time series
            'pressure_history': [s['peak_pressure'] for s in stats],
            'risk_history': [s['max_risk'] for s in stats],
            'time_min': np.array(sim_results['time_history']) / 60,

            # Time to critical thresholds
            'time_to_50pct_damage': find_time_to_threshold([s['max_risk'] for s in stats], 0.5, sim_results['time_history']),
            'time_to_critical': find_time_to_threshold([s['max_risk'] for s in stats], 0.75, sim_results['time_history']),
        }

        print(f"    Done in {elapsed:.1f}s | Peak P: {results[name]['peak_pressure']:.1f} mmHg | "
              f"Max Risk: {results[name]['max_damage']:.3f} | Critical: {results[name]['critical_cells']} cells")

    return results


def find_time_to_threshold(values, threshold, times):
    """Find the first time a threshold is exceeded."""
    for i, v in enumerate(values):
        if v >= threshold:
            return times[i] / 60  # Return in minutes
    return None  # Never reached


def create_comparison_plot(results, comparison_name, save_path):
    """Create visualization comparing strategies."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    names = list(results.keys())
    n_strategies = len(names)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_strategies))

    # 1. Peak Pressure Over Time
    ax = axes[0, 0]
    for i, name in enumerate(names):
        r = results[name]
        ax.plot(r['time_min'], r['pressure_history'], label=name,
                color=colors[i], linewidth=2)
    ax.axhline(32, color='red', linestyle='--', alpha=0.7, label='Capillary Threshold')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Peak Pressure (mmHg)')
    ax.set_title('Peak Interface Pressure Over Time')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 2. Risk Accumulation Over Time
    ax = axes[0, 1]
    for i, name in enumerate(names):
        r = results[name]
        ax.plot(r['time_min'], r['risk_history'], label=name,
                color=colors[i], linewidth=2)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='50% Damage')
    ax.axhline(0.75, color='red', linestyle='--', alpha=0.7, label='Critical')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Maximum Damage Index')
    ax.set_title('Risk Accumulation Over Time')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Final Risk Comparison (Bar Chart)
    ax = axes[0, 2]
    x = np.arange(n_strategies)
    max_damages = [results[n]['max_damage'] for n in names]
    bars = ax.bar(x, max_damages, color=colors)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.7)
    ax.axhline(0.75, color='red', linestyle='--', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Maximum Damage Index')
    ax.set_title('Final Maximum Damage')
    ax.set_ylim(0, 1.05)

    # 4. Critical/High Risk Cells
    ax = axes[1, 0]
    width = 0.35
    critical = [results[n]['critical_cells'] for n in names]
    high = [results[n]['high_cells'] for n in names]
    ax.bar(x - width/2, critical, width, label='Critical', color='#e74c3c')
    ax.bar(x + width/2, high, width, label='High', color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Number of Cells')
    ax.set_title('Critical & High Risk Cells at End')
    ax.legend()

    # 5. Time to Critical Damage
    ax = axes[1, 1]
    times_to_critical = []
    for n in names:
        t = results[n]['time_to_critical']
        times_to_critical.append(t if t is not None else 60)  # Cap at 60 min if never reached
    bars = ax.bar(x, times_to_critical, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Time (minutes)')
    ax.set_title('Time to Reach Critical Damage (longer = better)')
    # Mark bars that never reached critical
    for i, t in enumerate(times_to_critical):
        if results[names[i]]['time_to_critical'] is None:
            bars[i].set_color('#27ae60')
            ax.text(i, t + 1, 'Never', ha='center', fontsize=8, color='green')

    # 6. Summary Score (lower is better)
    ax = axes[1, 2]
    scores = []
    for n in names:
        r = results[n]
        # Composite score: weighted combination of risk factors
        score = (
            r['max_damage'] * 40 +  # Max damage (0-1) × 40
            r['critical_cells'] * 0.5 +  # Critical cells × 0.5
            r['high_cells'] * 0.2 +  # High risk cells × 0.2
            (60 - (r['time_to_critical'] or 60)) * 0.3  # Time to critical (inverted)
        )
        scores.append(score)

    bars = ax.bar(x, scores, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Risk Score (lower = better)')
    ax.set_title('Composite Risk Score')

    # Highlight best
    best_idx = np.argmin(scores)
    bars[best_idx].set_color('#27ae60')
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(3)

    plt.suptitle(f'QUADRIPLEGIC BED OPTIMIZATION: {comparison_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_path}")
    plt.close()

    return scores, names[best_idx]


def print_comparison_table(results, comparison_name):
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"RESULTS TABLE: {comparison_name}")
    print(f"{'='*90}")
    print(f"{'Strategy':<30} {'Peak P':>10} {'Max Risk':>10} {'Critical':>10} {'Time→Crit':>12}")
    print(f"{'':30} {'(mmHg)':>10} {'(0-1)':>10} {'(cells)':>10} {'(min)':>12}")
    print("-"*90)

    for name, r in results.items():
        time_crit = f"{r['time_to_critical']:.1f}" if r['time_to_critical'] else "Never"
        print(f"{name:<30} {r['peak_pressure']:>10.1f} {r['max_damage']:>10.3f} "
              f"{r['critical_cells']:>10} {time_crit:>12}")
    print("="*90)


def main():
    """Run all comparisons."""
    print("="*70)
    print("QUADRIPLEGIC BED OPTIMIZATION STUDY")
    print("Finding the best automated bed settings to minimize nurse intervention")
    print("="*70)
    print(f"\nSimulation duration: {BASE_CONFIG['simulation_time']/60:.0f} minutes")
    print(f"Patient: {BASE_CONFIG['body_mass']} kg quadriplegic")
    print(f"Quadriplegic factors applied:")
    print(f"  - Damage rate: {QUADRIPLEGIC_FACTORS['damage_rate_multiplier']}x normal")
    print(f"  - Muscle atrophy: {QUADRIPLEGIC_FACTORS['muscle_atrophy_factor']*100:.0f}% of normal")
    print(f"  - Baseline moisture: {QUADRIPLEGIC_FACTORS['baseline_moisture']*100:.0f}%")

    all_results = {}
    best_strategies = {}

    # Comparison 1: Lateral Rotation Period
    results = run_strategy_comparison(
        ROTATION_PERIOD_STRATEGIES,
        "Lateral Rotation Period"
    )
    print_comparison_table(results, "Lateral Rotation Period")
    scores, best = create_comparison_plot(
        results,
        "Lateral Rotation Period",
        "comparison_rotation_period.png"
    )
    all_results['rotation_period'] = results
    best_strategies['rotation_period'] = best

    # Comparison 2: Lateral Rotation Angle
    results = run_strategy_comparison(
        ROTATION_ANGLE_STRATEGIES,
        "Lateral Rotation Angle"
    )
    print_comparison_table(results, "Lateral Rotation Angle")
    scores, best = create_comparison_plot(
        results,
        "Lateral Rotation Angle",
        "comparison_rotation_angle.png"
    )
    all_results['rotation_angle'] = results
    best_strategies['rotation_angle'] = best

    # Comparison 3: Alternating Pressure Cycle
    results = run_strategy_comparison(
        ALTERNATING_STRATEGIES,
        "Alternating Pressure Cycle Period"
    )
    print_comparison_table(results, "Alternating Pressure Cycle Period")
    scores, best = create_comparison_plot(
        results,
        "Alternating Pressure Cycle",
        "comparison_alternating.png"
    )
    all_results['alternating'] = results
    best_strategies['alternating'] = best

    # Comparison 4: Combined Interventions
    results = run_strategy_comparison(
        COMBINED_STRATEGIES,
        "Combined Intervention Strategies"
    )
    print_comparison_table(results, "Combined Intervention Strategies")
    scores, best = create_comparison_plot(
        results,
        "Combined Interventions",
        "comparison_combined.png"
    )
    all_results['combined'] = results
    best_strategies['combined'] = best

    # Comparison 5: Optimized Combinations
    results = run_strategy_comparison(
        OPTIMIZED_STRATEGIES,
        "Optimized Combinations"
    )
    print_comparison_table(results, "Optimized Combinations")
    scores, best = create_comparison_plot(
        results,
        "Optimized Combinations",
        "comparison_optimized.png"
    )
    all_results['optimized'] = results
    best_strategies['optimized'] = best

    # Comparison 6: Moisture Management (NEW)
    results = run_strategy_comparison(
        MOISTURE_STRATEGIES,
        "Moisture Management Levels"
    )
    print_comparison_table(results, "Moisture Management Levels")
    scores, best = create_comparison_plot(
        results,
        "Moisture Management",
        "comparison_moisture.png"
    )
    all_results['moisture'] = results
    best_strategies['moisture'] = best

    # Comparison 7: Optimized Bed + Moisture Control
    results = run_strategy_comparison(
        OPTIMIZED_WITH_MOISTURE,
        "Optimized Bed + Moisture Control"
    )
    print_comparison_table(results, "Optimized Bed + Moisture Control")
    scores, best = create_comparison_plot(
        results,
        "Optimized + Moisture Control",
        "comparison_optimized_moisture.png"
    )
    all_results['optimized_moisture'] = results
    best_strategies['optimized_moisture'] = best

    # Final Summary
    print("\n" + "="*70)
    print("FINAL RECOMMENDATIONS")
    print("="*70)
    print("\nBest strategy in each category:")
    for category, best in best_strategies.items():
        print(f"  {category}: {best}")

    print("\n" + "-"*70)
    print("CONCLUSION:")
    print("-"*70)

    # Find overall best from optimized strategies
    optimized = all_results['optimized']
    best_optimized = min(optimized.keys(),
                         key=lambda k: optimized[k]['max_damage'] + optimized[k]['critical_cells']*0.01)

    print(f"\nBest overall strategy: {best_optimized}")
    print(f"  - Max damage: {optimized[best_optimized]['max_damage']:.3f}")
    print(f"  - Critical cells: {optimized[best_optimized]['critical_cells']}")
    print(f"  - Time to critical: {optimized[best_optimized]['time_to_critical'] or 'Never'}")

    # Check if any strategy prevents critical damage
    no_critical = [k for k, v in optimized.items() if v['critical_cells'] == 0]
    if no_critical:
        print(f"\nStrategies that PREVENTED critical damage: {no_critical}")
    else:
        print("\nWARNING: No strategy completely prevented critical damage.")
        print("Nurse repositioning may still be required.")

    print("\n" + "="*70)
    print("Output files:")
    print("  - comparison_rotation_period.png")
    print("  - comparison_rotation_angle.png")
    print("  - comparison_alternating.png")
    print("  - comparison_combined.png")
    print("  - comparison_optimized.png")
    print("  - comparison_moisture.png")
    print("  - comparison_optimized_moisture.png")
    print("="*70)

    return all_results, best_strategies


if __name__ == "__main__":
    all_results, best_strategies = main()
