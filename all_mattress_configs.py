#!/usr/bin/env python3
"""
All Mattress Configurations - Animated Comparison with Cumulative Damage
=========================================================================
Shows ALL bed configurations tested over 120 minutes with:
- Cumulative tissue damage tracking based on Braden Scale risk factors
- Pressure-time exposure model
- Risk scoring based on clinical thresholds

Evidence Base:
- Dweekat et al. 2023: Braden Scale + Random Forest prediction model
- Braden Scale: Clinical risk assessment (scores 6-23, lower = higher risk)
- Capillary closing pressure: 32 mmHg (Landis 1930)
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import zoom
from multidynamic_mattress_optimization import (
    SMPLBodyPressureModel, MultiDynamicAirMattress,
    MOVEMENT_PATTERNS, CAPILLARY_CLOSING_PRESSURE
)
from evolved_pattern import EvolvedOptimalPattern

# =============================================================================
# BRADEN SCALE-BASED DAMAGE MODEL (Dweekat et al. 2023)
# =============================================================================

# Pressure threshold (Landis 1930)
PRESSURE_THRESHOLD = 32  # mmHg - capillary closing pressure

# Braden Scale risk levels and corresponding damage multipliers
# Total Braden score ranges from 6-23 (lower = higher risk)
# Risk categories from clinical literature:
BRADEN_RISK_LEVELS = {
    'very_high': {'score_range': (6, 9), 'damage_multiplier': 2.0},    # Very high risk
    'high': {'score_range': (10, 12), 'damage_multiplier': 1.5},       # High risk
    'moderate': {'score_range': (13, 14), 'damage_multiplier': 1.2},   # Moderate risk
    'at_risk': {'score_range': (15, 18), 'damage_multiplier': 1.0},    # At risk
    'no_risk': {'score_range': (19, 23), 'damage_multiplier': 0.5},    # No risk
}

# Default patient: Quadriplegic (very high risk, Braden ~8)
# - Sensory perception: 1 (completely limited)
# - Moisture: 2 (often moist)
# - Activity: 1 (bedfast)
# - Mobility: 1 (completely immobile)
# - Nutrition: 2 (probably inadequate)
# - Friction/Shear: 1 (problem)
# Total: 8 = Very High Risk
DEFAULT_BRADEN_SCORE = 8
DAMAGE_MULTIPLIER = BRADEN_RISK_LEVELS['very_high']['damage_multiplier']

# Time-based damage thresholds (Dweekat et al. findings)
# ICU patients develop HAPI within median 5-7 days
# High-risk patients: damage accumulates faster
# Key finding: Braden subscales + time in ICU are primary predictors

# Damage accumulation rate (per minute above threshold, per mmHg excess)
# Calibrated so that 2 hours at 50 mmHg = 100% risk for very high risk patient
# (50-32) × 120 min × rate × 2.0 multiplier = 100
# 18 × 120 × rate × 2.0 = 100 → rate = 0.023
BASE_DAMAGE_RATE = 0.023

# Recovery rate when pressure relieved
RECOVERY_RATE = 0.3  # 30% of damage rate during relief

# Critical threshold (100 = high probability of HAPI)
CRITICAL_DAMAGE = 100

# Shear stress parameters
SHEAR_THRESHOLD = 10  # mmHg equivalent - shear above this compounds damage
SHEAR_SYNERGY = 1.5   # Multiplier when both pressure AND shear present
QUADRIPLEGIC_SHEAR_FACTOR = 1.3  # 30% higher shear due to spasticity

# Physical air mattress constraints
TRANSITION_TIME = 45.0  # Seconds for cell to fully inflate/deflate (realistic pump speed)


class RealisticMattressState:
    """Tracks actual cell states with realistic transition physics."""

    def __init__(self, rows: int, cols: int, transition_time: float = TRANSITION_TIME):
        self.rows = rows
        self.cols = cols
        self.transition_time = transition_time
        self.max_rate = 1.0 / transition_time  # Max change per second

        # Actual cell states (start fully inflated)
        self.actual_state = np.ones((rows, cols))
        self.last_time = 0.0

    def update(self, target_state: np.ndarray, time_seconds: float) -> np.ndarray:
        """
        Move actual states toward target at limited rate.

        Args:
            target_state: Desired cell states from pattern
            time_seconds: Current simulation time

        Returns:
            Actual cell states after realistic transition
        """
        dt = time_seconds - self.last_time
        self.last_time = time_seconds

        if dt <= 0:
            return self.actual_state

        # Maximum change possible in this time step
        max_change = self.max_rate * dt

        # Move each cell toward target at limited rate
        diff = target_state - self.actual_state
        change = np.clip(diff, -max_change, max_change)
        self.actual_state = self.actual_state + change

        return self.actual_state.copy()


def calculate_cumulative_damage(pressure_history, shear_history, time_step_min, braden_score=DEFAULT_BRADEN_SCORE):
    """
    Calculate cumulative tissue damage using Braden Scale-based model with shear forces.

    Based on Dweekat et al. 2023: Integrated Braden Scale and Random Forest
    model for predicting when Hospital-Acquired Pressure Injuries occur.

    Shear model based on:
    - Shear stress compounds pressure damage (multiplicative effect)
    - Quadriplegic patients have 30% higher shear due to spasticity
    - Combined pressure + shear synergy factor of 1.5×

    Args:
        pressure_history: List of pressure maps over time
        shear_history: List of shear stress maps over time
        time_step_min: Time between frames in minutes
        braden_score: Patient's Braden Scale score (6-23)

    Returns:
        cumulative_damage: Final damage map (100 = HAPI likely)
        damage_history: Damage at each time point
    """
    # Determine damage multiplier based on Braden score
    if braden_score <= 9:
        multiplier = BRADEN_RISK_LEVELS['very_high']['damage_multiplier']
    elif braden_score <= 12:
        multiplier = BRADEN_RISK_LEVELS['high']['damage_multiplier']
    elif braden_score <= 14:
        multiplier = BRADEN_RISK_LEVELS['moderate']['damage_multiplier']
    elif braden_score <= 18:
        multiplier = BRADEN_RISK_LEVELS['at_risk']['damage_multiplier']
    else:
        multiplier = BRADEN_RISK_LEVELS['no_risk']['damage_multiplier']

    shape = pressure_history[0].shape
    cumulative_damage = np.zeros(shape)
    damage_history = []

    for i, pressure_map in enumerate(pressure_history):
        shear_map = shear_history[i]

        # Cells above threshold accumulate damage
        excess_pressure = np.maximum(0, pressure_map - PRESSURE_THRESHOLD)
        excess_shear = np.maximum(0, shear_map - SHEAR_THRESHOLD)

        # Shear amplifies pressure damage (1.0 to 2.0× based on shear level)
        shear_factor = 1 + np.clip(shear_map / SHEAR_THRESHOLD, 0, 1)

        # Pressure-shear synergy: when BOTH are elevated, damage is multiplied
        has_both = (excess_pressure > 0) & (excess_shear > 0)
        synergy_factor = np.where(has_both, SHEAR_SYNERGY, 1.0)

        # Combined damage calculation
        # Base pressure damage × shear amplification × synergy × Braden risk
        pressure_damage = excess_pressure * BASE_DAMAGE_RATE * time_step_min
        shear_damage = excess_shear * BASE_DAMAGE_RATE * 0.5 * time_step_min  # Shear alone is 50% as damaging

        damage_increment = (pressure_damage * shear_factor + shear_damage) * synergy_factor * multiplier
        cumulative_damage += damage_increment

        # Cells below BOTH thresholds recover partially
        below_both = (pressure_map <= PRESSURE_THRESHOLD) & (shear_map <= SHEAR_THRESHOLD)
        recovery = RECOVERY_RATE * BASE_DAMAGE_RATE * time_step_min * multiplier
        cumulative_damage[below_both] = np.maximum(
            0, cumulative_damage[below_both] - recovery
        )

        damage_history.append(cumulative_damage.copy())

    return cumulative_damage, damage_history


def create_animated_comparison():
    """Create animated comparison of ALL mattress configurations with cumulative damage."""

    print("Generating Animated Mattress Comparison with Cumulative Damage...")
    print("=" * 60)

    # Generate body pressure and shear maps
    model = SMPLBodyPressureModel(75, 30)
    body_pressure, body_shear = model.calculate_pressure_map(40, 18)

    # Apply quadriplegic shear factor (spasticity increases shear)
    body_shear = body_shear * QUADRIPLEGIC_SHEAR_FACTOR

    print(f"\nBase body pressure: Peak={body_pressure.max():.1f} mmHg")
    print(f"Base body shear: Peak={body_shear.max():.1f} mmHg equivalent")

    # All configurations to test (Standard Foam is baseline)
    configs = {
        'Standard Foam': {'type': 'foam'},
    }

    # Add all movement patterns
    for name, pattern in MOVEMENT_PATTERNS.items():
        display_name = pattern.name if hasattr(pattern, 'name') else name.replace('_', ' ').title()
        configs[display_name] = {'type': 'apm', 'pattern': pattern, 'key': name}

    # Add genetically evolved pattern
    configs['Evolved Optimal'] = {'type': 'apm', 'pattern': EvolvedOptimalPattern(), 'key': 'evolved_optimal'}

    print(f"\nTesting {len(configs)} configurations:")
    for name in configs:
        print(f"  - {name}")

    # Simulate 120 minutes (2 hours) - clinical repositioning interval
    total_time = 120 * 60  # 120 minutes in seconds
    cycle_period = 300     # 5-minute cycles
    n_frames = 48          # 48 frames = every 2.5 minutes
    time_points = np.linspace(0, total_time, n_frames, endpoint=False)
    time_step_min = (time_points[1] - time_points[0]) / 60  # Minutes between frames

    # Calculate pressure and cumulative damage for each config
    # Use realistic transition physics (45 seconds for full inflate/deflate)
    all_results = {}

    for config_name, config in configs.items():
        pressure_history = []
        shear_history = []

        # Create mattress ONCE and track state across all time steps
        if config['type'] != 'foam':
            pattern = config['pattern']
            mattress = MultiDynamicAirMattress(cell_size_cm=5, movement_pattern=pattern)
            realistic_state = RealisticMattressState(mattress.rows, mattress.cols)

            # Resample body maps to mattress grid (do once)
            scale_y = mattress.rows / body_pressure.shape[0]
            scale_x = mattress.cols / body_pressure.shape[1]
            body_resampled = zoom(body_pressure, (scale_y, scale_x), order=1)
            shear_resampled = zoom(body_shear, (scale_y, scale_x), order=1)

        for time_sec in time_points:
            if config['type'] == 'foam':
                effective_pressure = body_pressure * 0.85
                effective_shear = body_shear * 0.85  # Foam reduces shear similarly
            else:
                # Get target state from pattern
                mattress.update(time_sec)
                target_state = mattress.cell_state.copy()

                # Apply realistic transition physics (45-second full transition)
                actual_state = realistic_state.update(target_state, time_sec)

                # Calculate pressure with ACTUAL (not target) cell states
                effective_pressure = body_resampled.copy()
                for row in range(mattress.rows):
                    for col in range(mattress.cols):
                        inflation = actual_state[row, col]
                        if inflation < 0.5:
                            relief_factor = 1 - inflation
                            pressure_relieved = effective_pressure[row, col] * relief_factor * 0.7
                            effective_pressure[row, col] -= pressure_relieved

                # Shear is reduced when cells deflate (less friction surface)
                effective_shear = shear_resampled * actual_state

                # Resample back to display size
                scale_y_back = body_pressure.shape[0] / effective_pressure.shape[0]
                scale_x_back = body_pressure.shape[1] / effective_pressure.shape[1]
                effective_pressure = zoom(effective_pressure, (scale_y_back, scale_x_back), order=1)
                effective_shear = zoom(effective_shear, (scale_y_back, scale_x_back), order=1)

            pressure_history.append(effective_pressure)
            shear_history.append(effective_shear)

        # Calculate cumulative damage (now includes shear)
        final_damage, damage_history = calculate_cumulative_damage(pressure_history, shear_history, time_step_min)

        # Store results
        all_results[config_name] = []
        for i, time_sec in enumerate(time_points):
            all_results[config_name].append({
                'pressure': pressure_history[i],
                'shear': shear_history[i],
                'damage': damage_history[i],
                'peak_pressure': pressure_history[i].max(),
                'peak_shear': shear_history[i].max(),
                'cells_over_32': (pressure_history[i] > 32).sum(),
                'cells_high_shear': (shear_history[i] > SHEAR_THRESHOLD).sum(),
                'cells_at_risk': (damage_history[i] >= CRITICAL_DAMAGE).sum(),
                'max_damage': damage_history[i].max(),
                'total_damage': damage_history[i].sum(),
                'time_min': time_sec / 60,
            })

        avg_shear = np.mean([s.max() for s in shear_history])
        print(f"  {config_name}: Max damage = {final_damage.max():.1f}, At risk = {(final_damage >= CRITICAL_DAMAGE).sum()}, Avg peak shear = {avg_shear:.1f}")

    # Create figure layout - 4 rows x 4 cols = 16 slots (13 configs)
    n_configs = len(configs)
    n_cols = 4
    n_rows = (n_configs + n_cols - 1) // n_cols

    config_names = list(configs.keys())

    # Calculate FINAL damage metrics for each config (after 120 min)
    final_metrics = {}
    for name in config_names:
        final_frame = all_results[name][-1]  # Last frame = end of 120 min
        avg_shear = np.mean([r['peak_shear'] for r in all_results[name]])
        final_metrics[name] = {
            'cells_at_risk': final_frame['cells_at_risk'],
            'max_damage': final_frame['max_damage'],
            'total_damage': final_frame['total_damage'],
            'avg_peak_shear': avg_shear,
        }

    # Create subplot titles showing FINAL DAMAGE (compared to Standard Foam baseline)
    baseline_risk = final_metrics['Standard Foam']['cells_at_risk']
    baseline_damage = final_metrics['Standard Foam']['total_damage']
    first_frame_titles = []
    for name in config_names:
        fm = final_metrics[name]
        if baseline_damage > 0:
            reduction = (1 - fm['total_damage'] / baseline_damage) * 100
        else:
            reduction = 0
        if name == 'Standard Foam':
            first_frame_titles.append(f"<b>{name} (Baseline)</b><br>Max: {fm['max_damage']:.0f} | At Risk: {fm['cells_at_risk']}")
        else:
            first_frame_titles.append(f"<b>{name}</b><br>Max: {fm['max_damage']:.0f} | Damage: {reduction:+.0f}%")

    # Pad with empty titles
    while len(first_frame_titles) < n_rows * n_cols:
        first_frame_titles.append("")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=first_frame_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    # Colorscale for cumulative damage (green = safe, red = HAPI likely)
    # Scale: 0 = no damage, 100 = HAPI likely (based on Braden risk model)
    damage_colorscale = [
        [0, 'rgb(0, 100, 0)'],        # 0 - safe
        [0.2, 'rgb(144, 238, 144)'],   # Low risk
        [0.4, 'rgb(255, 255, 0)'],     # Moderate risk
        [0.6, 'rgb(255, 165, 0)'],     # High risk
        [0.8, 'rgb(255, 0, 0)'],       # Very high risk
        [1.0, 'rgb(100, 0, 0)']        # HAPI likely (>100)
    ]

    # Add initial frame (time 0 - no damage yet)
    for idx, name in enumerate(config_names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        r = all_results[name][0]

        fig.add_trace(
            go.Heatmap(
                z=r['damage'],
                colorscale=damage_colorscale,
                zmin=0,
                zmax=CRITICAL_DAMAGE * 1.5,  # 150 to show cells that exceeded threshold
                showscale=(idx == 0),
                colorbar=dict(
                    title='Risk Score',
                    tickvals=[0, 25, 50, 75, 100, 150],
                    ticktext=['0', '25', '50', '75', '100<br>(HAPI)', '150'],
                    len=0.3,
                    y=0.85,
                    x=1.02
                ) if idx == 0 else None,
                hovertemplate=f'{name}<br>Risk: %{{z:.0f}}<br>(100 = HAPI likely)<extra></extra>'
            ),
            row=row, col=col
        )

    # Create animation frames showing damage accumulation
    frames = []

    for frame_idx, time_sec in enumerate(time_points):
        time_min = time_sec / 60
        frame_data = []

        # Count total cells at risk across all configs at this time
        total_at_risk = sum(all_results[name][frame_idx]['cells_at_risk'] for name in config_names)

        for idx, name in enumerate(config_names):
            r = all_results[name][frame_idx]
            frame_data.append(
                go.Heatmap(
                    z=r['damage'],
                    colorscale=damage_colorscale,
                    zmin=0,
                    zmax=CRITICAL_DAMAGE * 1.5,
                    showscale=(idx == 0),
                )
            )

        frames.append(go.Frame(
            data=frame_data,
            name=str(frame_idx),
            layout=go.Layout(
                title=dict(
                    text=f'<b>Braden Scale Risk Model with Shear Forces - 120 Minute Simulation</b><br>'
                         f'<sup>Time: {time_min:.1f} min | '
                         f'Risk ≥100 = HAPI likely | Includes pressure + shear synergy (1.5×)</sup>'
                )
            )
        ))

    fig.frames = frames

    # Animation controls
    fig.update_layout(
        title=dict(
            text='<b>Braden Scale Risk Model with Shear Forces - 120 Minute Simulation</b><br>'
                 '<sup>Time: 0 min | Risk ≥100 = HAPI likely | Includes pressure + shear synergy (1.5×)</sup>',
            x=0.5,
            font=dict(size=18)
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.08,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': 200, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 100}
                        }]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'prefix': 'Time: ',
                'suffix': ' min',
                'visible': True,
                'xanchor': 'center'
            },
            'transition': {'duration': 100},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.05,
            'y': 0,
            'steps': [
                {
                    'args': [[str(i)], {
                        'frame': {'duration': 100, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 100}
                    }],
                    'label': f'{time_points[i]/60:.0f}',
                    'method': 'animate'
                }
                for i in range(n_frames)
            ]
        }],
        height=1000,
        width=1400,
    )

    fig.write_html('all_mattress_configs.html', include_plotlyjs=True, full_html=True)
    print(f"\nSaved: all_mattress_configs.html")

    # Print summary with Braden Scale-based damage metrics
    print("\n" + "=" * 85)
    print("SUMMARY - BRADEN SCALE RISK MODEL WITH SHEAR FORCES (120 MINUTES)")
    print("=" * 85)
    print(f"{'Configuration':<30} {'At Risk':>10} {'Max Risk':>10} {'Avg Shear':>10} {'vs Foam':>10}")
    print("-" * 85)

    baseline_damage = final_metrics['Standard Foam']['total_damage']

    for name in config_names:
        fm = final_metrics[name]
        reduction = (1 - fm['total_damage'] / baseline_damage) * 100 if baseline_damage > 0 else 0
        print(f"{name:<30} {fm['cells_at_risk']:>10} {fm['max_damage']:>10.0f} {fm['avg_peak_shear']:>10.1f} {reduction:>+9.0f}%")

    print("-" * 85)
    print("Evidence base:")
    print("  - Dweekat et al. 2023: Braden Scale + Random Forest HAPI prediction")
    print("  - Patient modeled: Quadriplegic (Braden score 8 = very high risk)")
    print("  - Braden subscales: Sensory=1, Mobility=1, Activity=1, Moisture=2, Nutrition=2, Friction=1")
    print(f"  - Shear model: {QUADRIPLEGIC_SHEAR_FACTOR}× spasticity factor, {SHEAR_SYNERGY}× pressure-shear synergy")
    print(f"  - Cells at Risk = risk score ≥ {CRITICAL_DAMAGE} (HAPI likely)")
    print(f"  - Pressure threshold: {PRESSURE_THRESHOLD} mmHg | Shear threshold: {SHEAR_THRESHOLD} mmHg")
    print(f"  - Realistic physics: {TRANSITION_TIME:.0f} seconds for full cell inflate/deflate")

    return all_results


if __name__ == "__main__":
    create_animated_comparison()
