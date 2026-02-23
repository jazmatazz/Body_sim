#!/usr/bin/env python3
"""
All Mattress Configurations - Comprehensive Tissue Damage Simulation
=====================================================================

Evidence-based pressure ulcer simulation incorporating 8 key factors
from peer-reviewed literature:

1. Deep Tissue Injury - Pressure at bone-muscle interface (Gefen 2009)
2. Reactive Hyperemia - Blood reperfusion recovery (Tzen 2017)
3. Temperature Effects - Microclimate vulnerability (Kottner 2018)
4. Moisture Effects - Skin maceration (Beeckman 2014)
5. Patient Micro-movements - Spontaneous movements (Stinson 2013)
6. Body Position - Position-specific pressure (Moore 2011)
7. Tissue Stiffness - Age/condition variation (Gefen 2005)
8. Time-Dependent Tolerance - Cumulative insults (Gefen & Weihs 2016)

See tissue_damage_model.py for full citations.
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
from tissue_damage_model import (
    ComprehensiveTissueDamageModel,
    PatientProfile,
    BodyPosition,
    PatientCondition,
    create_default_patient,
    CAPILLARY_CLOSING_PRESSURE as PRESSURE_THRESHOLD,
)

# Physical air mattress constraints
TRANSITION_TIME = 45.0  # Seconds for cell to fully inflate/deflate

# Re-export for evolve_optimal_mattress.py compatibility
QUADRIPLEGIC_SHEAR_FACTOR = 1.3
CRITICAL_PTI = 20.0
SHEAR_THRESHOLD = 10


class RealisticMattressState:
    """Tracks actual cell states with realistic transition physics."""

    def __init__(self, rows: int, cols: int, transition_time: float = TRANSITION_TIME):
        self.rows = rows
        self.cols = cols
        self.transition_time = transition_time
        self.max_rate = 1.0 / transition_time

        self.actual_state = np.ones((rows, cols))
        self.last_time = 0.0

    def update(self, target_state: np.ndarray, time_seconds: float) -> np.ndarray:
        """Move actual states toward target at limited rate."""
        dt = time_seconds - self.last_time
        self.last_time = time_seconds

        if dt <= 0:
            return self.actual_state

        max_change = self.max_rate * dt
        diff = target_state - self.actual_state
        change = np.clip(diff, -max_change, max_change)
        self.actual_state = self.actual_state + change

        return self.actual_state.copy()


def calculate_cumulative_damage(pressure_history, shear_history, time_step_min,
                                braden_score=8, use_comprehensive=True):
    """
    Calculate cumulative tissue damage using comprehensive model.

    Args:
        pressure_history: List of pressure maps over time
        shear_history: List of shear stress maps over time
        time_step_min: Time between frames in minutes
        braden_score: Patient's Braden Scale score (6-23)
        use_comprehensive: If True, use full 8-factor model

    Returns:
        cumulative_damage: Final damage map (PTI in mmHg·hours)
        damage_history: Damage at each time point
    """
    shape = pressure_history[0].shape
    dt_hours = time_step_min / 60

    if use_comprehensive:
        # Create patient profile based on Braden score
        patient = create_default_patient('quadriplegic')
        patient.braden_score = braden_score

        # Create comprehensive model
        model = ComprehensiveTissueDamageModel(
            patient=patient,
            position=BodyPosition.SUPINE,
            grid_shape=shape
        )

        damage_history = []
        for i, pressure_map in enumerate(pressure_history):
            shear_map = shear_history[i]
            result = model.calculate_damage(pressure_map, shear_map, dt_hours)
            damage_history.append(result['combined_pti'].copy())

        return model.state.surface_pti + model.state.deep_pti * 4, damage_history

    else:
        # Simplified PTI calculation (legacy)
        cumulative_pti = np.zeros(shape)
        damage_history = []

        for i, pressure_map in enumerate(pressure_history):
            shear_map = shear_history[i]
            excess_pressure = np.maximum(0, pressure_map - PRESSURE_THRESHOLD)
            excess_shear = np.maximum(0, shear_map - SHEAR_THRESHOLD)

            shear_factor = 1 + np.clip(shear_map / SHEAR_THRESHOLD, 0, 1)
            has_both = (excess_pressure > 0) & (excess_shear > 0)
            synergy_factor = np.where(has_both, 1.5, 1.0)

            pti = (excess_pressure * shear_factor + excess_shear * 0.5) * synergy_factor * dt_hours
            cumulative_pti += pti

            below_both = (pressure_map <= PRESSURE_THRESHOLD) & (shear_map <= SHEAR_THRESHOLD)
            recovery_pti = 0.3 * PRESSURE_THRESHOLD * dt_hours
            cumulative_pti[below_both] = np.maximum(0, cumulative_pti[below_both] - recovery_pti)

            damage_history.append(cumulative_pti.copy())

        return cumulative_pti, damage_history


def create_animated_comparison():
    """Create animated comparison of ALL mattress configurations."""

    print("=" * 70)
    print("COMPREHENSIVE TISSUE DAMAGE SIMULATION")
    print("=" * 70)
    print("\nEvidence-based model incorporating 8 factors from peer-reviewed literature:")
    print("  1. Deep Tissue Injury (Gefen 2009)")
    print("  2. Reactive Hyperemia (Tzen 2017)")
    print("  3. Temperature Effects (Kottner 2018)")
    print("  4. Moisture/Maceration (Beeckman 2014)")
    print("  5. Micro-movements (Stinson 2013)")
    print("  6. Body Position (Moore 2011)")
    print("  7. Tissue Stiffness (Gefen 2005)")
    print("  8. Cumulative Tolerance (Gefen & Weihs 2016)")
    print("=" * 70)

    # Generate body pressure and shear maps
    model = SMPLBodyPressureModel(75, 30)
    body_pressure, body_shear = model.calculate_pressure_map(40, 18)

    # Apply quadriplegic shear factor (spasticity increases shear)
    body_shear = body_shear * QUADRIPLEGIC_SHEAR_FACTOR

    print(f"\nBase body pressure: Peak={body_pressure.max():.1f} mmHg")
    print(f"Base body shear: Peak={body_shear.max():.1f} mmHg equivalent")

    # All configurations to test
    configs = {
        'Standard Foam': {'type': 'foam'},
    }

    for name, pattern in MOVEMENT_PATTERNS.items():
        display_name = pattern.name if hasattr(pattern, 'name') else name.replace('_', ' ').title()
        configs[display_name] = {'type': 'apm', 'pattern': pattern, 'key': name}

    configs['Evolved Optimal'] = {'type': 'apm', 'pattern': EvolvedOptimalPattern(), 'key': 'evolved_optimal'}

    print(f"\nTesting {len(configs)} configurations:")
    for name in configs:
        print(f"  - {name}")

    # Simulate 120 minutes
    total_time = 120 * 60
    n_frames = 48
    time_points = np.linspace(0, total_time, n_frames, endpoint=False)
    time_step_min = (time_points[1] - time_points[0]) / 60

    all_results = {}

    for config_name, config in configs.items():
        pressure_history = []
        shear_history = []

        if config['type'] != 'foam':
            pattern = config['pattern']
            mattress = MultiDynamicAirMattress(cell_size_cm=5, movement_pattern=pattern)
            realistic_state = RealisticMattressState(mattress.rows, mattress.cols)

            scale_y = mattress.rows / body_pressure.shape[0]
            scale_x = mattress.cols / body_pressure.shape[1]
            body_resampled = zoom(body_pressure, (scale_y, scale_x), order=1)
            shear_resampled = zoom(body_shear, (scale_y, scale_x), order=1)

        for time_sec in time_points:
            if config['type'] == 'foam':
                effective_pressure = body_pressure * 0.85
                effective_shear = body_shear * 0.85
            else:
                mattress.update(time_sec)
                target_state = mattress.cell_state.copy()
                actual_state = realistic_state.update(target_state, time_sec)

                effective_pressure = body_resampled.copy()
                for row in range(mattress.rows):
                    for col in range(mattress.cols):
                        inflation = actual_state[row, col]
                        if inflation < 0.5:
                            relief_factor = 1 - inflation
                            pressure_relieved = effective_pressure[row, col] * relief_factor * 0.7
                            effective_pressure[row, col] -= pressure_relieved

                effective_shear = shear_resampled * actual_state

                scale_y_back = body_pressure.shape[0] / effective_pressure.shape[0]
                scale_x_back = body_pressure.shape[1] / effective_pressure.shape[1]
                effective_pressure = zoom(effective_pressure, (scale_y_back, scale_x_back), order=1)
                effective_shear = zoom(effective_shear, (scale_y_back, scale_x_back), order=1)

            pressure_history.append(effective_pressure)
            shear_history.append(effective_shear)

        # Calculate damage using comprehensive model
        final_damage, damage_history = calculate_cumulative_damage(
            pressure_history, shear_history, time_step_min,
            braden_score=8, use_comprehensive=True
        )

        all_results[config_name] = []
        for i, time_sec in enumerate(time_points):
            # Adjust critical threshold for high-risk patient
            critical_threshold = CRITICAL_PTI / 2.0  # Braden 8 = 2× multiplier
            all_results[config_name].append({
                'pressure': pressure_history[i],
                'shear': shear_history[i],
                'damage': damage_history[i],
                'peak_pressure': pressure_history[i].max(),
                'peak_shear': shear_history[i].max(),
                'cells_over_32': (pressure_history[i] > 32).sum(),
                'cells_at_risk': (damage_history[i] >= critical_threshold).sum(),
                'max_damage': damage_history[i].max(),
                'total_damage': damage_history[i].sum(),
                'time_min': time_sec / 60,
            })

        print(f"  {config_name}: Max PTI = {final_damage.max():.1f} mmHg·h, "
              f"At risk = {(final_damage >= critical_threshold).sum()}")

    # Create visualization
    n_configs = len(configs)
    n_cols = 4
    n_rows = (n_configs + n_cols - 1) // n_cols
    config_names = list(configs.keys())

    # Final metrics
    final_metrics = {}
    critical_threshold = CRITICAL_PTI / 2.0
    for name in config_names:
        final_frame = all_results[name][-1]
        avg_shear = np.mean([r['peak_shear'] for r in all_results[name]])
        final_metrics[name] = {
            'cells_at_risk': (final_frame['damage'] >= critical_threshold).sum(),
            'max_damage': final_frame['max_damage'],
            'total_damage': final_frame['total_damage'],
            'avg_peak_shear': avg_shear,
        }

    # Subplot titles
    baseline_damage = final_metrics['Standard Foam']['total_damage']
    first_frame_titles = []
    for name in config_names:
        fm = final_metrics[name]
        change = (fm['total_damage'] / baseline_damage - 1) * 100 if baseline_damage > 0 else 0
        if name == 'Standard Foam':
            first_frame_titles.append(f"<b>{name} (Baseline)</b><br>Peak: {fm['max_damage']:.0f} mmHg·h")
        else:
            first_frame_titles.append(f"<b>{name}</b><br>Peak: {fm['max_damage']:.0f} | vs Foam: {change:+.0f}%")

    while len(first_frame_titles) < n_rows * n_cols:
        first_frame_titles.append("")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=first_frame_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    # Colorscale
    pti_colorscale = [
        [0, 'rgb(0, 100, 0)'],
        [0.2, 'rgb(144, 238, 144)'],
        [0.4, 'rgb(255, 255, 0)'],
        [0.6, 'rgb(255, 165, 0)'],
        [0.8, 'rgb(255, 0, 0)'],
        [1.0, 'rgb(100, 0, 0)']
    ]

    # Determine max value for colorscale
    max_pti = max(final_metrics[n]['max_damage'] for n in config_names)
    zmax = min(max_pti * 1.2, 500)  # Cap at 500 for visibility

    # Initial frame
    for idx, name in enumerate(config_names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        r = all_results[name][0]

        fig.add_trace(
            go.Heatmap(
                z=r['damage'],
                colorscale=pti_colorscale,
                zmin=0,
                zmax=zmax,
                showscale=(idx == 0),
                colorbar=dict(
                    title='PTI<br>(mmHg·h)',
                    len=0.3,
                    y=0.85,
                    x=1.02
                ) if idx == 0 else None,
                hovertemplate=f'{name}<br>PTI: %{{z:.1f}} mmHg·h<extra></extra>'
            ),
            row=row, col=col
        )

    # Animation frames
    frames = []
    for frame_idx, time_sec in enumerate(time_points):
        time_min = time_sec / 60
        frame_data = []

        for idx, name in enumerate(config_names):
            r = all_results[name][frame_idx]
            frame_data.append(
                go.Heatmap(
                    z=r['damage'],
                    colorscale=pti_colorscale,
                    zmin=0,
                    zmax=zmax,
                    showscale=(idx == 0),
                )
            )

        frames.append(go.Frame(
            data=frame_data,
            name=str(frame_idx),
            layout=go.Layout(
                title=dict(
                    text=f'<b>Comprehensive Tissue Damage Model - 8 Evidence-Based Factors</b><br>'
                         f'<sup>Time: {time_min:.1f} min | PTI includes DTI, temperature, moisture, tolerance</sup>'
                )
            )
        ))

    fig.frames = frames

    # Layout
    fig.update_layout(
        title=dict(
            text='<b>Comprehensive Tissue Damage Model - 8 Evidence-Based Factors</b><br>'
                 '<sup>Time: 0 min | PTI includes deep tissue injury, temperature, moisture, tolerance</sup>',
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
                    dict(label='Play', method='animate',
                         args=[None, {'frame': {'duration': 200, 'redraw': True},
                                     'fromcurrent': True, 'transition': {'duration': 100}}]),
                    dict(label='Pause', method='animate',
                         args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                       'mode': 'immediate', 'transition': {'duration': 0}}])
                ]
            )
        ],
        sliders=[{
            'active': 0,
            'currentvalue': {'prefix': 'Time: ', 'suffix': ' min', 'visible': True},
            'transition': {'duration': 100},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.05,
            'y': 0,
            'steps': [
                {'args': [[str(i)], {'frame': {'duration': 100, 'redraw': True},
                                    'mode': 'immediate', 'transition': {'duration': 100}}],
                 'label': f'{time_points[i]/60:.0f}',
                 'method': 'animate'}
                for i in range(n_frames)
            ]
        }],
        height=1600,
        width=1800,
    )

    # Axis labels
    for i in range(n_rows * n_cols):
        axis_suffix = '' if i == 0 else str(i + 1)
        fig.update_layout(**{
            f'xaxis{axis_suffix}': dict(
                title='Width (cm)' if i >= (n_rows - 1) * n_cols else None,
                tickvals=[0, 9, 18],
                ticktext=['0', '45', '90'],
                constrain='domain',
            ),
            f'yaxis{axis_suffix}': dict(
                title='Length (cm)' if i % n_cols == 0 else None,
                tickvals=[0, 20, 40],
                ticktext=['0', '100', '200'],
                scaleanchor=f'x{axis_suffix}' if axis_suffix else 'x',
                scaleratio=1,
            ),
        })

    fig.write_html('all_mattress_configs.html', include_plotlyjs=True, full_html=True)
    print(f"\nSaved: all_mattress_configs.html")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY - COMPREHENSIVE 8-FACTOR MODEL (120 MINUTES)")
    print("=" * 90)
    print(f"{'Configuration':<30} {'At Risk':>10} {'Peak PTI':>12} {'Avg Shear':>10} {'vs Foam':>10}")
    print(f"{'':30} {'(cells)':>10} {'(mmHg·h)':>12} {'(mmHg)':>10} {'':>10}")
    print("-" * 90)

    for name in config_names:
        fm = final_metrics[name]
        change = (fm['total_damage'] / baseline_damage - 1) * 100 if baseline_damage > 0 else 0
        print(f"{name:<30} {fm['cells_at_risk']:>10} {fm['max_damage']:>12.1f} {fm['avg_peak_shear']:>10.1f} {change:>+9.0f}%")

    print("-" * 90)
    print("\nModel includes all 8 evidence-based factors:")
    print("  1. Deep Tissue Injury: Muscle 4× more sensitive than skin (Gefen 2009)")
    print("  2. Reactive Hyperemia: 30% recovery when relieved, diminishes with cycles (Tzen 2017)")
    print("  3. Temperature: +10% damage per °C rise, up to 3°C under occlusion (Kottner 2018)")
    print("  4. Moisture: 4.99× risk with incontinence, maceration over 2h (Beeckman 2014)")
    print("  5. Micro-movements: SCI patients ~2/hour vs 20/hour able-bodied (Stinson 2013)")
    print("  6. Body Position: Supine sacrum 1.0×, lateral 90° trochanter 2.5× (Moore 2011)")
    print("  7. Tissue Stiffness: Spasticity 1.4×, diabetes 1.5×, elderly 1.3× (Gefen 2005)")
    print("  8. Cumulative Tolerance: -10% per insult, min 30% (Gefen & Weihs 2016)")
    print(f"\nPatient: Quadriplegic (Braden 8), Critical PTI: {critical_threshold:.1f} mmHg·h")

    return all_results


if __name__ == "__main__":
    create_animated_comparison()
