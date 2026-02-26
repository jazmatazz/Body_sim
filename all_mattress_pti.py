#!/usr/bin/env python3
"""
All Mattress Configurations - Cumulative PTI (Pressure-Time Integral)
======================================================================
Shows cumulative pressure exposure over time in mmHg·hours.

PTI = Σ (Pressure × Time)

See also: all_mattress_pressure.html for instantaneous pressure
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

# Physical air mattress constraints
TRANSITION_TIME = 45.0  # Seconds for cell to fully inflate/deflate

# =============================================================================
# REALISTIC PRESSURE REDISTRIBUTION FACTORS
# =============================================================================
# Based on clinical pressure mapping studies (Defloor 2000, Reenalda 2009)
FOAM_REDISTRIBUTION = 0.15      # Foam reduces peak pressure by ~85%
APM_INFLATED_REDISTRIBUTION = 0.15  # Inflated APM cells similar to foam
APM_DEFLATED_RELIEF = 0.85      # Deflated cells provide additional 85% relief

# =============================================================================
# SHEAR AND MOISTURE FACTORS
# =============================================================================
# Shear amplification (Bennett 1979): shear reduces pressure tolerance by ~50%
SHEAR_THRESHOLD = 5.0      # mmHg - threshold for significant shear
SHEAR_AMPLIFICATION = 1.5  # Multiplier when shear present

# Moisture levels (based on Braden scale moisture subscale)
# 1 = constantly moist, 2 = very moist, 3 = occasionally moist, 4 = rarely moist
MOISTURE_MULTIPLIERS = {
    1: 1.5,   # Constantly moist - 50% increased risk (Beeckman 2014)
    2: 1.3,   # Very moist - 30% increased risk
    3: 1.1,   # Occasionally moist - 10% increased risk
    4: 1.0,   # Rarely moist - baseline
}
DEFAULT_MOISTURE_LEVEL = 2  # Assume "very moist" for high-risk patient


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


def create_pti_comparison():
    """Create animated comparison showing cumulative PTI."""

    print("Generating Cumulative PTI Comparison...")
    print("=" * 60)

    # Generate body pressure map
    model = SMPLBodyPressureModel(75, 30)
    body_pressure, body_shear = model.calculate_pressure_map(40, 18)

    print(f"\nBase body pressure: Peak={body_pressure.max():.1f} mmHg")

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
    dt_hours = (time_points[1] - time_points[0]) / 3600  # Time step in hours

    # Calculate cumulative PTI for each config at each time point
    all_results = {}

    for config_name, config in configs.items():
        if config['type'] != 'foam':
            pattern = config['pattern']
            mattress = MultiDynamicAirMattress(cell_size_cm=5, movement_pattern=pattern)
            realistic_state = RealisticMattressState(mattress.rows, mattress.cols)

            scale_y = mattress.rows / body_pressure.shape[0]
            scale_x = mattress.cols / body_pressure.shape[1]
            body_resampled = zoom(body_pressure, (scale_y, scale_x), order=1)
            shear_resampled = zoom(body_shear, (scale_y, scale_x), order=1)

        all_results[config_name] = []
        cumulative_pti = np.zeros_like(body_pressure)

        for frame_idx, time_sec in enumerate(time_points):
            if config['type'] == 'foam':
                # Foam mattress provides ~85% pressure redistribution
                effective_pressure = body_pressure * FOAM_REDISTRIBUTION
                effective_shear = body_shear * FOAM_REDISTRIBUTION
            else:
                mattress.update(time_sec)
                target_state = mattress.cell_state.copy()
                actual_state = realistic_state.update(target_state, time_sec)

                # Start with inflated APM baseline (similar to foam)
                effective_pressure = body_resampled * APM_INFLATED_REDISTRIBUTION
                effective_shear = shear_resampled * APM_INFLATED_REDISTRIBUTION
                for row in range(mattress.rows):
                    for col in range(mattress.cols):
                        inflation = actual_state[row, col]
                        if inflation < 0.5:
                            # Deflated cells provide additional pressure relief
                            relief_factor = 1 - inflation
                            pressure_relieved = effective_pressure[row, col] * relief_factor * APM_DEFLATED_RELIEF
                            effective_pressure[row, col] -= pressure_relieved
                            # Shear also reduced when cells deflate
                            effective_shear[row, col] *= inflation

                # Resample back to display size
                scale_y_back = body_pressure.shape[0] / effective_pressure.shape[0]
                scale_x_back = body_pressure.shape[1] / effective_pressure.shape[1]
                effective_pressure = zoom(effective_pressure, (scale_y_back, scale_x_back), order=1)
                effective_shear = zoom(effective_shear, (scale_y_back, scale_x_back), order=1)

            # Accumulate PTI: Pressure × Time
            cumulative_pti += effective_pressure * dt_hours

            all_results[config_name].append({
                'pti': cumulative_pti.copy(),
                'peak_pti': cumulative_pti.max(),
                'mean_pti': cumulative_pti[cumulative_pti > 0].mean(),
                'time_min': time_sec / 60,
            })

        # Report final metrics
        final = all_results[config_name][-1]
        print(f"  {config_name}: Final Peak PTI={final['peak_pti']:.1f} mmHg·h")

    # Calculate final metrics for comparison
    final_metrics = {}
    for name in configs:
        final = all_results[name][-1]
        final_metrics[name] = {
            'peak_pti': final['peak_pti'],
            'mean_pti': final['mean_pti'],
        }

    # Create figure
    n_configs = len(configs)
    n_cols = 4
    n_rows = (n_configs + n_cols - 1) // n_cols
    config_names = list(configs.keys())

    # Subplot titles
    baseline_peak = final_metrics['Standard Foam']['peak_pti']
    first_frame_titles = []
    for name in config_names:
        fm = final_metrics[name]
        change = (fm['peak_pti'] / baseline_peak - 1) * 100 if baseline_peak > 0 else 0
        if name == 'Standard Foam':
            first_frame_titles.append(f"<b>{name} (Baseline)</b><br>Peak PTI: {fm['peak_pti']:.1f} mmHg·h")
        else:
            first_frame_titles.append(f"<b>{name}</b><br>Peak: {fm['peak_pti']:.1f} | vs Foam: {change:+.0f}%")

    while len(first_frame_titles) < n_rows * n_cols:
        first_frame_titles.append("")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=first_frame_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    # Fixed color scale max (values above this show as dark red)
    zmax = 50.0

    # Dynamic colorbar ticks
    tick_step = zmax / 5
    colorbar_tickvals = [0, tick_step, tick_step*2, tick_step*3, tick_step*4, zmax]
    colorbar_ticktext = ['0', f'{tick_step:.0f}', f'{tick_step*2:.0f}',
                         f'{tick_step*3:.0f}', f'{tick_step*4:.0f}', f'{zmax:.0f}']

    # Colorscale for PTI (green = low exposure, red = high exposure)
    pti_colorscale = [
        [0, 'rgb(0, 100, 0)'],
        [0.2, 'rgb(144, 238, 144)'],
        [0.4, 'rgb(255, 255, 0)'],
        [0.6, 'rgb(255, 165, 0)'],
        [0.8, 'rgb(255, 0, 0)'],
        [1.0, 'rgb(100, 0, 0)']
    ]

    # Initial frame
    for idx, name in enumerate(config_names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        r = all_results[name][0]

        fig.add_trace(
            go.Heatmap(
                z=np.flipud(r['pti']),
                colorscale=pti_colorscale,
                zmin=0,
                zmax=zmax,
                showscale=(idx == 0),
                colorbar=dict(
                    title='PTI<br>(mmHg·h)',
                    tickvals=colorbar_tickvals,
                    ticktext=colorbar_ticktext,
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
                    z=np.flipud(r['pti']),
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
                    text=f'<b>Cumulative PTI (Pressure-Time Integral)</b><br>'
                         f'<sup>Time: {time_min:.1f} min | Units: mmHg·hours</sup>'
                )
            )
        ))

    fig.frames = frames

    # Layout
    fig.update_layout(
        title=dict(
            text='<b>Cumulative PTI (Pressure-Time Integral)</b><br>'
                 '<sup>Time: 0 min | Units: mmHg·hours | Simple pressure × time metric</sup>',
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
                ticktext=['200', '100', '0'],  # Head at top, feet at bottom
                scaleanchor=f'x{axis_suffix}' if axis_suffix else 'x',
                scaleratio=1,
            ),
        })

    fig.write_html('all_mattress_pti.html', include_plotlyjs=True, full_html=True)
    print(f"\nSaved: all_mattress_pti.html")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - CUMULATIVE PTI (120 MINUTES)")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Peak PTI':>12} {'Mean PTI':>12} {'vs Foam':>10}")
    print(f"{'':30} {'(mmHg·h)':>12} {'(mmHg·h)':>12} {'':>10}")
    print("-" * 80)

    for name in config_names:
        fm = final_metrics[name]
        change = (fm['peak_pti'] / baseline_peak - 1) * 100 if baseline_peak > 0 else 0
        print(f"{name:<30} {fm['peak_pti']:>12.1f} {fm['mean_pti']:>12.1f} {change:>+9.0f}%")

    print("-" * 80)
    print("\nPTI = Σ (Pressure × Time)  [mmHg·hours]")

    return all_results


if __name__ == "__main__":
    create_pti_comparison()
