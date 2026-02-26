#!/usr/bin/env python3
"""
All Mattress Configurations - Real-Time Interface Pressure
===========================================================
Shows instantaneous interface pressure (mmHg) at each moment:
- What the mattress is doing RIGHT NOW
- Pressure relief cycles visible in animation
- Cells above 32 mmHg capillary closing threshold

See also: all_mattress_pti.html for cumulative pressure exposure.
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
from all_mattress_pti import RealisticMattressState


# =============================================================================
# REALISTIC PRESSURE REDISTRIBUTION FACTORS
# =============================================================================
# Based on clinical pressure mapping studies (Defloor 2000, Reenalda 2009)
# Body pressure is theoretical contact pressure without support
# Real mattresses redistribute this significantly

FOAM_REDISTRIBUTION = 0.15      # Foam reduces peak pressure by ~85%
APM_INFLATED_REDISTRIBUTION = 0.15  # Inflated APM cells similar to foam
APM_DEFLATED_RELIEF = 0.85      # Deflated cells provide additional 85% relief


def create_pressure_comparison():
    """Create animated comparison showing real-time interface pressure."""

    print("Generating Real-Time Interface Pressure Comparison...")
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

    # Calculate instantaneous pressure for each config at each time point
    all_results = {}

    for config_name, config in configs.items():
        if config['type'] != 'foam':
            pattern = config['pattern']
            mattress = MultiDynamicAirMattress(cell_size_cm=5, movement_pattern=pattern)
            realistic_state = RealisticMattressState(mattress.rows, mattress.cols)

            scale_y = mattress.rows / body_pressure.shape[0]
            scale_x = mattress.cols / body_pressure.shape[1]
            body_resampled = zoom(body_pressure, (scale_y, scale_x), order=1)

        all_results[config_name] = []

        for time_sec in time_points:
            if config['type'] == 'foam':
                # Foam mattress provides ~85% pressure redistribution
                effective_pressure = body_pressure * FOAM_REDISTRIBUTION
            else:
                mattress.update(time_sec)
                target_state = mattress.cell_state.copy()
                actual_state = realistic_state.update(target_state, time_sec)

                # Start with inflated APM baseline (similar to foam)
                effective_pressure = body_resampled * APM_INFLATED_REDISTRIBUTION
                for row in range(mattress.rows):
                    for col in range(mattress.cols):
                        inflation = actual_state[row, col]
                        if inflation < 0.5:
                            # Deflated cells provide additional pressure relief
                            relief_factor = 1 - inflation
                            pressure_relieved = effective_pressure[row, col] * relief_factor * APM_DEFLATED_RELIEF
                            effective_pressure[row, col] -= pressure_relieved

                # Resample back to display size
                scale_y_back = body_pressure.shape[0] / effective_pressure.shape[0]
                scale_x_back = body_pressure.shape[1] / effective_pressure.shape[1]
                effective_pressure = zoom(effective_pressure, (scale_y_back, scale_x_back), order=1)

            all_results[config_name].append({
                'pressure': effective_pressure,
                'peak_pressure': effective_pressure.max(),
                'cells_over_32': (effective_pressure > 32).sum(),
                'mean_pressure': effective_pressure[effective_pressure > 0].mean(),
                'time_min': time_sec / 60,
            })

        # Calculate average metrics over time
        avg_peak = np.mean([r['peak_pressure'] for r in all_results[config_name]])
        avg_cells = np.mean([r['cells_over_32'] for r in all_results[config_name]])
        print(f"  {config_name}: Avg Peak={avg_peak:.1f} mmHg, Avg >32mmHg: {avg_cells:.0f} cells")

    # Calculate metrics for comparison
    avg_metrics = {}
    for name in configs:
        peaks = [r['peak_pressure'] for r in all_results[name]]
        cells = [r['cells_over_32'] for r in all_results[name]]
        avg_metrics[name] = {
            'avg_peak': np.mean(peaks),
            'min_peak': np.min(peaks),
            'max_peak': np.max(peaks),
            'avg_cells_over_32': np.mean(cells),
        }

    # Create figure
    n_configs = len(configs)
    n_cols = 4
    n_rows = (n_configs + n_cols - 1) // n_cols
    config_names = list(configs.keys())

    # Subplot titles
    baseline_avg = avg_metrics['Standard Foam']['avg_peak']
    first_frame_titles = []
    for name in config_names:
        am = avg_metrics[name]
        change = (am['avg_peak'] / baseline_avg - 1) * 100 if baseline_avg > 0 else 0
        if name == 'Standard Foam':
            first_frame_titles.append(f"<b>{name} (Baseline)</b><br>Avg Peak: {am['avg_peak']:.0f} mmHg")
        else:
            first_frame_titles.append(f"<b>{name}</b><br>Peak: {am['avg_peak']:.0f} | vs Foam: {change:+.0f}%")

    while len(first_frame_titles) < n_rows * n_cols:
        first_frame_titles.append("")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=first_frame_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    # Colorscale for pressure (green = safe, red = dangerous)
    pressure_colorscale = [
        [0, 'rgb(0, 100, 0)'],        # 0 mmHg
        [0.32, 'rgb(144, 238, 144)'], # Below 32 mmHg
        [0.40, 'rgb(255, 255, 0)'],   # At threshold (32 mmHg)
        [0.60, 'rgb(255, 165, 0)'],   # Above threshold
        [0.80, 'rgb(255, 0, 0)'],     # High pressure
        [1.0, 'rgb(100, 0, 0)']       # Very high
    ]

    zmax = 100  # Fixed scale for pressure

    # Initial frame
    for idx, name in enumerate(config_names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        r = all_results[name][0]

        fig.add_trace(
            go.Heatmap(
                z=np.flipud(r['pressure']),
                colorscale=pressure_colorscale,
                zmin=0,
                zmax=zmax,
                showscale=(idx == 0),
                colorbar=dict(
                    title='Pressure<br>(mmHg)',
                    tickvals=[0, 32, 50, 75, 100],
                    ticktext=['0', '32<br>(threshold)', '50', '75', '100'],
                    len=0.3,
                    y=0.85,
                    x=1.02
                ) if idx == 0 else None,
                hovertemplate=f'{name}<br>Pressure: %{{z:.1f}} mmHg<extra></extra>'
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
                    z=np.flipud(r['pressure']),
                    colorscale=pressure_colorscale,
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
                    text=f'<b>Real-Time Interface Pressure (mmHg)</b><br>'
                         f'<sup>Time: {time_min:.1f} min | Yellow/Red = >32 mmHg (capillary occlusion)</sup>'
                )
            )
        ))

    fig.frames = frames

    # Layout
    fig.update_layout(
        title=dict(
            text='<b>Real-Time Interface Pressure (mmHg)</b><br>'
                 '<sup>Time: 0 min | Yellow/Red = >32 mmHg (capillary occlusion) | Watch APM cells cycle!</sup>',
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
                    dict(label='▶ Play', method='animate',
                         args=[None, {'frame': {'duration': 200, 'redraw': True},
                                     'fromcurrent': True, 'transition': {'duration': 100}}]),
                    dict(label='⏸ Pause', method='animate',
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

    fig.write_html('all_mattress_pressure.html', include_plotlyjs=True, full_html=True)
    print(f"\nSaved: all_mattress_pressure.html")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - REAL-TIME INTERFACE PRESSURE (120 MINUTES)")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Avg Peak':>10} {'Min Peak':>10} {'Max Peak':>10} {'vs Foam':>10}")
    print(f"{'':30} {'(mmHg)':>10} {'(mmHg)':>10} {'(mmHg)':>10} {'':>10}")
    print("-" * 80)

    for name in config_names:
        am = avg_metrics[name]
        change = (am['avg_peak'] / baseline_avg - 1) * 100 if baseline_avg > 0 else 0
        print(f"{name:<30} {am['avg_peak']:>10.1f} {am['min_peak']:>10.1f} {am['max_peak']:>10.1f} {change:>+9.0f}%")

    print("-" * 80)
    print("\nThis shows INSTANTANEOUS pressure - what the mattress is doing NOW.")
    print("APM mattresses cycle cells, so pressure varies over time.")
    print("Compare with all_mattress_configs.html for CUMULATIVE damage.")

    return all_results


if __name__ == "__main__":
    create_pressure_comparison()
