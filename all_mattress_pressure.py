#!/usr/bin/env python3
"""
All Mattress Configurations - Animated Pressure Map Comparison
==============================================================
Shows ALL bed configurations tested over 120 minutes with:
- Real-time pressure distribution
- Cells above 32 mmHg threshold
- Comparison vs Standard Foam baseline
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


def create_pressure_comparison():
    """Create animated comparison of ALL mattress configurations showing pressure maps."""

    print("Generating Animated Pressure Map Comparison...")
    print("=" * 60)

    # Generate body pressure map
    model = SMPLBodyPressureModel(75, 30)
    body_pressure, _ = model.calculate_pressure_map(40, 18)

    print(f"\nBase body pressure: Peak={body_pressure.max():.1f} mmHg")

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
    n_frames = 48          # 48 frames = every 2.5 minutes
    time_points = np.linspace(0, total_time, n_frames, endpoint=False)

    # Calculate pressure for each config at each time point
    all_results = {}

    for config_name, config in configs.items():
        all_results[config_name] = []

        for time_sec in time_points:
            if config['type'] == 'foam':
                effective = body_pressure * 0.85
            else:
                pattern = config['pattern']
                mattress = MultiDynamicAirMattress(cell_size_cm=5, movement_pattern=pattern)

                # Resample to match mattress grid
                scale_y = mattress.rows / body_pressure.shape[0]
                scale_x = mattress.cols / body_pressure.shape[1]
                body_resampled = zoom(body_pressure, (scale_y, scale_x), order=1)

                mattress.update(time_sec)
                effective = mattress.get_effective_interface_pressure(body_resampled)

                # Resample back to display size
                scale_y = body_pressure.shape[0] / effective.shape[0]
                scale_x = body_pressure.shape[1] / effective.shape[1]
                effective = zoom(effective, (scale_y, scale_x), order=1)

            all_results[config_name].append({
                'pressure': effective,
                'peak': effective.max(),
                'cells_over_32': (effective > 32).sum(),
                'time_min': time_sec / 60,
            })

        print(f"  {config_name}: Peak={all_results[config_name][0]['peak']:.1f} mmHg, >32: {all_results[config_name][0]['cells_over_32']} cells")

    # Calculate cycle averages for each config
    avg_metrics = {}
    for name in configs:
        peaks = [r['peak'] for r in all_results[name]]
        cells = [r['cells_over_32'] for r in all_results[name]]
        avg_metrics[name] = {
            'avg_peak': np.mean(peaks),
            'avg_cells_over_32': np.mean(cells),
        }

    # Create figure layout
    n_configs = len(configs)
    n_cols = 4
    n_rows = (n_configs + n_cols - 1) // n_cols

    config_names = list(configs.keys())

    # Create subplot titles with averages (compared to Standard Foam baseline)
    baseline_cells = avg_metrics['Standard Foam']['avg_cells_over_32']
    first_frame_titles = []
    for name in config_names:
        am = avg_metrics[name]
        if baseline_cells > 0:
            reduction = (1 - am['avg_cells_over_32'] / baseline_cells) * 100
        else:
            reduction = 0
        if name == 'Standard Foam':
            first_frame_titles.append(f"<b>{name} (Baseline)</b><br>Avg Peak: {am['avg_peak']:.0f} | Avg >32: {am['avg_cells_over_32']:.0f}")
        else:
            first_frame_titles.append(f"<b>{name}</b><br>Avg Peak: {am['avg_peak']:.0f} | >32: {reduction:+.0f}%")

    # Pad with empty titles
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
        [0, 'rgb(0, 100, 0)'],        # 0 mmHg - no pressure
        [0.32, 'rgb(144, 238, 144)'], # Below threshold
        [0.40, 'rgb(255, 255, 0)'],   # At threshold (32 mmHg)
        [0.60, 'rgb(255, 165, 0)'],   # Above threshold
        [0.80, 'rgb(255, 0, 0)'],     # High pressure
        [1.0, 'rgb(100, 0, 0)']       # Very high pressure
    ]

    # Add initial frame
    for idx, name in enumerate(config_names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        r = all_results[name][0]

        fig.add_trace(
            go.Heatmap(
                z=r['pressure'],
                colorscale=pressure_colorscale,
                zmin=0,
                zmax=100,
                showscale=(idx == 0),
                colorbar=dict(
                    title='mmHg',
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

    # Create animation frames
    frames = []

    for frame_idx, time_sec in enumerate(time_points):
        time_min = time_sec / 60
        frame_data = []

        for idx, name in enumerate(config_names):
            r = all_results[name][frame_idx]
            frame_data.append(
                go.Heatmap(
                    z=r['pressure'],
                    colorscale=pressure_colorscale,
                    zmin=0,
                    zmax=100,
                    showscale=(idx == 0),
                )
            )

        frames.append(go.Frame(
            data=frame_data,
            name=str(frame_idx),
            layout=go.Layout(
                title=dict(
                    text=f'<b>Interface Pressure Maps - 120 Minute Simulation</b><br>'
                         f'<sup>Time: {time_min:.1f} min | '
                         f'Yellow/Red = pressure > 32 mmHg (capillary occlusion)</sup>'
                )
            )
        ))

    fig.frames = frames

    # Animation controls
    fig.update_layout(
        title=dict(
            text='<b>Interface Pressure Maps - 120 Minute Simulation</b><br>'
                 '<sup>Time: 0 min | Yellow/Red = pressure > 32 mmHg (capillary occlusion)</sup>',
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

    fig.write_html('all_mattress_pressure.html', include_plotlyjs=True, full_html=True)
    print(f"\nSaved: all_mattress_pressure.html")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - AVERAGE PRESSURE METRICS OVER 120 MINUTES")
    print("=" * 70)
    print(f"{'Configuration':<30} {'Avg Peak':>10} {'Avg >32':>10} {'Damage ↓':>10}")
    print("-" * 70)

    for name in config_names:
        am = avg_metrics[name]
        reduction = (1 - am['avg_cells_over_32'] / baseline_cells) * 100 if baseline_cells > 0 else 0
        print(f"{name:<30} {am['avg_peak']:>10.1f} {am['avg_cells_over_32']:>10.1f} {reduction:>9.0f}%")

    print("-" * 70)
    print(f"Threshold: 32 mmHg (capillary closing pressure)")

    return all_results


if __name__ == "__main__":
    create_pressure_comparison()
