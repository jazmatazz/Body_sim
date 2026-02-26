#!/usr/bin/env python3
"""
All Mattress Configurations - FEM Stress Analysis
==================================================
Shows tissue stress computed via 3D Finite Element Model.

Uses the same display format as all_mattress_pti.py but shows
max principal stress (kPa) instead of PTI.

Note: FEM is computed at sacrum region for each configuration.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import zoom
from multidynamic_mattress_optimization import (
    SMPLBodyPressureModel, MultiDynamicAirMattress,
    MOVEMENT_PATTERNS
)
from evolved_pattern import EvolvedOptimalPattern
from all_mattress_pti import RealisticMattressState
from fem_tissue_model import LayeredTissueMesh, FEMTissueSolver


# Cache FEM solvers for efficiency
FEM_CACHE = {}

# =============================================================================
# REALISTIC PRESSURE REDISTRIBUTION FACTORS
# =============================================================================
# Based on clinical pressure mapping studies (Defloor 2000, Reenalda 2009)
FOAM_REDISTRIBUTION = 0.15      # Foam reduces peak pressure by ~85%
APM_INFLATED_REDISTRIBUTION = 0.15  # Inflated APM cells similar to foam
APM_DEFLATED_RELIEF = 0.85      # Deflated cells provide additional 85% relief


def get_fem_solver(region: str = 'sacrum', moisture: float = 0.3):
    """Get or create cached FEM solver."""
    key = (region, moisture)
    if key not in FEM_CACHE:
        mesh = LayeredTissueMesh(region=region, resolution=6, moisture_level=moisture)
        FEM_CACHE[key] = FEMTissueSolver(mesh)
    return FEM_CACHE[key]


def compute_stress_from_pressure(pressure_mmHg: float, region: str = 'sacrum') -> dict:
    """
    Compute tissue stress using FEM for given pressure.

    Args:
        pressure_mmHg: Interface pressure in mmHg
        region: Body region

    Returns:
        Dict with stress metrics
    """
    if pressure_mmHg < 1.0:
        return {'max_stress': 0.0, 'max_shear': 0.0, 'compression': 0.0}

    solver = get_fem_solver(region)

    # Reset displacement for new solve
    solver.displacement = np.zeros(solver.n_dofs)

    # Solve FEM
    results = solver.solve(pressure_mmHg, max_iter=10)

    return {
        'max_stress': np.max(results['max_principal_stress']),
        'max_shear': np.max(results['max_shear_stress']),
        'compression': results['max_compression_mm'],
    }


def create_stress_comparison():
    """Create animated comparison showing FEM-computed stress."""

    print("Generating FEM Stress Comparison...")
    print("=" * 60)
    print("Using 3D Finite Element Model with:")
    print("  - Layered tissue (skin, fat, muscle)")
    print("  - Ogden hyperelastic material")
    print("  - Bone as rigid boundary")
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

    # Simulate 120 minutes with fewer frames for FEM (expensive)
    # Start at 5 minutes to skip initial transient (cells need time to deflate)
    start_time = 5 * 60  # 5 minutes
    total_time = 120 * 60
    n_frames = 12  # Reduced for FEM computation time
    time_points = np.linspace(start_time, total_time, n_frames, endpoint=False)

    # Sacrum location in pressure map
    h, w = body_pressure.shape
    sacrum_row, sacrum_col = int(0.4 * h), int(0.5 * w)

    all_results = {}

    for config_name, config in configs.items():
        print(f"\n  Processing: {config_name}")

        if config['type'] != 'foam':
            pattern = config['pattern']
            mattress = MultiDynamicAirMattress(cell_size_cm=5, movement_pattern=pattern)
            realistic_state = RealisticMattressState(mattress.rows, mattress.cols)

            scale_y = mattress.rows / body_pressure.shape[0]
            scale_x = mattress.cols / body_pressure.shape[1]
            body_resampled = zoom(body_pressure, (scale_y, scale_x), order=1)

        all_results[config_name] = []
        max_stress_so_far = 0.0

        for frame_idx, time_sec in enumerate(time_points):
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

            # Get sacrum pressure (sample 3x3 area)
            r_start = max(0, sacrum_row - 1)
            r_end = min(h, sacrum_row + 2)
            c_start = max(0, sacrum_col - 1)
            c_end = min(w, sacrum_col + 2)
            sacrum_pressure = effective_pressure[r_start:r_end, c_start:c_end].max()

            # Compute FEM stress
            stress_data = compute_stress_from_pressure(sacrum_pressure, 'sacrum')

            # Track max stress over time
            max_stress_so_far = max(max_stress_so_far, stress_data['max_stress'])

            # Create stress map (simplified: scale pressure map by stress/pressure ratio)
            if sacrum_pressure > 0:
                stress_ratio = stress_data['max_stress'] / (sacrum_pressure * 0.1333)  # Convert mmHg to kPa
            else:
                stress_ratio = 0

            # Create approximate stress map
            stress_map = effective_pressure * 0.1333 * max(0.5, min(2.0, stress_ratio))

            all_results[config_name].append({
                'stress_map': stress_map,
                'max_stress': stress_data['max_stress'],
                'max_shear': stress_data['max_shear'],
                'compression': stress_data['compression'],
                'cumulative_max_stress': max_stress_so_far,
                'time_min': time_sec / 60,
            })

        # Report final metrics
        final = all_results[config_name][-1]
        print(f"    Max Stress: {final['cumulative_max_stress']:.1f} kPa")

    # Calculate final metrics for comparison
    final_metrics = {}
    for name in configs:
        final = all_results[name][-1]
        final_metrics[name] = {
            'max_stress': final['cumulative_max_stress'],
            'avg_stress': np.mean([r['max_stress'] for r in all_results[name]]),
        }

    # Create figure
    n_configs = len(configs)
    n_cols = 4
    n_rows = (n_configs + n_cols - 1) // n_cols
    config_names = list(configs.keys())

    # Subplot titles - use AVERAGE stress for comparison (more meaningful for cycling patterns)
    baseline_stress = final_metrics['Standard Foam']['avg_stress']
    first_frame_titles = []
    for name in config_names:
        fm = final_metrics[name]
        change = (fm['avg_stress'] / baseline_stress - 1) * 100 if baseline_stress > 0 else 0
        if name == 'Standard Foam':
            first_frame_titles.append(f"<b>{name} (Baseline)</b><br>Avg Stress: {fm['avg_stress']:.0f} kPa")
        else:
            first_frame_titles.append(f"<b>{name}</b><br>Avg: {fm['avg_stress']:.0f} kPa | vs Foam: {change:+.0f}%")

    while len(first_frame_titles) < n_rows * n_cols:
        first_frame_titles.append("")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=first_frame_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    # Fixed color scale max
    zmax = 100.0  # kPa

    # ==========================================================================
    # STRESS THRESHOLD ZONES (based on tissue mechanics literature)
    # ==========================================================================
    # < 10 kPa:  Safe - no damage
    # 10-20 kPa: Moderate risk - damage in 2-4 hours
    # 20-30 kPa: High risk - damage in 1-2 hours
    # 30-50 kPa: Very high risk - damage in 30-60 min
    # > 50 kPa:  Critical - damage in < 30 min
    STRESS_THRESHOLDS = {
        'safe': 10.0,
        'moderate': 20.0,
        'high': 30.0,
        'very_high': 50.0,
        'critical': 100.0,
    }

    # Colorbar ticks at threshold boundaries
    colorbar_tickvals = [0, 10, 20, 30, 50, 100]
    colorbar_ticktext = [
        '0 (Safe)',
        '10',
        '20',
        '30',
        '50',
        '100'
    ]

    # Colorscale aligned with damage thresholds
    # 0-10 kPa (0-0.1): Green (safe)
    # 10-20 kPa (0.1-0.2): Yellow (moderate)
    # 20-30 kPa (0.2-0.3): Orange (high)
    # 30-50 kPa (0.3-0.5): Red (very high)
    # 50-100 kPa (0.5-1.0): Dark red (critical)
    stress_colorscale = [
        [0.0, 'rgb(0, 128, 0)'],       # Green - safe
        [0.10, 'rgb(144, 238, 144)'],  # Light green - approaching threshold
        [0.10, 'rgb(255, 255, 0)'],    # Yellow - moderate risk starts
        [0.20, 'rgb(255, 200, 0)'],    # Gold
        [0.20, 'rgb(255, 165, 0)'],    # Orange - high risk starts
        [0.30, 'rgb(255, 100, 0)'],    # Dark orange
        [0.30, 'rgb(255, 0, 0)'],      # Red - very high risk starts
        [0.50, 'rgb(200, 0, 0)'],      # Dark red
        [0.50, 'rgb(139, 0, 0)'],      # Darker red - critical starts
        [1.0, 'rgb(80, 0, 0)']         # Very dark red - extreme
    ]

    # Initial frame
    for idx, name in enumerate(config_names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        r = all_results[name][0]

        fig.add_trace(
            go.Heatmap(
                z=np.flipud(r['stress_map']),
                colorscale=stress_colorscale,
                zmin=0,
                zmax=zmax,
                showscale=(idx == 0),
                colorbar=dict(
                    title='Stress (kPa)<br>─────────<br>Risk Level',
                    tickvals=[0, 10, 20, 30, 50, 100],
                    ticktext=[
                        '0 Safe',
                        '10 ───',
                        '20 Moderate',
                        '30 High',
                        '50 V.High',
                        '100 Critical'
                    ],
                    len=0.4,
                    y=0.80,
                    x=1.02,
                    tickfont=dict(size=10),
                ) if idx == 0 else None,
                hovertemplate=f'{name}<br>Stress: %{{z:.1f}} kPa<extra></extra>'
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
                    z=np.flipud(r['stress_map']),
                    colorscale=stress_colorscale,
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
                    text=f'<b>FEM Tissue Stress Analysis</b><br>'
                         f'<sup>Time: {time_min:.0f} min | Thresholds: <10 kPa Safe, 10-20 Moderate, 20-30 High, 30-50 V.High, >50 Critical</sup>'
                )
            )
        ))

    fig.frames = frames

    # Layout
    fig.update_layout(
        title=dict(
            text='<b>FEM Tissue Stress Analysis</b><br>'
                 '<sup>Time: 0 min | Thresholds: <10 kPa Safe, 10-20 Moderate, 20-30 High, 30-50 V.High, >50 Critical</sup>',
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
                         args=[None, {'frame': {'duration': 500, 'redraw': True},
                                     'fromcurrent': True, 'transition': {'duration': 200}}]),
                    dict(label='Pause', method='animate',
                         args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                       'mode': 'immediate', 'transition': {'duration': 0}}])
                ]
            )
        ],
        sliders=[{
            'active': 0,
            'currentvalue': {'prefix': 'Time: ', 'suffix': ' min', 'visible': True},
            'transition': {'duration': 200},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.05,
            'y': 0,
            'steps': [
                {'args': [[str(i)], {'frame': {'duration': 200, 'redraw': True},
                                    'mode': 'immediate', 'transition': {'duration': 200}}],
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

    fig.write_html('all_mattress_stress.html', include_plotlyjs=True, full_html=True)
    print(f"\nSaved: all_mattress_stress.html")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - FEM TISSUE STRESS (120 MINUTES)")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Max Stress':>12} {'Avg Stress':>12} {'vs Foam':>12}")
    print(f"{'':30} {'(kPa)':>12} {'(kPa)':>12} {'':>12}")
    print("-" * 80)

    for name in config_names:
        fm = final_metrics[name]
        change = (fm['avg_stress'] / baseline_stress - 1) * 100 if baseline_stress > 0 else 0
        print(f"{name:<30} {fm['max_stress']:>12.1f} {fm['avg_stress']:>12.1f} {change:>+11.0f}%")

    print("-" * 80)
    print("\nStress computed using 3D FEM with:")
    print("  - Layered tissue: skin (2mm), fat (15mm), muscle (20mm)")
    print("  - Ogden hyperelastic material model")
    print("  - Bone modeled as rigid boundary")
    print("\nDAMAGE RISK THRESHOLDS:")
    print("  < 10 kPa:  SAFE - no tissue damage expected")
    print("  10-20 kPa: MODERATE - damage in 2-4 hours")
    print("  20-30 kPa: HIGH - damage in 1-2 hours")
    print("  30-50 kPa: VERY HIGH - damage in 30-60 minutes")
    print("  > 50 kPa:  CRITICAL - damage in < 30 minutes")

    return all_results


if __name__ == "__main__":
    create_stress_comparison()
