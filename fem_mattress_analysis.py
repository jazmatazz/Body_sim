#!/usr/bin/env python3
"""
FEM-Based Mattress Analysis
============================

Integrates the 3D FEM tissue model with mattress pressure simulation
to compute realistic stress/strain distributions in tissue.

Features:
- 3D FEM analysis at key body regions (sacrum, heel, etc.)
- Layered tissue geometry (skin, fat, muscle)
- Nonlinear hyperelastic material properties
- Moisture-modified tissue parameters
- Visualization of stress distributions
"""

import numpy as np
from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fem_tissue_model import (
    LayeredTissueMesh, FEMTissueSolver, FEMPressureAnalyzer,
    REGIONAL_THICKNESS, TISSUE_LAYERS
)
from multidynamic_mattress_optimization import (
    SMPLBodyPressureModel, MultiDynamicAirMattress, MOVEMENT_PATTERNS
)
from evolved_pattern import EvolvedOptimalPattern
from all_mattress_pti import RealisticMattressState
from scipy.ndimage import zoom


def analyze_mattress_with_fem(config_name: str, pressure_map: np.ndarray,
                               moisture_level: float = 0.3) -> Dict:
    """
    Perform FEM analysis for a mattress configuration.

    Args:
        config_name: Name of the mattress configuration
        pressure_map: 2D pressure map (mmHg)
        moisture_level: Skin moisture (0-1)

    Returns:
        Dictionary with FEM results
    """
    analyzer = FEMPressureAnalyzer(moisture_level=moisture_level)

    # Analyze key body regions
    fem_results = analyzer.analyze_pressure_map(
        pressure_map,
        regions=['sacrum', 'heel']
    )

    # Compute damage metrics
    damage_metrics = analyzer.compute_damage_metrics(fem_results)

    return {
        'config_name': config_name,
        'fem_results': fem_results,
        'damage_metrics': damage_metrics,
    }


def run_fem_comparison():
    """Run FEM analysis for multiple mattress configurations."""

    print("=" * 70)
    print("FEM-BASED TISSUE STRESS ANALYSIS")
    print("=" * 70)
    print("\n3D Finite Element Model with:")
    print("  - Layered tissue geometry (skin, fat, muscle)")
    print("  - Ogden hyperelastic material model")
    print("  - Bone as rigid boundary")
    print("  - Moisture-modified parameters")
    print("=" * 70)

    # Generate body pressure map
    model = SMPLBodyPressureModel(75, 30)
    body_pressure, body_shear = model.calculate_pressure_map(40, 18)

    print(f"\nBase body pressure: Peak={body_pressure.max():.1f} mmHg")

    # Configurations to test
    configs = {
        'Standard Foam': {'type': 'foam'},
        'Evolved Optimal': {'type': 'apm', 'pattern': EvolvedOptimalPattern()},
    }

    # Add a couple of key patterns
    for name in ['zone_based', 'row_groups_2']:
        if name in MOVEMENT_PATTERNS:
            pattern = MOVEMENT_PATTERNS[name]
            display_name = pattern.name if hasattr(pattern, 'name') else name.replace('_', ' ').title()
            configs[display_name] = {'type': 'apm', 'pattern': pattern}

    print(f"\nAnalyzing {len(configs)} configurations with FEM:")
    for name in configs:
        print(f"  - {name}")

    # Analyze each configuration
    all_results = {}
    moisture_level = 0.3  # Moderate moisture

    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Configuration: {config_name}")
        print(f"{'='*60}")

        if config['type'] == 'foam':
            # Static pressure (85% of body pressure for foam)
            effective_pressure = body_pressure * 0.85
        else:
            # Get pressure at a specific time point (e.g., 60 minutes)
            pattern = config['pattern']
            mattress = MultiDynamicAirMattress(cell_size_cm=5, movement_pattern=pattern)
            realistic_state = RealisticMattressState(mattress.rows, mattress.cols)

            # Resample body pressure to mattress grid
            scale_y = mattress.rows / body_pressure.shape[0]
            scale_x = mattress.cols / body_pressure.shape[1]
            body_resampled = zoom(body_pressure, (scale_y, scale_x), order=1)

            # Simulate at 60 minutes
            time_sec = 60 * 60
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

            # Resample back
            scale_y_back = body_pressure.shape[0] / effective_pressure.shape[0]
            scale_x_back = body_pressure.shape[1] / effective_pressure.shape[1]
            effective_pressure = zoom(effective_pressure, (scale_y_back, scale_x_back), order=1)

        # Run FEM analysis
        result = analyze_mattress_with_fem(
            config_name, effective_pressure, moisture_level
        )
        all_results[config_name] = result

    # Print summary
    print("\n" + "=" * 80)
    print("FEM ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<25} {'Region':<10} {'Stress':>12} {'Strain':>10} {'Damage':>10}")
    print(f"{'':25} {'':10} {'(kPa)':>12} {'(%)':>10} {'Index':>10}")
    print("-" * 80)

    for config_name, result in all_results.items():
        for region, metrics in result['damage_metrics'].items():
            print(f"{config_name:<25} {region:<10} "
                  f"{metrics['max_stress_kPa']:>12.1f} "
                  f"{metrics['max_strain']*100:>10.1f} "
                  f"{metrics['combined_damage']:>10.2f}")

    print("-" * 80)
    print("\nDamage Index Interpretation:")
    print("  < 1.0: Low risk (within tissue tolerance)")
    print("  1.0-2.0: Moderate risk (approaching threshold)")
    print("  > 2.0: High risk (exceeds damage threshold)")

    # Create visualization
    create_fem_visualization(all_results)

    return all_results


def create_fem_visualization(all_results: Dict):
    """Create visualization of FEM results."""

    n_configs = len(all_results)
    config_names = list(all_results.keys())

    # Create subplots
    fig = make_subplots(
        rows=2, cols=n_configs,
        subplot_titles=[f"{name}" for name in config_names] * 2,
        row_titles=['Max Stress (kPa)', 'Damage Index'],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )

    regions = ['sacrum', 'heel']
    colors = {'sacrum': 'royalblue', 'heel': 'coral'}

    for col, config_name in enumerate(config_names, 1):
        result = all_results[config_name]
        metrics = result['damage_metrics']

        x = list(metrics.keys())
        stress = [metrics[r]['max_stress_kPa'] for r in x]
        damage = [metrics[r]['combined_damage'] for r in x]

        # Stress bar chart
        fig.add_trace(
            go.Bar(
                x=x, y=stress,
                marker_color=[colors.get(r, 'gray') for r in x],
                name='Stress',
                showlegend=(col == 1)
            ),
            row=1, col=col
        )

        # Damage index bar chart
        fig.add_trace(
            go.Bar(
                x=x, y=damage,
                marker_color=[colors.get(r, 'gray') for r in x],
                name='Damage',
                showlegend=False
            ),
            row=2, col=col
        )

        # Add threshold line
        fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                      annotation_text="Threshold", row=2, col=col)

    fig.update_layout(
        title=dict(
            text='<b>FEM Tissue Stress Analysis</b><br>'
                 '<sup>3D Finite Element Model with Layered Tissue Geometry</sup>',
            x=0.5,
            font=dict(size=18)
        ),
        height=700,
        width=400 * n_configs,
        showlegend=True,
    )

    fig.write_html('fem_analysis.html', include_plotlyjs=True, full_html=True)
    print(f"\nSaved: fem_analysis.html")


if __name__ == "__main__":
    results = run_fem_comparison()

