#!/usr/bin/env python3
"""Basic pressure simulation example.

This example demonstrates how to:
1. Set up a pressure simulator with configurable grid
2. Load a body with predefined pose
3. Apply an alternating pressure pattern
4. Run simulation and analyze results
5. Visualize pressure distribution

Usage:
    python examples/basic_simulation.py

Requirements:
    - SMPL models must be downloaded to ./models/smpl
    - Or use --mock flag to run without real SMPL models
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from body_sim.core.types import SimulationConfig, SMPLParameters, Gender
from body_sim.simulation import PressureSimulator
from body_sim.mattress.patterns import AlternatingPattern, WavePattern
from body_sim.body.pose_library import get_supine_pose
from body_sim.metrics import PressureAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Basic pressure simulation example")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock SMPL model (no real model files needed)",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=16,
        help="Number of grid rows (default: 16)",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=32,
        help="Number of grid columns (default: 32)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Simulation duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=70.0,
        help="Body weight in kg (default: 70)",
    )
    parser.add_argument(
        "--pattern",
        choices=["alternating", "wave"],
        default="alternating",
        help="Pressure pattern type (default: alternating)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization plots",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Body Pressure Simulator - Basic Example")
    print("=" * 60)

    # Create configuration
    config = SimulationConfig(
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        body_weight=args.weight,
        time_step=0.5,  # 0.5 second steps
    )

    print(f"\nConfiguration:")
    print(f"  Grid size: {config.grid_rows} x {config.grid_cols}")
    print(f"  Mattress: {config.mattress_width:.2f}m x {config.mattress_height:.2f}m")
    print(f"  Body weight: {config.body_weight} kg")
    print(f"  Time step: {config.time_step}s")

    # Initialize simulator
    print("\nInitializing simulator...")
    sim = PressureSimulator(
        config=config,
        use_mock_smpl=args.mock,
    )

    # Load body in supine position
    print("Loading body mesh...")
    body_params = SMPLParameters(
        betas=np.zeros(10, dtype=np.float32),  # Average body shape
        pose=get_supine_pose(),
        trans=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        gender=Gender.NEUTRAL,
    )

    mesh = sim.load_body(body_params, position_on_mattress=True)
    print(f"  Loaded mesh with {mesh.num_vertices} vertices")

    # Set up pressure pattern
    if args.pattern == "alternating":
        pattern = AlternatingPattern(
            cycle_duration=10.0,
            low_height=0.3,
            high_height=1.0,
        )
        print(f"\nPattern: Alternating (10s cycle)")
    else:
        pattern = WavePattern(
            cycle_duration=20.0,
            wavelength=4.0,
            direction="head_to_foot",
        )
        print(f"\nPattern: Wave (20s cycle)")

    sim.set_pattern(pattern)

    # Run simulation
    print(f"\nRunning simulation for {args.duration}s...")

    def progress_callback(progress: float):
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "=" * filled + "-" * (bar_width - filled)
        print(f"\r  [{bar}] {progress*100:.0f}%", end="", flush=True)

    results = sim.run(
        duration=args.duration,
        progress_callback=progress_callback,
    )
    print()  # New line after progress bar

    # Analyze results
    print("\n" + "=" * 60)
    print("Results Analysis")
    print("=" * 60)

    summary = sim.get_pressure_summary()
    print(f"\nSimulation Summary:")
    print(f"  Total steps: {summary['num_steps']}")
    print(f"  Duration: {summary['duration']:.1f}s")
    print(f"  Max peak pressure: {summary['max_peak_pressure']:.0f} Pa")
    print(f"  Avg peak pressure: {summary['avg_peak_pressure']:.0f} Pa")
    print(f"  Avg average pressure: {summary['avg_average_pressure']:.0f} Pa")
    print(f"  Avg contact area: {summary['avg_contact_area']:.1%}")

    # Detailed pressure analysis
    analyzer = PressureAnalyzer(history=results)
    detailed = analyzer.get_summary()

    print(f"\nPressure Analysis:")
    print(f"  Time above capillary threshold: {detailed['time_above_threshold']:.1f}s")
    print(f"  Cumulative exposure: {detailed['cumulative_exposure']:.0f} Pa·s")
    print(f"  Above threshold fraction: {detailed['above_threshold_fraction']:.1%}")

    # Region analysis on final state
    if results:
        regions = analyzer.region_analysis(results[-1])
        print(f"\nPressure by Body Region (final state):")
        for region, metrics in regions.items():
            if metrics["peak"] > 0:
                print(
                    f"  {region:15s}: Peak={metrics['peak']:6.0f} Pa, "
                    f"Avg={metrics['average']:6.0f} Pa"
                )

    # Visualization
    if args.visualize:
        print("\nGenerating visualizations...")
        try:
            import matplotlib.pyplot as plt
            from body_sim.visualization import (
                create_pressure_heatmap,
                create_time_series_plot,
                create_region_pressure_chart,
            )

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Pressure heatmap
            if results:
                create_pressure_heatmap(results[-1], ax=axes[0, 0])

            # Time series
            create_time_series_plot(results, ax=axes[0, 1])

            # Region chart
            if results:
                create_region_pressure_chart(results[-1], ax=axes[1, 0])

            # Contact area over time
            times = [r.timestamp for r in results]
            areas = [r.contact_area * 100 for r in results]
            axes[1, 1].plot(times, areas, "g-", linewidth=2)
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Contact Area (%)")
            axes[1, 1].set_title("Contact Area Over Time")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("simulation_results.png", dpi=150)
            print("  Saved: simulation_results.png")
            plt.show()

        except ImportError as e:
            print(f"  Visualization requires matplotlib: {e}")

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
