#!/usr/bin/env python3
"""Live 3D visualization of pressure simulation with real-time updates."""

import argparse
import time
from pathlib import Path

import numpy as np

# Set up paths
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from body_sim.core.types import SimulationConfig, SMPLParameters, Gender
from body_sim.simulation.simulator import PressureSimulator
from body_sim.mattress.patterns import AlternatingPattern, WavePattern
from body_sim.body.pose_library import get_supine_pose


def run_live_visualization(duration: float = 30.0, pattern_type: str = "alternating"):
    """Run live 3D visualization of the simulation.

    Args:
        duration: Simulation duration in seconds
        pattern_type: Pattern type ('alternating' or 'wave')
    """
    try:
        import pyvista as pv
        pv.global_theme.allow_empty_mesh = True
    except ImportError:
        print("PyVista is required for 3D visualization.")
        print("Install with: pip install pyvista")
        return

    print("=" * 60)
    print("Live Pressure Simulation Visualization")
    print("=" * 60)

    # Configuration - using larger cell height range for visible movement
    config = SimulationConfig(
        grid_rows=16,
        grid_cols=32,
        mattress_width=0.762,
        mattress_height=1.854,
        body_weight=70.0,
        time_step=0.5,
        cell_min_height=0.02,   # 2cm min
        cell_max_height=0.25,   # 25cm max - makes movement very visible
    )

    print(f"\nConfiguration:")
    print(f"  Grid size: {config.grid_rows} x {config.grid_cols}")
    print(f"  Pattern: {pattern_type}")
    print(f"  Duration: {duration}s")

    # Initialize simulator
    print("\nInitializing simulator...")
    model_path = Path(__file__).parent.parent / "models" / "smpl"
    sim = PressureSimulator(config, smpl_model_path=model_path)

    # Load body
    print("Loading body mesh...")
    body_params = SMPLParameters(
        betas=np.zeros(10, dtype=np.float32),
        pose=get_supine_pose(),
        trans=np.zeros(3, dtype=np.float32),
        gender=Gender.NEUTRAL,
    )
    sim.load_body(body_params)
    print(f"  Loaded mesh with {sim.current_mesh.vertices.shape[0]} vertices")

    # Set pattern - using full height range for maximum visible effect
    if pattern_type == "wave":
        pattern = WavePattern(cycle_duration=8.0, wavelength=4.0, amplitude=0.5, baseline=0.5)
    else:
        # low_height=0.0 means cells go to minimum, high_height=1.0 means maximum
        pattern = AlternatingPattern(cycle_duration=4.0, high_height=1.0, low_height=0.0)
    sim.set_pattern(pattern)

    # Create PyVista plotter
    print("\nStarting live visualization...")
    print("  - Red/yellow = high pressure on body")
    print("  - Mattress cells change height with pattern")
    print("  - Close the window to stop")
    print()

    plotter = pv.Plotter(title="Pressure Simulation - Live View")
    plotter.set_background("white")

    # Initial pressure calculation (this also updates the body mesh position)
    pressure_dist = sim.step()
    grid = sim.grid

    # Get body mesh from simulator (now dynamically positioned on mattress surface)
    mesh = sim.current_mesh
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
    body_pv = pv.PolyData(mesh.vertices, faces)

    # Create vertex pressure array
    vertex_pressure = np.zeros(mesh.vertices.shape[0], dtype=np.float32)
    for i, v in enumerate(mesh.vertices):
        col = int(np.clip(v[0] / grid.cell_width, 0, grid.cols - 1))
        row = int(np.clip(v[1] / grid.cell_length, 0, grid.rows - 1))
        vertex_pressure[i] = pressure_dist.grid_pressure[row, col]

    body_pv["pressure"] = vertex_pressure

    # Add body mesh
    plotter.add_mesh(
        body_pv,
        scalars="pressure",
        cmap="hot",
        clim=[0, 1500],
        show_scalar_bar=True,
        scalar_bar_args={"title": "Pressure (Pa)", "vertical": True},
        opacity=0.9,
        name="body",
    )

    # Add mattress cells as individual boxes
    def add_mattress_to_plotter(plotter, name_prefix="mattress"):
        """Add mattress cells to plotter."""
        actors = []
        for row in range(grid.rows):
            for col in range(grid.cols):
                height = max(grid.get_cell_height(row, col), 0.01)
                x_min = col * grid.cell_width
                x_max = (col + 1) * grid.cell_width
                y_min = row * grid.cell_length
                y_max = (row + 1) * grid.cell_length

                box = pv.Box(bounds=(x_min, x_max, y_min, y_max, -0.01, height))

                # Color by height
                height_ratio = (height - grid.min_cell_height) / (grid.max_cell_height - grid.min_cell_height + 0.001)
                color = [0.2, 0.3 + 0.5 * height_ratio, 0.8 - 0.5 * height_ratio]

                actor = plotter.add_mesh(
                    box,
                    color=color,
                    opacity=0.5,
                    show_edges=True,
                    name=f"{name_prefix}_{row}_{col}",
                )
                actors.append((row, col, f"{name_prefix}_{row}_{col}"))
        return actors

    mattress_actors = add_mattress_to_plotter(plotter)

    # Add floor
    floor = pv.Plane(
        center=(grid.width / 2, grid.height / 2, -0.03),
        direction=(0, 0, 1),
        i_size=grid.width * 1.2,
        j_size=grid.height * 1.2,
    )
    plotter.add_mesh(floor, color="lightgray", opacity=0.5, name="floor")

    # Set camera
    plotter.camera_position = [
        (grid.width * 2, -grid.height * 0.3, grid.height * 1.0),
        (grid.width / 2, grid.height / 2, 0.1),
        (0, 0, 1),
    ]

    # Add initial text
    plotter.add_text(
        f"Time: 0.0s | Peak: {pressure_dist.peak_pressure:.0f} Pa | Pattern: {pattern_type}",
        position="upper_left",
        font_size=12,
        color="black",
        name="info_text",
    )

    # Show window and animate
    plotter.show(interactive_update=True, auto_close=False)

    simulation_time = 0.0
    step_count = 0

    print("\nRunning simulation...")
    try:
        while simulation_time < duration:
            # Step simulation (this updates body position based on mattress surface)
            pressure_dist = sim.step()
            simulation_time += config.time_step
            step_count += 1

            # Get updated mesh from simulator
            mesh = sim.current_mesh

            # Update body mesh vertices in visualization (in-place update triggers render)
            body_pv.points[:] = mesh.vertices

            # Update vertex pressures
            for i, v in enumerate(mesh.vertices):
                col = int(np.clip(v[0] / grid.cell_width, 0, grid.cols - 1))
                row = int(np.clip(v[1] / grid.cell_length, 0, grid.rows - 1))
                vertex_pressure[i] = pressure_dist.grid_pressure[row, col]

            body_pv["pressure"] = vertex_pressure

            # Update mattress cells
            for row, col, name in mattress_actors:
                plotter.remove_actor(name)
            mattress_actors.clear()
            mattress_actors.extend(add_mattress_to_plotter(plotter))

            # Update text
            plotter.remove_actor("info_text")
            plotter.add_text(
                f"Time: {simulation_time:.1f}s | Peak: {pressure_dist.peak_pressure:.0f} Pa | Avg: {pressure_dist.average_pressure:.0f} Pa",
                position="upper_left",
                font_size=12,
                color="black",
                name="info_text",
            )

            # Update display - render is needed to show mesh vertex changes
            plotter.render()
            plotter.update()

            # Print progress with body Z position
            if step_count % 4 == 0:
                z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
                print(f"  Time: {simulation_time:.1f}s | Body Z: {z_min:.3f}-{z_max:.3f}m | Peak: {pressure_dist.peak_pressure:.0f} Pa")

            # Small delay for visualization
            time.sleep(0.1)

    except Exception as e:
        print(f"\nVisualization interrupted: {e}")

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print(f"  Total time: {simulation_time:.1f}s")
    print(f"  Total steps: {step_count}")
    print("=" * 60)

    # Keep window open for inspection
    print("\nClose the visualization window to exit.")
    plotter.show()


def main():
    parser = argparse.ArgumentParser(description="Live 3D pressure simulation visualization")
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Simulation duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--pattern", type=str, default="alternating",
        choices=["alternating", "wave"],
        help="Pressure pattern type (default: alternating)"
    )
    args = parser.parse_args()

    run_live_visualization(duration=args.duration, pattern_type=args.pattern)


if __name__ == "__main__":
    main()
