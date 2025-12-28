"""3D mesh rendering using PyVista."""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from body_sim.core.types import BodyMesh, PressureDistribution
from body_sim.mattress.grid import AirCellGrid


class MeshRenderer:
    """3D mesh renderer using PyVista.

    Provides visualization of body mesh, mattress grid, and combined views.
    """

    def __init__(self, background_color: str = "white"):
        """Initialize mesh renderer.

        Args:
            background_color: Background color for plots
        """
        self.background_color = background_color
        self._plotter = None

    def _get_pyvista(self):
        """Lazy import PyVista."""
        try:
            import pyvista as pv

            return pv
        except ImportError:
            raise ImportError(
                "PyVista is required for 3D visualization. "
                "Install with: pip install pyvista"
            )

    def render_body_mesh(
        self,
        mesh: BodyMesh,
        color: str = "lightblue",
        opacity: float = 1.0,
        show_edges: bool = False,
        show_joints: bool = True,
        interactive: bool = True,
    ) -> None:
        """Render body mesh in 3D.

        Args:
            mesh: Body mesh to render
            color: Mesh color
            opacity: Mesh opacity
            show_edges: Whether to show mesh edges
            show_joints: Whether to show joint positions
            interactive: Whether to show interactive window
        """
        pv = self._get_pyvista()

        # Create PyVista mesh
        faces = np.hstack(
            [np.full((len(mesh.faces), 1), 3), mesh.faces]
        ).ravel()
        pv_mesh = pv.PolyData(mesh.vertices, faces)

        # Create plotter
        plotter = pv.Plotter(off_screen=not interactive)
        plotter.set_background(self.background_color)

        # Add mesh
        plotter.add_mesh(
            pv_mesh,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
        )

        # Add joints as spheres
        if show_joints:
            joint_cloud = pv.PolyData(mesh.joints)
            plotter.add_mesh(
                joint_cloud,
                color="red",
                point_size=10,
                render_points_as_spheres=True,
            )

        # Set camera
        plotter.camera_position = "xy"
        plotter.camera.zoom(1.5)

        if interactive:
            plotter.show()
        else:
            self._plotter = plotter

    def render_mattress_grid(
        self,
        grid: AirCellGrid,
        color_by_height: bool = True,
        show_edges: bool = True,
        interactive: bool = True,
    ) -> None:
        """Render mattress grid in 3D.

        Args:
            grid: Air cell grid to render
            color_by_height: Color cells by height
            show_edges: Whether to show cell edges
            interactive: Whether to show interactive window
        """
        pv = self._get_pyvista()

        plotter = pv.Plotter(off_screen=not interactive)
        plotter.set_background(self.background_color)

        # Create cell meshes
        for row, col, state in grid.iterate_cells():
            if not state.enabled:
                continue

            x_min = col * grid.cell_width
            x_max = (col + 1) * grid.cell_width
            y_min = row * grid.cell_length
            y_max = (row + 1) * grid.cell_length
            z_max = state.height

            # Create box for cell
            box = pv.Box(bounds=(x_min, x_max, y_min, y_max, 0, z_max))

            if color_by_height:
                # Color based on height (blue=low, red=high)
                height_ratio = (state.height - grid.min_cell_height) / (
                    grid.max_cell_height - grid.min_cell_height
                )
                color = [height_ratio, 0.3, 1 - height_ratio]
            else:
                color = "lightgray"

            plotter.add_mesh(
                box,
                color=color,
                opacity=0.8,
                show_edges=show_edges,
            )

        # Add floor plane
        floor = pv.Plane(
            center=(grid.width / 2, grid.height / 2, 0),
            direction=(0, 0, 1),
            i_size=grid.width,
            j_size=grid.height,
        )
        plotter.add_mesh(floor, color="gray", opacity=0.3)

        plotter.camera_position = "iso"

        if interactive:
            plotter.show()
        else:
            self._plotter = plotter

    def render_combined(
        self,
        mesh: BodyMesh,
        grid: AirCellGrid,
        pressure_dist: Optional[PressureDistribution] = None,
        show_pressure_colors: bool = True,
        interactive: bool = True,
    ) -> None:
        """Render body mesh and mattress grid together.

        Args:
            mesh: Body mesh
            grid: Air cell grid
            pressure_dist: Optional pressure distribution for coloring
            show_pressure_colors: Color mattress cells by pressure
            interactive: Whether to show interactive window
        """
        pv = self._get_pyvista()

        plotter = pv.Plotter(off_screen=not interactive)
        plotter.set_background(self.background_color)

        # Add body mesh
        faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
        pv_mesh = pv.PolyData(mesh.vertices, faces)
        plotter.add_mesh(pv_mesh, color="lightblue", opacity=0.7)

        # Add mattress grid
        for row in range(grid.rows):
            for col in range(grid.cols):
                height = grid.get_cell_height(row, col)

                x_min = col * grid.cell_width
                x_max = (col + 1) * grid.cell_width
                y_min = row * grid.cell_length
                y_max = (row + 1) * grid.cell_length

                box = pv.Box(bounds=(x_min, x_max, y_min, y_max, 0, height))

                if show_pressure_colors and pressure_dist is not None:
                    pressure = pressure_dist.grid_pressure[row, col]
                    max_pressure = pressure_dist.peak_pressure
                    if max_pressure > 0:
                        ratio = min(1.0, pressure / max_pressure)
                        color = [ratio, 0.2, 0.2]
                    else:
                        color = "lightgray"
                else:
                    height_ratio = (height - grid.min_cell_height) / (
                        grid.max_cell_height - grid.min_cell_height
                    )
                    color = [0.3, height_ratio, 0.3]

                plotter.add_mesh(box, color=color, opacity=0.6, show_edges=True)

        plotter.camera_position = "iso"
        plotter.camera.zoom(1.2)

        if interactive:
            plotter.show()
        else:
            self._plotter = plotter

    def render_pressure_on_mesh(
        self,
        mesh: BodyMesh,
        contact_pressures: NDArray[np.float32],
        interactive: bool = True,
    ) -> None:
        """Render body mesh with pressure-colored vertices.

        Args:
            mesh: Body mesh
            contact_pressures: Pressure value per vertex
            interactive: Whether to show interactive window
        """
        pv = self._get_pyvista()

        faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
        pv_mesh = pv.PolyData(mesh.vertices, faces)

        # Add pressure as scalar data
        pv_mesh["pressure"] = contact_pressures

        plotter = pv.Plotter(off_screen=not interactive)
        plotter.set_background(self.background_color)
        plotter.add_mesh(
            pv_mesh,
            scalars="pressure",
            cmap="hot",
            show_scalar_bar=True,
            scalar_bar_args={"title": "Pressure (Pa)"},
        )

        plotter.camera_position = "xy"

        if interactive:
            plotter.show()
        else:
            self._plotter = plotter

    def screenshot(
        self,
        filename: str,
        window_size: tuple[int, int] = (1920, 1080),
    ) -> None:
        """Save current view to image file.

        Args:
            filename: Output filename
            window_size: Image size (width, height)
        """
        if self._plotter is None:
            raise RuntimeError("No active render. Call a render method first.")

        self._plotter.window_size = window_size
        self._plotter.screenshot(filename)

    def create_animation(
        self,
        history: list[tuple[BodyMesh, AirCellGrid, PressureDistribution]],
        filename: str,
        fps: int = 10,
    ) -> None:
        """Create animation from simulation history.

        Args:
            history: List of (mesh, grid, pressure) tuples
            filename: Output filename (should end in .gif or .mp4)
            fps: Frames per second
        """
        pv = self._get_pyvista()

        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(self.background_color)
        plotter.open_gif(filename) if filename.endswith(".gif") else plotter.open_movie(
            filename, framerate=fps
        )

        for mesh, grid, pressure_dist in history:
            plotter.clear()

            # Add body
            faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
            pv_mesh = pv.PolyData(mesh.vertices, faces)
            plotter.add_mesh(pv_mesh, color="lightblue", opacity=0.7)

            # Add grid with pressure colors
            for row in range(grid.rows):
                for col in range(grid.cols):
                    height = grid.get_cell_height(row, col)
                    x_min = col * grid.cell_width
                    x_max = (col + 1) * grid.cell_width
                    y_min = row * grid.cell_length
                    y_max = (row + 1) * grid.cell_length

                    box = pv.Box(bounds=(x_min, x_max, y_min, y_max, 0, height))

                    pressure = pressure_dist.grid_pressure[row, col]
                    if pressure_dist.peak_pressure > 0:
                        ratio = min(1.0, pressure / pressure_dist.peak_pressure)
                    else:
                        ratio = 0
                    color = [ratio, 0.2, 0.2]

                    plotter.add_mesh(box, color=color, opacity=0.6)

            plotter.write_frame()

        plotter.close()
