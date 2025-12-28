"""Visualization module for pressure maps and 3D rendering."""

from body_sim.visualization.pressure_map import (
    create_pressure_heatmap,
    create_pressure_contour,
    create_threshold_map,
    create_region_pressure_chart,
    create_time_series_plot,
    create_comparison_plot,
    save_pressure_figure,
)
from body_sim.visualization.mesh_renderer import MeshRenderer
from body_sim.visualization.dashboard import (
    SimulationDashboard,
    create_plotly_heatmap,
    create_plotly_3d_surface,
    create_plotly_time_series,
)

__all__ = [
    # Matplotlib visualizations
    "create_pressure_heatmap",
    "create_pressure_contour",
    "create_threshold_map",
    "create_region_pressure_chart",
    "create_time_series_plot",
    "create_comparison_plot",
    "save_pressure_figure",
    # 3D rendering
    "MeshRenderer",
    # Dashboard
    "SimulationDashboard",
    "create_plotly_heatmap",
    "create_plotly_3d_surface",
    "create_plotly_time_series",
]
