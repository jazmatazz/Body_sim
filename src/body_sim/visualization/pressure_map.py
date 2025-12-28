"""2D pressure map visualization."""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from body_sim.core.types import PressureDistribution
from body_sim.core.constants import CAPILLARY_CLOSING_PRESSURE_PA, BODY_REGION_ROWS


def create_pressure_heatmap(
    pressure_dist: PressureDistribution,
    ax: Optional[Axes] = None,
    cmap: str = "hot",
    show_colorbar: bool = True,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
) -> tuple[Figure, Axes]:
    """Create 2D heatmap of pressure distribution.

    Args:
        pressure_dist: Pressure distribution to visualize
        ax: Optional matplotlib axes (creates new figure if None)
        cmap: Colormap name
        show_colorbar: Whether to show colorbar
        vmax: Maximum value for colormap (auto if None)
        title: Plot title

    Returns:
        (figure, axes) tuple
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
    else:
        fig = ax.figure

    grid = pressure_dist.grid_pressure

    # Use vmax if provided, otherwise auto-scale
    if vmax is None:
        vmax = grid.max() if grid.max() > 0 else 1.0

    im = ax.imshow(
        grid,
        cmap=cmap,
        aspect="auto",
        origin="upper",
        vmin=0,
        vmax=vmax,
    )

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, label="Pressure (Pa)")

    # Labels
    ax.set_xlabel("Column (Width)")
    ax.set_ylabel("Row (Head → Foot)")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"Pressure Map\nPeak: {pressure_dist.peak_pressure:.0f} Pa, "
            f"Avg: {pressure_dist.average_pressure:.0f} Pa"
        )

    return fig, ax


def create_pressure_contour(
    pressure_dist: PressureDistribution,
    ax: Optional[Axes] = None,
    levels: int = 10,
    filled: bool = True,
    show_colorbar: bool = True,
) -> tuple[Figure, Axes]:
    """Create contour plot of pressure distribution.

    Args:
        pressure_dist: Pressure distribution
        ax: Optional axes
        levels: Number of contour levels
        filled: Whether to use filled contours
        show_colorbar: Whether to show colorbar

    Returns:
        (figure, axes) tuple
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
    else:
        fig = ax.figure

    grid = pressure_dist.grid_pressure
    rows, cols = grid.shape

    X, Y = np.meshgrid(range(cols), range(rows))

    if filled:
        contour = ax.contourf(X, Y, grid, levels=levels, cmap="hot")
    else:
        contour = ax.contour(X, Y, grid, levels=levels, cmap="hot")

    if show_colorbar:
        fig.colorbar(contour, ax=ax, label="Pressure (Pa)")

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Pressure Contour")
    ax.invert_yaxis()

    return fig, ax


def create_threshold_map(
    pressure_dist: PressureDistribution,
    threshold: float = CAPILLARY_CLOSING_PRESSURE_PA,
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """Create binary map showing areas above pressure threshold.

    Args:
        pressure_dist: Pressure distribution
        threshold: Pressure threshold in Pa
        ax: Optional axes

    Returns:
        (figure, axes) tuple
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
    else:
        fig = ax.figure

    above_threshold = pressure_dist.grid_pressure > threshold

    # Custom colormap: below threshold = green, above = red
    cmap = mcolors.ListedColormap(["lightgreen", "red"])

    ax.imshow(above_threshold, cmap=cmap, aspect="auto", origin="upper")

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(f"Pressure > {threshold:.0f} Pa (Capillary Closing Pressure)")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="lightgreen", label=f"Below {threshold:.0f} Pa"),
        Patch(facecolor="red", label=f"Above {threshold:.0f} Pa"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    return fig, ax


def create_region_pressure_chart(
    pressure_dist: PressureDistribution,
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """Create bar chart of pressure by body region.

    Args:
        pressure_dist: Pressure distribution
        ax: Optional axes

    Returns:
        (figure, axes) tuple
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    rows = pressure_dist.grid_pressure.shape[0]

    region_pressures = {}
    for region, (start_frac, end_frac) in BODY_REGION_ROWS.items():
        start_row = int(start_frac * rows)
        end_row = int(end_frac * rows)
        region_data = pressure_dist.grid_pressure[start_row:end_row, :]
        mask = region_data > 0
        if mask.any():
            region_pressures[region] = {
                "peak": float(region_data.max()),
                "avg": float(region_data[mask].mean()),
            }
        else:
            region_pressures[region] = {"peak": 0, "avg": 0}

    regions = list(region_pressures.keys())
    peaks = [region_pressures[r]["peak"] for r in regions]
    avgs = [region_pressures[r]["avg"] for r in regions]

    x = np.arange(len(regions))
    width = 0.35

    bars1 = ax.bar(x - width / 2, peaks, width, label="Peak", color="red", alpha=0.7)
    bars2 = ax.bar(x + width / 2, avgs, width, label="Average", color="blue", alpha=0.7)

    # Add threshold line
    ax.axhline(
        y=CAPILLARY_CLOSING_PRESSURE_PA,
        color="orange",
        linestyle="--",
        label=f"Capillary Pressure ({CAPILLARY_CLOSING_PRESSURE_PA:.0f} Pa)",
    )

    ax.set_xlabel("Body Region")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_title("Pressure by Body Region")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()

    return fig, ax


def create_time_series_plot(
    history: list[PressureDistribution],
    ax: Optional[Axes] = None,
    show_peak: bool = True,
    show_average: bool = True,
    show_threshold: bool = True,
) -> tuple[Figure, Axes]:
    """Create time series plot of pressure metrics.

    Args:
        history: List of pressure distributions over time
        ax: Optional axes
        show_peak: Show peak pressure line
        show_average: Show average pressure line
        show_threshold: Show threshold reference line

    Returns:
        (figure, axes) tuple
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    times = [pd.timestamp for pd in history]
    peaks = [pd.peak_pressure for pd in history]
    avgs = [pd.average_pressure for pd in history]

    if show_peak:
        ax.plot(times, peaks, "r-", label="Peak Pressure", linewidth=2)

    if show_average:
        ax.plot(times, avgs, "b-", label="Average Pressure", linewidth=2)

    if show_threshold:
        ax.axhline(
            y=CAPILLARY_CLOSING_PRESSURE_PA,
            color="orange",
            linestyle="--",
            label=f"Capillary Threshold ({CAPILLARY_CLOSING_PRESSURE_PA:.0f} Pa)",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_title("Pressure Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def create_comparison_plot(
    distributions: dict[str, PressureDistribution],
    figsize: tuple[int, int] = (15, 5),
) -> Figure:
    """Create side-by-side comparison of multiple pressure distributions.

    Args:
        distributions: Dictionary mapping names to distributions
        figsize: Figure size

    Returns:
        Figure with comparison plots
    """
    n = len(distributions)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    # Find global max for consistent colormap
    vmax = max(pd.grid_pressure.max() for pd in distributions.values())

    for ax, (name, pd) in zip(axes, distributions.items()):
        im = ax.imshow(
            pd.grid_pressure,
            cmap="hot",
            aspect="auto",
            origin="upper",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(f"{name}\nPeak: {pd.peak_pressure:.0f} Pa")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig.colorbar(im, ax=axes, label="Pressure (Pa)")
    fig.suptitle("Pressure Distribution Comparison")
    fig.tight_layout()

    return fig


def save_pressure_figure(
    pressure_dist: PressureDistribution,
    filepath: str,
    dpi: int = 150,
    **kwargs,
) -> None:
    """Save pressure heatmap to file.

    Args:
        pressure_dist: Pressure distribution
        filepath: Output file path
        dpi: Figure resolution
        **kwargs: Additional arguments for create_pressure_heatmap
    """
    fig, ax = create_pressure_heatmap(pressure_dist, **kwargs)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
