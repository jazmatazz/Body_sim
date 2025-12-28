"""Air cell grid representation for the adjustable mattress."""

from typing import Iterator, Optional

import numpy as np
from numpy.typing import NDArray

from body_sim.core.types import CellState, SimulationConfig
from body_sim.core.constants import (
    DEFAULT_MATTRESS_WIDTH,
    DEFAULT_MATTRESS_HEIGHT,
    DEFAULT_CELL_MAX_HEIGHT,
    DEFAULT_CELL_MIN_HEIGHT,
    DEFAULT_GRID_ROWS,
    DEFAULT_GRID_COLS,
)


class AirCellGrid:
    """Represents the adjustable air mattress grid.

    The grid consists of individually controllable air cells that can
    inflate/deflate to different heights. The grid uses a coordinate system where:
    - Row 0 is at the head of the bed (Y = 0)
    - Row max is at the foot (Y = mattress_height)
    - Column 0 is on the left side (X = 0)
    - Column max is on the right (X = mattress_width)

    Attributes:
        rows: Number of cell rows (along mattress length)
        cols: Number of cell columns (along mattress width)
        width: Physical mattress width in meters
        height: Physical mattress length in meters
        max_cell_height: Maximum cell inflation height in meters
        min_cell_height: Minimum cell height (never fully flat)
    """

    def __init__(
        self,
        rows: int = DEFAULT_GRID_ROWS,
        cols: int = DEFAULT_GRID_COLS,
        width: float = DEFAULT_MATTRESS_WIDTH,
        height: float = DEFAULT_MATTRESS_HEIGHT,
        max_cell_height: float = DEFAULT_CELL_MAX_HEIGHT,
        min_cell_height: float = DEFAULT_CELL_MIN_HEIGHT,
    ):
        """Initialize air cell grid.

        Args:
            rows: Number of cell rows
            cols: Number of cell columns
            width: Mattress width in meters
            height: Mattress length in meters
            max_cell_height: Maximum inflation height in meters
            min_cell_height: Minimum height (cells never fully deflate)
        """
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.max_cell_height = max_cell_height
        self.min_cell_height = min_cell_height

        # Calculate cell dimensions
        self._cell_width = width / cols
        self._cell_length = height / rows

        # Initialize all cells at 50% height
        default_height = (max_cell_height + min_cell_height) / 2
        self._heights = np.full((rows, cols), default_height, dtype=np.float32)
        self._enabled = np.ones((rows, cols), dtype=bool)

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "AirCellGrid":
        """Create grid from simulation configuration.

        Args:
            config: Simulation configuration

        Returns:
            Configured AirCellGrid
        """
        return cls(
            rows=config.grid_rows,
            cols=config.grid_cols,
            width=config.mattress_width,
            height=config.mattress_height,
            max_cell_height=config.cell_max_height,
            min_cell_height=config.cell_min_height,
        )

    @property
    def cell_width(self) -> float:
        """Width of each cell in meters."""
        return self._cell_width

    @property
    def cell_length(self) -> float:
        """Length of each cell in meters."""
        return self._cell_length

    @property
    def cell_area(self) -> float:
        """Area of each cell in square meters."""
        return self._cell_width * self._cell_length

    @property
    def heights(self) -> NDArray[np.float32]:
        """Current height state of all cells (rows, cols)."""
        return self._heights.copy()

    @property
    def enabled(self) -> NDArray[np.bool_]:
        """Enabled state of all cells."""
        return self._enabled.copy()

    def set_cell_height(self, row: int, col: int, height: float) -> None:
        """Set height of a specific cell.

        Args:
            row: Row index
            col: Column index
            height: Target height in meters (clamped to valid range)
        """
        self._heights[row, col] = np.clip(
            height, self.min_cell_height, self.max_cell_height
        )

    def set_cell_enabled(self, row: int, col: int, enabled: bool) -> None:
        """Enable or disable a cell.

        Args:
            row: Row index
            col: Column index
            enabled: Whether cell is active
        """
        self._enabled[row, col] = enabled

    def set_all_heights(self, height: float) -> None:
        """Set all cells to same height.

        Args:
            height: Target height in meters
        """
        self._heights.fill(np.clip(height, self.min_cell_height, self.max_cell_height))

    def set_pattern(self, heights: NDArray[np.float32]) -> None:
        """Set heights for all cells from array.

        Args:
            heights: Array of shape (rows, cols) with height values

        Raises:
            ValueError: If shape doesn't match grid
        """
        if heights.shape != (self.rows, self.cols):
            raise ValueError(
                f"Height array shape {heights.shape} doesn't match grid ({self.rows}, {self.cols})"
            )
        self._heights = np.clip(
            heights.astype(np.float32), self.min_cell_height, self.max_cell_height
        )

    def set_normalized_pattern(self, pattern: NDArray[np.float32]) -> None:
        """Set heights from normalized (0-1) pattern.

        Args:
            pattern: Array of shape (rows, cols) with values in [0, 1]
        """
        height_range = self.max_cell_height - self.min_cell_height
        heights = self.min_cell_height + pattern * height_range
        self.set_pattern(heights)

    def get_cell_height(self, row: int, col: int) -> float:
        """Get height of specific cell.

        Args:
            row: Row index
            col: Column index

        Returns:
            Cell height in meters
        """
        return float(self._heights[row, col])

    def get_cell_center(self, row: int, col: int) -> tuple[float, float, float]:
        """Get 3D center position of cell.

        Args:
            row: Row index
            col: Column index

        Returns:
            (x, y, z) position where z is top surface height
        """
        x = (col + 0.5) * self._cell_width
        y = (row + 0.5) * self._cell_length
        z = self._heights[row, col]
        return (x, y, z)

    def get_cell_bounds(
        self, row: int, col: int
    ) -> tuple[float, float, float, float, float]:
        """Get cell boundary coordinates.

        Args:
            row: Row index
            col: Column index

        Returns:
            (x_min, x_max, y_min, y_max, z_top)
        """
        x_min = col * self._cell_width
        x_max = (col + 1) * self._cell_width
        y_min = row * self._cell_length
        y_max = (row + 1) * self._cell_length
        z_top = self._heights[row, col]
        return (x_min, x_max, y_min, y_max, z_top)

    def get_cell_at_position(self, x: float, y: float) -> Optional[tuple[int, int]]:
        """Find cell index at given x,y position.

        Args:
            x: X coordinate (across width)
            y: Y coordinate (along length)

        Returns:
            (row, col) tuple or None if outside grid
        """
        col = int(x / self._cell_width)
        row = int(y / self._cell_length)

        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)
        return None

    def get_surface_height_at(self, x: float, y: float) -> float:
        """Get surface height at position (no interpolation).

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Surface height in meters, or 0 if outside grid
        """
        cell = self.get_cell_at_position(x, y)
        if cell is None:
            return 0.0
        return float(self._heights[cell])

    def get_interpolated_height_at(self, x: float, y: float) -> float:
        """Get bilinearly interpolated surface height.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Interpolated surface height
        """
        # Convert to cell-relative coordinates
        col_f = x / self._cell_width - 0.5
        row_f = y / self._cell_length - 0.5

        # Clamp to valid range
        col_f = np.clip(col_f, 0, self.cols - 1)
        row_f = np.clip(row_f, 0, self.rows - 1)

        # Get surrounding cells
        col0 = int(col_f)
        row0 = int(row_f)
        col1 = min(col0 + 1, self.cols - 1)
        row1 = min(row0 + 1, self.rows - 1)

        # Interpolation weights
        col_w = col_f - col0
        row_w = row_f - row0

        # Bilinear interpolation
        h00 = self._heights[row0, col0]
        h01 = self._heights[row0, col1]
        h10 = self._heights[row1, col0]
        h11 = self._heights[row1, col1]

        h0 = h00 * (1 - col_w) + h01 * col_w
        h1 = h10 * (1 - col_w) + h11 * col_w
        return float(h0 * (1 - row_w) + h1 * row_w)

    def iterate_cells(self) -> Iterator[tuple[int, int, CellState]]:
        """Iterate over all cells with their states.

        Yields:
            (row, col, CellState) for each cell
        """
        for row in range(self.rows):
            for col in range(self.cols):
                yield row, col, CellState(
                    height=float(self._heights[row, col]),
                    pressure=float(
                        (self._heights[row, col] - self.min_cell_height)
                        / (self.max_cell_height - self.min_cell_height)
                    ),
                    enabled=bool(self._enabled[row, col]),
                )

    def get_row_heights(self, row: int) -> NDArray[np.float32]:
        """Get heights for all cells in a row.

        Args:
            row: Row index

        Returns:
            Array of heights for the row
        """
        return self._heights[row, :].copy()

    def set_row_heights(self, row: int, heights: NDArray[np.float32]) -> None:
        """Set heights for all cells in a row.

        Args:
            row: Row index
            heights: Array of heights (length must match cols)
        """
        if len(heights) != self.cols:
            raise ValueError(f"Expected {self.cols} heights, got {len(heights)}")
        self._heights[row, :] = np.clip(
            heights, self.min_cell_height, self.max_cell_height
        )

    def get_zone_mask(
        self, row_start: float, row_end: float
    ) -> NDArray[np.bool_]:
        """Get mask for cells in a row range (as fractions).

        Args:
            row_start: Start fraction (0 = head)
            row_end: End fraction (1 = foot)

        Returns:
            Boolean mask of shape (rows, cols)
        """
        start_row = int(row_start * self.rows)
        end_row = int(row_end * self.rows)

        mask = np.zeros((self.rows, self.cols), dtype=bool)
        mask[start_row:end_row, :] = True
        return mask

    def reset(self) -> None:
        """Reset all cells to default 50% height."""
        default_height = (self.max_cell_height + self.min_cell_height) / 2
        self._heights.fill(default_height)
        self._enabled.fill(True)

    def __repr__(self) -> str:
        return (
            f"AirCellGrid(rows={self.rows}, cols={self.cols}, "
            f"size={self.width:.3f}x{self.height:.3f}m)"
        )
