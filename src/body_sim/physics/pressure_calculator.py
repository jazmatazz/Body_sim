"""Pressure distribution calculation from body-mattress contact."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from body_sim.core.types import BodyMesh, ContactPoint, PressureDistribution
from body_sim.core.constants import (
    GRAVITY,
    SMPL_NUM_VERTICES,
    SMPL_VERTEX_AREA_M2,
)
from body_sim.mattress.grid import AirCellGrid


class PressureCalculator:
    """Calculates pressure distribution from body-mattress contact.

    Computes pressure based on:
    - Body weight distribution across contacting vertices
    - Cell height (softer/higher cells reduce pressure)
    - Contact area per cell

    Attributes:
        body_weight: Body weight in kg
        grid: Air cell mattress grid
        vertex_weights: Per-vertex weight distribution (normalized)
    """

    def __init__(
        self,
        body_weight_kg: float,
        grid: AirCellGrid,
        vertex_weights: Optional[NDArray[np.float32]] = None,
    ):
        """Initialize pressure calculator.

        Args:
            body_weight_kg: Total body weight in kg
            grid: Air cell mattress grid
            vertex_weights: Optional per-vertex weight distribution (will be normalized)
        """
        self.body_weight = body_weight_kg
        self.grid = grid
        self.gravity = GRAVITY

        # Default: uniform weight distribution
        if vertex_weights is None:
            self._vertex_weights = np.ones(SMPL_NUM_VERTICES, dtype=np.float32)
            self._vertex_weights /= self._vertex_weights.sum()
        else:
            self._vertex_weights = vertex_weights.astype(np.float32)
            self._vertex_weights /= self._vertex_weights.sum()

    def set_vertex_weights(self, weights: NDArray[np.float32]) -> None:
        """Update vertex weight distribution.

        Args:
            weights: New weight distribution (will be normalized)
        """
        self._vertex_weights = weights.astype(np.float32)
        self._vertex_weights /= self._vertex_weights.sum()

    def calculate_pressure(
        self,
        mesh: BodyMesh,
        contacts: list[ContactPoint],
        smooth_sigma: float = 0.5,
    ) -> PressureDistribution:
        """Calculate pressure distribution on mattress grid.

        Args:
            mesh: Body mesh
            contacts: List of contact points from ContactDetector
            smooth_sigma: Gaussian smoothing sigma (0 for no smoothing)

        Returns:
            PressureDistribution with grid pressure values
        """
        # Initialize grids
        pressure_grid = np.zeros((self.grid.rows, self.grid.cols), dtype=np.float32)
        force_grid = np.zeros_like(pressure_grid)
        vertex_count = np.zeros_like(pressure_grid, dtype=np.int32)

        # Total gravitational force
        total_force = self.body_weight * self.gravity  # Newtons

        # Distribute force to contacting cells
        for contact in contacts:
            row, col = contact.cell_idx
            vertex_idx = contact.vertex_idx

            # Force contribution from this vertex
            if vertex_idx < len(self._vertex_weights):
                vertex_weight = self._vertex_weights[vertex_idx]
            else:
                vertex_weight = 1.0 / SMPL_NUM_VERTICES

            vertex_force = total_force * vertex_weight

            # Adjust force based on cell height
            # Higher cells (more inflated) provide more support, reducing pressure
            cell_height = self.grid.get_cell_height(row, col)
            height_range = self.grid.max_cell_height - self.grid.min_cell_height
            height_ratio = (cell_height - self.grid.min_cell_height) / height_range

            # Pressure reduction factor: higher cells = lower pressure
            # At max height, pressure is 50% of what it would be at min height
            pressure_factor = 1.0 - (height_ratio * 0.5)

            # Penetration-based adjustment (deeper penetration = more pressure)
            penetration_factor = 1.0 + contact.penetration_depth * 10

            # Accumulate force on cell
            force_grid[row, col] += vertex_force * pressure_factor * penetration_factor
            vertex_count[row, col] += 1

        # Convert force to pressure (Pa = N/m²)
        cell_area = self.grid.cell_area
        pressure_grid = force_grid / cell_area

        # Apply Gaussian smoothing for more realistic distribution
        if smooth_sigma > 0:
            pressure_grid = gaussian_filter(pressure_grid, sigma=smooth_sigma)

        # Calculate statistics
        contact_mask = vertex_count > 0
        contact_area = float(contact_mask.sum()) / (self.grid.rows * self.grid.cols)

        if contact_mask.any():
            peak_pressure = float(pressure_grid.max())
            avg_pressure = float(pressure_grid[contact_mask].mean())
        else:
            peak_pressure = 0.0
            avg_pressure = 0.0

        return PressureDistribution(
            grid_pressure=pressure_grid,
            peak_pressure=peak_pressure,
            average_pressure=avg_pressure,
            contact_area=contact_area,
            timestamp=0.0,
        )

    def calculate_with_vertex_areas(
        self,
        mesh: BodyMesh,
        contacts: list[ContactPoint],
        vertex_areas: NDArray[np.float32],
    ) -> PressureDistribution:
        """Calculate pressure using actual vertex areas.

        More accurate than uniform weight distribution.

        Args:
            mesh: Body mesh
            contacts: List of contact points
            vertex_areas: Area associated with each vertex

        Returns:
            PressureDistribution
        """
        # Use areas as weights
        contact_vertex_indices = [c.vertex_idx for c in contacts]
        if not contact_vertex_indices:
            return PressureDistribution(
                grid_pressure=np.zeros((self.grid.rows, self.grid.cols), dtype=np.float32),
                peak_pressure=0.0,
                average_pressure=0.0,
                contact_area=0.0,
                timestamp=0.0,
            )

        # Normalize weights for contacting vertices only
        contact_areas = vertex_areas[contact_vertex_indices]
        weights = contact_areas / contact_areas.sum()

        # Create full weight array
        full_weights = np.zeros(len(vertex_areas), dtype=np.float32)
        for idx, vertex_idx in enumerate(contact_vertex_indices):
            full_weights[vertex_idx] = weights[idx]

        # Temporarily set weights and calculate
        old_weights = self._vertex_weights
        self._vertex_weights = full_weights
        result = self.calculate_pressure(mesh, contacts)
        self._vertex_weights = old_weights

        return result

    def calculate_region_pressure(
        self,
        pressure_dist: PressureDistribution,
        row_start: float,
        row_end: float,
    ) -> dict[str, float]:
        """Calculate pressure statistics for a body region.

        Args:
            pressure_dist: Pressure distribution
            row_start: Start fraction (0 = head)
            row_end: End fraction (1 = foot)

        Returns:
            Dictionary with peak, average, and total pressure
        """
        start_row = int(row_start * self.grid.rows)
        end_row = int(row_end * self.grid.rows)

        region = pressure_dist.grid_pressure[start_row:end_row, :]
        contact_mask = region > 0

        if not contact_mask.any():
            return {"peak": 0.0, "average": 0.0, "total": 0.0}

        return {
            "peak": float(region.max()),
            "average": float(region[contact_mask].mean()),
            "total": float(region.sum()),
        }

    def identify_high_pressure_zones(
        self,
        pressure_dist: PressureDistribution,
        threshold_pa: float,
    ) -> NDArray[np.bool_]:
        """Identify cells above pressure threshold.

        Args:
            pressure_dist: Pressure distribution
            threshold_pa: Pressure threshold in Pascals

        Returns:
            Boolean mask of high-pressure cells
        """
        return pressure_dist.grid_pressure > threshold_pa

    def calculate_pressure_variance(
        self,
        pressure_dist: PressureDistribution,
    ) -> float:
        """Calculate variance in pressure over contact area.

        Lower variance indicates more uniform pressure distribution.

        Args:
            pressure_dist: Pressure distribution

        Returns:
            Pressure variance (Pa²)
        """
        contact_mask = pressure_dist.grid_pressure > 0
        if not contact_mask.any():
            return 0.0

        return float(pressure_dist.grid_pressure[contact_mask].var())

    def estimate_tissue_risk(
        self,
        pressure_dist: PressureDistribution,
        capillary_pressure_pa: float = 4266.0,  # ~32 mmHg
    ) -> dict[str, float]:
        """Estimate tissue damage risk based on pressure.

        Args:
            pressure_dist: Pressure distribution
            capillary_pressure_pa: Capillary closing pressure threshold

        Returns:
            Risk metrics dictionary
        """
        grid = pressure_dist.grid_pressure
        contact_mask = grid > 0

        if not contact_mask.any():
            return {
                "above_threshold_fraction": 0.0,
                "max_pressure_ratio": 0.0,
                "risk_score": 0.0,
            }

        above_threshold = grid > capillary_pressure_pa
        above_fraction = float(above_threshold.sum()) / float(contact_mask.sum())

        max_pressure_ratio = pressure_dist.peak_pressure / capillary_pressure_pa

        # Simple risk score (0-1)
        risk_score = min(1.0, (above_fraction * 0.5) + (max_pressure_ratio - 1) * 0.5)
        risk_score = max(0.0, risk_score)

        return {
            "above_threshold_fraction": above_fraction,
            "max_pressure_ratio": max_pressure_ratio,
            "risk_score": risk_score,
        }
