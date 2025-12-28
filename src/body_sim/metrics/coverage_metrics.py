"""Coverage and contact area metrics."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from body_sim.core.types import PressureDistribution, BodyMesh
from body_sim.core.constants import BODY_REGION_ROWS
from body_sim.mattress.grid import AirCellGrid


class CoverageAnalyzer:
    """Analyzes contact coverage and distribution patterns."""

    def __init__(self, grid: AirCellGrid):
        """Initialize coverage analyzer.

        Args:
            grid: Air cell mattress grid
        """
        self.grid = grid

    def contact_area_fraction(self, dist: PressureDistribution) -> float:
        """Calculate fraction of mattress in contact.

        Args:
            dist: Pressure distribution

        Returns:
            Contact area fraction (0-1)
        """
        return dist.contact_area

    def contact_area_m2(self, dist: PressureDistribution) -> float:
        """Calculate actual contact area in square meters.

        Args:
            dist: Pressure distribution

        Returns:
            Contact area in m²
        """
        total_area = self.grid.width * self.grid.height
        return dist.contact_area * total_area

    def region_coverage(self, dist: PressureDistribution) -> dict[str, float]:
        """Calculate contact coverage by body region.

        Args:
            dist: Pressure distribution

        Returns:
            Dictionary mapping region names to coverage fractions
        """
        rows = dist.grid_pressure.shape[0]
        results = {}

        for region, (start_frac, end_frac) in BODY_REGION_ROWS.items():
            start_row = int(start_frac * rows)
            end_row = int(end_frac * rows)

            region_data = dist.grid_pressure[start_row:end_row, :]
            contact_mask = region_data > 0

            results[region] = float(contact_mask.sum()) / float(region_data.size)

        return results

    def contact_symmetry(self, dist: PressureDistribution) -> float:
        """Calculate left-right symmetry of contact.

        Values close to 1 indicate symmetric contact.

        Args:
            dist: Pressure distribution

        Returns:
            Symmetry score (0-1)
        """
        grid = dist.grid_pressure
        mid_col = grid.shape[1] // 2

        left_half = grid[:, :mid_col]
        right_half = grid[:, mid_col:][:, ::-1]  # Flip for comparison

        # Ensure same size
        min_cols = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_cols]
        right_half = right_half[:, :min_cols]

        # Calculate correlation
        left_flat = left_half.flatten()
        right_flat = right_half.flatten()

        if left_flat.std() == 0 or right_flat.std() == 0:
            return 1.0 if np.allclose(left_flat, right_flat) else 0.0

        correlation = np.corrcoef(left_flat, right_flat)[0, 1]
        return max(0.0, correlation)  # Clip negative correlations

    def contact_distribution_entropy(self, dist: PressureDistribution) -> float:
        """Calculate entropy of contact distribution.

        Higher entropy indicates more spread out contact.

        Args:
            dist: Pressure distribution

        Returns:
            Normalized entropy (0-1)
        """
        grid = dist.grid_pressure
        contact_mask = grid > 0

        if not contact_mask.any():
            return 0.0

        # Normalize to probability distribution
        prob = grid / grid.sum()
        prob = prob[prob > 0]  # Remove zeros

        # Calculate entropy
        entropy = -np.sum(prob * np.log2(prob))

        # Normalize by maximum possible entropy
        max_entropy = np.log2(grid.size)

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def contact_center_of_mass(
        self, dist: PressureDistribution
    ) -> tuple[float, float]:
        """Calculate center of mass of contact pressure.

        Args:
            dist: Pressure distribution

        Returns:
            (row_center, col_center) as fractions of grid dimensions
        """
        grid = dist.grid_pressure
        total = grid.sum()

        if total == 0:
            return (0.5, 0.5)

        rows, cols = grid.shape
        row_indices = np.arange(rows)[:, np.newaxis]
        col_indices = np.arange(cols)[np.newaxis, :]

        row_center = (grid * row_indices).sum() / total / rows
        col_center = (grid * col_indices).sum() / total / cols

        return (float(row_center), float(col_center))

    def contact_spread(self, dist: PressureDistribution) -> dict[str, float]:
        """Calculate spread of contact area.

        Args:
            dist: Pressure distribution

        Returns:
            Dictionary with row and column spread metrics
        """
        grid = dist.grid_pressure
        contact_mask = grid > 0

        if not contact_mask.any():
            return {"row_span": 0.0, "col_span": 0.0, "area_spread": 0.0}

        contact_rows = np.any(contact_mask, axis=1)
        contact_cols = np.any(contact_mask, axis=0)

        row_span = contact_rows.sum() / len(contact_rows)
        col_span = contact_cols.sum() / len(contact_cols)

        # Weighted spread based on pressure
        row_center, col_center = self.contact_center_of_mass(dist)

        rows, cols = grid.shape
        row_indices = np.arange(rows)[:, np.newaxis] / rows
        col_indices = np.arange(cols)[np.newaxis, :] / cols

        total = grid.sum()
        row_variance = (grid * (row_indices - row_center) ** 2).sum() / total
        col_variance = (grid * (col_indices - col_center) ** 2).sum() / total

        area_spread = np.sqrt(row_variance + col_variance)

        return {
            "row_span": float(row_span),
            "col_span": float(col_span),
            "area_spread": float(area_spread),
        }

    def effective_support_area(
        self,
        dist: PressureDistribution,
        threshold_fraction: float = 0.1,
    ) -> float:
        """Calculate area providing effective support.

        Cells must have at least threshold_fraction of peak pressure
        to count as providing support.

        Args:
            dist: Pressure distribution
            threshold_fraction: Fraction of peak pressure for support

        Returns:
            Effective support area fraction (0-1)
        """
        if dist.peak_pressure == 0:
            return 0.0

        threshold = dist.peak_pressure * threshold_fraction
        support_mask = dist.grid_pressure >= threshold

        return float(support_mask.sum()) / float(dist.grid_pressure.size)

    def get_summary(self, dist: PressureDistribution) -> dict[str, float]:
        """Get summary of coverage metrics.

        Args:
            dist: Pressure distribution

        Returns:
            Dictionary of metric names to values
        """
        center = self.contact_center_of_mass(dist)
        spread = self.contact_spread(dist)

        return {
            "contact_fraction": self.contact_area_fraction(dist),
            "contact_area_m2": self.contact_area_m2(dist),
            "symmetry": self.contact_symmetry(dist),
            "entropy": self.contact_distribution_entropy(dist),
            "center_row": center[0],
            "center_col": center[1],
            "row_span": spread["row_span"],
            "col_span": spread["col_span"],
            "area_spread": spread["area_spread"],
            "effective_support": self.effective_support_area(dist),
        }
