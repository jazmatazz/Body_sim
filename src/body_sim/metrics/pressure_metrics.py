"""Pressure distribution metrics and analysis."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from body_sim.core.types import PressureDistribution
from body_sim.core.constants import (
    CAPILLARY_CLOSING_PRESSURE_PA,
    HIGH_RISK_PRESSURE_PA,
    BODY_REGION_ROWS,
)


class PressureAnalyzer:
    """Analyzes pressure distributions for medical metrics.

    Provides various pressure metrics relevant to pressure ulcer
    prevention and tissue health assessment.
    """

    def __init__(
        self,
        history: Optional[list[PressureDistribution]] = None,
        capillary_threshold: float = CAPILLARY_CLOSING_PRESSURE_PA,
        high_risk_threshold: float = HIGH_RISK_PRESSURE_PA,
    ):
        """Initialize pressure analyzer.

        Args:
            history: List of pressure distributions to analyze
            capillary_threshold: Capillary closing pressure (Pa)
            high_risk_threshold: High risk pressure threshold (Pa)
        """
        self.history = history or []
        self.capillary_threshold = capillary_threshold
        self.high_risk_threshold = high_risk_threshold

    def add_distribution(self, dist: PressureDistribution) -> None:
        """Add pressure distribution to history.

        Args:
            dist: Pressure distribution to add
        """
        self.history.append(dist)

    def clear_history(self) -> None:
        """Clear analysis history."""
        self.history.clear()

    # Instantaneous metrics

    def peak_pressure(self, dist: Optional[PressureDistribution] = None) -> float:
        """Get peak pressure from distribution.

        Args:
            dist: Distribution to analyze (uses last in history if None)

        Returns:
            Peak pressure in Pa
        """
        dist = dist or (self.history[-1] if self.history else None)
        if dist is None:
            return 0.0
        return dist.peak_pressure

    def average_pressure(self, dist: Optional[PressureDistribution] = None) -> float:
        """Get average pressure over contact area.

        Args:
            dist: Distribution to analyze

        Returns:
            Average pressure in Pa
        """
        dist = dist or (self.history[-1] if self.history else None)
        if dist is None:
            return 0.0
        return dist.average_pressure

    def pressure_variance(self, dist: Optional[PressureDistribution] = None) -> float:
        """Calculate pressure variance over contact area.

        Lower variance indicates more uniform distribution.

        Args:
            dist: Distribution to analyze

        Returns:
            Pressure variance (Pa²)
        """
        dist = dist or (self.history[-1] if self.history else None)
        if dist is None:
            return 0.0

        contact_mask = dist.grid_pressure > 0
        if not contact_mask.any():
            return 0.0

        return float(dist.grid_pressure[contact_mask].var())

    def pressure_gradient(self, dist: Optional[PressureDistribution] = None) -> float:
        """Calculate maximum pressure gradient.

        High gradients indicate rapid pressure changes between adjacent cells.

        Args:
            dist: Distribution to analyze

        Returns:
            Maximum gradient magnitude (Pa/cell)
        """
        dist = dist or (self.history[-1] if self.history else None)
        if dist is None:
            return 0.0

        grid = dist.grid_pressure
        grad_y, grad_x = np.gradient(grid)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        return float(gradient_mag.max())

    def above_threshold_fraction(
        self,
        dist: Optional[PressureDistribution] = None,
        threshold: Optional[float] = None,
    ) -> float:
        """Calculate fraction of contact area above threshold.

        Args:
            dist: Distribution to analyze
            threshold: Pressure threshold (uses capillary if None)

        Returns:
            Fraction of contact area above threshold (0-1)
        """
        dist = dist or (self.history[-1] if self.history else None)
        if dist is None:
            return 0.0

        threshold = threshold or self.capillary_threshold

        contact_mask = dist.grid_pressure > 0
        if not contact_mask.any():
            return 0.0

        above_threshold = dist.grid_pressure > threshold
        return float((above_threshold & contact_mask).sum()) / float(contact_mask.sum())

    def identify_hotspots(
        self,
        dist: Optional[PressureDistribution] = None,
        percentile: float = 90,
    ) -> list[tuple[int, int]]:
        """Identify high-pressure hotspot locations.

        Args:
            dist: Distribution to analyze
            percentile: Percentile threshold for hotspot

        Returns:
            List of (row, col) tuples for hotspot cells
        """
        dist = dist or (self.history[-1] if self.history else None)
        if dist is None:
            return []

        contact_mask = dist.grid_pressure > 0
        if not contact_mask.any():
            return []

        threshold = np.percentile(dist.grid_pressure[contact_mask], percentile)
        hotspots = np.argwhere(dist.grid_pressure >= threshold)

        return [tuple(h) for h in hotspots]

    def region_analysis(
        self,
        dist: Optional[PressureDistribution] = None,
    ) -> dict[str, dict[str, float]]:
        """Analyze pressure by body region.

        Args:
            dist: Distribution to analyze

        Returns:
            Dictionary mapping region names to metrics
        """
        dist = dist or (self.history[-1] if self.history else None)
        if dist is None:
            return {}

        rows = dist.grid_pressure.shape[0]
        results = {}

        for region, (start_frac, end_frac) in BODY_REGION_ROWS.items():
            start_row = int(start_frac * rows)
            end_row = int(end_frac * rows)

            region_data = dist.grid_pressure[start_row:end_row, :]
            contact_mask = region_data > 0

            if contact_mask.any():
                results[region] = {
                    "peak": float(region_data.max()),
                    "average": float(region_data[contact_mask].mean()),
                    "above_threshold": float(
                        (region_data > self.capillary_threshold).sum()
                    )
                    / float(region_data.size),
                    "contact_fraction": float(contact_mask.sum())
                    / float(region_data.size),
                }
            else:
                results[region] = {
                    "peak": 0.0,
                    "average": 0.0,
                    "above_threshold": 0.0,
                    "contact_fraction": 0.0,
                }

        return results

    # Temporal metrics (require history)

    def max_peak_pressure(self) -> float:
        """Get maximum peak pressure over history.

        Returns:
            Maximum peak pressure in Pa
        """
        if not self.history:
            return 0.0
        return max(d.peak_pressure for d in self.history)

    def mean_peak_pressure(self) -> float:
        """Get average peak pressure over history.

        Returns:
            Mean peak pressure in Pa
        """
        if not self.history:
            return 0.0
        return float(np.mean([d.peak_pressure for d in self.history]))

    def time_above_threshold(self, threshold: Optional[float] = None) -> float:
        """Calculate total time with peak pressure above threshold.

        Args:
            threshold: Pressure threshold (uses capillary if None)

        Returns:
            Time in seconds
        """
        if not self.history:
            return 0.0

        threshold = threshold or self.capillary_threshold

        # Calculate time step from timestamps
        if len(self.history) < 2:
            return 0.0

        dt = self.history[1].timestamp - self.history[0].timestamp
        above_count = sum(1 for d in self.history if d.peak_pressure > threshold)

        return above_count * dt

    def cumulative_pressure_exposure(self) -> float:
        """Calculate cumulative pressure exposure (pressure × time).

        This is a simple metric for total tissue loading.

        Returns:
            Cumulative exposure in Pa·s
        """
        if len(self.history) < 2:
            return 0.0

        dt = self.history[1].timestamp - self.history[0].timestamp
        total = sum(d.average_pressure * dt for d in self.history)

        return total

    def pressure_relief_periods(
        self,
        threshold: Optional[float] = None,
        min_duration: float = 1.0,
    ) -> list[tuple[float, float]]:
        """Identify periods when pressure was below threshold.

        Args:
            threshold: Pressure threshold
            min_duration: Minimum relief duration to count (seconds)

        Returns:
            List of (start_time, end_time) tuples for relief periods
        """
        if not self.history:
            return []

        threshold = threshold or self.capillary_threshold
        relief_periods = []

        relief_start = None
        for dist in self.history:
            is_relief = dist.peak_pressure < threshold

            if is_relief and relief_start is None:
                relief_start = dist.timestamp
            elif not is_relief and relief_start is not None:
                duration = dist.timestamp - relief_start
                if duration >= min_duration:
                    relief_periods.append((relief_start, dist.timestamp))
                relief_start = None

        # Handle ongoing relief at end
        if relief_start is not None:
            duration = self.history[-1].timestamp - relief_start
            if duration >= min_duration:
                relief_periods.append((relief_start, self.history[-1].timestamp))

        return relief_periods

    def get_summary(self) -> dict[str, float]:
        """Get summary of all key metrics.

        Returns:
            Dictionary of metric names to values
        """
        latest = self.history[-1] if self.history else None

        return {
            "current_peak": self.peak_pressure(latest),
            "current_average": self.average_pressure(latest),
            "current_variance": self.pressure_variance(latest),
            "above_threshold_fraction": self.above_threshold_fraction(latest),
            "max_peak_over_time": self.max_peak_pressure(),
            "mean_peak_over_time": self.mean_peak_pressure(),
            "time_above_threshold": self.time_above_threshold(),
            "cumulative_exposure": self.cumulative_pressure_exposure(),
            "num_samples": len(self.history),
        }
