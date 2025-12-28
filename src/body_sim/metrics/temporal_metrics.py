"""Temporal analysis metrics for pressure exposure over time."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from body_sim.core.types import PressureDistribution
from body_sim.core.constants import CAPILLARY_CLOSING_PRESSURE_PA, BODY_REGION_ROWS


@dataclass
class ExposureEvent:
    """Represents a period of high pressure exposure."""

    start_time: float
    end_time: float
    peak_pressure: float
    average_pressure: float
    region: Optional[str] = None

    @property
    def duration(self) -> float:
        """Duration of exposure in seconds."""
        return self.end_time - self.start_time


class TemporalAnalyzer:
    """Analyzes pressure patterns over time for tissue health assessment."""

    def __init__(
        self,
        history: Optional[list[PressureDistribution]] = None,
        threshold: float = CAPILLARY_CLOSING_PRESSURE_PA,
    ):
        """Initialize temporal analyzer.

        Args:
            history: List of pressure distributions over time
            threshold: Pressure threshold for exposure tracking
        """
        self.history = history or []
        self.threshold = threshold

    def add_distribution(self, dist: PressureDistribution) -> None:
        """Add distribution to history.

        Args:
            dist: Pressure distribution
        """
        self.history.append(dist)

    def clear_history(self) -> None:
        """Clear history."""
        self.history.clear()

    @property
    def duration(self) -> float:
        """Total duration of recorded history."""
        if len(self.history) < 2:
            return 0.0
        return self.history[-1].timestamp - self.history[0].timestamp

    @property
    def time_step(self) -> float:
        """Estimated time step between samples."""
        if len(self.history) < 2:
            return 0.1
        return self.history[1].timestamp - self.history[0].timestamp

    def exposure_duration(self, threshold: Optional[float] = None) -> float:
        """Calculate total time with peak pressure above threshold.

        Args:
            threshold: Pressure threshold (uses default if None)

        Returns:
            Total exposure time in seconds
        """
        threshold = threshold or self.threshold
        dt = self.time_step

        exposure_count = sum(
            1 for d in self.history if d.peak_pressure > threshold
        )

        return exposure_count * dt

    def exposure_fraction(self, threshold: Optional[float] = None) -> float:
        """Calculate fraction of time above threshold.

        Args:
            threshold: Pressure threshold

        Returns:
            Exposure fraction (0-1)
        """
        if not self.history:
            return 0.0

        threshold = threshold or self.threshold
        above_count = sum(
            1 for d in self.history if d.peak_pressure > threshold
        )

        return above_count / len(self.history)

    def cumulative_exposure(
        self,
        threshold: Optional[float] = None,
        use_average: bool = False,
    ) -> float:
        """Calculate cumulative pressure exposure (pressure × time).

        Only counts pressure above threshold.

        Args:
            threshold: Minimum pressure to count
            use_average: Use average instead of peak pressure

        Returns:
            Cumulative exposure in Pa·s
        """
        threshold = threshold or 0.0
        dt = self.time_step

        total = 0.0
        for dist in self.history:
            pressure = dist.average_pressure if use_average else dist.peak_pressure
            if pressure > threshold:
                total += (pressure - threshold) * dt

        return total

    def find_exposure_events(
        self,
        threshold: Optional[float] = None,
        min_duration: float = 0.0,
    ) -> list[ExposureEvent]:
        """Find continuous periods of high pressure exposure.

        Args:
            threshold: Pressure threshold
            min_duration: Minimum event duration to include

        Returns:
            List of ExposureEvent objects
        """
        if not self.history:
            return []

        threshold = threshold or self.threshold
        events = []

        event_start = None
        event_pressures = []

        for dist in self.history:
            above = dist.peak_pressure > threshold

            if above:
                if event_start is None:
                    event_start = dist.timestamp
                    event_pressures = []
                event_pressures.append(dist.peak_pressure)

            elif event_start is not None:
                # End of event
                duration = dist.timestamp - event_start
                if duration >= min_duration:
                    events.append(
                        ExposureEvent(
                            start_time=event_start,
                            end_time=dist.timestamp,
                            peak_pressure=max(event_pressures),
                            average_pressure=float(np.mean(event_pressures)),
                        )
                    )
                event_start = None
                event_pressures = []

        # Handle ongoing event at end
        if event_start is not None:
            duration = self.history[-1].timestamp - event_start
            if duration >= min_duration:
                events.append(
                    ExposureEvent(
                        start_time=event_start,
                        end_time=self.history[-1].timestamp,
                        peak_pressure=max(event_pressures),
                        average_pressure=float(np.mean(event_pressures)),
                    )
                )

        return events

    def relief_periods(
        self,
        threshold: Optional[float] = None,
        min_duration: float = 1.0,
    ) -> list[tuple[float, float]]:
        """Find periods of pressure relief (below threshold).

        Args:
            threshold: Pressure threshold
            min_duration: Minimum relief duration

        Returns:
            List of (start_time, end_time) tuples
        """
        if not self.history:
            return []

        threshold = threshold or self.threshold
        periods = []

        relief_start = None

        for dist in self.history:
            is_relief = dist.peak_pressure < threshold

            if is_relief and relief_start is None:
                relief_start = dist.timestamp
            elif not is_relief and relief_start is not None:
                duration = dist.timestamp - relief_start
                if duration >= min_duration:
                    periods.append((relief_start, dist.timestamp))
                relief_start = None

        # Handle ongoing relief
        if relief_start is not None:
            duration = self.history[-1].timestamp - relief_start
            if duration >= min_duration:
                periods.append((relief_start, self.history[-1].timestamp))

        return periods

    def pressure_variability(self) -> dict[str, float]:
        """Calculate pressure variability over time.

        Returns:
            Dictionary with variability metrics
        """
        if not self.history:
            return {"std": 0.0, "cv": 0.0, "range": 0.0}

        peaks = [d.peak_pressure for d in self.history]

        mean_val = float(np.mean(peaks))
        std_val = float(np.std(peaks))
        range_val = float(max(peaks) - min(peaks))

        cv = std_val / mean_val if mean_val > 0 else 0.0

        return {
            "std": std_val,
            "cv": cv,  # Coefficient of variation
            "range": range_val,
        }

    def pressure_trend(self) -> float:
        """Calculate linear trend in peak pressure over time.

        Returns:
            Slope of pressure trend (Pa/s), positive = increasing
        """
        if len(self.history) < 2:
            return 0.0

        times = np.array([d.timestamp for d in self.history])
        peaks = np.array([d.peak_pressure for d in self.history])

        # Linear regression
        slope, _ = np.polyfit(times, peaks, 1)

        return float(slope)

    def effective_relief_ratio(
        self,
        threshold: Optional[float] = None,
    ) -> float:
        """Calculate ratio of effective relief time.

        This considers both duration and magnitude of relief.

        Args:
            threshold: Pressure threshold

        Returns:
            Effective relief ratio (0-1)
        """
        if not self.history:
            return 0.0

        threshold = threshold or self.threshold

        relief_score = 0.0
        total_score = 0.0
        dt = self.time_step

        for dist in self.history:
            if dist.peak_pressure < threshold:
                # Relief is better when pressure is further below threshold
                relief_depth = (threshold - dist.peak_pressure) / threshold
                relief_score += relief_depth * dt
            total_score += dt

        return relief_score / total_score if total_score > 0 else 0.0

    def per_cell_exposure(self) -> NDArray[np.float32]:
        """Calculate cumulative pressure exposure per cell.

        Returns:
            Array of cumulative exposure (Pa·s) per cell
        """
        if not self.history:
            first = self.history[0] if self.history else None
            if first is None:
                return np.array([[0.0]])
            return np.zeros_like(first.grid_pressure)

        dt = self.time_step
        cumulative = np.zeros_like(self.history[0].grid_pressure)

        for dist in self.history:
            cumulative += dist.grid_pressure * dt

        return cumulative

    def get_summary(self) -> dict[str, float]:
        """Get summary of temporal metrics.

        Returns:
            Dictionary of metric names to values
        """
        variability = self.pressure_variability()
        events = self.find_exposure_events()

        return {
            "duration": self.duration,
            "exposure_duration": self.exposure_duration(),
            "exposure_fraction": self.exposure_fraction(),
            "cumulative_exposure": self.cumulative_exposure(),
            "num_exposure_events": len(events),
            "max_event_duration": max((e.duration for e in events), default=0.0),
            "pressure_std": variability["std"],
            "pressure_cv": variability["cv"],
            "pressure_trend": self.pressure_trend(),
            "effective_relief_ratio": self.effective_relief_ratio(),
        }
