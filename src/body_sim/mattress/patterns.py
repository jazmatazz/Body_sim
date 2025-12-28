"""Pattern generators for mattress height sequences."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class PatternGenerator(ABC):
    """Abstract base for mattress height pattern generators.

    Patterns generate normalized height values [0, 1] that are then
    scaled to actual cell heights by the grid.
    """

    @abstractmethod
    def generate(self, rows: int, cols: int, phase: float) -> NDArray[np.float32]:
        """Generate height pattern for given grid size and time phase.

        Args:
            rows: Number of grid rows
            cols: Number of grid columns
            phase: Time phase in range [0, 1] for cyclic patterns

        Returns:
            Array of shape (rows, cols) with normalized heights [0, 1]
        """
        pass

    @abstractmethod
    def get_cycle_duration(self) -> float:
        """Return duration of one complete pattern cycle in seconds."""
        pass

    @property
    def name(self) -> str:
        """Pattern name."""
        return self.__class__.__name__


@dataclass
class AlternatingPattern(PatternGenerator):
    """Two-phase alternating pressure pattern (checkerboard).

    Alternates between two groups of cells in a checkerboard pattern,
    similar to clinical alternating pressure mattresses.

    Attributes:
        cycle_duration: Duration of one complete cycle in seconds
        low_height: Normalized height for deflated cells (0-1)
        high_height: Normalized height for inflated cells (0-1)
        transition_smoothness: How smooth the transition is (0=instant, 1=sine wave)
    """

    cycle_duration: float = 10.0
    low_height: float = 0.3
    high_height: float = 1.0
    transition_smoothness: float = 0.5

    def generate(self, rows: int, cols: int, phase: float) -> NDArray[np.float32]:
        pattern = np.zeros((rows, cols), dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                # Checkerboard: alternate based on row + col parity
                is_group_a = (r + c) % 2 == 0

                # Calculate height based on phase
                if self.transition_smoothness > 0:
                    # Smooth sinusoidal transition
                    t = np.sin(phase * 2 * np.pi) * 0.5 + 0.5
                    t = t * self.transition_smoothness + (1 - self.transition_smoothness) * (
                        0 if phase < 0.5 else 1
                    )
                else:
                    t = 0 if phase < 0.5 else 1

                if is_group_a:
                    height = self.low_height + (self.high_height - self.low_height) * t
                else:
                    height = self.high_height - (self.high_height - self.low_height) * t

                pattern[r, c] = height

        return pattern

    def get_cycle_duration(self) -> float:
        return self.cycle_duration


@dataclass
class WavePattern(PatternGenerator):
    """Wave propagation pattern along mattress length.

    Creates a sinusoidal wave that travels from head to foot or vice versa.

    Attributes:
        cycle_duration: Duration of one wave cycle in seconds
        wavelength: Wavelength in number of cells
        direction: 'head_to_foot' or 'foot_to_head'
        amplitude: Wave amplitude (0-0.5, added to baseline)
        baseline: Base height level (0-1)
    """

    cycle_duration: float = 20.0
    wavelength: float = 4.0
    direction: str = "head_to_foot"
    amplitude: float = 0.35
    baseline: float = 0.65

    def generate(self, rows: int, cols: int, phase: float) -> NDArray[np.float32]:
        pattern = np.zeros((rows, cols), dtype=np.float32)

        for r in range(rows):
            # Calculate wave position
            if self.direction == "head_to_foot":
                wave_pos = (r / self.wavelength - phase) * 2 * np.pi
            else:
                wave_pos = ((rows - r) / self.wavelength - phase) * 2 * np.pi

            # Sinusoidal height
            height = self.baseline + self.amplitude * np.sin(wave_pos)
            height = np.clip(height, 0, 1)
            pattern[r, :] = height

        return pattern

    def get_cycle_duration(self) -> float:
        return self.cycle_duration


@dataclass
class ZoneConfig:
    """Configuration for a single zone in ZonePattern."""

    row_start: float  # Fraction (0 = head, 1 = foot)
    row_end: float
    pattern_type: str = "constant"  # 'constant', 'alternating', 'wave'
    height: float = 0.5
    phase_offset: float = 0.0


@dataclass
class ZonePattern(PatternGenerator):
    """Zone-based pattern with independent control of body regions.

    Allows different pressure patterns for different body regions,
    such as reducing pressure at sacrum while alternating at shoulders.

    Attributes:
        zones: Dictionary of zone configurations
        cycle_duration: Duration of one complete cycle
    """

    zones: dict[str, ZoneConfig] = field(default_factory=dict)
    cycle_duration: float = 30.0

    def __post_init__(self):
        # Default zones if none provided
        if not self.zones:
            self.zones = {
                "head": ZoneConfig(0.0, 0.15, "constant", 0.7),
                "shoulders": ZoneConfig(0.15, 0.30, "alternating", 0.8),
                "back": ZoneConfig(0.30, 0.50, "alternating", 0.8, 0.25),
                "sacrum": ZoneConfig(0.50, 0.65, "alternating", 0.7, 0.5),
                "legs": ZoneConfig(0.65, 0.90, "wave", 0.6),
                "heels": ZoneConfig(0.90, 1.0, "constant", 0.9),
            }

    def generate(self, rows: int, cols: int, phase: float) -> NDArray[np.float32]:
        pattern = np.ones((rows, cols), dtype=np.float32) * 0.5  # Default

        for zone_name, zone in self.zones.items():
            row_start = int(zone.row_start * rows)
            row_end = int(zone.row_end * rows)

            # Apply zone-specific pattern
            zone_phase = (phase + zone.phase_offset) % 1.0

            if zone.pattern_type == "constant":
                pattern[row_start:row_end, :] = zone.height

            elif zone.pattern_type == "alternating":
                for r in range(row_start, row_end):
                    for c in range(cols):
                        is_group_a = (r + c) % 2 == 0
                        if zone_phase < 0.5:
                            height = zone.height if is_group_a else zone.height * 0.4
                        else:
                            height = zone.height * 0.4 if is_group_a else zone.height
                        pattern[r, c] = height

            elif zone.pattern_type == "wave":
                for r in range(row_start, row_end):
                    wave_pos = ((r - row_start) / 2 - zone_phase) * 2 * np.pi
                    height = zone.height + 0.2 * np.sin(wave_pos)
                    pattern[r, :] = np.clip(height, 0, 1)

        return pattern

    def get_cycle_duration(self) -> float:
        return self.cycle_duration


@dataclass
class RowAlternatingPattern(PatternGenerator):
    """Row-based alternating pattern.

    Alternates entire rows between high and low, useful for
    simpler pressure relief strategies.

    Attributes:
        cycle_duration: Cycle duration in seconds
        row_group_size: Number of rows that alternate together
        low_height: Height for deflated rows
        high_height: Height for inflated rows
    """

    cycle_duration: float = 15.0
    row_group_size: int = 2
    low_height: float = 0.3
    high_height: float = 1.0

    def generate(self, rows: int, cols: int, phase: float) -> NDArray[np.float32]:
        pattern = np.zeros((rows, cols), dtype=np.float32)

        for r in range(rows):
            group = (r // self.row_group_size) % 2

            if phase < 0.5:
                height = self.high_height if group == 0 else self.low_height
            else:
                height = self.low_height if group == 0 else self.high_height

            pattern[r, :] = height

        return pattern

    def get_cycle_duration(self) -> float:
        return self.cycle_duration


@dataclass
class CustomPattern(PatternGenerator):
    """Custom pattern from user-defined keyframes.

    Allows specification of exact height patterns at different phases.

    Attributes:
        keyframes: Dict mapping phase (0-1) to height patterns
        cycle_duration: Cycle duration in seconds
        interpolation: 'linear' or 'nearest'
    """

    keyframes: dict[float, NDArray[np.float32]] = field(default_factory=dict)
    cycle_duration: float = 10.0
    interpolation: str = "linear"
    _rows: int = 0
    _cols: int = 0

    def generate(self, rows: int, cols: int, phase: float) -> NDArray[np.float32]:
        if not self.keyframes:
            return np.ones((rows, cols), dtype=np.float32) * 0.5

        # Find surrounding keyframes
        phases = sorted(self.keyframes.keys())

        # Handle wraparound
        if phase <= phases[0]:
            if self.interpolation == "nearest":
                return self.keyframes[phases[0]].copy()
            # Interpolate from last to first
            prev_phase = phases[-1] - 1
            next_phase = phases[0]
            prev_pattern = self.keyframes[phases[-1]]
            next_pattern = self.keyframes[phases[0]]
        elif phase >= phases[-1]:
            if self.interpolation == "nearest":
                return self.keyframes[phases[-1]].copy()
            # Interpolate from last to first
            prev_phase = phases[-1]
            next_phase = phases[0] + 1
            prev_pattern = self.keyframes[phases[-1]]
            next_pattern = self.keyframes[phases[0]]
        else:
            # Find surrounding keyframes
            for i, p in enumerate(phases):
                if p > phase:
                    prev_phase = phases[i - 1]
                    next_phase = p
                    prev_pattern = self.keyframes[prev_phase]
                    next_pattern = self.keyframes[next_phase]
                    break
            else:
                return self.keyframes[phases[-1]].copy()

        if self.interpolation == "nearest":
            if abs(phase - prev_phase) < abs(phase - next_phase):
                return prev_pattern.copy()
            return next_pattern.copy()

        # Linear interpolation
        t = (phase - prev_phase) / (next_phase - prev_phase)
        return (prev_pattern * (1 - t) + next_pattern * t).astype(np.float32)

    def get_cycle_duration(self) -> float:
        return self.cycle_duration

    def add_keyframe(self, phase: float, pattern: NDArray[np.float32]) -> None:
        """Add a keyframe pattern at specified phase.

        Args:
            phase: Phase value (0-1)
            pattern: Height pattern array (rows, cols)
        """
        self.keyframes[phase] = pattern.astype(np.float32)


class StaticPattern(PatternGenerator):
    """Static pattern with no time-based changes.

    Useful for testing or constant-pressure scenarios.
    """

    def __init__(self, pattern: Optional[NDArray[np.float32]] = None, height: float = 0.5):
        """Initialize static pattern.

        Args:
            pattern: Optional fixed pattern array
            height: Default height if no pattern specified
        """
        self._pattern = pattern
        self._height = height

    def generate(self, rows: int, cols: int, phase: float) -> NDArray[np.float32]:
        if self._pattern is not None:
            return self._pattern.copy()
        return np.full((rows, cols), self._height, dtype=np.float32)

    def get_cycle_duration(self) -> float:
        return 1.0  # Arbitrary, pattern doesn't change


# Pattern registry for easy access
PATTERN_TYPES = {
    "alternating": AlternatingPattern,
    "wave": WavePattern,
    "zone": ZonePattern,
    "row_alternating": RowAlternatingPattern,
    "static": StaticPattern,
    "custom": CustomPattern,
}


def create_pattern(pattern_type: str, **kwargs) -> PatternGenerator:
    """Create pattern by type name.

    Args:
        pattern_type: Pattern type name
        **kwargs: Pattern-specific parameters

    Returns:
        Configured pattern generator

    Raises:
        ValueError: If pattern type is unknown
    """
    if pattern_type not in PATTERN_TYPES:
        available = ", ".join(PATTERN_TYPES.keys())
        raise ValueError(f"Unknown pattern type '{pattern_type}'. Available: {available}")

    return PATTERN_TYPES[pattern_type](**kwargs)
