"""Time controller for pattern phase management."""

from dataclasses import dataclass, field
from typing import Callable, Optional

from body_sim.mattress.patterns import PatternGenerator


@dataclass
class PatternScheduleEntry:
    """Entry in pattern schedule."""

    pattern: PatternGenerator
    duration: float  # Duration to run this pattern (seconds), 0 = indefinite
    repeat: int = 1  # Number of times to repeat


class TimeController:
    """Controls time progression and pattern scheduling.

    Manages simulation time, pattern phase calculation, and
    pattern scheduling for complex multi-pattern scenarios.

    Attributes:
        time_step: Simulation time step in seconds
        current_time: Current simulation time
        paused: Whether simulation is paused
    """

    def __init__(self, time_step: float = 0.1):
        """Initialize time controller.

        Args:
            time_step: Simulation time step in seconds
        """
        self.time_step = time_step
        self._current_time = 0.0
        self._paused = False

        # Pattern scheduling
        self._schedule: list[PatternScheduleEntry] = []
        self._current_schedule_index = 0
        self._schedule_start_time = 0.0
        self._schedule_repeat_count = 0

        # Callbacks
        self._on_pattern_change: Optional[Callable[[PatternGenerator], None]] = None
        self._on_time_tick: Optional[Callable[[float], None]] = None

    @property
    def current_time(self) -> float:
        """Current simulation time in seconds."""
        return self._current_time

    @property
    def paused(self) -> bool:
        """Whether simulation is paused."""
        return self._paused

    def set_time(self, time: float) -> None:
        """Set current time.

        Args:
            time: New time value in seconds
        """
        self._current_time = max(0, time)

    def pause(self) -> None:
        """Pause simulation."""
        self._paused = True

    def resume(self) -> None:
        """Resume simulation."""
        self._paused = False

    def toggle_pause(self) -> bool:
        """Toggle pause state.

        Returns:
            New pause state
        """
        self._paused = not self._paused
        return self._paused

    def tick(self) -> float:
        """Advance time by one step.

        Returns:
            New current time
        """
        if not self._paused:
            self._current_time += self.time_step

            if self._on_time_tick:
                self._on_time_tick(self._current_time)

            # Check schedule transitions
            self._check_schedule()

        return self._current_time

    def get_phase(self, pattern: PatternGenerator) -> float:
        """Get current phase for a pattern.

        Args:
            pattern: Pattern to get phase for

        Returns:
            Phase value in range [0, 1]
        """
        cycle_duration = pattern.get_cycle_duration()
        if cycle_duration <= 0:
            return 0.0
        return (self._current_time % cycle_duration) / cycle_duration

    def get_phase_at_time(self, pattern: PatternGenerator, time: float) -> float:
        """Get phase for a pattern at specific time.

        Args:
            pattern: Pattern generator
            time: Time to calculate phase for

        Returns:
            Phase value in range [0, 1]
        """
        cycle_duration = pattern.get_cycle_duration()
        if cycle_duration <= 0:
            return 0.0
        return (time % cycle_duration) / cycle_duration

    def reset(self) -> None:
        """Reset time to zero."""
        self._current_time = 0.0
        self._current_schedule_index = 0
        self._schedule_start_time = 0.0
        self._schedule_repeat_count = 0

    # Pattern scheduling

    def add_to_schedule(
        self,
        pattern: PatternGenerator,
        duration: float = 0,
        repeat: int = 1,
    ) -> None:
        """Add pattern to schedule.

        Args:
            pattern: Pattern to add
            duration: How long to run (0 = indefinite)
            repeat: Number of times to repeat
        """
        self._schedule.append(
            PatternScheduleEntry(pattern=pattern, duration=duration, repeat=repeat)
        )

    def clear_schedule(self) -> None:
        """Clear pattern schedule."""
        self._schedule.clear()
        self._current_schedule_index = 0
        self._schedule_start_time = 0.0
        self._schedule_repeat_count = 0

    def get_current_scheduled_pattern(self) -> Optional[PatternGenerator]:
        """Get currently active pattern from schedule.

        Returns:
            Current pattern or None if no schedule
        """
        if not self._schedule or self._current_schedule_index >= len(self._schedule):
            return None
        return self._schedule[self._current_schedule_index].pattern

    def _check_schedule(self) -> None:
        """Check and handle schedule transitions."""
        if not self._schedule:
            return

        if self._current_schedule_index >= len(self._schedule):
            return

        entry = self._schedule[self._current_schedule_index]

        # Check if current pattern duration is complete
        if entry.duration > 0:
            elapsed = self._current_time - self._schedule_start_time
            if elapsed >= entry.duration:
                # Check repeats
                self._schedule_repeat_count += 1
                if self._schedule_repeat_count >= entry.repeat:
                    # Move to next pattern
                    self._current_schedule_index += 1
                    self._schedule_repeat_count = 0

                self._schedule_start_time = self._current_time

                # Notify of pattern change
                if self._on_pattern_change and self._current_schedule_index < len(
                    self._schedule
                ):
                    self._on_pattern_change(
                        self._schedule[self._current_schedule_index].pattern
                    )

    def set_pattern_change_callback(
        self, callback: Callable[[PatternGenerator], None]
    ) -> None:
        """Set callback for pattern changes.

        Args:
            callback: Function called when pattern changes
        """
        self._on_pattern_change = callback

    def set_time_tick_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback for time ticks.

        Args:
            callback: Function called on each tick with current time
        """
        self._on_time_tick = callback

    def get_schedule_progress(self) -> tuple[int, int, float]:
        """Get current schedule progress.

        Returns:
            (current_index, total_entries, entry_progress)
        """
        if not self._schedule:
            return (0, 0, 0.0)

        if self._current_schedule_index >= len(self._schedule):
            return (len(self._schedule), len(self._schedule), 1.0)

        entry = self._schedule[self._current_schedule_index]
        if entry.duration > 0:
            elapsed = self._current_time - self._schedule_start_time
            progress = elapsed / entry.duration
        else:
            progress = 0.0

        return (self._current_schedule_index, len(self._schedule), min(1.0, progress))
