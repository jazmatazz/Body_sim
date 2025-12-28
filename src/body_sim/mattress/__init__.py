"""Mattress module for air cell grid and pattern management."""

from body_sim.mattress.grid import AirCellGrid
from body_sim.mattress.patterns import (
    PatternGenerator,
    AlternatingPattern,
    WavePattern,
    ZonePattern,
    ZoneConfig,
    RowAlternatingPattern,
    StaticPattern,
    CustomPattern,
    PATTERN_TYPES,
    create_pattern,
)
from body_sim.mattress.pattern_loader import (
    load_pattern,
    load_pattern_config,
    pattern_from_config,
    save_pattern_config,
)

__all__ = [
    # Grid
    "AirCellGrid",
    # Patterns
    "PatternGenerator",
    "AlternatingPattern",
    "WavePattern",
    "ZonePattern",
    "ZoneConfig",
    "RowAlternatingPattern",
    "StaticPattern",
    "CustomPattern",
    "PATTERN_TYPES",
    "create_pattern",
    # Pattern I/O
    "load_pattern",
    "load_pattern_config",
    "pattern_from_config",
    "save_pattern_config",
]
