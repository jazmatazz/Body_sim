"""YAML configuration loader for mattress patterns."""

from pathlib import Path
from typing import Any, Union

import yaml
import numpy as np

from body_sim.mattress.patterns import (
    PatternGenerator,
    AlternatingPattern,
    WavePattern,
    ZonePattern,
    ZoneConfig,
    RowAlternatingPattern,
    StaticPattern,
    CustomPattern,
    create_pattern,
)


def load_pattern_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load pattern configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Parsed configuration dictionary
    """
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pattern_from_config(config: dict[str, Any]) -> PatternGenerator:
    """Create pattern generator from configuration dictionary.

    Args:
        config: Configuration dictionary with 'pattern' key

    Returns:
        Configured pattern generator
    """
    pattern_config = config.get("pattern", config)
    pattern_type = pattern_config.get("type", "alternating")

    if pattern_type == "alternating":
        return _create_alternating_from_config(pattern_config)
    elif pattern_type == "wave":
        return _create_wave_from_config(pattern_config)
    elif pattern_type == "zone":
        return _create_zone_from_config(pattern_config)
    elif pattern_type == "row_alternating":
        return _create_row_alternating_from_config(pattern_config)
    elif pattern_type == "static":
        return _create_static_from_config(pattern_config)
    elif pattern_type == "custom":
        return _create_custom_from_config(pattern_config)
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")


def _create_alternating_from_config(config: dict) -> AlternatingPattern:
    """Create AlternatingPattern from config."""
    cycle_duration = config.get("cycle_duration", 10.0)

    # Handle levels specification
    levels = config.get("levels", {})
    high_height = levels.get("high", config.get("high_height", 1.0))
    low_height = levels.get("low", config.get("low_height", 0.3))

    transition = config.get("transition_time", 0.0)
    smoothness = transition / cycle_duration if cycle_duration > 0 else 0

    return AlternatingPattern(
        cycle_duration=cycle_duration,
        high_height=high_height,
        low_height=low_height,
        transition_smoothness=smoothness,
    )


def _create_wave_from_config(config: dict) -> WavePattern:
    """Create WavePattern from config."""
    wave_config = config.get("wave", {})

    return WavePattern(
        cycle_duration=config.get("cycle_duration", 20.0),
        wavelength=wave_config.get("wavelength", 4.0),
        direction=wave_config.get("direction", "head_to_foot"),
        amplitude=wave_config.get("amplitude", 0.35),
        baseline=wave_config.get("baseline", 0.65),
    )


def _create_zone_from_config(config: dict) -> ZonePattern:
    """Create ZonePattern from config."""
    zones_config = config.get("zones", {})
    zones = {}

    for zone_name, zone_def in zones_config.items():
        zones[zone_name] = ZoneConfig(
            row_start=zone_def.get("row_start", 0.0),
            row_end=zone_def.get("row_end", 1.0),
            pattern_type=zone_def.get("type", "constant"),
            height=zone_def.get("height", 0.5),
            phase_offset=zone_def.get("phase_offset", 0.0),
        )

    return ZonePattern(
        zones=zones,
        cycle_duration=config.get("cycle_duration", 30.0),
    )


def _create_row_alternating_from_config(config: dict) -> RowAlternatingPattern:
    """Create RowAlternatingPattern from config."""
    levels = config.get("levels", {})

    return RowAlternatingPattern(
        cycle_duration=config.get("cycle_duration", 15.0),
        row_group_size=config.get("row_group_size", 2),
        high_height=levels.get("high", 1.0),
        low_height=levels.get("low", 0.3),
    )


def _create_static_from_config(config: dict) -> StaticPattern:
    """Create StaticPattern from config."""
    height = config.get("height", 0.5)

    # Check for pattern array
    pattern_data = config.get("pattern_data")
    if pattern_data is not None:
        pattern = np.array(pattern_data, dtype=np.float32)
        return StaticPattern(pattern=pattern)

    return StaticPattern(height=height)


def _create_custom_from_config(config: dict) -> CustomPattern:
    """Create CustomPattern from config."""
    keyframes_config = config.get("keyframes", {})
    keyframes = {}

    for phase_str, pattern_data in keyframes_config.items():
        phase = float(phase_str)
        pattern = np.array(pattern_data, dtype=np.float32)
        keyframes[phase] = pattern

    return CustomPattern(
        keyframes=keyframes,
        cycle_duration=config.get("cycle_duration", 10.0),
        interpolation=config.get("interpolation", "linear"),
    )


def load_pattern(path: Union[str, Path]) -> PatternGenerator:
    """Load pattern from YAML file.

    Args:
        path: Path to pattern configuration file

    Returns:
        Configured pattern generator
    """
    config = load_pattern_config(path)
    return pattern_from_config(config)


def save_pattern_config(
    pattern: PatternGenerator,
    path: Union[str, Path],
    name: str = "Custom Pattern",
    description: str = "",
) -> None:
    """Save pattern configuration to YAML file.

    Args:
        pattern: Pattern generator to save
        path: Output file path
        name: Pattern name for config
        description: Pattern description
    """
    config = {
        "pattern": {
            "name": name,
            "description": description,
            "type": pattern.name.lower().replace("pattern", ""),
            "cycle_duration": pattern.get_cycle_duration(),
        }
    }

    # Add type-specific configuration
    if isinstance(pattern, AlternatingPattern):
        config["pattern"]["levels"] = {
            "high": pattern.high_height,
            "low": pattern.low_height,
        }
        config["pattern"]["transition_smoothness"] = pattern.transition_smoothness

    elif isinstance(pattern, WavePattern):
        config["pattern"]["wave"] = {
            "wavelength": pattern.wavelength,
            "direction": pattern.direction,
            "amplitude": pattern.amplitude,
            "baseline": pattern.baseline,
        }

    elif isinstance(pattern, ZonePattern):
        zones = {}
        for zone_name, zone in pattern.zones.items():
            zones[zone_name] = {
                "row_start": zone.row_start,
                "row_end": zone.row_end,
                "type": zone.pattern_type,
                "height": zone.height,
                "phase_offset": zone.phase_offset,
            }
        config["pattern"]["zones"] = zones

    path = Path(path)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
