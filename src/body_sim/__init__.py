"""
Body Pressure Simulator

A simulator for analyzing pressure distribution on adjustable grid air mattresses
using SMPL body models from the bodies-at-rest project.

Main components:
- body: SMPL model wrapper and pose library
- mattress: Configurable air cell grid and patterns
- physics: Contact detection and pressure calculation
- simulation: Main simulator orchestrator
- visualization: Pressure maps and 3D rendering
- metrics: Pressure, coverage, and temporal analysis

Quick start:
    from body_sim.simulation import PressureSimulator
    from body_sim.mattress.patterns import AlternatingPattern
    from body_sim.core.types import SimulationConfig

    config = SimulationConfig(grid_rows=16, grid_cols=32)
    sim = PressureSimulator(config, use_mock_smpl=True)
    sim.load_body_from_pose("supine")
    sim.set_pattern(AlternatingPattern(cycle_duration=10.0))
    results = sim.run(duration=60.0)
"""

__version__ = "0.1.0"
__author__ = "Jasper Metz"

# Core types
from body_sim.core.types import (
    Gender,
    BodyRegion,
    SMPLParameters,
    BodyMesh,
    CellState,
    ContactPoint,
    PressureDistribution,
    SimulationConfig,
    SimulationState,
)

# Main simulator
from body_sim.simulation import PressureSimulator

# Patterns
from body_sim.mattress.patterns import (
    PatternGenerator,
    AlternatingPattern,
    WavePattern,
    ZonePattern,
    StaticPattern,
)

# Mattress grid
from body_sim.mattress import AirCellGrid

__all__ = [
    # Version
    "__version__",
    # Core types
    "Gender",
    "BodyRegion",
    "SMPLParameters",
    "BodyMesh",
    "CellState",
    "ContactPoint",
    "PressureDistribution",
    "SimulationConfig",
    "SimulationState",
    # Simulator
    "PressureSimulator",
    # Patterns
    "PatternGenerator",
    "AlternatingPattern",
    "WavePattern",
    "ZonePattern",
    "StaticPattern",
    # Grid
    "AirCellGrid",
]
