"""Core types and constants for the body pressure simulator."""

from body_sim.core.constants import (
    GRAVITY,
    SMPL_NUM_VERTICES,
    SMPL_NUM_JOINTS,
    SMPL_NUM_BETAS,
    SMPL_NUM_POSE_PARAMS,
    DEFAULT_MATTRESS_WIDTH,
    DEFAULT_MATTRESS_HEIGHT,
    DEFAULT_CELL_MAX_HEIGHT,
    DEFAULT_GRID_ROWS,
    DEFAULT_GRID_COLS,
    CAPILLARY_CLOSING_PRESSURE_PA,
    HIGH_RISK_PRESSURE_PA,
    BODY_REGION_ROWS,
)
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

__all__ = [
    # Constants
    "GRAVITY",
    "SMPL_NUM_VERTICES",
    "SMPL_NUM_JOINTS",
    "SMPL_NUM_BETAS",
    "SMPL_NUM_POSE_PARAMS",
    "DEFAULT_MATTRESS_WIDTH",
    "DEFAULT_MATTRESS_HEIGHT",
    "DEFAULT_CELL_MAX_HEIGHT",
    "DEFAULT_GRID_ROWS",
    "DEFAULT_GRID_COLS",
    "CAPILLARY_CLOSING_PRESSURE_PA",
    "HIGH_RISK_PRESSURE_PA",
    "BODY_REGION_ROWS",
    # Types
    "Gender",
    "BodyRegion",
    "SMPLParameters",
    "BodyMesh",
    "CellState",
    "ContactPoint",
    "PressureDistribution",
    "SimulationConfig",
    "SimulationState",
]
