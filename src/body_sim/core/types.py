"""Core type definitions for the body pressure simulator."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class Gender(str, Enum):
    """Gender options for SMPL model."""

    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class BodyRegion(str, Enum):
    """Body regions for pressure analysis."""

    HEAD = "head"
    SHOULDERS = "shoulders"
    UPPER_BACK = "upper_back"
    LOWER_BACK = "lower_back"
    SACRUM = "sacrum"
    HIPS = "hips"
    THIGHS = "thighs"
    CALVES = "calves"
    HEELS = "heels"
    FULL_BODY = "full_body"


@dataclass
class SMPLParameters:
    """SMPL model parameters for body generation.

    Attributes:
        betas: Shape parameters (10,) controlling body morphology
        pose: Pose parameters (69,) for 23 joints × 3 axis-angle
        trans: Global translation (3,)
        gender: Body gender for model selection
    """

    betas: NDArray[np.float32] = field(default_factory=lambda: np.zeros(10, dtype=np.float32))
    pose: NDArray[np.float32] = field(default_factory=lambda: np.zeros(69, dtype=np.float32))
    trans: NDArray[np.float32] = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    gender: Gender = Gender.NEUTRAL

    def __post_init__(self) -> None:
        """Validate parameter shapes."""
        self.betas = np.asarray(self.betas, dtype=np.float32)
        self.pose = np.asarray(self.pose, dtype=np.float32)
        self.trans = np.asarray(self.trans, dtype=np.float32)

        if self.betas.shape != (10,):
            raise ValueError(f"betas must have shape (10,), got {self.betas.shape}")
        if self.pose.shape != (69,):
            raise ValueError(f"pose must have shape (69,), got {self.pose.shape}")
        if self.trans.shape != (3,):
            raise ValueError(f"trans must have shape (3,), got {self.trans.shape}")


@dataclass
class BodyMesh:
    """Generated body mesh from SMPL model.

    Attributes:
        vertices: Mesh vertices (6890, 3)
        faces: Triangle face indices (13776, 3)
        joints: Joint positions (24, 3)
        parameters: Original SMPL parameters used to generate mesh
    """

    vertices: NDArray[np.float32]
    faces: NDArray[np.int64]
    joints: NDArray[np.float32]
    parameters: SMPLParameters

    @property
    def num_vertices(self) -> int:
        """Number of vertices in mesh."""
        return self.vertices.shape[0]

    @property
    def bounds(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Get axis-aligned bounding box (min, max)."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


@dataclass
class CellState:
    """State of a single air cell in the mattress.

    Attributes:
        height: Current inflation height in meters (0.0 to max_height)
        pressure: Internal air pressure (normalized 0-1)
        enabled: Whether cell is active or disabled
    """

    height: float
    pressure: float = 0.5
    enabled: bool = True


@dataclass
class ContactPoint:
    """Contact between a body vertex and mattress cell.

    Attributes:
        vertex_idx: Index of contacting body mesh vertex
        cell_idx: Grid cell indices (row, col)
        penetration_depth: How far vertex penetrates below cell surface (meters)
        normal_force: Estimated normal force at contact (Newtons)
    """

    vertex_idx: int
    cell_idx: tuple[int, int]
    penetration_depth: float
    normal_force: float = 0.0


@dataclass
class PressureDistribution:
    """Pressure map over the mattress grid.

    Attributes:
        grid_pressure: Pressure values per cell (rows, cols) in Pascals
        peak_pressure: Maximum pressure value
        average_pressure: Mean pressure over contact area
        contact_area: Fraction of grid in contact (0-1)
        timestamp: Simulation time when captured
    """

    grid_pressure: NDArray[np.float32]
    peak_pressure: float
    average_pressure: float
    contact_area: float
    timestamp: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        """Grid dimensions (rows, cols)."""
        return self.grid_pressure.shape  # type: ignore

    def get_region_pressure(self, row_start: int, row_end: int) -> float:
        """Get average pressure for a row region."""
        region = self.grid_pressure[row_start:row_end, :]
        mask = region > 0
        if not mask.any():
            return 0.0
        return float(region[mask].mean())


@dataclass
class SimulationConfig:
    """Complete simulation configuration.

    Attributes:
        mattress_width: Physical width in meters
        mattress_height: Physical length in meters
        grid_rows: Number of cell rows (along length)
        grid_cols: Number of cell columns (along width)
        cell_max_height: Maximum cell inflation height in meters
        time_step: Simulation time step in seconds
        body_weight: Subject body weight in kg
    """

    # Mattress dimensions (default: standard hospital bed)
    mattress_width: float = 0.762  # 30 inches
    mattress_height: float = 1.854  # 73 inches

    # Grid resolution
    grid_rows: int = 16
    grid_cols: int = 32

    # Cell properties
    cell_max_height: float = 0.127  # 5 inches
    cell_min_height: float = 0.025  # 1 inch

    # Simulation parameters
    time_step: float = 0.1  # seconds
    body_weight: float = 80.0  # kg

    @property
    def cell_width(self) -> float:
        """Width of each cell in meters."""
        return self.mattress_width / self.grid_cols

    @property
    def cell_length(self) -> float:
        """Length of each cell in meters."""
        return self.mattress_height / self.grid_rows

    @property
    def cell_area(self) -> float:
        """Area of each cell in square meters."""
        return self.cell_width * self.cell_length


@dataclass
class SimulationState:
    """Complete state of simulation at a point in time.

    Attributes:
        time: Current simulation time
        body_mesh: Current body mesh
        cell_heights: Current mattress cell heights
        pressure_distribution: Current pressure distribution
        contacts: List of contact points
    """

    time: float
    body_mesh: Optional[BodyMesh]
    cell_heights: NDArray[np.float32]
    pressure_distribution: Optional[PressureDistribution]
    contacts: list[ContactPoint] = field(default_factory=list)
