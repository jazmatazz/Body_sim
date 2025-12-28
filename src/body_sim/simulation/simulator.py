"""Main simulation orchestrator for pressure point simulation."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from numpy.typing import NDArray

from body_sim.core.types import (
    BodyMesh,
    PressureDistribution,
    SimulationConfig,
    SimulationState,
    SMPLParameters,
)
from body_sim.body.smpl_wrapper import SMPLPytorchWrapper, MockSMPLWrapper, SMPLInterface
from body_sim.body.mesh_processor import position_body_on_mattress
from body_sim.mattress.grid import AirCellGrid
from body_sim.mattress.patterns import PatternGenerator, StaticPattern
from body_sim.physics.contact_detector import ContactDetector
from body_sim.physics.pressure_calculator import PressureCalculator


class PressureSimulator:
    """Main simulation orchestrator.

    Coordinates body mesh, mattress grid, pattern application,
    and pressure calculation.

    Attributes:
        config: Simulation configuration
        grid: Air cell mattress grid
        smpl: SMPL model wrapper
        pattern: Current pattern generator
        current_mesh: Current body mesh
        simulation_time: Current simulation time
        history: List of pressure distributions over time
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        smpl_model_path: Optional[str | Path] = None,
        use_mock_smpl: bool = False,
        device: str = "cpu",
    ):
        """Initialize pressure simulator.

        Args:
            config: Simulation configuration (uses defaults if None)
            smpl_model_path: Path to SMPL model directory
            use_mock_smpl: Use mock SMPL for testing
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.config = config or SimulationConfig()

        # Initialize mattress grid
        self.grid = AirCellGrid.from_config(self.config)

        # Initialize SMPL wrapper
        if use_mock_smpl:
            self._smpl: SMPLInterface = MockSMPLWrapper()
        else:
            model_path = smpl_model_path or "./models/smpl"
            self._smpl = SMPLPytorchWrapper(model_root=model_path, device=device)

        # Initialize physics
        self._contact_detector = ContactDetector(self.grid)
        self._pressure_calc = PressureCalculator(
            body_weight_kg=self.config.body_weight,
            grid=self.grid,
        )

        # State
        self._pattern: PatternGenerator = StaticPattern(height=0.5)
        self._current_mesh: Optional[BodyMesh] = None
        self._simulation_time = 0.0
        self._history: list[PressureDistribution] = []

    @property
    def pattern(self) -> PatternGenerator:
        """Current pattern generator."""
        return self._pattern

    @property
    def current_mesh(self) -> Optional[BodyMesh]:
        """Current body mesh."""
        return self._current_mesh

    @property
    def simulation_time(self) -> float:
        """Current simulation time in seconds."""
        return self._simulation_time

    @property
    def history(self) -> list[PressureDistribution]:
        """List of pressure distributions from simulation."""
        return self._history.copy()

    def load_body(
        self,
        params: SMPLParameters,
        position_on_mattress: bool = True,
        bed_angle: float = 0.0,
    ) -> BodyMesh:
        """Load body with SMPL parameters.

        Args:
            params: SMPL parameters for body generation
            position_on_mattress: Whether to position body on mattress
            bed_angle: Head elevation angle in radians

        Returns:
            Generated body mesh
        """
        # Load model for gender if needed
        self._smpl.load_model(params.gender)

        # Generate mesh
        mesh = self._smpl.generate_from_params(params)

        # Position on mattress
        if position_on_mattress:
            mesh = position_body_on_mattress(mesh, self.config, bed_angle)

        self._current_mesh = mesh
        return mesh

    def load_body_from_pose(
        self,
        pose_name: str,
        betas: Optional[NDArray[np.float32]] = None,
        position_on_mattress: bool = True,
    ) -> BodyMesh:
        """Load body using predefined pose.

        Args:
            pose_name: Name of pose from pose library
            betas: Optional shape parameters (zeros if None)
            position_on_mattress: Whether to position on mattress

        Returns:
            Generated body mesh
        """
        from body_sim.body.pose_library import get_pose
        from body_sim.core.types import Gender

        pose = get_pose(pose_name)
        params = SMPLParameters(
            betas=betas if betas is not None else np.zeros(10, dtype=np.float32),
            pose=pose,
            trans=np.zeros(3, dtype=np.float32),
            gender=Gender.NEUTRAL,
        )

        return self.load_body(params, position_on_mattress)

    def set_pattern(self, pattern: PatternGenerator) -> None:
        """Set the mattress height pattern.

        Args:
            pattern: Pattern generator to use
        """
        self._pattern = pattern

    def set_body_weight(self, weight_kg: float) -> None:
        """Update body weight.

        Args:
            weight_kg: New body weight in kg
        """
        self.config.body_weight = weight_kg
        self._pressure_calc = PressureCalculator(
            body_weight_kg=weight_kg,
            grid=self.grid,
        )

    def step(self) -> PressureDistribution:
        """Advance simulation by one time step.

        Updates pattern phase, applies to grid, detects contacts,
        and calculates pressure distribution.

        Returns:
            Pressure distribution for current step

        Raises:
            RuntimeError: If no body mesh is loaded
        """
        if self._current_mesh is None:
            raise RuntimeError("No body loaded. Call load_body() first.")

        # Update pattern phase and apply to grid
        cycle_duration = self._pattern.get_cycle_duration()
        phase = (self._simulation_time % cycle_duration) / cycle_duration

        heights = self._pattern.generate(self.grid.rows, self.grid.cols, phase)
        self.grid.set_normalized_pattern(heights)

        # Detect contacts
        contacts = self._contact_detector.detect_contacts(self._current_mesh)

        # Calculate pressure
        pressure_dist = self._pressure_calc.calculate_pressure(
            self._current_mesh, contacts
        )
        pressure_dist.timestamp = self._simulation_time

        # Record history
        self._history.append(pressure_dist)

        # Advance time
        self._simulation_time += self.config.time_step

        return pressure_dist

    def run(
        self,
        duration: float,
        callback: Optional[Callable[[PressureDistribution], None]] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list[PressureDistribution]:
        """Run simulation for specified duration.

        Args:
            duration: Simulation duration in seconds
            callback: Optional callback for each step
            progress_callback: Optional progress callback (0-1)

        Returns:
            List of pressure distributions for entire simulation
        """
        results = []
        start_time = self._simulation_time
        end_time = start_time + duration

        while self._simulation_time < end_time:
            pressure = self.step()
            results.append(pressure)

            if callback:
                callback(pressure)

            if progress_callback:
                progress = (self._simulation_time - start_time) / duration
                progress_callback(min(1.0, progress))

        return results

    def get_current_state(self) -> SimulationState:
        """Get complete current simulation state.

        Returns:
            SimulationState snapshot
        """
        contacts = []
        pressure_dist = None

        if self._current_mesh is not None:
            contacts = self._contact_detector.detect_contacts(self._current_mesh)
            pressure_dist = self._pressure_calc.calculate_pressure(
                self._current_mesh, contacts
            )
            pressure_dist.timestamp = self._simulation_time

        return SimulationState(
            time=self._simulation_time,
            body_mesh=self._current_mesh,
            cell_heights=self.grid.heights,
            pressure_distribution=pressure_dist,
            contacts=contacts,
        )

    def reset(self, clear_history: bool = True) -> None:
        """Reset simulation state.

        Args:
            clear_history: Whether to clear pressure history
        """
        self._simulation_time = 0.0
        self.grid.reset()

        if clear_history:
            self._history.clear()

    def get_pressure_summary(self) -> dict:
        """Get summary statistics from simulation history.

        Returns:
            Dictionary with summary statistics
        """
        if not self._history:
            return {
                "num_steps": 0,
                "duration": 0.0,
                "max_peak_pressure": 0.0,
                "avg_peak_pressure": 0.0,
                "avg_average_pressure": 0.0,
                "avg_contact_area": 0.0,
            }

        peaks = [p.peak_pressure for p in self._history]
        avgs = [p.average_pressure for p in self._history]
        areas = [p.contact_area for p in self._history]

        return {
            "num_steps": len(self._history),
            "duration": self._simulation_time,
            "max_peak_pressure": max(peaks),
            "avg_peak_pressure": np.mean(peaks),
            "avg_average_pressure": np.mean(avgs),
            "avg_contact_area": np.mean(areas),
        }
