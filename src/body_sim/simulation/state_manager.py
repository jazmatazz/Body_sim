"""State management for simulation persistence and snapshots."""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from body_sim.core.types import (
    PressureDistribution,
    SimulationConfig,
    SimulationState,
    SMPLParameters,
)


class StateManager:
    """Manages simulation state persistence and snapshots.

    Provides functionality to save/load simulation states,
    create checkpoints, and export results.

    Attributes:
        save_dir: Directory for saving states
        auto_save_interval: Automatic save interval (0 = disabled)
    """

    def __init__(
        self,
        save_dir: Union[str, Path] = "./simulation_states",
        auto_save_interval: float = 0.0,
    ):
        """Initialize state manager.

        Args:
            save_dir: Directory for saving states
            auto_save_interval: Auto-save interval in seconds (0 = disabled)
        """
        self.save_dir = Path(save_dir)
        self.auto_save_interval = auto_save_interval
        self._last_auto_save = 0.0
        self._snapshots: dict[str, SimulationState] = {}

    def ensure_save_dir(self) -> None:
        """Create save directory if it doesn't exist."""
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, state: SimulationState, name: str) -> None:
        """Save named snapshot in memory.

        Args:
            state: Simulation state to snapshot
            name: Snapshot name
        """
        self._snapshots[name] = state

    def load_snapshot(self, name: str) -> Optional[SimulationState]:
        """Load named snapshot from memory.

        Args:
            name: Snapshot name

        Returns:
            Simulation state or None if not found
        """
        return self._snapshots.get(name)

    def list_snapshots(self) -> list[str]:
        """List available snapshot names.

        Returns:
            List of snapshot names
        """
        return list(self._snapshots.keys())

    def clear_snapshots(self) -> None:
        """Clear all in-memory snapshots."""
        self._snapshots.clear()

    def save_state_to_file(
        self,
        state: SimulationState,
        filename: Optional[str] = None,
    ) -> Path:
        """Save simulation state to file.

        Args:
            state: State to save
            filename: Optional filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        self.ensure_save_dir()

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"state_{timestamp}.pkl"

        filepath = self.save_dir / filename

        # Prepare state for serialization
        state_dict = self._state_to_dict(state)

        with open(filepath, "wb") as f:
            pickle.dump(state_dict, f)

        return filepath

    def load_state_from_file(self, filepath: Union[str, Path]) -> SimulationState:
        """Load simulation state from file.

        Args:
            filepath: Path to state file

        Returns:
            Loaded simulation state
        """
        with open(filepath, "rb") as f:
            state_dict = pickle.load(f)

        return self._dict_to_state(state_dict)

    def _state_to_dict(self, state: SimulationState) -> dict[str, Any]:
        """Convert state to serializable dictionary."""
        result = {
            "time": state.time,
            "cell_heights": state.cell_heights.tolist(),
        }

        if state.body_mesh is not None:
            result["body_mesh"] = {
                "vertices": state.body_mesh.vertices.tolist(),
                "faces": state.body_mesh.faces.tolist(),
                "joints": state.body_mesh.joints.tolist(),
                "parameters": {
                    "betas": state.body_mesh.parameters.betas.tolist(),
                    "pose": state.body_mesh.parameters.pose.tolist(),
                    "trans": state.body_mesh.parameters.trans.tolist(),
                    "gender": state.body_mesh.parameters.gender.value,
                },
            }

        if state.pressure_distribution is not None:
            result["pressure_distribution"] = {
                "grid_pressure": state.pressure_distribution.grid_pressure.tolist(),
                "peak_pressure": state.pressure_distribution.peak_pressure,
                "average_pressure": state.pressure_distribution.average_pressure,
                "contact_area": state.pressure_distribution.contact_area,
                "timestamp": state.pressure_distribution.timestamp,
            }

        result["contacts"] = [
            {
                "vertex_idx": c.vertex_idx,
                "cell_idx": c.cell_idx,
                "penetration_depth": c.penetration_depth,
                "normal_force": c.normal_force,
            }
            for c in state.contacts
        ]

        return result

    def _dict_to_state(self, data: dict[str, Any]) -> SimulationState:
        """Convert dictionary to simulation state."""
        from body_sim.core.types import BodyMesh, ContactPoint, Gender

        body_mesh = None
        if "body_mesh" in data:
            bm = data["body_mesh"]
            params = SMPLParameters(
                betas=np.array(bm["parameters"]["betas"], dtype=np.float32),
                pose=np.array(bm["parameters"]["pose"], dtype=np.float32),
                trans=np.array(bm["parameters"]["trans"], dtype=np.float32),
                gender=Gender(bm["parameters"]["gender"]),
            )
            body_mesh = BodyMesh(
                vertices=np.array(bm["vertices"], dtype=np.float32),
                faces=np.array(bm["faces"], dtype=np.int64),
                joints=np.array(bm["joints"], dtype=np.float32),
                parameters=params,
            )

        pressure_dist = None
        if "pressure_distribution" in data:
            pd = data["pressure_distribution"]
            pressure_dist = PressureDistribution(
                grid_pressure=np.array(pd["grid_pressure"], dtype=np.float32),
                peak_pressure=pd["peak_pressure"],
                average_pressure=pd["average_pressure"],
                contact_area=pd["contact_area"],
                timestamp=pd["timestamp"],
            )

        contacts = [
            ContactPoint(
                vertex_idx=c["vertex_idx"],
                cell_idx=tuple(c["cell_idx"]),
                penetration_depth=c["penetration_depth"],
                normal_force=c["normal_force"],
            )
            for c in data.get("contacts", [])
        ]

        return SimulationState(
            time=data["time"],
            body_mesh=body_mesh,
            cell_heights=np.array(data["cell_heights"], dtype=np.float32),
            pressure_distribution=pressure_dist,
            contacts=contacts,
        )

    def save_config(self, config: SimulationConfig, filename: str = "config.json") -> Path:
        """Save simulation configuration to JSON.

        Args:
            config: Configuration to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        self.ensure_save_dir()
        filepath = self.save_dir / filename

        config_dict = {
            "mattress_width": config.mattress_width,
            "mattress_height": config.mattress_height,
            "grid_rows": config.grid_rows,
            "grid_cols": config.grid_cols,
            "cell_max_height": config.cell_max_height,
            "cell_min_height": config.cell_min_height,
            "time_step": config.time_step,
            "body_weight": config.body_weight,
        }

        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)

        return filepath

    def load_config(self, filepath: Union[str, Path]) -> SimulationConfig:
        """Load simulation configuration from JSON.

        Args:
            filepath: Path to config file

        Returns:
            Loaded configuration
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        return SimulationConfig(**data)

    def export_pressure_history(
        self,
        history: list[PressureDistribution],
        filename: str = "pressure_history.csv",
    ) -> Path:
        """Export pressure history to CSV.

        Args:
            history: List of pressure distributions
            filename: Output filename

        Returns:
            Path to exported file
        """
        self.ensure_save_dir()
        filepath = self.save_dir / filename

        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "peak_pressure", "average_pressure", "contact_area"]
            )
            for pd in history:
                writer.writerow(
                    [pd.timestamp, pd.peak_pressure, pd.average_pressure, pd.contact_area]
                )

        return filepath

    def export_pressure_grids(
        self,
        history: list[PressureDistribution],
        filename: str = "pressure_grids.npz",
    ) -> Path:
        """Export pressure grid history to NPZ.

        Args:
            history: List of pressure distributions
            filename: Output filename

        Returns:
            Path to exported file
        """
        self.ensure_save_dir()
        filepath = self.save_dir / filename

        grids = np.stack([pd.grid_pressure for pd in history])
        timestamps = np.array([pd.timestamp for pd in history])

        np.savez(filepath, grids=grids, timestamps=timestamps)

        return filepath

    def should_auto_save(self, current_time: float) -> bool:
        """Check if auto-save should trigger.

        Args:
            current_time: Current simulation time

        Returns:
            True if should auto-save
        """
        if self.auto_save_interval <= 0:
            return False

        if current_time - self._last_auto_save >= self.auto_save_interval:
            self._last_auto_save = current_time
            return True

        return False
