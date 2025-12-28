"""SMPL model wrapper for body mesh generation."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

from body_sim.core.types import BodyMesh, Gender, SMPLParameters
from body_sim.core.constants import SMPL_NUM_VERTICES


class SMPLInterface(ABC):
    """Abstract interface for SMPL model implementations."""

    @abstractmethod
    def load_model(self, gender: Gender = Gender.NEUTRAL) -> None:
        """Load SMPL model for specified gender."""
        pass

    @abstractmethod
    def forward(
        self,
        betas: torch.Tensor,
        pose: torch.Tensor,
        trans: Optional[torch.Tensor] = None,
    ) -> BodyMesh:
        """Generate body mesh from parameters."""
        pass

    @abstractmethod
    def get_vertex_weights(self) -> NDArray[np.float32]:
        """Get per-vertex body part weights for region mapping."""
        pass


class SMPLPytorchWrapper(SMPLInterface):
    """SMPL wrapper using smplpytorch library.

    This provides Python 3 compatible SMPL model loading and mesh generation.
    """

    def __init__(
        self,
        model_root: str | Path,
        device: str = "cpu",
    ):
        """Initialize SMPL wrapper.

        Args:
            model_root: Path to directory containing SMPL model files
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.model_root = Path(model_root)
        self.device = torch.device(device)
        self.model: Optional[object] = None
        self.gender: Optional[Gender] = None
        self._faces: Optional[NDArray[np.int64]] = None

    def load_model(self, gender: Gender = Gender.NEUTRAL) -> None:
        """Load SMPL model for specified gender.

        Args:
            gender: Body gender (male, female, or neutral)
        """
        try:
            from smplpytorch.pytorch.smpl_layer import SMPL_Layer
        except ImportError as e:
            raise ImportError(
                "smplpytorch is required. Install with: pip install smplpytorch"
            ) from e

        self.gender = gender
        self.model = SMPL_Layer(
            center_idx=0,
            gender=gender.value,
            model_root=str(self.model_root),
        ).to(self.device)

        # Cache faces
        self._faces = np.array(self.model.faces, dtype=np.int64)

    def forward(
        self,
        betas: torch.Tensor,
        pose: torch.Tensor,
        trans: Optional[torch.Tensor] = None,
    ) -> BodyMesh:
        """Generate body mesh from SMPL parameters.

        Args:
            betas: Shape parameters (1, 10)
            pose: Pose parameters (1, 72) including global orientation
            trans: Translation (1, 3), optional

        Returns:
            BodyMesh with vertices, faces, and joints
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure correct shapes and device
        if betas.dim() == 1:
            betas = betas.unsqueeze(0)
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)

        betas = betas.to(self.device).float()
        pose = pose.to(self.device).float()

        # Pad pose to 72 if only 69 provided (add global orientation)
        if pose.shape[1] == 69:
            global_orient = torch.zeros(pose.shape[0], 3, device=self.device)
            pose = torch.cat([global_orient, pose], dim=1)

        # Generate mesh
        verts, joints = self.model(pose, betas)

        # Apply translation if provided
        if trans is not None:
            if trans.dim() == 1:
                trans = trans.unsqueeze(0)
            trans = trans.to(self.device).float()
            verts = verts + trans.unsqueeze(1)
            joints = joints + trans.unsqueeze(1)

        # Convert to numpy
        vertices = verts.cpu().numpy().squeeze().astype(np.float32)
        joints_np = joints.cpu().numpy().squeeze().astype(np.float32)

        # Create parameters object
        params = SMPLParameters(
            betas=betas.cpu().numpy().squeeze(),
            pose=pose[:, 3:].cpu().numpy().squeeze(),  # Remove global orient
            trans=trans.cpu().numpy().squeeze() if trans is not None else np.zeros(3),
            gender=self.gender or Gender.NEUTRAL,
        )

        return BodyMesh(
            vertices=vertices,
            faces=self._faces,
            joints=joints_np,
            parameters=params,
        )

    def get_vertex_weights(self) -> NDArray[np.float32]:
        """Get per-vertex weights based on distance from joints.

        Returns:
            Array of shape (6890,) with normalized weights
        """
        # Default uniform weights
        return np.ones(SMPL_NUM_VERTICES, dtype=np.float32) / SMPL_NUM_VERTICES

    def generate_from_params(self, params: SMPLParameters) -> BodyMesh:
        """Generate mesh from SMPLParameters dataclass.

        Args:
            params: SMPL parameters

        Returns:
            Generated body mesh
        """
        if self.gender != params.gender:
            self.load_model(params.gender)

        betas = torch.from_numpy(params.betas).float()
        pose = torch.from_numpy(params.pose).float()
        trans = torch.from_numpy(params.trans).float()

        return self.forward(betas, pose, trans)


class MockSMPLWrapper(SMPLInterface):
    """Mock SMPL wrapper for testing without actual model files.

    Generates a simple ellipsoid mesh approximating a human body.
    """

    def __init__(self):
        self.gender: Optional[Gender] = None
        self._vertices: Optional[NDArray[np.float32]] = None
        self._faces: Optional[NDArray[np.int64]] = None

    def load_model(self, gender: Gender = Gender.NEUTRAL) -> None:
        """Mock model loading."""
        self.gender = gender
        self._generate_mock_mesh()

    def _generate_mock_mesh(self) -> None:
        """Generate a simple ellipsoid mesh."""
        # Create ellipsoid vertices
        n_lat, n_lon = 50, 50
        phi = np.linspace(0, np.pi, n_lat)
        theta = np.linspace(0, 2 * np.pi, n_lon)
        phi, theta = np.meshgrid(phi, theta)

        # Ellipsoid dimensions (approximate human body lying down)
        a, b, c = 0.25, 0.15, 0.9  # width, depth, height

        x = a * np.sin(phi) * np.cos(theta)
        y = b * np.sin(phi) * np.sin(theta)
        z = c * np.cos(phi)

        self._vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(
            np.float32
        )

        # Generate faces (simple triangulation)
        faces = []
        for i in range(n_lat - 1):
            for j in range(n_lon - 1):
                idx = i * n_lon + j
                faces.append([idx, idx + 1, idx + n_lon])
                faces.append([idx + 1, idx + n_lon + 1, idx + n_lon])
        self._faces = np.array(faces, dtype=np.int64)

    def forward(
        self,
        betas: torch.Tensor,
        pose: torch.Tensor,
        trans: Optional[torch.Tensor] = None,
    ) -> BodyMesh:
        """Generate mock mesh."""
        if self._vertices is None:
            self._generate_mock_mesh()

        vertices = self._vertices.copy()

        # Apply translation
        if trans is not None:
            trans_np = trans.cpu().numpy().squeeze()
            vertices += trans_np

        # Mock joints (24 points along body axis)
        joints = np.zeros((24, 3), dtype=np.float32)
        for i in range(24):
            joints[i, 2] = 0.9 - (i / 24) * 1.8  # Distribute along z

        params = SMPLParameters(
            betas=betas.cpu().numpy().squeeze() if betas.dim() > 0 else np.zeros(10),
            pose=pose.cpu().numpy().squeeze()[:69]
            if pose.numel() >= 69
            else np.zeros(69),
            trans=trans.cpu().numpy().squeeze() if trans is not None else np.zeros(3),
            gender=self.gender or Gender.NEUTRAL,
        )

        return BodyMesh(
            vertices=vertices,
            faces=self._faces,
            joints=joints,
            parameters=params,
        )

    def get_vertex_weights(self) -> NDArray[np.float32]:
        """Get uniform vertex weights."""
        n_verts = self._vertices.shape[0] if self._vertices is not None else 2500
        return np.ones(n_verts, dtype=np.float32) / n_verts
