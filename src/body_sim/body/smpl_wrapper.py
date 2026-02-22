"""SMPL model wrapper for body mesh generation using smplx library."""

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


class SMPLXWrapper(SMPLInterface):
    """SMPL wrapper using smplx library.

    This provides Python 3.10+ compatible SMPL model loading and mesh generation.
    Uses the smplx library which doesn't require chumpy.
    """

    # Model filename patterns
    MODEL_FILES = {
        Gender.MALE: [
            "SMPL_MALE.pkl",
            "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
            "basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
        ],
        Gender.FEMALE: [
            "SMPL_FEMALE.pkl",
            "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
            "basicmodel_f_lbs_10_207_0_v1.0.0.pkl",
        ],
        Gender.NEUTRAL: [
            "SMPL_NEUTRAL.pkl",
            "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
            "basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl",
        ],
    }

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
        self.model = None
        self.gender: Optional[Gender] = None
        self._faces: Optional[NDArray[np.int64]] = None

    def _find_model_file(self, gender: Gender) -> Optional[Path]:
        """Find the model file for a given gender."""
        for filename in self.MODEL_FILES[gender]:
            filepath = self.model_root / filename
            if filepath.exists():
                return filepath
        return None

    def _prepare_model_files(self) -> None:
        """Prepare model files in the structure smplx expects.

        smplx expects: {model_path}/smpl/SMPL_MALE.pkl, etc.
        So we create a 'smpl' subdirectory with symlinks.
        """
        # Create smpl subdirectory if needed
        smpl_dir = self.model_root / "smpl"
        smpl_dir.mkdir(exist_ok=True)

        renames = {
            Gender.MALE: "SMPL_MALE.pkl",
            Gender.FEMALE: "SMPL_FEMALE.pkl",
            Gender.NEUTRAL: "SMPL_NEUTRAL.pkl",
        }

        for gender, target_name in renames.items():
            target_path = smpl_dir / target_name
            if target_path.exists():
                continue

            # Find source file in parent directory
            source_file = self._find_model_file(gender)
            if source_file:
                try:
                    # Create relative symlink from smpl/ to parent
                    target_path.symlink_to(f"../{source_file.name}")
                except OSError:
                    import shutil
                    shutil.copy2(source_file, target_path)

    def load_model(self, gender: Gender = Gender.NEUTRAL) -> None:
        """Load SMPL model for specified gender.

        Args:
            gender: Body gender (male, female, or neutral)
        """
        try:
            import smplx
        except ImportError as e:
            raise ImportError(
                "smplx is required. Install with: pip install smplx"
            ) from e

        # Prepare model files (create symlinks if needed)
        self._prepare_model_files()

        self.gender = gender

        # smplx expects gender as string
        gender_str = gender.value

        # Create SMPL model
        self.model = smplx.create(
            model_path=str(self.model_root),
            model_type='smpl',
            gender=gender_str,
            batch_size=1,
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_transl=True,
        ).to(self.device)

        # Cache faces
        self._faces = self.model.faces.astype(np.int64)

    def forward(
        self,
        betas: torch.Tensor,
        pose: torch.Tensor,
        trans: Optional[torch.Tensor] = None,
    ) -> BodyMesh:
        """Generate body mesh from SMPL parameters.

        Args:
            betas: Shape parameters (1, 10) or (10,)
            pose: Pose parameters (1, 69) or (69,) - body pose only
            trans: Translation (1, 3) or (3,), optional

        Returns:
            BodyMesh with vertices, faces, and joints
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure correct shapes
        if betas.dim() == 1:
            betas = betas.unsqueeze(0)
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)

        betas = betas.to(self.device).float()
        pose = pose.to(self.device).float()

        # smplx expects body_pose to be (batch, 69) for SMPL
        # and global_orient to be (batch, 3)
        if pose.shape[1] >= 72:
            global_orient = pose[:, :3]
            body_pose = pose[:, 3:72]
        elif pose.shape[1] == 69:
            global_orient = torch.zeros(pose.shape[0], 3, device=self.device)
            body_pose = pose
        else:
            # Pad if needed
            global_orient = torch.zeros(pose.shape[0], 3, device=self.device)
            body_pose = torch.zeros(pose.shape[0], 69, device=self.device)
            body_pose[:, :pose.shape[1]] = pose

        # Handle translation
        if trans is not None:
            if trans.dim() == 1:
                trans = trans.unsqueeze(0)
            trans = trans.to(self.device).float()
        else:
            trans = torch.zeros(1, 3, device=self.device)

        # Generate mesh
        output = self.model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=trans,
            return_verts=True,
        )

        vertices = output.vertices.detach().cpu().numpy().squeeze().astype(np.float32)
        joints = output.joints.detach().cpu().numpy().squeeze().astype(np.float32)

        # Create parameters object
        params = SMPLParameters(
            betas=betas.cpu().numpy().squeeze(),
            pose=body_pose.cpu().numpy().squeeze(),
            trans=trans.cpu().numpy().squeeze(),
            gender=self.gender or Gender.NEUTRAL,
        )

        return BodyMesh(
            vertices=vertices,
            faces=self._faces,
            joints=joints[:24],  # SMPL has 24 joints
            parameters=params,
        )

    def get_vertex_weights(self) -> NDArray[np.float32]:
        """Get per-vertex weights based on distance from joints.

        Returns:
            Array of shape (6890,) with normalized weights
        """
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


# Alias for backwards compatibility
SMPLPytorchWrapper = SMPLXWrapper


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
        """Generate a simple ellipsoid mesh approximating a lying human."""
        # Create a more human-like shape with segments
        vertices = []
        faces = []

        # Body segments: head, torso, arms, legs
        # Main torso (elongated ellipsoid)
        n_lat, n_lon = 20, 20

        for segment in ['torso', 'head', 'left_leg', 'right_leg']:
            if segment == 'torso':
                a, b, c = 0.20, 0.12, 0.50  # width, depth, length
                offset = np.array([0.0, 0.0, 0.0])
            elif segment == 'head':
                a, b, c = 0.10, 0.10, 0.12
                offset = np.array([0.0, 0.0, 0.55])
            elif segment == 'left_leg':
                a, b, c = 0.08, 0.08, 0.45
                offset = np.array([-0.10, 0.0, -0.70])
            else:  # right_leg
                a, b, c = 0.08, 0.08, 0.45
                offset = np.array([0.10, 0.0, -0.70])

            phi = np.linspace(0, np.pi, n_lat)
            theta = np.linspace(0, 2 * np.pi, n_lon)
            phi, theta = np.meshgrid(phi, theta)

            x = a * np.sin(phi) * np.cos(theta) + offset[0]
            y = b * np.sin(phi) * np.sin(theta) + offset[1]
            z = c * np.cos(phi) + offset[2]

            base_idx = len(vertices)
            segment_verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
            vertices.extend(segment_verts)

            # Generate faces for this segment
            for i in range(n_lat - 1):
                for j in range(n_lon - 1):
                    idx = base_idx + i * n_lon + j
                    faces.append([idx, idx + 1, idx + n_lon])
                    faces.append([idx + 1, idx + n_lon + 1, idx + n_lon])

        self._vertices = np.array(vertices, dtype=np.float32)
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
        # Distribute joints along the body
        joint_z = np.linspace(0.6, -0.9, 24)
        joints[:, 2] = joint_z

        params = SMPLParameters(
            betas=betas.cpu().numpy().squeeze() if betas.dim() > 0 else np.zeros(10),
            pose=pose.cpu().numpy().squeeze()[:69] if pose.numel() >= 69 else np.zeros(69),
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
        n_verts = self._vertices.shape[0] if self._vertices is not None else 1600
        return np.ones(n_verts, dtype=np.float32) / n_verts

    def generate_from_params(self, params: SMPLParameters) -> BodyMesh:
        """Generate mesh from parameters."""
        if self.gender != params.gender:
            self.load_model(params.gender)

        betas = torch.from_numpy(params.betas).float()
        pose = torch.from_numpy(params.pose).float()
        trans = torch.from_numpy(params.trans).float()

        return self.forward(betas, pose, trans)
