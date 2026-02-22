"""Body module for SMPL model handling and mesh processing."""

from body_sim.body.smpl_wrapper import SMPLInterface, SMPLXWrapper, SMPLPytorchWrapper, MockSMPLWrapper
from body_sim.body.mesh_processor import (
    position_body_on_mattress,
    rotate_body,
    get_body_regions_mask,
    compute_vertex_areas,
)
from body_sim.body.pose_library import (
    get_pose,
    get_supine_pose,
    get_lateral_left_pose,
    get_lateral_right_pose,
    get_prone_pose,
    get_fowler_pose,
    POSE_LIBRARY,
)

__all__ = [
    # SMPL wrappers
    "SMPLInterface",
    "SMPLXWrapper",
    "SMPLPytorchWrapper",  # Alias for SMPLXWrapper
    "MockSMPLWrapper",
    # Mesh processing
    "position_body_on_mattress",
    "rotate_body",
    "get_body_regions_mask",
    "compute_vertex_areas",
    # Poses
    "get_pose",
    "get_supine_pose",
    "get_lateral_left_pose",
    "get_lateral_right_pose",
    "get_prone_pose",
    "get_fowler_pose",
    "POSE_LIBRARY",
]
