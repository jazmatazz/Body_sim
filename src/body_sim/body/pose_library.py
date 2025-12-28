"""Predefined body poses for clinical simulation scenarios."""

import numpy as np
from numpy.typing import NDArray


def get_supine_pose() -> NDArray[np.float32]:
    """Get SMPL pose parameters for supine position (lying on back).

    Returns:
        Pose array of shape (69,) - neutral pose (all zeros works for supine)
    """
    # For supine, we use near-zero pose (T-pose rotated to lying down)
    # The mesh processor handles the rotation from standing to lying
    pose = np.zeros(69, dtype=np.float32)

    # Slight arm adjustment to lie at sides
    # Left shoulder (joint 16): rotate slightly down
    pose[16 * 3 + 2] = 0.3  # Z rotation
    # Right shoulder (joint 17): rotate slightly down
    pose[17 * 3 + 2] = -0.3

    return pose


def get_lateral_left_pose() -> NDArray[np.float32]:
    """Get pose for left lateral position (lying on left side).

    Returns:
        Pose array of shape (69,)
    """
    pose = np.zeros(69, dtype=np.float32)

    # Rotate torso slightly
    pose[0] = 0.1  # Pelvis
    pose[3] = 0.05  # Spine 1
    pose[6] = 0.05  # Spine 2
    pose[9] = 0.05  # Spine 3

    # Adjust arms
    # Left arm down
    pose[16 * 3 + 0] = 0.5
    pose[16 * 3 + 2] = 0.5
    # Right arm across body
    pose[17 * 3 + 0] = -0.3
    pose[17 * 3 + 2] = 0.8

    # Slight leg bend
    pose[1 * 3 + 0] = 0.2  # Left hip
    pose[4 * 3 + 0] = 0.3  # Left knee
    pose[2 * 3 + 0] = 0.3  # Right hip
    pose[5 * 3 + 0] = 0.4  # Right knee

    return pose


def get_lateral_right_pose() -> NDArray[np.float32]:
    """Get pose for right lateral position (lying on right side).

    Returns:
        Pose array of shape (69,)
    """
    pose = np.zeros(69, dtype=np.float32)

    # Rotate torso slightly (opposite of left lateral)
    pose[0] = -0.1
    pose[3] = -0.05
    pose[6] = -0.05
    pose[9] = -0.05

    # Adjust arms (mirrored from left lateral)
    pose[17 * 3 + 0] = 0.5
    pose[17 * 3 + 2] = -0.5
    pose[16 * 3 + 0] = -0.3
    pose[16 * 3 + 2] = -0.8

    # Leg bend (mirrored)
    pose[2 * 3 + 0] = 0.2
    pose[5 * 3 + 0] = 0.3
    pose[1 * 3 + 0] = 0.3
    pose[4 * 3 + 0] = 0.4

    return pose


def get_prone_pose() -> NDArray[np.float32]:
    """Get pose for prone position (lying on stomach).

    Returns:
        Pose array of shape (69,)
    """
    pose = np.zeros(69, dtype=np.float32)

    # Head turned to side
    pose[12 * 3 + 1] = 0.5  # Neck Y rotation

    # Arms at sides or slightly bent
    pose[16 * 3 + 2] = -0.2
    pose[17 * 3 + 2] = 0.2

    return pose


def get_fowler_pose(angle: float = 0.75) -> NDArray[np.float32]:
    """Get pose for Fowler's position (semi-reclined with bent knees).

    This is common for patients who need upper body elevation.

    Args:
        angle: Degree of hip/knee bend (0-1 scale, higher = more bent)

    Returns:
        Pose array of shape (69,)
    """
    pose = np.zeros(69, dtype=np.float32)

    # Hip flexion (sitting up angle)
    hip_angle = angle * 0.8
    pose[1 * 3 + 0] = hip_angle  # Left hip
    pose[2 * 3 + 0] = hip_angle  # Right hip

    # Knee bend
    knee_angle = angle * 0.5
    pose[4 * 3 + 0] = knee_angle  # Left knee
    pose[5 * 3 + 0] = knee_angle  # Right knee

    # Slight spine curve
    for i in [3, 6, 9]:  # Spine joints
        pose[i] = angle * 0.1

    # Arms resting at sides
    pose[16 * 3 + 2] = 0.4
    pose[17 * 3 + 2] = -0.4

    return pose


def get_semi_fowler_pose() -> NDArray[np.float32]:
    """Get Semi-Fowler's position (30-45 degree elevation).

    Returns:
        Pose array of shape (69,)
    """
    return get_fowler_pose(angle=0.5)


def get_high_fowler_pose() -> NDArray[np.float32]:
    """Get High Fowler's position (60-90 degree elevation).

    Returns:
        Pose array of shape (69,)
    """
    return get_fowler_pose(angle=1.0)


# Pose library dictionary for easy access
POSE_LIBRARY = {
    "supine": get_supine_pose,
    "lateral_left": get_lateral_left_pose,
    "lateral_right": get_lateral_right_pose,
    "prone": get_prone_pose,
    "fowler": get_fowler_pose,
    "semi_fowler": get_semi_fowler_pose,
    "high_fowler": get_high_fowler_pose,
}


def get_pose(name: str, **kwargs) -> NDArray[np.float32]:
    """Get pose by name from library.

    Args:
        name: Pose name (supine, lateral_left, lateral_right, prone, fowler, etc.)
        **kwargs: Additional arguments passed to pose function

    Returns:
        Pose array of shape (69,)

    Raises:
        ValueError: If pose name is not recognized
    """
    if name not in POSE_LIBRARY:
        available = ", ".join(POSE_LIBRARY.keys())
        raise ValueError(f"Unknown pose '{name}'. Available poses: {available}")

    pose_fn = POSE_LIBRARY[name]

    # Check if function accepts kwargs
    import inspect

    sig = inspect.signature(pose_fn)
    if len(sig.parameters) > 0:
        return pose_fn(**kwargs)
    return pose_fn()
