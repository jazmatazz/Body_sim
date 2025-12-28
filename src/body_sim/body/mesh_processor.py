"""Mesh processing utilities for body positioning and transformation."""

import numpy as np
from numpy.typing import NDArray

from body_sim.core.types import BodyMesh, SimulationConfig


def rotation_matrix_x(angle: float) -> NDArray[np.float32]:
    """Create rotation matrix around X axis.

    Args:
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[1, 0, 0], [0, c, -s], [0, s, c]],
        dtype=np.float32,
    )


def rotation_matrix_y(angle: float) -> NDArray[np.float32]:
    """Create rotation matrix around Y axis.

    Args:
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        dtype=np.float32,
    )


def rotation_matrix_z(angle: float) -> NDArray[np.float32]:
    """Create rotation matrix around Z axis.

    Args:
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[c, -s, 0], [s, c, 0], [0, 0, 1]],
        dtype=np.float32,
    )


def position_body_on_mattress(
    mesh: BodyMesh,
    config: SimulationConfig,
    bed_angle: float = 0.0,
) -> BodyMesh:
    """Position body mesh centered on mattress surface.

    The body is positioned supine (lying on back) with:
    - X axis: lateral (across mattress width)
    - Y axis: longitudinal (along mattress length, head toward Y=0)
    - Z axis: vertical (up from mattress surface)

    Args:
        mesh: Original body mesh
        config: Simulation configuration with mattress dimensions
        bed_angle: Head elevation angle in radians (for adjustable beds)

    Returns:
        New BodyMesh with transformed vertices
    """
    vertices = mesh.vertices.copy()
    joints = mesh.joints.copy()

    # Step 1: Rotate body to supine position (lying on back)
    # SMPL default is standing, so rotate 90° around X to lie down
    rot_supine = rotation_matrix_x(-np.pi / 2)
    vertices = vertices @ rot_supine.T
    joints = joints @ rot_supine.T

    # Step 2: Apply bed angle (head elevation)
    if bed_angle != 0:
        rot_bed = rotation_matrix_y(bed_angle)
        vertices = vertices @ rot_bed.T
        joints = joints @ rot_bed.T

    # Step 3: Center on mattress
    center_x = config.mattress_width / 2
    center_y = config.mattress_height / 2

    # Get current bounds
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    current_center = (min_bounds + max_bounds) / 2

    # Translate to center of mattress, with bottom of body at z=0
    translation = np.array(
        [
            center_x - current_center[0],
            center_y - current_center[1],
            -min_bounds[2],  # Lift so lowest point is at z=0
        ],
        dtype=np.float32,
    )

    vertices += translation
    joints += translation

    return BodyMesh(
        vertices=vertices,
        faces=mesh.faces,
        joints=joints,
        parameters=mesh.parameters,
    )


def rotate_body(
    mesh: BodyMesh,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> BodyMesh:
    """Rotate body mesh around its center.

    Args:
        mesh: Body mesh to rotate
        roll: Rotation around X axis (radians)
        pitch: Rotation around Y axis (radians)
        yaw: Rotation around Z axis (radians)

    Returns:
        Rotated body mesh
    """
    vertices = mesh.vertices.copy()
    joints = mesh.joints.copy()

    # Get center of body
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2

    # Translate to origin
    vertices -= center
    joints -= center

    # Apply rotations (Z * Y * X order)
    if roll != 0:
        rot = rotation_matrix_x(roll)
        vertices = vertices @ rot.T
        joints = joints @ rot.T

    if pitch != 0:
        rot = rotation_matrix_y(pitch)
        vertices = vertices @ rot.T
        joints = joints @ rot.T

    if yaw != 0:
        rot = rotation_matrix_z(yaw)
        vertices = vertices @ rot.T
        joints = joints @ rot.T

    # Translate back
    vertices += center
    joints += center

    return BodyMesh(
        vertices=vertices,
        faces=mesh.faces,
        joints=joints,
        parameters=mesh.parameters,
    )


def get_body_regions_mask(
    mesh: BodyMesh,
    config: SimulationConfig,
) -> dict[str, NDArray[np.bool_]]:
    """Get vertex masks for different body regions.

    Approximates body regions based on Y position (head to foot).

    Args:
        mesh: Body mesh
        config: Simulation configuration

    Returns:
        Dictionary mapping region names to boolean vertex masks
    """
    from body_sim.core.constants import BODY_REGION_ROWS

    vertices = mesh.vertices
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    y_range = y_max - y_min

    masks = {}
    for region, (start_frac, end_frac) in BODY_REGION_ROWS.items():
        y_start = y_min + start_frac * y_range
        y_end = y_min + end_frac * y_range
        masks[region] = (vertices[:, 1] >= y_start) & (vertices[:, 1] < y_end)

    return masks


def compute_vertex_areas(mesh: BodyMesh) -> NDArray[np.float32]:
    """Compute approximate area associated with each vertex.

    Uses 1/3 of the area of adjacent triangles.

    Args:
        mesh: Body mesh

    Returns:
        Array of shape (num_vertices,) with area per vertex
    """
    vertices = mesh.vertices
    faces = mesh.faces

    # Compute face areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Cross product gives 2x area
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)

    # Distribute to vertices (1/3 per vertex)
    vertex_areas = np.zeros(len(vertices), dtype=np.float32)
    for i, face in enumerate(faces):
        for v_idx in face:
            vertex_areas[v_idx] += face_areas[i] / 3

    return vertex_areas
