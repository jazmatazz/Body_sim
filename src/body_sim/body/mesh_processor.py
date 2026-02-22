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

    # SMPL with supine pose already outputs body lying down:
    # - X: left/right (arm span ~1.7m in T-pose)
    # - Y: head/foot (body length ~1.7m, head at +Y, feet at -Y)
    # - Z: front/back (thickness ~0.3m, front at +Z, back at -Z)
    #
    # For mattress coordinate system:
    # - X: across mattress width
    # - Y: along mattress length (head at Y=0, feet at Y=max)
    # - Z: vertical (up from mattress surface)
    #
    # We just need to translate so back (-Z) is at mattress surface (Z=0)
    # and body is centered on mattress

    # Step 1: Apply bed angle (head elevation) if needed
    if bed_angle != 0:
        rot_bed = rotation_matrix_x(-bed_angle)
        vertices = vertices @ rot_bed.T
        joints = joints @ rot_bed.T

    # Step 2: Center on mattress
    center_x = config.mattress_width / 2
    center_y = config.mattress_height / 2

    # Get current bounds
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    current_center = (min_bounds + max_bounds) / 2

    # Translate to center of mattress, with back of body at z=0
    translation = np.array(
        [
            center_x - current_center[0],
            center_y - current_center[1],
            -min_bounds[2],  # Lift so lowest point (back) is at z=0
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


def update_body_on_surface(
    mesh: BodyMesh,
    grid: "AirCellGrid",
    base_offset: float = 0.0,
) -> BodyMesh:
    """Update body vertex Z positions to rest on the mattress surface.

    Each vertex's Z coordinate is adjusted so the body follows the
    contour of the mattress surface. This simulates the body sinking
    into lower cells and being pushed up by higher cells.

    Args:
        mesh: Body mesh to update
        grid: Air cell mattress grid with current heights
        base_offset: Additional vertical offset (e.g., for body thickness)

    Returns:
        New BodyMesh with updated vertex positions
    """
    vertices = mesh.vertices.copy()
    joints = mesh.joints.copy()

    # Get the minimum Z of the original mesh (the "bottom" of the body)
    min_z = vertices[:, 2].min()

    # For each vertex, compute how much it should be lifted based on
    # the mattress surface height at its (x, y) position
    for i, vertex in enumerate(vertices):
        x, y, z = vertex

        # Get the mattress surface height at this position
        surface_height = grid.get_interpolated_height_at(x, y)

        # The vertex's relative height above the body's bottom
        relative_z = z - min_z

        # New Z = surface height + relative height within body + offset
        vertices[i, 2] = surface_height + relative_z + base_offset

    # Update joints similarly (approximate - use average surface height under body)
    avg_surface = np.mean([
        grid.get_interpolated_height_at(j[0], j[1])
        for j in joints
        if 0 <= j[0] <= grid.width and 0 <= j[1] <= grid.height
    ] or [grid.min_cell_height])

    joint_min_z = joints[:, 2].min()
    joints[:, 2] = avg_surface + (joints[:, 2] - joint_min_z) + base_offset

    return BodyMesh(
        vertices=vertices,
        faces=mesh.faces,
        joints=joints,
        parameters=mesh.parameters,
    )


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
