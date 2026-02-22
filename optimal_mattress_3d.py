#!/usr/bin/env python3
"""
3D Visualization of Optimal Mattress Configuration
===================================================
Visualizes the Optimized Hybrid air mattress with:
- 15cm cells (13 x 6 grid)
- Optimized Hybrid pattern - combines best strategies per body region
- 5-minute cycle period
- SMPL body model on inclined surface
- 30-degree head-of-bed incline

Optimized Hybrid achieved 75% damage reduction vs foam and 25% fewer
at-risk cells than Zone-Based Adaptive, making it the best pattern tested.

Hybrid strategy (optimized per region):
- Sacrum: 3× cycling speed (critical zone, 36% of ulcers)
- Heels: 2× cycling speed
- Scapulae: 1.2× cycling speed (optimal for this region)
- Trochanters/Occiput: 1.3× cycling speed
"""

import numpy as np
import plotly.graph_objects as go
import math

# Mattress parameters (optimal configuration)
MATTRESS_LENGTH = 200  # cm
MATTRESS_WIDTH = 90    # cm
CELL_SIZE = 15         # cm (optimal)
CYCLE_PERIOD = 300     # seconds (5 minutes)

N_ROWS = int(MATTRESS_LENGTH / CELL_SIZE)  # 13
N_COLS = int(MATTRESS_WIDTH / CELL_SIZE)   # 6

# Cell dimensions
CELL_HEIGHT_MAX = 12  # cm when fully inflated
CELL_HEIGHT_MIN = 4   # cm when deflated
CELL_GAP = 1          # cm gap between cells

# Bed incline (30 degrees at waist for quadriplegic patients)
INCLINE_ANGLE = 30    # degrees
HINGE_POSITION = 0.45  # 45% from head (waist/hip area)

# Body parameters
BODY_LENGTH = 175  # cm


def load_smpl_body_on_inclined_bed(apply_deformation: bool = True):
    """
    Load SMPL body model and position it on the inclined bed surface.
    Returns vertices, faces for 3D mesh rendering.

    Args:
        apply_deformation: If True, apply tissue deformation at contact points
    """
    try:
        import smplx
        import torch

        print("Loading SMPL body model...")

        # Load SMPL model
        smpl_dir = '/Users/jaspermetz/Documents/Body_Sim/smpl'
        model = smplx.create(smpl_dir, model_type='smpl', gender='neutral')

        # Pose: supine with arms at sides
        body_pose = torch.zeros(1, 69)
        body_pose[0, 45 + 2] = -1.5   # L_shoulder Z-rot (arm down)
        body_pose[0, 48 + 2] = 1.5    # R_shoulder Z-rot (arm down)

        output = model(body_pose=body_pose)
        verts = output.vertices.detach().numpy()[0]  # (6890, 3)
        faces = model.faces  # (13776, 3)

        # Rotate from standing to supine (lying on back)
        angle = -math.pi / 2
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)],
        ])
        v = verts @ Rx.T
        v *= 100  # metres → cm

        # Scale to match body length
        raw_length = v[:, 2].max() - v[:, 2].min()
        scale = BODY_LENGTH / raw_length
        v *= scale

        # After rotation: X = width, Y = depth (height off bed), Z = body length
        # Head is at negative Z after rotation, feet at positive Z
        # Shift so head is at Z=0, feet at Z=body_length
        v[:, 2] -= v[:, 2].min()

        # Center X on mattress width
        v[:, 0] -= v[:, 0].mean()
        v[:, 0] += MATTRESS_WIDTH / 2

        # Y is height above bed surface - shift so back touches bed
        v[:, 1] -= v[:, 1].min()

        # Now position body on inclined bed surface
        # Use the same hinge position as the mattress cells
        hinge_row = int(N_ROWS * HINGE_POSITION)
        hinge_y_cm = hinge_row * CELL_SIZE  # Match cell hinge position
        angle_rad = np.radians(INCLINE_ANGLE)

        # Create output arrays for final positions
        vx = v[:, 0].copy()  # Width stays the same
        vy = np.zeros(len(v))  # Will be computed based on incline
        vz = np.zeros(len(v))  # Will be computed based on incline

        # Calculate tissue deformation parameters for each vertex
        # Tissue thickness and compression depends on body part
        tissue_params = get_tissue_deformation_params(v)

        # Position body on the inclined bed surface using smooth continuous transforms
        # (not discrete row-based) to avoid squiggly appearance
        hinge_y_cm = int(N_ROWS * HINGE_POSITION) * CELL_SIZE  # Hinge at ~75cm

        for i in range(len(v)):
            body_pos = v[i, 2]  # Position along body (0=head, BODY_LENGTH=feet)
            depth = v[i, 1]     # Height above bed surface

            # Offset body to start 12cm from head of mattress
            bed_pos = body_pos + 12

            # Apply tissue deformation for vertices near the surface
            if apply_deformation and depth < 5.0:  # Within 5cm of surface
                compression = tissue_params['compression'][i]
                # Reduce depth based on tissue compression
                depth = max(0, depth - compression)

            if bed_pos < hinge_y_cm:
                # Above hinge - on inclined section
                dist_from_hinge = hinge_y_cm - bed_pos

                # Smooth continuous surface calculation
                surf_y = hinge_y_cm - dist_from_hinge * np.cos(angle_rad)
                surf_z = dist_from_hinge * np.sin(angle_rad) + CELL_HEIGHT_MAX

                # Normal direction (perpendicular to inclined surface)
                norm_y = -np.sin(angle_rad)
                norm_z = np.cos(angle_rad)

                # Position vertex on surface + depth along normal
                vy[i] = surf_y + depth * norm_y
                vz[i] = surf_z + depth * norm_z
            else:
                # Below hinge - on flat section
                vy[i] = bed_pos
                vz[i] = CELL_HEIGHT_MAX + depth

        return vx, vy, vz, faces, tissue_params

    except Exception as e:
        print(f"Could not load SMPL body: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def get_tissue_deformation_params(vertices):
    """
    Calculate tissue deformation parameters for each vertex.

    Based on biomechanical properties:
    - Bony prominences (sacrum, heels) - minimal compression, high pressure
    - Soft tissue (buttocks, thighs) - significant compression
    """
    n_verts = len(vertices)

    # Tissue thickness by body region (mm)
    tissue_thickness = np.ones(n_verts) * 20.0  # Default

    # Tissue stiffness factor (higher = stiffer = less compression)
    stiffness = np.ones(n_verts) * 1.0

    # Compression amount (cm) - will be calculated
    compression = np.zeros(n_verts)

    # Body part assignment based on position
    for i, v in enumerate(vertices):
        x, y, z = v  # x=width, y=height above bed, z=position along body

        # Normalize z position (0=head, 1=feet)
        z_norm = z / BODY_LENGTH
        # Normalize x position relative to body center
        x_norm = (x - MATTRESS_WIDTH/2) / (45/2)  # Assume 45cm body width

        # Assign tissue properties by body region
        # Head region (0-10%)
        if z_norm < 0.10:
            tissue_thickness[i] = 8.0
            stiffness[i] = 2.0

        # Shoulder region (10-25%)
        elif z_norm < 0.25:
            if abs(x_norm) > 0.5:
                tissue_thickness[i] = 6.0  # Scapula
                stiffness[i] = 3.0
            else:
                tissue_thickness[i] = 15.0  # Upper back
                stiffness[i] = 1.2

        # Upper back (25-40%)
        elif z_norm < 0.40:
            if abs(x_norm) < 0.2:
                tissue_thickness[i] = 5.0  # Spine
                stiffness[i] = 4.0
            else:
                tissue_thickness[i] = 15.0
                stiffness[i] = 1.2

        # Sacrum region (40-55%) - CRITICAL
        elif z_norm < 0.55:
            if abs(x_norm) < 0.35:
                tissue_thickness[i] = 3.0  # Sacrum - very thin
                stiffness[i] = 6.0  # Very stiff (bone)
            else:
                tissue_thickness[i] = 25.0  # Buttocks - thick
                stiffness[i] = 0.5  # Very soft

        # Buttocks/ischial (55-65%)
        elif z_norm < 0.65:
            if abs(x_norm) < 0.25:
                tissue_thickness[i] = 4.0  # Ischial
                stiffness[i] = 4.0
            else:
                tissue_thickness[i] = 25.0  # Buttocks
                stiffness[i] = 0.5

        # Thighs (65-80%)
        elif z_norm < 0.80:
            tissue_thickness[i] = 30.0
            stiffness[i] = 0.8

        # Calves (80-92%)
        elif z_norm < 0.92:
            tissue_thickness[i] = 20.0
            stiffness[i] = 1.0

        # Heels/ankles (92-100%)
        else:
            if abs(x_norm) > 0.4:
                tissue_thickness[i] = 3.0  # Malleolus
                stiffness[i] = 5.0
            else:
                tissue_thickness[i] = 5.0  # Heel
                stiffness[i] = 3.0

        # Calculate compression based on tissue properties and contact
        # Vertices near the surface (small y) compress more
        if y < 5.0:  # Within 5cm of bed surface
            contact_factor = 1.0 - (y / 5.0)  # 1.0 at surface, 0 at 5cm

            # Soft tissue compresses more than stiff tissue
            max_compression = (tissue_thickness[i] / 10.0) * 0.6  # Up to 60% of tissue thickness

            # Higher pressure regions compress more
            # Sacrum region (with 30° incline) has highest pressure
            pressure_factor = 1.0
            if 0.40 <= z_norm < 0.55 and abs(x_norm) < 0.35:
                pressure_factor = 2.5  # Sacrum with incline - max pressure
            elif z_norm >= 0.92:
                pressure_factor = 2.0  # Heels - high pressure
            elif 0.10 <= z_norm < 0.25 and abs(x_norm) > 0.5:
                pressure_factor = 1.5  # Scapulae

            # Final compression (cm)
            compression[i] = (max_compression * contact_factor * pressure_factor) / stiffness[i]
            compression[i] = min(compression[i], tissue_thickness[i] / 10.0 * 0.65)  # Max 65% compression

    return {
        'tissue_thickness': tissue_thickness,
        'stiffness': stiffness,
        'compression': compression,
    }


def create_body_mesh(show_compression: bool = True):
    """Create SMPL body mesh trace for the visualization.

    Args:
        show_compression: If True, color body based on tissue compression
    """
    result = load_smpl_body_on_inclined_bed(apply_deformation=True)
    if result[0] is None:
        print("SMPL body not available, skipping body mesh")
        return []

    vx, vy, vz, faces, tissue_params = result

    # Create double-sided mesh by including both face orientations
    # This prevents the see-through effect from sides/bottom
    fi = np.concatenate([faces[:, 0], faces[:, 0]])
    fj = np.concatenate([faces[:, 1], faces[:, 2]])  # Swap j,k for reverse
    fk = np.concatenate([faces[:, 2], faces[:, 1]])

    traces = []

    if show_compression and tissue_params is not None:
        # Color body based on tissue compression
        compression = tissue_params['compression']

        # Normalize compression for coloring (0-2cm range typical)
        compression_normalized = np.clip(compression / 2.0, 0, 1)

        # Create vertex colors based on compression
        # Low compression = normal skin tone, High compression = reddish (pressure)
        vertex_colors = np.zeros((len(vx), 3))
        for i in range(len(vx)):
            c = compression_normalized[i]
            if c < 0.1:
                # Normal skin tone
                vertex_colors[i] = [232, 190, 172]
            elif c < 0.3:
                # Light pressure - slightly pink
                blend = (c - 0.1) / 0.2
                vertex_colors[i] = [
                    232 + (255 - 232) * blend,
                    190 - 40 * blend,
                    172 - 50 * blend
                ]
            elif c < 0.6:
                # Moderate pressure - pink/red
                blend = (c - 0.3) / 0.3
                vertex_colors[i] = [
                    255,
                    150 - 50 * blend,
                    122 - 42 * blend
                ]
            else:
                # High pressure - red (warning)
                blend = min((c - 0.6) / 0.4, 1.0)
                vertex_colors[i] = [
                    255,
                    100 - 50 * blend,
                    80 - 30 * blend
                ]

        # Create intensity values for colorscale
        intensity = compression_normalized

        # Main body mesh with compression coloring
        body_trace = go.Mesh3d(
            x=vx, y=vy, z=vz,
            i=fi, j=fj, k=fk,
            intensity=np.concatenate([intensity, intensity]),  # Double for both face orientations
            colorscale=[
                [0.0, 'rgb(232, 190, 172)'],   # Normal skin tone
                [0.2, 'rgb(245, 180, 160)'],   # Light pressure
                [0.4, 'rgb(255, 150, 130)'],   # Moderate pressure
                [0.6, 'rgb(255, 100, 100)'],   # High pressure
                [1.0, 'rgb(255, 50, 50)'],     # Critical pressure
            ],
            cmin=0,
            cmax=1,
            opacity=1.0,
            flatshading=False,
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                specular=0.3,
                roughness=0.4,
                fresnel=0.2
            ),
            lightposition=dict(x=100, y=-200, z=300),
            showscale=True,
            colorbar=dict(
                title='Tissue<br>Compression',
                x=1.02,
                len=0.4,
                tickvals=[0, 0.5, 1],
                ticktext=['None', '1 cm', '2+ cm']
            ),
            hovertemplate='Tissue Compression: %{intensity:.1%}<extra>Patient Body</extra>',
            name='Patient'
        )
        traces.append(body_trace)

    else:
        # Simple skin tone coloring
        body_trace = go.Mesh3d(
            x=vx, y=vy, z=vz,
            i=fi, j=fj, k=fk,
            color='rgb(232, 190, 172)',  # Skin tone
            opacity=1.0,
            flatshading=False,
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                specular=0.3,
                roughness=0.4,
                fresnel=0.2
            ),
            lightposition=dict(x=100, y=-200, z=300),
            showscale=False,
            hovertemplate='Patient Body<extra></extra>',
            name='Patient'
        )
        traces.append(body_trace)

    return traces

# Zone definitions (row_start, row_end, col_start, col_end, freq_mult, name, color)
ZONES = [
    # Sacrum - highest risk, fastest cycle (3x)
    (0.40, 0.55, 0.30, 0.70, 3.0, 'Sacrum', '#ff4444'),
    # Heels - high risk (2.5x)
    (0.90, 1.00, 0.20, 0.45, 2.5, 'Left Heel', '#ff8844'),
    (0.90, 1.00, 0.55, 0.80, 2.5, 'Right Heel', '#ff8844'),
    # Scapulae (2x)
    (0.10, 0.25, 0.15, 0.40, 2.0, 'Left Scapula', '#ffaa44'),
    (0.10, 0.25, 0.60, 0.85, 2.0, 'Right Scapula', '#ffaa44'),
    # Occiput (1.5x)
    (0.00, 0.10, 0.35, 0.65, 1.5, 'Occiput', '#ffcc44'),
]


def get_zone_info(row, col):
    """Get zone information for a cell."""
    row_pos = row / N_ROWS
    col_pos = col / N_COLS

    for r_start, r_end, c_start, c_end, freq, name, color in ZONES:
        if r_start <= row_pos < r_end and c_start <= col_pos <= c_end:
            return freq, name, color

    return 1.0, 'Standard', '#44aa44'  # Default zone


def get_cell_inflation(row, col, phase):
    """
    Get cell inflation state (0-1) at given phase using Optimized Hybrid pattern.

    Optimized Hybrid combines best strategies per body region:
    - Sacrum: 3× cycling speed (critical zone, 36% of ulcers)
    - Heels: 2× cycling speed
    - Scapulae: 1.2× cycling speed (optimal for this region)
    - Trochanters/Occiput: 1.3× cycling speed

    Achieves lowest max damage AND fewest at-risk cells of all patterns.
    """
    row_pos = row / N_ROWS
    col_pos = col / N_COLS

    # Sacrum zone: VERY FAST cycling (3x) - critical zone
    if 0.40 <= row_pos < 0.55 and 0.35 <= col_pos <= 0.65:
        local_phase = (phase * 3.0) % 1.0
        return 0.5 + 0.5 * np.cos(2 * np.pi * local_phase)

    # Heel zones: FAST cycling (2x)
    elif row_pos >= 0.92 and (0.30 <= col_pos <= 0.42 or 0.58 <= col_pos <= 0.70):
        local_phase = (phase * 2.0) % 1.0
        return 0.5 + 0.5 * np.cos(2 * np.pi * local_phase)

    # Scapulae zone: moderate cycling (1.2x) - optimal for this region
    elif 0.10 <= row_pos < 0.25:
        local_phase = (phase * 1.2) % 1.0
        return 0.5 + 0.5 * np.cos(2 * np.pi * local_phase)

    # Trochanter zones: moderate cycling (1.3x)
    elif 0.45 <= row_pos < 0.55 and (col_pos < 0.30 or col_pos > 0.70):
        local_phase = (phase * 1.3) % 1.0
        return 0.5 + 0.5 * np.cos(2 * np.pi * local_phase)

    # Occiput: moderate cycling (1.3x)
    elif row_pos < 0.08:
        local_phase = (phase * 1.3) % 1.0
        return 0.5 + 0.5 * np.cos(2 * np.pi * local_phase)

    # Default zones: standard cycle
    else:
        return 0.5 + 0.5 * np.cos(2 * np.pi * phase)


def get_bed_surface_position(row):
    """
    Get the Y (length) and Z (height) position for a cell row on the inclined bed.

    The bed hinges at HINGE_POSITION (45% from head).
    Head section is elevated by INCLINE_ANGLE degrees.
    """
    hinge_row = int(N_ROWS * HINGE_POSITION)
    angle_rad = np.radians(INCLINE_ANGLE)

    row_y = row * CELL_SIZE  # Base Y position
    row_z = 0  # Base Z position

    if row <= hinge_row:
        # Above hinge - elevated section
        distance_from_hinge = (hinge_row - row) * CELL_SIZE
        # Rotate around hinge point
        row_y = hinge_row * CELL_SIZE - distance_from_hinge * np.cos(angle_rad)
        row_z = distance_from_hinge * np.sin(angle_rad)
    else:
        # Below hinge - flat section
        row_y = hinge_row * CELL_SIZE + (row - hinge_row) * CELL_SIZE
        row_z = 0

    return row_y, row_z


def create_cell_mesh(row, col, inflation):
    """Create mesh for a single cell on the inclined bed surface."""
    # Cell position in X (width)
    x0 = col * CELL_SIZE + CELL_GAP
    x1 = (col + 1) * CELL_SIZE - CELL_GAP

    # Get Y and Z positions for this row on the inclined surface
    base_y0, base_z0 = get_bed_surface_position(row)
    base_y1, base_z1 = get_bed_surface_position(row + 1)

    # Adjust for cell gap
    hinge_row = int(N_ROWS * HINGE_POSITION)
    angle_rad = np.radians(INCLINE_ANGLE)

    if row <= hinge_row:
        # Inclined section - adjust gap along incline
        gap_y = CELL_GAP * np.cos(angle_rad)
        gap_z = CELL_GAP * np.sin(angle_rad)
        y0 = base_y0 + gap_y
        z0 = base_z0 - gap_z
        y1 = base_y1 - gap_y
        z1 = base_z1 + gap_z
    else:
        # Flat section
        y0 = base_y0 + CELL_GAP
        z0 = base_z0
        y1 = base_y1 - CELL_GAP
        z1 = base_z1

    height = CELL_HEIGHT_MIN + (CELL_HEIGHT_MAX - CELL_HEIGHT_MIN) * inflation

    # Get zone color
    freq, zone_name, base_color = get_zone_info(row, col)

    # Adjust color based on inflation
    # Deflated = greener (relief), Inflated = redder (support)
    if inflation < 0.5:
        # Deflated - show as green (relief)
        r = int(50 + 100 * inflation)
        g = int(200 - 50 * inflation)
        b = 50
    else:
        # Inflated - use orange/red for support
        blend = (inflation - 0.5) * 2
        r = int(220 * blend + 100 * (1 - blend))
        g = int(100 * blend + 150 * (1 - blend))
        b = int(50 * blend + 50 * (1 - blend))

    color = f'rgb({r},{g},{b})'

    # Calculate normal direction for cell top (perpendicular to bed surface)
    if row <= hinge_row:
        # Inclined section - normal points up and back
        nx, ny, nz = 0, -np.sin(angle_rad), np.cos(angle_rad)
    else:
        # Flat section - normal points straight up
        nx, ny, nz = 0, 0, 1

    # Create box vertices on inclined surface
    # Bottom corners (on bed surface)
    vertices = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z1],
        [x0, y1, z1],
        # Top corners (raised by height along normal)
        [x0, y0 + height * ny, z0 + height * nz],
        [x1, y0 + height * ny, z0 + height * nz],
        [x1, y1 + height * ny, z1 + height * nz],
        [x0, y1 + height * ny, z1 + height * nz],
    ], dtype=float)

    # Triangular faces
    i = [0, 0, 0, 0, 4, 4, 0, 1, 1, 2, 2, 3]
    j = [1, 2, 3, 4, 5, 6, 1, 5, 2, 6, 3, 7]
    k = [2, 3, 4, 5, 6, 7, 5, 6, 6, 7, 7, 4]

    # Determine zone for hover info
    row_pos = row / N_ROWS
    col_pos = col / N_COLS
    if 0.40 <= row_pos < 0.55 and 0.35 <= col_pos <= 0.65:
        zone_label = "Sacrum (2x cycle)"
    elif row_pos >= 0.92 and (0.32 <= col_pos <= 0.40 or 0.58 <= col_pos <= 0.68):
        zone_label = "Heel (1.5x cycle)"
    elif 0.10 <= row_pos < 0.25:
        zone_label = "Scapula (1.2x cycle)"
    else:
        zone_label = "Standard"

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i, j=j, k=k,
        color=color,
        opacity=0.9,
        flatshading=True,
        showlegend=False,
        hovertemplate=f'Zone: {zone_label}<br>Inflation: {inflation:.0%}<extra></extra>'
    )


def create_body_outline():
    """Create a simple body outline overlay following the inclined bed."""
    traces = []

    angle_rad = np.radians(INCLINE_ANGLE)
    hinge_y = int(N_ROWS * HINGE_POSITION) * CELL_SIZE

    def get_body_position(row_cm, height_above_bed):
        """Get Y, Z position for a point on the body."""
        if row_cm < hinge_y:
            # Inclined section
            dist_from_hinge = hinge_y - row_cm
            y = hinge_y - dist_from_hinge * np.cos(angle_rad)
            z = dist_from_hinge * np.sin(angle_rad) + height_above_bed
        else:
            # Flat section
            y = row_cm
            z = height_above_bed
        return y, z

    # Body regions as ellipses (row_cm, col_cm, length, width, label)
    # row_cm = distance from head along body
    body_parts = [
        # Head (0-18 cm from top)
        (10, 45, 18, 14, 'Head'),
        # Shoulders (20-45 cm)
        (32, 28, 22, 16, 'L Shoulder'),
        (32, 62, 22, 16, 'R Shoulder'),
        # Upper back / torso
        (55, 45, 30, 28, 'Upper Back'),
        # Sacrum region (80-100 cm) - CRITICAL with incline
        (90, 45, 20, 22, 'SACRUM'),
        # Buttocks
        (110, 38, 18, 14, 'L Buttock'),
        (110, 52, 18, 14, 'R Buttock'),
        # Thighs
        (140, 35, 35, 12, 'L Thigh'),
        (140, 55, 35, 12, 'R Thigh'),
        # Calves
        (172, 35, 28, 10, 'L Calf'),
        (172, 55, 28, 10, 'R Calf'),
        # Heels - high pressure
        (195, 35, 10, 10, 'L Heel'),
        (195, 55, 10, 10, 'R Heel'),
    ]

    height_above = CELL_HEIGHT_MAX + 5

    for row_cm, col_cm, length, width, label in body_parts:
        # Create ellipse points following inclined surface
        theta = np.linspace(0, 2 * np.pi, 30)

        ex = col_cm + (width / 2) * np.cos(theta)
        ey = np.zeros_like(theta)
        ez = np.zeros_like(theta)

        for i, t in enumerate(theta):
            row_offset = (length / 2) * np.sin(t)
            local_row = row_cm + row_offset
            y, z = get_body_position(local_row, height_above)
            ey[i] = y
            ez[i] = z

        # Color sacrum and heels differently (high risk)
        if 'SACRUM' in label:
            color = 'rgba(255, 100, 100, 0.9)'
            width_line = 4
        elif 'Heel' in label:
            color = 'rgba(255, 150, 100, 0.9)'
            width_line = 3
        else:
            color = 'rgba(255, 200, 150, 0.8)'
            width_line = 2

        traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode='lines',
            line=dict(color=color, width=width_line),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add body centerline following incline
    centerline_rows = np.linspace(5, 198, 50)
    centerline_x = np.ones(50) * 45
    centerline_y = np.zeros(50)
    centerline_z = np.zeros(50)

    for i, row_cm in enumerate(centerline_rows):
        y, z = get_body_position(row_cm, height_above)
        centerline_y[i] = y
        centerline_z[i] = z

    traces.append(go.Scatter3d(
        x=centerline_x, y=centerline_y, z=centerline_z,
        mode='lines',
        line=dict(color='rgba(255, 200, 150, 0.5)', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Add hinge line indicator
    hinge_x = np.array([0, MATTRESS_WIDTH])
    hinge_y_pos = np.array([hinge_y, hinge_y])
    hinge_z = np.array([0, 0])

    traces.append(go.Scatter3d(
        x=hinge_x, y=hinge_y_pos, z=hinge_z,
        mode='lines+text',
        line=dict(color='red', width=5),
        text=[f'Hinge ({INCLINE_ANGLE}°)', ''],
        textposition='top center',
        textfont=dict(size=12, color='red'),
        showlegend=False
    ))

    return traces


def create_zone_labels():
    """Create labels for high-risk zones on inclined bed."""
    angle_rad = np.radians(INCLINE_ANGLE)
    hinge_y = int(N_ROWS * HINGE_POSITION) * CELL_SIZE

    def get_label_pos(row_cm, height):
        if row_cm < hinge_y:
            dist = hinge_y - row_cm
            y = hinge_y - dist * np.cos(angle_rad)
            z = dist * np.sin(angle_rad) + height
        else:
            y = row_cm
            z = height
        return y, z

    traces = []

    # Sacrum label (most critical with incline)
    sacrum_y, sacrum_z = get_label_pos(90, CELL_HEIGHT_MAX + 25)
    traces.append(go.Scatter3d(
        x=[45], y=[sacrum_y], z=[sacrum_z],
        mode='text',
        text=['<b>SACRUM</b><br>Highest pressure<br>with 30° incline'],
        textfont=dict(size=11, color='#ff4444'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Head label
    head_y, head_z = get_label_pos(10, CELL_HEIGHT_MAX + 20)
    traces.append(go.Scatter3d(
        x=[45], y=[head_y], z=[head_z],
        mode='text',
        text=['HEAD<br>(elevated)'],
        textfont=dict(size=10, color='#666666'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Heels label
    traces.append(go.Scatter3d(
        x=[45], y=[195], z=[CELL_HEIGHT_MAX + 15],
        mode='text',
        text=['HEELS<br>High pressure'],
        textfont=dict(size=10, color='#ff8844'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Incline angle indicator
    traces.append(go.Scatter3d(
        x=[5], y=[hinge_y - 30], z=[25],
        mode='text',
        text=[f'<b>30° Incline</b><br>Weight shifts<br>to sacrum'],
        textfont=dict(size=10, color='red'),
        showlegend=False,
        hoverinfo='skip'
    ))

    return traces


def create_mattress_frame(phase):
    """Create all mattress cell meshes for a given phase."""
    traces = []

    for row in range(N_ROWS):
        for col in range(N_COLS):
            inflation = get_cell_inflation(row, col, phase)
            traces.append(create_cell_mesh(row, col, inflation))

    return traces


def create_animation():
    """Create animated 3D visualization of the optimal mattress."""

    # Create frames for animation
    n_frames = 60
    phases = np.linspace(0, 1, n_frames)

    frames = []
    for i, phase in enumerate(phases):
        frame_traces = create_mattress_frame(phase)
        frames.append(go.Frame(
            data=frame_traces,
            name=str(i),
            traces=list(range(N_ROWS * N_COLS))
        ))

    # Initial frame
    initial_traces = create_mattress_frame(0)

    # Add SMPL body mesh
    body_traces = create_body_mesh()

    # Add zone labels
    zone_labels = create_zone_labels()

    # Create figure
    fig = go.Figure(
        data=initial_traces + body_traces + zone_labels,
        frames=frames
    )

    # Animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.1,
                xanchor='right',
                yanchor='top',
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50}
                        }]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'prefix': 'Cycle: ',
                'suffix': '%',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 50},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [[str(i)], {
                        'frame': {'duration': 100, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 50}
                    }],
                    'label': str(int(i / n_frames * 100)),
                    'method': 'animate'
                }
                for i in range(n_frames)
            ]
        }]
    )

    # Layout - adjusted for inclined bed view with equal scale
    # Use manual aspect ratio matching actual axis ranges for 1:1 cm scaling
    # x range: 110 (width), y range: 240 (length), z range: 75 (height)
    fig.update_layout(
        title=dict(
            text='<b>Optimal Multi-Dynamic Air Mattress for Quadriplegic Patient</b><br>'
                 '<sup>15cm cells | Optimized Hybrid Pattern (75% damage reduction) | 5-minute cycle | 30° Head-of-Bed Incline</sup>',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(title='Width (cm)'),
            yaxis=dict(title='Length (cm) - Head to Feet'),
            zaxis=dict(title='Height (cm)'),
            aspectmode='data',  # Auto-scale based on data
            camera=dict(
                eye=dict(x=1.5, y=-2.0, z=1.0),
                center=dict(x=0, y=0, z=0),
            ),
            dragmode='orbit',  # Free rotation
        ),
        width=1400,
        height=900,
        showlegend=False,
        margin=dict(l=0, r=0, t=80, b=100),
    )

    # Add legend annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text='<b>Optimized Hybrid:</b><br>'
             '• Sacrum: 3x faster cycle<br>'
             '• Heels: 2x faster cycle<br>'
             '• Scapulae: 1.2x faster<br>'
             '• Best overall performance<br><br>'
             '<b>30° Incline Effects:</b><br>'
             '• Weight shifts to sacrum<br>'
             '• Shear stress from gravity<br>'
             '• Increased pressure at hinge<br><br>'
             '<b>Mattress Colors:</b><br>'
             '🟢 Green = Deflated (relief)<br>'
             '🟠 Orange = Inflated (support)<br><br>'
             '<b>Body Deformation:</b><br>'
             '🟤 Skin tone = No compression<br>'
             '🔴 Red = High compression',
        showarrow=False,
        font=dict(size=11),
        align='left',
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='gray',
        borderwidth=1
    )

    return fig


def create_static_comparison():
    """Create static comparison showing different cycle phases."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=[
            'Phase 0% - Sacrum Relief',
            'Phase 17% - Heels Relief',
            'Phase 33% - Transition',
            'Phase 50% - Opposite Relief',
            'Phase 67% - Recovery',
            'Phase 83% - Full Cycle'
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.1
    )

    phases = [0, 0.17, 0.33, 0.50, 0.67, 0.83]
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    for phase, (row, col) in zip(phases, positions):
        for cell_row in range(N_ROWS):
            for cell_col in range(N_COLS):
                inflation = get_cell_inflation(cell_row, cell_col, phase)

                x0 = cell_row * CELL_SIZE + CELL_GAP
                x1 = (cell_row + 1) * CELL_SIZE - CELL_GAP
                y0 = cell_col * CELL_SIZE + CELL_GAP
                y1 = (cell_col + 1) * CELL_SIZE - CELL_GAP
                height = CELL_HEIGHT_MIN + (CELL_HEIGHT_MAX - CELL_HEIGHT_MIN) * inflation

                freq, _, base_color = get_zone_info(cell_row, cell_col)

                if inflation < 0.5:
                    r, g, b = int(50 + 100 * inflation), int(200 - 50 * inflation), 50
                else:
                    r = int(base_color[1:3], 16)
                    g = int(base_color[3:5], 16)
                    b = int(base_color[5:7], 16)

                vertices = np.array([
                    [x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0],
                    [x0, y0, height], [x1, y0, height], [x1, y1, height], [x0, y1, height],
                ], dtype=float)

                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                    i=[0, 0, 0, 0, 4, 4, 0, 1, 1, 2, 2, 3],
                    j=[1, 2, 3, 4, 5, 6, 1, 5, 2, 6, 3, 7],
                    k=[2, 3, 4, 5, 6, 7, 5, 6, 6, 7, 7, 4],
                    color=f'rgb({r},{g},{b})',
                    opacity=0.9,
                    flatshading=True,
                    showlegend=False
                ), row=row, col=col)

    # Update all scenes
    scene_settings = dict(
        xaxis=dict(title='', showticklabels=False, range=[0, 200]),
        yaxis=dict(title='', showticklabels=False, range=[0, 90]),
        zaxis=dict(title='', showticklabels=False, range=[0, 15]),
        aspectmode='data',
        camera=dict(eye=dict(x=1.2, y=-1.2, z=0.8))
    )

    for i in range(1, 7):
        fig.update_layout(**{f'scene{i}': scene_settings})

    fig.update_layout(
        title=dict(
            text='<b>Optimized Hybrid Pattern - Cycle Phases</b><br>'
                 '<sup>High-risk zones cycle faster: Sacrum 3×, Heels 2×, Scapulae 1.5×</sup>',
            x=0.5,
            font=dict(size=18)
        ),
        height=800,
        width=1400,
        showlegend=False
    )

    return fig


if __name__ == "__main__":
    print("Creating optimal mattress 3D visualization...")

    # Create animated version
    fig_animated = create_animation()
    fig_animated.write_html('optimal_mattress_3d.html', include_plotlyjs=True, full_html=True)
    print("Saved: optimal_mattress_3d.html")

    # Create static comparison
    fig_static = create_static_comparison()
    fig_static.write_html('optimal_mattress_phases.html', include_plotlyjs=True, full_html=True)
    print("Saved: optimal_mattress_phases.html")

    print("\nDone! Open optimal_mattress_3d.html to see the animated visualization.")
