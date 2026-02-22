#!/usr/bin/env python3
"""
3D Visualization of Optimal Mattress Configuration
===================================================
Visualizes the Genetically Evolved Optimal air mattress with:
- 15cm cells (13 x 6 grid)
- Evolved pattern - optimized via genetic algorithm over 25 generations
- 5-minute cycle period
- SMPL body model on inclined surface
- 30-degree head-of-bed incline

Evolved Optimal achieved 98% damage reduction vs foam:
- Max damage: 94 (vs 3011 for foam)
- At-risk cells: 0 (vs 37 for foam)
- Avg shear: 8.3 (vs 33.8 for foam)

Key evolved strategies:
- Lower max inflation (0.6-0.7) prevents pressure spikes
- Asymmetric heel patterns (different left/right treatment)
- Square waves in scapulae for abrupt relief transitions
- Variable wave types per region (sine, square, triangle, sawtooth)
"""

import numpy as np
import plotly.graph_objects as go
import math
from evolved_pattern import EvolvedOptimalPattern
from multidynamic_mattress_optimization import SMPLBodyPressureModel

# Instantiate the evolved pattern
EVOLVED_PATTERN = EvolvedOptimalPattern()

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

# Cache for the body model (same model used in simulation)
_body_model_cache = None


def get_simulation_body_model():
    """
    Get the same SMPLBodyPressureModel used in the simulation.
    This ensures the 3D visualization matches the simulation exactly.
    """
    global _body_model_cache
    if _body_model_cache is None:
        print("Loading SMPL body model (same as simulation)...")
        _body_model_cache = SMPLBodyPressureModel(body_mass=75, incline_angle=INCLINE_ANGLE)
    return _body_model_cache


def load_smpl_body_on_inclined_bed(apply_deformation: bool = True):
    """
    Load SMPL body model using the SAME model as the simulation.
    Returns vertices, faces for 3D mesh rendering.

    The simulation model uses coordinates:
      x = width, y = height above bed, z = length along body

    The visualization uses:
      x = width, y = length along body, z = height

    This function transforms the simulation coordinates to visualization coordinates.
    """
    try:
        model = get_simulation_body_model()

        if model.vertices is None:
            print("SMPL model not available")
            return None, None, None, None, None

        # Get vertices and faces from simulation model
        sim_verts = model.vertices.copy()
        faces = model.faces

        # Transform from simulation coords (x, y_height, z_length)
        # to visualization coords (x, y_length, z_height)
        vx = sim_verts[:, 0]  # Width stays the same
        vy = sim_verts[:, 2]  # Length (sim z -> viz y)
        vz = sim_verts[:, 1] + CELL_HEIGHT_MAX  # Height (sim y -> viz z) + cell offset

        # Calculate compression based on tissue thickness and position
        # Using the simulation model's tissue thickness data
        n_verts = len(vx)
        compression = np.zeros(n_verts)

        if hasattr(model, 'vertex_tissue_thickness') and model.vertex_tissue_thickness is not None:
            tissue_thickness = model.vertex_tissue_thickness
            for i in range(n_verts):
                # Vertices near the surface compress based on tissue thickness
                height_above_bed = sim_verts[i, 1]
                if height_above_bed < 5.0:  # Within 5cm of surface
                    contact_factor = 1.0 - (height_above_bed / 5.0)
                    # Thinner tissue = more stiff = less compression
                    # Thick tissue = soft = more compression
                    thickness_mm = tissue_thickness[i]
                    if thickness_mm < 5:  # Bony prominence
                        max_compression = 0.3  # cm
                    elif thickness_mm < 20:
                        max_compression = 1.0  # cm
                    else:
                        max_compression = 2.0  # cm (soft tissue)
                    compression[i] = max_compression * contact_factor

        tissue_params = {
            'compression': compression,
            'tissue_thickness': model.vertex_tissue_thickness if hasattr(model, 'vertex_tissue_thickness') else None,
            'body_parts': model.vertex_body_parts if hasattr(model, 'vertex_body_parts') else None,
        }

        print(f"  Using simulation model: {len(vx)} vertices")

        return vx, vy, vz, faces, tissue_params

    except Exception as e:
        print(f"Could not load SMPL body: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


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
    Get cell inflation state (0-1) at given phase using Genetically Evolved pattern.

    Evolved via genetic algorithm over 25 generations to minimize:
    - Max cumulative damage
    - Number of at-risk cells
    - Average shear stress

    Key evolved strategies:
    - Lower max inflation (0.6-0.7) prevents pressure spikes
    - Asymmetric heel patterns (different left/right)
    - Square waves in scapulae for abrupt relief
    - Variable wave types per region

    Uses smooth=True for realistic visualization (gradual transitions).
    """
    return EVOLVED_PATTERN.get_cell_state(row, col, N_ROWS, N_COLS, phase, smooth=True)


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
                 '<sup>15cm cells | Genetically Evolved Pattern (90% damage reduction, realistic 45s transitions) | 5-minute cycle | 30° Incline</sup>',
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
        text='<b>Evolved Optimal (GA):</b><br>'
             '• 25 generations evolved<br>'
             '• 98% damage reduction<br>'
             '• 0 at-risk cells<br>'
             '• Asymmetric heel patterns<br>'
             '• Variable wave types<br><br>'
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
            text='<b>Genetically Evolved Pattern - Cycle Phases</b><br>'
                 '<sup>Evolved over 25 generations | Variable wave types | Asymmetric regions</sup>',
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
    print("Using Genetically Evolved Pattern (90% damage reduction with realistic 45s transitions)")

    # Create animated version
    fig_animated = create_animation()
    fig_animated.write_html('optimal_mattress_3d.html', include_plotlyjs=True, full_html=True)
    print("Saved: optimal_mattress_3d.html")

    # Create static comparison
    fig_static = create_static_comparison()
    fig_static.write_html('optimal_mattress_phases.html', include_plotlyjs=True, full_html=True)
    print("Saved: optimal_mattress_phases.html")

    print("\nDone! Open optimal_mattress_3d.html to see the animated visualization.")
