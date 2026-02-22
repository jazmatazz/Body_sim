#!/usr/bin/env python3
"""
3D Bed Visualization with Patient
==================================
Shows the lateral rotation bed AND alternating pressure mattress
in 3D with animated cells and patient body.
"""

import numpy as np
import plotly.graph_objects as go

# Bed parameters
BED_LENGTH = 200  # cm
BED_WIDTH = 90    # cm
BED_HEIGHT = 20   # cm
CELL_ROWS = 8     # mattress cells along length
CELL_COLS = 3     # mattress cells along width
CELL_HEIGHT_MAX = 18  # cm max cell height (inflated)
CELL_HEIGHT_MIN = 6   # cm min cell height (deflated)
MAX_ANGLE = 30    # degrees

# Alternating pressure parameters
ALTERNATING_GROUPS = 2  # cells alternate in 2 groups

def load_body_mesh():
    """Load SMPL body mesh for visualization."""
    try:
        import smplx
        import torch

        smpl_dir = '/Users/jaspermetz/Documents/Body_Sim/smpl'
        model = smplx.create(smpl_dir, model_type='smpl', gender='neutral')

        # Pose: supine with arms at sides
        body_pose = torch.zeros(1, 69)
        body_pose[0, 47] = -1.5  # L_shoulder Z-rot
        body_pose[0, 50] = 1.5   # R_shoulder Z-rot

        output = model(body_pose=body_pose)
        verts = output.vertices.detach().numpy()[0]
        faces = model.faces

        # Rotate from standing to supine
        angle = -np.pi / 2
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ])
        v = verts @ Rx.T
        v *= 100  # meters to cm

        # Scale to body length
        raw_length = v[:, 2].max() - v[:, 2].min()
        scale = 170 / raw_length
        v *= scale

        # Position: head at front, centered on bed
        v[:, 2] = v[:, 2].max() - v[:, 2] + v[:, 2].min()
        v[:, 2] -= v[:, 2].min()
        v[:, 2] += 15  # offset from head of bed

        v[:, 1] -= v[:, 1].min()
        v[:, 1] += BED_HEIGHT + CELL_HEIGHT_MAX  # on top of mattress

        v[:, 0] -= v[:, 0].mean()
        v[:, 0] += BED_WIDTH / 2  # center on bed

        return v, faces

    except Exception as e:
        print(f"Could not load SMPL model: {e}")
        print("Using simplified body shape...")
        return create_simple_body()


def create_simple_body():
    """Create a simplified body mesh if SMPL not available."""
    # Create ellipsoid-based body parts
    verts = []
    faces = []

    def add_ellipsoid(cx, cy, cz, rx, ry, rz, n_lat=10, n_lon=20):
        start_idx = len(verts)
        for i in range(n_lat + 1):
            lat = np.pi * i / n_lat - np.pi/2
            for j in range(n_lon):
                lon = 2 * np.pi * j / n_lon
                x = cx + rx * np.cos(lat) * np.cos(lon)
                y = cy + ry * np.cos(lat) * np.sin(lon)
                z = cz + rz * np.sin(lat)
                verts.append([x, y, z])

        for i in range(n_lat):
            for j in range(n_lon):
                p1 = start_idx + i * n_lon + j
                p2 = start_idx + i * n_lon + (j + 1) % n_lon
                p3 = start_idx + (i + 1) * n_lon + j
                p4 = start_idx + (i + 1) * n_lon + (j + 1) % n_lon
                faces.append([p1, p2, p3])
                faces.append([p2, p4, p3])

    body_y = BED_HEIGHT + CELL_HEIGHT_MAX + 10

    # Head
    add_ellipsoid(BED_WIDTH/2, body_y, 25, 8, 10, 12)
    # Torso
    add_ellipsoid(BED_WIDTH/2, body_y, 70, 18, 12, 35)
    # Pelvis
    add_ellipsoid(BED_WIDTH/2, body_y, 115, 16, 10, 15)
    # Left leg
    add_ellipsoid(BED_WIDTH/2 - 10, body_y, 155, 7, 8, 35)
    # Right leg
    add_ellipsoid(BED_WIDTH/2 + 10, body_y, 155, 7, 8, 35)

    return np.array(verts), np.array(faces)


def rotate_points(points, angle_deg, pivot_y=BED_WIDTH/2, pivot_z=BED_HEIGHT):
    """Rotate points around the bed's length axis."""
    angle_rad = np.radians(angle_deg)
    rotated = points.copy()

    # Translate to pivot
    rotated[:, 1] -= pivot_y
    rotated[:, 2] -= pivot_z

    # Rotate around X axis
    y_new = rotated[:, 1] * np.cos(angle_rad) - rotated[:, 2] * np.sin(angle_rad)
    z_new = rotated[:, 1] * np.sin(angle_rad) + rotated[:, 2] * np.cos(angle_rad)

    rotated[:, 1] = y_new + pivot_y
    rotated[:, 2] = z_new + pivot_z

    return rotated


def create_mattress_cells(angle_deg=0, alternating_phase=0):
    """
    Create 3D mattress cell meshes at a given tilt angle and alternating phase.

    Args:
        angle_deg: Lateral rotation angle
        alternating_phase: 0-1 phase of alternating pressure cycle
    """
    cell_length = BED_LENGTH / CELL_ROWS
    cell_width = BED_WIDTH / CELL_COLS

    cells = []

    for row in range(CELL_ROWS):
        for col in range(CELL_COLS):
            x0 = row * cell_length + 2
            x1 = (row + 1) * cell_length - 2
            y0 = col * cell_width + 2
            y1 = (col + 1) * cell_width - 2

            # Determine cell group for alternating pressure
            # Cells alternate in a checkerboard-like pattern along length
            cell_group = row % ALTERNATING_GROUPS

            # Calculate cell height based on alternating phase
            # Group 0 inflates when phase is 0-0.5, deflates 0.5-1
            # Group 1 is opposite
            if cell_group == 0:
                inflation = 0.5 + 0.5 * np.cos(2 * np.pi * alternating_phase)
            else:
                inflation = 0.5 - 0.5 * np.cos(2 * np.pi * alternating_phase)

            cell_height = CELL_HEIGHT_MIN + (CELL_HEIGHT_MAX - CELL_HEIGHT_MIN) * inflation

            # Box vertices (before rotation)
            verts = np.array([
                [x0, y0, BED_HEIGHT],
                [x1, y0, BED_HEIGHT],
                [x1, y1, BED_HEIGHT],
                [x0, y1, BED_HEIGHT],
                [x0, y0, BED_HEIGHT + cell_height],
                [x1, y0, BED_HEIGHT + cell_height],
                [x1, y1, BED_HEIGHT + cell_height],
                [x0, y1, BED_HEIGHT + cell_height],
            ], dtype=float)

            # Rotate for lateral tilt
            rotated = rotate_points(verts, angle_deg)

            # Pressure/color based on inflation (higher = more pressure = redder)
            # Also factor in lateral tilt
            tilt_factor = 1 + 0.3 * np.sin(np.radians(angle_deg)) * (col - 1)
            pressure = inflation * tilt_factor

            cells.append({
                'verts': rotated,
                'pressure': pressure,
                'inflation': inflation,
                'group': cell_group
            })

    return cells


def create_3d_bed_with_body():
    """Create animated 3D bed with body visualization."""

    print("Loading body mesh...")
    body_verts, body_faces = load_body_mesh()

    # Swap axes for plotly (X=length, Y=width, Z=height)
    # SMPL gives us: X=width, Y=height, Z=length
    # We want: X=width, Y=length(Z), Z=height(Y) -> then swap for bed orientation
    body_plot = np.zeros_like(body_verts)
    body_plot[:, 0] = body_verts[:, 0]  # width stays
    body_plot[:, 1] = body_verts[:, 2]  # length from Z
    body_plot[:, 2] = body_verts[:, 1]  # height from Y

    print("Creating animation frames...")
    n_frames = 60
    frames = []

    # Animation cycles:
    # - Lateral rotation: 1 full cycle over n_frames
    # - Alternating pressure: 2 full cycles over n_frames (faster)
    LATERAL_CYCLES = 1
    ALTERNATING_CYCLES = 3

    for i in range(n_frames):
        frame_phase = i / n_frames

        # Lateral rotation angle
        angle = MAX_ANGLE * np.sin(2 * np.pi * LATERAL_CYCLES * frame_phase)

        # Alternating pressure phase
        alt_phase = (ALTERNATING_CYCLES * frame_phase) % 1.0

        frame_data = []

        # Get cells first to calculate average height for body positioning
        cells = create_mattress_cells(angle, alt_phase)

        # Calculate body height adjustment based on cell inflation under body
        avg_cell_height = np.mean([CELL_HEIGHT_MIN + (CELL_HEIGHT_MAX - CELL_HEIGHT_MIN) * c['inflation']
                                   for c in cells])

        # Adjust body height based on mattress
        body_adjusted = body_plot.copy()
        body_adjusted[:, 2] = body_adjusted[:, 2] - (CELL_HEIGHT_MAX - avg_cell_height) * 0.3

        # Rotated body
        rotated_body = rotate_points(body_adjusted, angle)

        frame_data.append(go.Mesh3d(
            x=rotated_body[:, 1],  # swap for bed: length along X
            y=rotated_body[:, 0],  # width along Y
            z=rotated_body[:, 2],
            i=body_faces[:, 0],
            j=body_faces[:, 1],
            k=body_faces[:, 2],
            color='peachpuff',
            opacity=0.9,
            name='Patient'
        ))

        # Mattress cells with color based on inflation
        for idx, cell in enumerate(cells):
            v = cell['verts']
            inflation = cell['inflation']
            group = cell['group']

            # Color: inflated = blue/purple (high pressure), deflated = green (relieved)
            if inflation > 0.5:
                # Inflated - blue to red gradient
                r = int(80 + 150 * (inflation - 0.5) * 2)
                g = int(80)
                b = int(220 - 100 * (inflation - 0.5) * 2)
            else:
                # Deflated - green gradient (pressure relief)
                r = int(80)
                g = int(150 + 80 * (0.5 - inflation) * 2)
                b = int(80)

            frame_data.append(go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=[0, 0, 0, 0, 4, 4, 0, 1, 1, 2, 2, 3],
                j=[1, 2, 3, 4, 5, 6, 1, 5, 2, 6, 3, 7],
                k=[2, 3, 4, 5, 6, 7, 5, 6, 6, 7, 7, 4],
                color=f'rgb({r},{g},{b})',
                opacity=0.9,
                flatshading=True,
                name=f'Cell {idx}'
            ))

        # Bed frame (base - doesn't rotate)
        frame_data.append(go.Scatter3d(
            x=[0, BED_LENGTH, BED_LENGTH, 0, 0],
            y=[0, 0, BED_WIDTH, BED_WIDTH, 0],
            z=[0, 0, 0, 0, 0],
            mode='lines',
            line=dict(color='saddlebrown', width=10),
            name='Base'
        ))

        # Bed frame top (rotates)
        top_corners = np.array([
            [0, 0, BED_HEIGHT],
            [BED_LENGTH, 0, BED_HEIGHT],
            [BED_LENGTH, BED_WIDTH, BED_HEIGHT],
            [0, BED_WIDTH, BED_HEIGHT],
            [0, 0, BED_HEIGHT],
        ], dtype=float)
        rotated_top = rotate_points(top_corners, angle)

        frame_data.append(go.Scatter3d(
            x=rotated_top[:, 0],
            y=rotated_top[:, 1],
            z=rotated_top[:, 2],
            mode='lines',
            line=dict(color='saddlebrown', width=10),
            name='Frame'
        ))

        # Vertical supports
        for x in [5, BED_LENGTH-5]:
            for y in [5, BED_WIDTH-5]:
                top_pt = rotate_points(np.array([[x, y, BED_HEIGHT]], dtype=float), angle)[0]
                frame_data.append(go.Scatter3d(
                    x=[x, top_pt[0]],
                    y=[y, top_pt[1]],
                    z=[0, top_pt[2]],
                    mode='lines',
                    line=dict(color='saddlebrown', width=8),
                    showlegend=False
                ))

        # Status indicator text
        direction = "RIGHT" if angle > 1 else "LEFT" if angle < -1 else "CENTER"
        alt_status = "A" if alt_phase < 0.5 else "B"
        frame_data.append(go.Scatter3d(
            x=[BED_LENGTH/2],
            y=[BED_WIDTH/2],
            z=[85],
            mode='text',
            text=[f'Tilt: {angle:.1f}° {direction}<br>Alternating: Group {alt_status}'],
            textfont=dict(size=14, color='darkblue'),
            showlegend=False
        ))

        frames.append(go.Frame(data=frame_data, name=str(i)))

    # Initial figure
    print("Building initial figure...")
    fig = go.Figure()

    # Add initial body
    fig.add_trace(go.Mesh3d(
        x=body_plot[:, 1],
        y=body_plot[:, 0],
        z=body_plot[:, 2],
        i=body_faces[:, 0],
        j=body_faces[:, 1],
        k=body_faces[:, 2],
        color='peachpuff',
        opacity=0.9,
        name='Patient'
    ))

    # Initial cells with alternating pressure
    cells = create_mattress_cells(0, 0)
    for idx, cell in enumerate(cells):
        v = cell['verts']
        inflation = cell['inflation']

        if inflation > 0.5:
            r, g, b = int(80 + 150 * (inflation - 0.5) * 2), 80, int(220 - 100 * (inflation - 0.5) * 2)
        else:
            r, g, b = 80, int(150 + 80 * (0.5 - inflation) * 2), 80

        fig.add_trace(go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=[0, 0, 0, 0, 4, 4, 0, 1, 1, 2, 2, 3],
            j=[1, 2, 3, 4, 5, 6, 1, 5, 2, 6, 3, 7],
            k=[2, 3, 4, 5, 6, 7, 5, 6, 6, 7, 7, 4],
            color=f'rgb({r},{g},{b})',
            opacity=0.9,
            flatshading=True,
            name=f'Cell {idx}'
        ))

    # Frame
    fig.add_trace(go.Scatter3d(
        x=[0, BED_LENGTH, BED_LENGTH, 0, 0],
        y=[0, 0, BED_WIDTH, BED_WIDTH, 0],
        z=[0, 0, 0, 0, 0],
        mode='lines',
        line=dict(color='saddlebrown', width=10),
        name='Base'
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, BED_LENGTH, BED_LENGTH, 0, 0],
        y=[0, 0, BED_WIDTH, BED_WIDTH, 0],
        z=[BED_HEIGHT]*5,
        mode='lines',
        line=dict(color='saddlebrown', width=10),
        name='Frame'
    ))

    for x in [5, BED_LENGTH-5]:
        for y in [5, BED_WIDTH-5]:
            fig.add_trace(go.Scatter3d(
                x=[x, x],
                y=[y, y],
                z=[0, BED_HEIGHT],
                mode='lines',
                line=dict(color='saddlebrown', width=8),
                showlegend=False
            ))

    fig.add_trace(go.Scatter3d(
        x=[BED_LENGTH/2],
        y=[BED_WIDTH/2],
        z=[85],
        mode='text',
        text=['Tilt: 0.0° CENTER<br>Alternating: Group A'],
        textfont=dict(size=14, color='darkblue'),
        showlegend=False
    ))

    fig.frames = frames

    fig.update_layout(
        title=dict(
            text='<b>Lateral Rotation + Alternating Pressure Bed</b><br><sup>Green = deflated (pressure relief) | Blue/Red = inflated (supporting)</sup>',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(title='Length (cm)', range=[-20, BED_LENGTH+20]),
            yaxis=dict(title='Width (cm)', range=[-20, BED_WIDTH+20]),
            zaxis=dict(title='Height (cm)', range=[-10, 90]),
            aspectmode='data',
            camera=dict(eye=dict(x=1.8, y=1.2, z=0.8))
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0.1,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=80, redraw=True),
                            fromcurrent=True,
                            mode='immediate'
                        )]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate'
                        )]
                    )
                ]
            )
        ],
        sliders=[{
            'active': 0,
            'currentvalue': {'prefix': 'Tilt: ', 'visible': True},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.05,
            'y': 0,
            'steps': [
                {
                    'args': [[str(i)], {'frame': {'duration': 80, 'redraw': True}, 'mode': 'immediate'}],
                    'label': f'{MAX_ANGLE * np.sin(2 * np.pi * i / n_frames):.0f}°',
                    'method': 'animate'
                }
                for i in range(0, n_frames, 2)
            ]
        }],
        showlegend=False,
        width=1100,
        height=850
    )

    print("Saving HTML...")
    fig.write_html('bed_3d_rotation.html', include_plotlyjs=True, full_html=True)
    print("Saved: bed_3d_rotation.html")

    return fig


if __name__ == "__main__":
    print("Creating 3D bed + body rotation visualization...")
    fig = create_3d_bed_with_body()
    print("\nDone! Open bed_3d_rotation.html to see the animated 3D bed with patient.")
