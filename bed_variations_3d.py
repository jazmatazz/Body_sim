#!/usr/bin/env python3
"""
3D Bed Variations Visualization
================================
Shows all the different bed configurations tested in the comparison:
- Lateral rotation angles
- Alternating pressure speeds
- Combined strategies
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Bed parameters
BED_LENGTH = 80   # cm (scaled down for grid view)
BED_WIDTH = 36    # cm
BED_HEIGHT = 8    # cm
CELL_ROWS = 4     # mattress cells along length
CELL_COLS = 2     # mattress cells along width
CELL_HEIGHT_MAX = 8
CELL_HEIGHT_MIN = 3

def create_bed_mesh(x_offset, y_offset, angle_deg=0, alt_phase=0, label=""):
    """Create a single bed visualization at given offset."""

    traces = []

    cell_length = BED_LENGTH / CELL_ROWS
    cell_width = BED_WIDTH / CELL_COLS

    # Create mattress cells
    for row in range(CELL_ROWS):
        for col in range(CELL_COLS):
            x0 = x_offset + row * cell_length + 1
            x1 = x_offset + (row + 1) * cell_length - 1
            y0 = y_offset + col * cell_width + 1
            y1 = y_offset + (col + 1) * cell_width - 1

            # Alternating pressure
            cell_group = row % 2
            if cell_group == 0:
                inflation = 0.5 + 0.5 * np.cos(2 * np.pi * alt_phase)
            else:
                inflation = 0.5 - 0.5 * np.cos(2 * np.pi * alt_phase)

            cell_height = CELL_HEIGHT_MIN + (CELL_HEIGHT_MAX - CELL_HEIGHT_MIN) * inflation

            # Vertices
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

            # Apply rotation around bed center
            pivot_y = y_offset + BED_WIDTH / 2
            pivot_z = BED_HEIGHT
            angle_rad = np.radians(angle_deg)

            for i in range(len(verts)):
                y_centered = verts[i, 1] - pivot_y
                z_centered = verts[i, 2] - pivot_z
                verts[i, 1] = y_centered * np.cos(angle_rad) - z_centered * np.sin(angle_rad) + pivot_y
                verts[i, 2] = y_centered * np.sin(angle_rad) + z_centered * np.cos(angle_rad) + pivot_z

            # Color based on inflation
            if inflation > 0.5:
                r = int(100 + 120 * (inflation - 0.5) * 2)
                g = 80
                b = int(200 - 80 * (inflation - 0.5) * 2)
            else:
                r = 80
                g = int(140 + 80 * (0.5 - inflation) * 2)
                b = 80

            traces.append(go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=[0, 0, 0, 0, 4, 4, 0, 1, 1, 2, 2, 3],
                j=[1, 2, 3, 4, 5, 6, 1, 5, 2, 6, 3, 7],
                k=[2, 3, 4, 5, 6, 7, 5, 6, 6, 7, 7, 4],
                color=f'rgb({r},{g},{b})',
                opacity=0.9,
                flatshading=True,
                showlegend=False
            ))

    # Bed frame
    frame_corners = np.array([
        [x_offset, y_offset, BED_HEIGHT],
        [x_offset + BED_LENGTH, y_offset, BED_HEIGHT],
        [x_offset + BED_LENGTH, y_offset + BED_WIDTH, BED_HEIGHT],
        [x_offset, y_offset + BED_WIDTH, BED_HEIGHT],
        [x_offset, y_offset, BED_HEIGHT],
    ], dtype=float)

    # Rotate frame
    pivot_y = y_offset + BED_WIDTH / 2
    angle_rad = np.radians(angle_deg)
    for i in range(len(frame_corners)):
        y_centered = frame_corners[i, 1] - pivot_y
        z_centered = frame_corners[i, 2] - BED_HEIGHT
        frame_corners[i, 1] = y_centered * np.cos(angle_rad) - z_centered * np.sin(angle_rad) + pivot_y
        frame_corners[i, 2] = y_centered * np.sin(angle_rad) + z_centered * np.cos(angle_rad) + BED_HEIGHT

    traces.append(go.Scatter3d(
        x=frame_corners[:, 0],
        y=frame_corners[:, 1],
        z=frame_corners[:, 2],
        mode='lines',
        line=dict(color='saddlebrown', width=6),
        showlegend=False
    ))

    # Base
    traces.append(go.Scatter3d(
        x=[x_offset, x_offset + BED_LENGTH, x_offset + BED_LENGTH, x_offset, x_offset],
        y=[y_offset, y_offset, y_offset + BED_WIDTH, y_offset + BED_WIDTH, y_offset],
        z=[0, 0, 0, 0, 0],
        mode='lines',
        line=dict(color='gray', width=4),
        showlegend=False
    ))

    # Label
    traces.append(go.Scatter3d(
        x=[x_offset + BED_LENGTH/2],
        y=[y_offset + BED_WIDTH/2],
        z=[35],
        mode='text',
        text=[label],
        textfont=dict(size=11, color='black'),
        showlegend=False
    ))

    return traces


def create_all_variations():
    """Create visualization showing all bed variations."""

    fig = go.Figure()

    # Define all variations to show
    variations = [
        # Row 1: Rotation angles
        {'x': 0, 'y': 0, 'angle': 0, 'alt': 0, 'label': 'No Rotation<br>0°'},
        {'x': 100, 'y': 0, 'angle': 15, 'alt': 0, 'label': 'Gentle<br>15°'},
        {'x': 200, 'y': 0, 'angle': 25, 'alt': 0, 'label': 'Moderate<br>25°'},
        {'x': 300, 'y': 0, 'angle': 35, 'alt': 0, 'label': 'Aggressive<br>35°'},
        {'x': 400, 'y': 0, 'angle': 45, 'alt': 0, 'label': 'Maximum<br>45°'},

        # Row 2: Alternating pressure phases (no rotation)
        {'x': 0, 'y': 60, 'angle': 0, 'alt': 0.0, 'label': 'Alt Phase 0%<br>Group A up'},
        {'x': 100, 'y': 60, 'angle': 0, 'alt': 0.25, 'label': 'Alt Phase 25%<br>Transitioning'},
        {'x': 200, 'y': 60, 'angle': 0, 'alt': 0.5, 'label': 'Alt Phase 50%<br>Group B up'},
        {'x': 300, 'y': 60, 'angle': 0, 'alt': 0.75, 'label': 'Alt Phase 75%<br>Transitioning'},
        {'x': 400, 'y': 60, 'angle': 0, 'alt': 1.0, 'label': 'Alt Phase 100%<br>Group A up'},

        # Row 3: Combined (rotation + alternating)
        {'x': 0, 'y': 120, 'angle': -30, 'alt': 0, 'label': 'Combined<br>Left 30° + Alt A'},
        {'x': 100, 'y': 120, 'angle': -15, 'alt': 0.25, 'label': 'Combined<br>Left 15° + Trans'},
        {'x': 200, 'y': 120, 'angle': 0, 'alt': 0.5, 'label': 'Combined<br>Center + Alt B'},
        {'x': 300, 'y': 120, 'angle': 15, 'alt': 0.75, 'label': 'Combined<br>Right 15° + Trans'},
        {'x': 400, 'y': 120, 'angle': 30, 'alt': 1.0, 'label': 'Combined<br>Right 30° + Alt A'},

        # Row 4: Different strategies
        {'x': 0, 'y': 180, 'angle': 0, 'alt': 0.5, 'label': 'Baseline<br>No interventions'},
        {'x': 100, 'y': 180, 'angle': 20, 'alt': 0.5, 'label': 'Conservative<br>20° rotation'},
        {'x': 200, 'y': 180, 'angle': 30, 'alt': 0.25, 'label': 'Moderate<br>30° + fast alt'},
        {'x': 300, 'y': 180, 'angle': 35, 'alt': 0.1, 'label': 'Aggressive<br>35° + v.fast alt'},
        {'x': 400, 'y': 180, 'angle': 40, 'alt': 0, 'label': 'Maximum<br>40° + sync alt'},
    ]

    # Add all beds
    for v in variations:
        traces = create_bed_mesh(v['x'], v['y'], v['angle'], v['alt'], v['label'])
        for trace in traces:
            fig.add_trace(trace)

    # Row labels
    row_labels = [
        {'y': -20, 'text': '<b>ROTATION ANGLES</b>'},
        {'y': 40, 'text': '<b>ALTERNATING PRESSURE PHASES</b>'},
        {'y': 100, 'text': '<b>COMBINED (Rotation + Alternating)</b>'},
        {'y': 160, 'text': '<b>OPTIMIZATION STRATEGIES</b>'},
    ]

    for label in row_labels:
        fig.add_trace(go.Scatter3d(
            x=[-30],
            y=[label['y'] + BED_WIDTH/2],
            z=[20],
            mode='text',
            text=[label['text']],
            textfont=dict(size=12, color='darkblue'),
            showlegend=False
        ))

    fig.update_layout(
        title=dict(
            text='<b>All Bed Configuration Variations</b><br><sup>Green = deflated (relief) | Blue/Red = inflated (support) | Tilted = lateral rotation</sup>',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(title='', range=[-50, 520], showticklabels=False),
            yaxis=dict(title='', range=[-40, 240], showticklabels=False),
            zaxis=dict(title='Height (cm)', range=[-5, 45]),
            aspectmode='data',
            camera=dict(
                eye=dict(x=0.8, y=-1.5, z=0.8)
            )
        ),
        showlegend=False,
        width=1400,
        height=900
    )

    # Add legend/key
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text='<b>Color Key:</b><br>🟢 Green = Deflated (pressure relief)<br>🔵 Blue = Partially inflated<br>🔴 Red = Fully inflated (max support)<br><br><b>Tilt:</b> Bed frame angle shows rotation',
        showarrow=False,
        font=dict(size=12),
        align='left',
        bgcolor='white',
        bordercolor='gray',
        borderwidth=1
    )

    fig.write_html('bed_variations_3d.html', include_plotlyjs=True, full_html=True)
    print("Saved: bed_variations_3d.html")

    return fig


if __name__ == "__main__":
    print("Creating all bed variations visualization...")
    fig = create_all_variations()
    print("\nDone! Open bed_variations_3d.html to see all configurations.")
