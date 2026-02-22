#!/usr/bin/env python3
"""
Bed Rotation Visualization
==========================
Creates an animation showing the lateral rotation bed tilting
with pressure redistribution visualization.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Ellipse, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# Simulation parameters
ROTATION_PERIOD = 10  # seconds for full cycle (sped up for visualization)
MAX_ANGLE = 30  # degrees
DURATION = 20  # seconds total
FPS = 20


def create_bed_animation():
    """Create animation of rotating bed with patient and pressure map."""

    fig = plt.figure(figsize=(14, 10))

    # Create subplots
    ax_side = fig.add_subplot(2, 2, 1)  # Side view of bed
    ax_front = fig.add_subplot(2, 2, 2)  # Front view showing tilt
    ax_pressure = fig.add_subplot(2, 2, 3)  # Pressure heatmap
    ax_angle = fig.add_subplot(2, 2, 4)  # Angle over time

    # Initialize data
    n_frames = DURATION * FPS
    times = np.linspace(0, DURATION, n_frames)
    angles = MAX_ANGLE * np.sin(2 * np.pi * times / ROTATION_PERIOD)

    # Create pressure grid (simplified body shape)
    grid_rows, grid_cols = 40, 18

    # Create body mask (elliptical body shape)
    y_coords, x_coords = np.ogrid[:grid_rows, :grid_cols]
    body_center_y, body_center_x = grid_rows // 2, grid_cols // 2
    body_mask = ((y_coords - body_center_y) / (grid_rows * 0.45)) ** 2 + \
                ((x_coords - body_center_x) / (grid_cols * 0.4)) ** 2 < 1

    # Base pressure distribution (higher at bony prominences)
    base_pressure = np.zeros((grid_rows, grid_cols))
    # Sacrum (high pressure)
    base_pressure[int(grid_rows*0.55):int(grid_rows*0.65),
                  int(grid_cols*0.35):int(grid_cols*0.65)] = 45
    # Heels
    base_pressure[int(grid_rows*0.85):int(grid_rows*0.95),
                  int(grid_cols*0.2):int(grid_cols*0.35)] = 35
    base_pressure[int(grid_rows*0.85):int(grid_rows*0.95),
                  int(grid_cols*0.65):int(grid_cols*0.8)] = 35
    # Shoulders
    base_pressure[int(grid_rows*0.15):int(grid_rows*0.25),
                  int(grid_cols*0.15):int(grid_cols*0.35)] = 30
    base_pressure[int(grid_rows*0.15):int(grid_rows*0.25),
                  int(grid_cols*0.65):int(grid_cols*0.85)] = 30
    # General body contact
    base_pressure[body_mask] = np.maximum(base_pressure[body_mask], 15)

    # Setup side view (top-down view of bed)
    ax_side.set_xlim(-10, 210)
    ax_side.set_ylim(-10, 100)
    ax_side.set_aspect('equal')
    ax_side.set_title('Top View - Patient on Bed', fontsize=12, fontweight='bold')
    ax_side.set_xlabel('Length (cm)')
    ax_side.set_ylabel('Width (cm)')

    # Draw bed frame (static)
    bed_frame = FancyBboxPatch((0, 0), 200, 90, boxstyle="round,pad=0.02",
                                facecolor='#8B4513', edgecolor='#5D3A1A', linewidth=3)
    ax_side.add_patch(bed_frame)

    # Draw mattress sections
    mattress_sections = []
    section_width = 200 / 8
    for i in range(8):
        section = Rectangle((i * section_width + 2, 2), section_width - 4, 86,
                           facecolor='#4A90D9', edgecolor='#2E5A8B', linewidth=1)
        ax_side.add_patch(section)
        mattress_sections.append(section)

    # Draw patient body (ellipses for body parts)
    # Head
    head = Ellipse((25, 45), 20, 25, facecolor='#FFD5B8', edgecolor='#CC9966', linewidth=2)
    ax_side.add_patch(head)
    # Torso
    torso = Ellipse((80, 45), 70, 50, facecolor='#FFD5B8', edgecolor='#CC9966', linewidth=2)
    ax_side.add_patch(torso)
    # Legs
    left_leg = Ellipse((155, 30), 60, 18, facecolor='#FFD5B8', edgecolor='#CC9966', linewidth=2)
    right_leg = Ellipse((155, 60), 60, 18, facecolor='#FFD5B8', edgecolor='#CC9966', linewidth=2)
    ax_side.add_patch(left_leg)
    ax_side.add_patch(right_leg)

    # Setup front view
    ax_front.set_xlim(-60, 60)
    ax_front.set_ylim(-10, 50)
    ax_front.set_aspect('equal')
    ax_front.set_title('Front View - Bed Tilt', fontsize=12, fontweight='bold')
    ax_front.set_xlabel('Width (cm)')
    ax_front.set_ylabel('Height (cm)')
    ax_front.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Setup pressure heatmap
    pressure_img = ax_pressure.imshow(base_pressure, cmap='YlOrRd',
                                       vmin=0, vmax=60, aspect='auto',
                                       origin='upper')
    ax_pressure.set_title('Pressure Distribution (mmHg)', fontsize=12, fontweight='bold')
    ax_pressure.set_xlabel('Left ← → Right')
    ax_pressure.set_ylabel('Head ← → Feet')
    cbar = plt.colorbar(pressure_img, ax=ax_pressure, label='Pressure (mmHg)')

    # Add threshold line to colorbar
    ax_pressure.axhline(y=0, color='white', alpha=0)  # placeholder

    # Setup angle plot
    ax_angle.set_xlim(0, DURATION)
    ax_angle.set_ylim(-MAX_ANGLE - 5, MAX_ANGLE + 5)
    ax_angle.set_title('Bed Rotation Angle', fontsize=12, fontweight='bold')
    ax_angle.set_xlabel('Time (seconds)')
    ax_angle.set_ylabel('Tilt Angle (degrees)')
    ax_angle.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_angle.fill_between([0, DURATION], -MAX_ANGLE, MAX_ANGLE, alpha=0.1, color='blue')
    angle_line, = ax_angle.plot([], [], 'b-', linewidth=2)
    angle_marker, = ax_angle.plot([], [], 'ro', markersize=10)

    # Text annotations
    angle_text = ax_front.text(0, 45, '', fontsize=14, ha='center', fontweight='bold')
    time_text = ax_angle.text(DURATION * 0.02, MAX_ANGLE * 0.85, '', fontsize=11)

    # Front view elements (will be updated)
    bed_surface = None
    patient_body = None

    def get_tilted_pressure(base_pressure, angle):
        """Calculate pressure redistribution based on tilt angle."""
        rows, cols = base_pressure.shape

        # Weight shift factor
        shift_factor = 0.015 * angle  # 1.5% per degree

        # Create gradient across columns
        col_positions = np.linspace(-1, 1, cols)
        weight_gradient = 1.0 + shift_factor * col_positions

        # Apply gradient
        tilted = base_pressure.copy()
        for i in range(rows):
            tilted[i, :] = base_pressure[i, :] * weight_gradient

        return np.clip(tilted, 0, 60)

    def init():
        angle_line.set_data([], [])
        angle_marker.set_data([], [])
        return []

    def animate(frame):
        nonlocal bed_surface, patient_body

        t = times[frame]
        angle = angles[frame]

        # Update front view - clear and redraw
        ax_front.clear()
        ax_front.set_xlim(-60, 60)
        ax_front.set_ylim(-10, 50)
        ax_front.set_aspect('equal')
        ax_front.set_title('Front View - Bed Tilt', fontsize=12, fontweight='bold')
        ax_front.set_xlabel('Width (cm)')
        ax_front.set_ylabel('Height (cm)')
        ax_front.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Draw tilted bed frame
        angle_rad = np.radians(angle)
        bed_width = 90
        bed_height = 8

        # Bed corners (centered at origin, then tilted)
        corners_x = np.array([-bed_width/2, bed_width/2, bed_width/2, -bed_width/2, -bed_width/2])
        corners_y = np.array([0, 0, bed_height, bed_height, 0])

        # Rotate
        rotated_x = corners_x * np.cos(angle_rad) - corners_y * np.sin(angle_rad)
        rotated_y = corners_x * np.sin(angle_rad) + corners_y * np.cos(angle_rad)

        # Draw bed
        ax_front.fill(rotated_x, rotated_y + 5, color='#4A90D9', edgecolor='#2E5A8B', linewidth=2)
        ax_front.plot(rotated_x, rotated_y + 5, 'k-', linewidth=2)

        # Draw patient on bed (ellipse)
        patient_width = 50
        patient_height = 25

        # Patient center on bed surface
        patient_cx = 0
        patient_cy = bed_height + patient_height/2 + 2

        # Rotate patient position
        rot_cx = patient_cx * np.cos(angle_rad) - patient_cy * np.sin(angle_rad)
        rot_cy = patient_cx * np.sin(angle_rad) + patient_cy * np.cos(angle_rad) + 5

        patient = Ellipse((rot_cx, rot_cy), patient_width, patient_height,
                         angle=angle, facecolor='#FFD5B8', edgecolor='#CC9966', linewidth=2)
        ax_front.add_patch(patient)

        # Draw bed frame/supports
        ax_front.plot([-55, -55], [0, 5], 'k-', linewidth=4)
        ax_front.plot([55, 55], [0, 5], 'k-', linewidth=4)

        # Angle indicator arc
        arc_angles = np.linspace(90, 90 - angle, 20)
        arc_x = 30 * np.cos(np.radians(arc_angles))
        arc_y = 30 * np.sin(np.radians(arc_angles))
        if abs(angle) > 1:
            ax_front.plot(arc_x, arc_y, 'r-', linewidth=2)

        # Angle text
        direction = "RIGHT" if angle > 0 else "LEFT" if angle < 0 else "CENTER"
        ax_front.text(0, 42, f'{abs(angle):.1f}° {direction}',
                     fontsize=14, ha='center', fontweight='bold',
                     color='red' if abs(angle) > 20 else 'blue')

        # Update pressure map
        tilted_pressure = get_tilted_pressure(base_pressure, angle)
        pressure_img.set_array(tilted_pressure)

        # Add arrow showing weight shift direction
        ax_pressure.clear()
        ax_pressure.imshow(tilted_pressure, cmap='YlOrRd', vmin=0, vmax=60,
                          aspect='auto', origin='upper')
        ax_pressure.set_title('Pressure Distribution (mmHg)', fontsize=12, fontweight='bold')
        ax_pressure.set_xlabel('Left ← → Right')
        ax_pressure.set_ylabel('Head ← → Feet')

        if abs(angle) > 2:
            arrow_dir = 1 if angle > 0 else -1
            ax_pressure.annotate('', xy=(grid_cols/2 + arrow_dir*5, grid_rows/2),
                               xytext=(grid_cols/2 - arrow_dir*2, grid_rows/2),
                               arrowprops=dict(arrowstyle='->', color='white', lw=3))
            ax_pressure.text(grid_cols/2, grid_rows - 3,
                           f'Weight shifting {"RIGHT" if angle > 0 else "LEFT"}',
                           ha='center', color='white', fontsize=10, fontweight='bold')

        # Update angle plot
        angle_line.set_data(times[:frame+1], angles[:frame+1])
        angle_marker.set_data([t], [angle])

        # Time text - remove old text objects first
        for txt in list(ax_angle.texts):
            txt.remove()
        ax_angle.text(DURATION * 0.02, MAX_ANGLE * 0.85, f'Time: {t:.1f}s', fontsize=11)

        # Highlight current zone
        if angle > 15:
            ax_angle.axhspan(15, MAX_ANGLE, alpha=0.2, color='green')
            ax_angle.text(DURATION * 0.98, 22, 'RIGHT RELIEF', ha='right', fontsize=9, color='green')
        elif angle < -15:
            ax_angle.axhspan(-MAX_ANGLE, -15, alpha=0.2, color='green')
            ax_angle.text(DURATION * 0.98, -22, 'LEFT RELIEF', ha='right', fontsize=9, color='green')

        return [angle_line, angle_marker]

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=1000/FPS, blit=False)

    plt.tight_layout()

    # Save as GIF
    print("Saving animation as bed_rotation.gif...")
    writer = PillowWriter(fps=FPS)
    anim.save('bed_rotation.gif', writer=writer)
    print("Saved: bed_rotation.gif")

    # Also save a few key frames as PNG
    for frame_idx, label in [(0, 'center'), (int(n_frames*0.25), 'right'),
                              (int(n_frames*0.5), 'center2'), (int(n_frames*0.75), 'left')]:
        animate(frame_idx)
        plt.savefig(f'bed_rotation_{label}.png', dpi=100, bbox_inches='tight')
        print(f"Saved: bed_rotation_{label}.png")

    plt.close()


if __name__ == "__main__":
    print("Creating bed rotation visualization...")
    create_bed_animation()
    print("\nDone! Files created:")
    print("  - bed_rotation.gif (animated)")
    print("  - bed_rotation_center.png")
    print("  - bed_rotation_right.png")
    print("  - bed_rotation_left.png")
