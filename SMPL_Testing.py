import numpy as np
import matplotlib.pyplot as plt

# 4x4 Matrices (copy from earlier)
pelvis = np.array([[1,0,0,0],[0,1,0,0.45],[0,0,1,0],[0,0,0,1]])
spine_lower = np.array([[1,0,0,0],[0,1,0,0.55],[0,0,1,0],[0,0,0,1]])
spine_upper = np.array([[1,0,0,0],[0,1,0,0.65],[0,0,1,0],[0,0,0,1]])
chest = np.array([[1,0,0,0],[0,1,0,0.75],[0,0,1,0],[0,0,0,1]])
shoulders = np.array([[1,0,0,0],[0,1,0,0.85],[0,0,1,0],[0,0,0,1]])
head = np.array([[1,0,0,0],[0,1,0,0.95],[0,0,1,0],[0,0,0,1]])
left_leg = np.array([[1,0,0,-0.1],[0,1,0,0.25],[0,0,1,0],[0,0,0,1]])
right_leg = np.array([[1,0,0,0.1],[0,1,0,0.25],[0,0,1,0],[0,0,0,1]])

body_segments = {
    "Pelvis": pelvis,
    "Spine Lower": spine_lower,
    "Spine Upper": spine_upper,
    "Chest": chest,
    "Shoulders": shoulders,
    "Head": head,
    "Left Leg": left_leg,
    "Right Leg": right_leg
}

# ---------------------------
# Visualization
# ---------------------------
# Extract translation components for plotting
positions = {name: mat[:3,3] for name, mat in body_segments.items()}

# Skeleton connections
connections = [
    ("Pelvis", "Spine Lower"),
    ("Spine Lower", "Spine Upper"),
    ("Spine Upper", "Chest"),
    ("Chest", "Shoulders"),
    ("Shoulders", "Head"),
    ("Pelvis", "Left Leg"),
    ("Pelvis", "Right Leg")
]

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for name, pos in positions.items():
    ax.scatter(pos[0], pos[1], pos[2], s=50, label=name)

for seg1, seg2 in connections:
    p1 = positions[seg1]
    p2 = positions[seg2]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', linewidth=2)

ax.set_xlabel("X (left-right)")
ax.set_ylabel("Y (height)")
ax.set_zlabel("Z (head-foot)")
ax.view_init(elev=20, azim=-90)
ax.set_title("Supine Human Skeleton (Approximate)")
plt.legend()
plt.show()
