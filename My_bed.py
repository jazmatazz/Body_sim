import pyvista as pv
import numpy as np
import time

# Mattress grid
ROWS, COLS = 30, 18
BASE = 20
AMP = 2.0
CYCLE_SEC = 10

x = np.linspace(-1, 1, COLS)
y = np.linspace(-1, 1, ROWS)
X, Y = np.meshgrid(x, y)
Z = np.full_like(X, BASE)

# Bed mask
corner_radius = 0.15
bed_mask = ((np.abs(X) <= 1 - corner_radius) & (np.abs(Y) <= 1 - corner_radius))
corner_mask = (((np.abs(X) - (1 - corner_radius))**2 + (np.abs(Y) - (1 - corner_radius))**2) <= corner_radius**2)
bed_mask = bed_mask | corner_mask
Z[~bed_mask] = np.nan

# Structured grid
points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = (COLS, ROWS, 1)

# BackgroundPlotter (supports add_callback)
plotter = pv.BackgroundPlotter(window_size=[1000, 600])
mesh_actor = plotter.add_mesh(grid, scalars=Z.flatten(), cmap="viridis", show_scalar_bar=True)

# Stick figure as a simple cylinder
plotter.add_mesh(pv.Cylinder(center=(0,0,BASE+AMP+0.5),
                             direction=(0,0,1),
                             radius=0.05, height=1),
                 color="black")

# Gaussian weight function
def gaussian_pressure(x0, y0, sigma_x, sigma_y, weight):
    return weight * np.exp(-(((X - x0)/sigma_x)**2 + ((Y - y0)/sigma_y)**2))

# Update function
def update():
    t = time.time()
    phase = int((t % CYCLE_SEC) / (CYCLE_SEC / 2))
    newZ = np.zeros_like(X)

    for i in range(ROWS):
        for j in range(COLS):
            if bed_mask[i, j]:
                if (i % 2 == 0 and phase == 0) or (i % 2 == 1 and phase == 1):
                    newZ[i, j] = BASE + AMP
                else:
                    newZ[i, j] = BASE - AMP
            else:
                newZ[i, j] = np.nan

    # Stick figure weight
    newZ += gaussian_pressure(0, 0, 0.2, 0.3, 8)      # torso/head
    newZ += gaussian_pressure(-0.2, 0.3, 0.1, 0.1, 3) # left arm
    newZ += gaussian_pressure(0.2, 0.3, 0.1, 0.1, 3)  # right arm
    newZ += gaussian_pressure(-0.2, -0.5, 0.1, 0.1, 4)# left leg
    newZ += gaussian_pressure(0.2, -0.5, 0.1, 0.1, 4) # right leg

    # Update mesh
    grid.points[:,2] = newZ.flatten()
    grid["Pressure"] = newZ.flatten()

# Animate
plotter.add_callback(update, interval=100)
