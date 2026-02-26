import numpy as np

# -----------------------------
# GLOBAL CONSTANTS
# -----------------------------
BODY_WEIGHT = 80 * 9.81        # Newtons
SIM_TIME_HOURS = 8
TIME_STEP_SEC = 30
TOTAL_STEPS = int((SIM_TIME_HOURS * 3600) / TIME_STEP_SEC)

CONTACT_POINTS = 50
CAPILLARY_PRESSURE = 4.7      # kPa threshold

# -----------------------------
# BODY PRESSURE MODEL
# -----------------------------
def initialize_body_pressure():
    """
    Creates baseline pressure distribution
    Higher pressure at sacrum, heels, shoulders
    """
    base = np.ones(CONTACT_POINTS)
    base[10:15] *= 1.8   # sacrum
    base[30:35] *= 1.6   # heels
    base[5:8]   *= 1.4   # shoulders
    return base / base.sum()

# -----------------------------
# AIR MATTRESS MODEL
# -----------------------------
def create_air_cells(cell_size_cm):
    """
    Returns number of cells supporting body
    """
    body_area_cm2 = 180 * 60  # approx contact area
    cell_area = cell_size_cm ** 2
    return int(body_area_cm2 / cell_area)

def inflation_pattern(num_cells, step, cycle_steps):
    """
    Alternating inflation pattern
    """
    phase = (step // cycle_steps) % 2
    pattern = np.ones(num_cells)
    if phase == 1:
        pattern[::2] *= 0.3   # deflated cells
    else:
        pattern[1::2] *= 0.3
    return pattern

# -----------------------------
# PRESSURE COMPUTATION
# -----------------------------
def compute_pressure(body_map, air_pattern):
    """
    Redistributes load based on inflated cells
    """
    support_factor = air_pattern.mean()
    pressure = body_map * BODY_WEIGHT / support_factor
    pressure_kpa = pressure / 1000
    return pressure_kpa

# -----------------------------
# RUN SIMULATION
# -----------------------------
def run_simulation(cell_size, cycle_minutes):
    body = initialize_body_pressure()
    num_cells = create_air_cells(cell_size)
    cycle_steps = int((cycle_minutes * 60) / TIME_STEP_SEC)

    pressure_history = []

    for step in range(TOTAL_STEPS):
        air = inflation_pattern(num_cells, step, cycle_steps)
        pressure = compute_pressure(body, air)
        pressure_history.append(pressure)

    pressure_history = np.array(pressure_history)

    mean_pressure = pressure_history.mean()
    peak_pressure = pressure_history.max()
    pti = pressure_history.sum() * TIME_STEP_SEC

    return mean_pressure, peak_pressure, pti

# -----------------------------
# EXPERIMENTAL RUNS
# -----------------------------
configs = [
    (6, 3), (6, 60), (6, 180),
    (10, 3), (10, 60), (10, 180),
    (14, 3), (14, 60), (14, 180)
]

results = {}

for cell, cycle in configs:
    results[(cell, cycle)] = run_simulation(cell, cycle)

# Manual repositioning baseline
manual_mean, manual_peak, manual_pti = run_simulation(14, 120)

print("Simulation complete.")
