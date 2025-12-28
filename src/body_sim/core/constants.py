"""Physical constants and default values for the simulator."""

# Physics constants
GRAVITY = 9.81  # m/s^2

# SMPL model constants
SMPL_NUM_VERTICES = 6890
SMPL_NUM_FACES = 13776
SMPL_NUM_JOINTS = 24
SMPL_NUM_BETAS = 10
SMPL_NUM_POSE_PARAMS = 69  # 23 joints × 3 axis-angle

# Approximate area per SMPL vertex (for pressure calculations)
SMPL_VERTEX_AREA_M2 = 0.0003  # ~3 cm² per vertex

# Default mattress dimensions (standard hospital bed)
DEFAULT_MATTRESS_WIDTH = 0.762  # 30 inches in meters
DEFAULT_MATTRESS_HEIGHT = 1.854  # 73 inches in meters

# Default air cell properties
DEFAULT_CELL_MAX_HEIGHT = 0.127  # 5 inches in meters
DEFAULT_CELL_MIN_HEIGHT = 0.025  # 1 inch in meters
DEFAULT_CELL_HEIGHT = 0.063  # 2.5 inches (50% inflation)

# Default grid resolution
DEFAULT_GRID_ROWS = 16
DEFAULT_GRID_COLS = 32

# Bodies-at-rest original sensor grid specs (for PMR compatibility)
BAR_TAXEL_ROWS = 64
BAR_TAXEL_COLS = 27
BAR_INTER_TAXEL_DISTANCE = 0.0286  # meters

# Pressure thresholds (medical reference values)
# Capillary closing pressure ~32 mmHg = 4266 Pa
CAPILLARY_CLOSING_PRESSURE_PA = 4266.0
# High risk threshold ~70 mmHg = 9333 Pa
HIGH_RISK_PRESSURE_PA = 9333.0

# Contact detection threshold
CONTACT_THRESHOLD_M = 0.01  # 1 cm

# Body region row ranges (as fractions of mattress length, head=0)
BODY_REGION_ROWS = {
    "head": (0.0, 0.12),
    "shoulders": (0.12, 0.25),
    "upper_back": (0.25, 0.40),
    "lower_back": (0.40, 0.52),
    "sacrum": (0.52, 0.60),
    "hips": (0.60, 0.70),
    "thighs": (0.70, 0.85),
    "calves": (0.85, 0.95),
    "heels": (0.95, 1.0),
}
