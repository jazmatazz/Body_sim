#!/usr/bin/env python3
"""
Pressure Ulcer Prevention Simulation
=====================================
A biomedical simulation modeling pressure distribution for a human body
in supine position on an adaptive air mattress.

Physical Assumptions:
- Body represented as 6 anatomical regions with mass distributions
- Mattress modeled as 2D grid of interconnected air cells
- Pressure follows modified Hertzian contact mechanics
- Viscoelastic creep modeled with exponential time decay
- Bed sore risk follows cumulative damage model (pressure × time)

Limitations:
- 2D simplification (no depth modeling)
- Homogeneous tissue properties per region
- Simplified airflow dynamics between cells
- No thermal effects considered
- Linear superposition of pressure contributions

Author: Biomedical Simulation Engineer
Dependencies: NumPy, Matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTS AND PARAMETERS
# =============================================================================

# Physical constants
GRAVITY = 9.81  # m/s²
MMHG_PER_PASCAL = 0.00750062  # Conversion factor

# Default body parameters (70 kg adult male, supine)
DEFAULT_BODY_MASS = 70.0  # kg
DEFAULT_BODY_LENGTH = 180  # cm
DEFAULT_BODY_WIDTH = 45  # cm

# Mattress parameters
DEFAULT_CELL_SIZE = 5  # cm per cell
DEFAULT_MATTRESS_LENGTH = 200  # cm
DEFAULT_MATTRESS_WIDTH = 90  # cm
DEFAULT_MAX_CELL_PRESSURE = 50  # mmHg maximum per cell
DEFAULT_CELL_STIFFNESS = 0.5  # kPa/cm (air cell stiffness)

# Tissue parameters
CAPILLARY_CLOSING_PRESSURE = 32  # mmHg - threshold for tissue damage
TISSUE_DAMAGE_RATE = 0.01  # damage accumulation rate per second above threshold

# Viscoelastic parameters
CREEP_TIME_CONSTANT = 300  # seconds (5 minutes for significant creep)
MAX_CREEP_FACTOR = 1.3  # Maximum increase due to creep

# Simulation parameters
DEFAULT_TIMESTEP = 10  # seconds
DEFAULT_SIMULATION_TIME = 3600  # 1 hour in seconds


# =============================================================================
# BODY MODEL - Anatomical Region Representation
# =============================================================================

class BodyModel:
    """
    Represents a human body in supine position using NumPy matrices.

    Each anatomical region has:
    - Mass distribution matrix (kg per grid cell)
    - Contact area distribution
    - Tissue stiffness properties

    Regions are arranged to enforce left-right symmetry.
    """

    # Anatomical region definitions with mass fractions and tissue properties
    # Mass fractions based on anthropometric data
    REGION_DEFINITIONS = {
        'head': {
            'mass_fraction': 0.08,      # 8% of body mass
            'length_fraction': 0.12,     # 12% of body length
            'width_fraction': 0.35,      # 35% of body width (narrower)
            'tissue_stiffness': 1.2,     # Relative stiffness (skull)
            'contact_depth': 0.6,        # Fraction making contact
        },
        'shoulders': {
            'mass_fraction': 0.12,
            'length_fraction': 0.08,
            'width_fraction': 0.95,      # Wide at shoulders
            'tissue_stiffness': 0.8,     # Moderate (muscle/bone)
            'contact_depth': 0.7,
        },
        'upper_torso': {
            'mass_fraction': 0.15,
            'length_fraction': 0.15,
            'width_fraction': 0.75,
            'tissue_stiffness': 0.6,     # Softer (organs)
            'contact_depth': 0.5,
        },
        'lower_torso': {
            'mass_fraction': 0.18,
            'length_fraction': 0.12,
            'width_fraction': 0.70,
            'tissue_stiffness': 0.7,
            'contact_depth': 0.6,
        },
        'pelvis': {
            'mass_fraction': 0.15,
            'length_fraction': 0.10,
            'width_fraction': 0.80,
            'tissue_stiffness': 0.9,     # Bony prominence (sacrum)
            'contact_depth': 0.8,        # High contact - risk area
        },
        'thighs': {
            'mass_fraction': 0.20,
            'length_fraction': 0.23,
            'width_fraction': 0.60,
            'tissue_stiffness': 0.5,     # Soft tissue
            'contact_depth': 0.55,
        },
        'calves': {
            'mass_fraction': 0.08,
            'length_fraction': 0.18,
            'width_fraction': 0.35,
            'tissue_stiffness': 0.6,
            'contact_depth': 0.5,
        },
        'heels': {
            'mass_fraction': 0.04,
            'length_fraction': 0.02,
            'width_fraction': 0.25,
            'tissue_stiffness': 1.0,     # Bony (calcaneus)
            'contact_depth': 0.9,        # High contact - risk area
        },
    }

    def __init__(self, body_mass=DEFAULT_BODY_MASS,
                 body_length=DEFAULT_BODY_LENGTH,
                 body_width=DEFAULT_BODY_WIDTH,
                 grid_resolution=1.0):
        """
        Initialize body model.

        Args:
            body_mass: Total body mass in kg
            body_length: Body length in cm (head to heel)
            body_width: Maximum body width in cm
            grid_resolution: cm per grid cell
        """
        self.body_mass = body_mass
        self.body_length = body_length
        self.body_width = body_width
        self.grid_resolution = grid_resolution

        # Grid dimensions
        self.grid_rows = int(body_length / grid_resolution)
        self.grid_cols = int(body_width / grid_resolution)

        # Initialize matrices
        self.mass_matrix = np.zeros((self.grid_rows, self.grid_cols))
        self.stiffness_matrix = np.zeros((self.grid_rows, self.grid_cols))
        self.contact_matrix = np.zeros((self.grid_rows, self.grid_cols))
        self.region_map = np.zeros((self.grid_rows, self.grid_cols), dtype=int)

        # Build the body representation
        self._build_body()

    def _build_body(self):
        """Construct anatomical regions with symmetric mass distributions."""
        current_row = 0
        region_id = 1

        for region_name, props in self.REGION_DEFINITIONS.items():
            # Calculate region dimensions in grid cells
            region_length = int(props['length_fraction'] * self.grid_rows)
            region_width = int(props['width_fraction'] * self.grid_cols)

            # Ensure minimum size
            region_length = max(region_length, 2)
            region_width = max(region_width, 2)

            # Calculate region mass
            region_mass = props['mass_fraction'] * self.body_mass

            # Center the region horizontally (enforce symmetry)
            col_start = (self.grid_cols - region_width) // 2
            col_end = col_start + region_width
            row_end = min(current_row + region_length, self.grid_rows)

            # Create symmetric mass distribution (Gaussian-like profile)
            for i in range(current_row, row_end):
                for j in range(col_start, col_end):
                    # Distance from center (normalized)
                    center_j = (col_start + col_end) / 2
                    dist_from_center = abs(j - center_j) / (region_width / 2)

                    # Gaussian-like weight (symmetric left-right)
                    weight = np.exp(-2 * dist_from_center**2)

                    # Longitudinal variation (slight tapering at edges)
                    row_pos = (i - current_row) / max(region_length - 1, 1)
                    long_weight = 1.0 - 0.3 * (2 * row_pos - 1)**2

                    self.mass_matrix[i, j] = weight * long_weight
                    self.stiffness_matrix[i, j] = props['tissue_stiffness']
                    self.contact_matrix[i, j] = props['contact_depth'] * weight
                    self.region_map[i, j] = region_id

            # Normalize mass for this region
            region_slice = self.mass_matrix[current_row:row_end, col_start:col_end]
            if region_slice.sum() > 0:
                region_slice *= region_mass / region_slice.sum()

            current_row = row_end
            region_id += 1

        # Ensure exact left-right symmetry
        self._enforce_symmetry()

    def _enforce_symmetry(self):
        """Enforce exact left-right symmetry in all matrices."""
        mid = self.grid_cols // 2

        # Average left and right halves
        for matrix in [self.mass_matrix, self.stiffness_matrix, self.contact_matrix]:
            left_half = matrix[:, :mid]
            right_half = matrix[:, -mid:][:, ::-1]  # Flip right half

            # Average and apply symmetrically
            avg = (left_half + right_half) / 2
            matrix[:, :mid] = avg
            matrix[:, -mid:] = avg[:, ::-1]

    def get_pressure_weights(self):
        """
        Calculate pressure distribution weights based on mass and contact.

        Returns:
            2D array of pressure weights (normalized mass × contact)
        """
        weights = self.mass_matrix * self.contact_matrix
        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum() * self.body_mass
        return weights

    def apply_micro_shift(self, shift_x=0, shift_y=0):
        """
        Apply a small positional shift to the body (micro-repositioning).

        Args:
            shift_x: Horizontal shift in grid cells
            shift_y: Vertical shift in grid cells
        """
        shift_x = int(np.clip(shift_x, -5, 5))
        shift_y = int(np.clip(shift_y, -5, 5))

        self.mass_matrix = np.roll(self.mass_matrix, shift_y, axis=0)
        self.mass_matrix = np.roll(self.mass_matrix, shift_x, axis=1)

        self.contact_matrix = np.roll(self.contact_matrix, shift_y, axis=0)
        self.contact_matrix = np.roll(self.contact_matrix, shift_x, axis=1)


# =============================================================================
# MATTRESS MODEL - Adaptive Air Cell System
# =============================================================================

class AdaptiveMattress:
    """
    Models an adaptive air mattress as a 2D grid of interconnected air cells.

    Features:
    - Individual cell pressure control
    - Airflow between neighboring cells
    - Pressure equalization dynamics
    - Deformation response to load
    """

    def __init__(self, length=DEFAULT_MATTRESS_LENGTH,
                 width=DEFAULT_MATTRESS_WIDTH,
                 cell_size=DEFAULT_CELL_SIZE,
                 max_cell_pressure=DEFAULT_MAX_CELL_PRESSURE,
                 cell_stiffness=DEFAULT_CELL_STIFFNESS):
        """
        Initialize mattress model.

        Args:
            length: Mattress length in cm
            width: Mattress width in cm
            cell_size: Size of each air cell in cm
            max_cell_pressure: Maximum pressure per cell (mmHg)
            cell_stiffness: Cell spring constant (kPa/cm)
        """
        self.length = length
        self.width = width
        self.cell_size = cell_size
        self.max_cell_pressure = max_cell_pressure
        self.cell_stiffness = cell_stiffness

        # Grid dimensions
        self.n_cells_length = int(length / cell_size)
        self.n_cells_width = int(width / cell_size)

        # Cell state matrices
        self.cell_pressures = np.ones((self.n_cells_length, self.n_cells_width)) * 20  # Initial 20 mmHg
        self.cell_deformation = np.zeros((self.n_cells_length, self.n_cells_width))  # cm
        self.target_pressures = self.cell_pressures.copy()

        # Airflow conductance between cells (normalized)
        self.airflow_conductance = 0.1  # Rate of pressure equalization

    def apply_load(self, load_matrix):
        """
        Apply body load to mattress and compute deformation.

        Args:
            load_matrix: 2D array of loads (kg) at mattress resolution

        Returns:
            Deformation matrix (cm)
        """
        # Resample load to mattress grid if needed
        load_resampled = self._resample_to_mattress(load_matrix)

        # Calculate force per cell (N)
        force_matrix = load_resampled * GRAVITY

        # Calculate deformation using Hooke's law: F = k * x
        # Convert stiffness to N/cm (from kPa/cm, considering cell area)
        cell_area = self.cell_size**2  # cm²
        k_effective = self.cell_stiffness * 1000 * cell_area / 100  # N/cm

        # Deformation (cm)
        self.cell_deformation = force_matrix / (k_effective + 1e-6)

        # Limit deformation (mattress cannot deform infinitely)
        max_deformation = self.cell_size * 0.5  # Max 50% of cell size
        self.cell_deformation = np.clip(self.cell_deformation, 0, max_deformation)

        return self.cell_deformation

    def _resample_to_mattress(self, input_matrix):
        """Resample input matrix to mattress grid resolution."""
        from scipy.ndimage import zoom

        # Use simple averaging if scipy not available
        try:
            zoom_factors = (self.n_cells_length / input_matrix.shape[0],
                           self.n_cells_width / input_matrix.shape[1])
            return zoom(input_matrix, zoom_factors, order=1)
        except ImportError:
            # Fallback: simple block averaging
            return self._block_average(input_matrix,
                                       self.n_cells_length,
                                       self.n_cells_width)

    def _block_average(self, matrix, new_rows, new_cols):
        """Simple block averaging for resampling."""
        result = np.zeros((new_rows, new_cols))
        row_ratio = matrix.shape[0] / new_rows
        col_ratio = matrix.shape[1] / new_cols

        for i in range(new_rows):
            for j in range(new_cols):
                r_start = int(i * row_ratio)
                r_end = int((i + 1) * row_ratio)
                c_start = int(j * col_ratio)
                c_end = int((j + 1) * col_ratio)
                result[i, j] = matrix[r_start:r_end, c_start:c_end].mean()

        return result

    def simulate_airflow(self, dt):
        """
        Simulate air redistribution between neighboring cells.

        Args:
            dt: Time step in seconds
        """
        # Laplacian for diffusion (4-neighbor averaging)
        kernel = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])

        # Compute pressure gradients using convolution
        from scipy.signal import convolve2d
        try:
            pressure_laplacian = convolve2d(self.cell_pressures, kernel,
                                            mode='same', boundary='fill')
        except ImportError:
            # Fallback: manual convolution
            pressure_laplacian = self._manual_laplacian(self.cell_pressures)

        # Update pressures based on airflow (diffusion equation)
        flow_rate = self.airflow_conductance * dt / 10  # Normalized rate
        self.cell_pressures += flow_rate * pressure_laplacian

        # Enforce pressure bounds
        self.cell_pressures = np.clip(self.cell_pressures, 5, self.max_cell_pressure)

    def _manual_laplacian(self, matrix):
        """Compute discrete Laplacian without scipy."""
        laplacian = np.zeros_like(matrix)
        laplacian[1:-1, 1:-1] = (
            matrix[:-2, 1:-1] + matrix[2:, 1:-1] +
            matrix[1:-1, :-2] + matrix[1:-1, 2:] -
            4 * matrix[1:-1, 1:-1]
        )
        return laplacian

    def set_zone_pressure(self, zone_mask, pressure):
        """Set target pressure for a specific zone."""
        self.target_pressures[zone_mask] = pressure

    def adjust_to_targets(self, rate=0.1):
        """Gradually adjust cell pressures toward targets."""
        diff = self.target_pressures - self.cell_pressures
        self.cell_pressures += rate * diff


# =============================================================================
# PRESSURE COMPUTATION ENGINE
# =============================================================================

class PressureCalculator:
    """
    Computes interface pressure between body and mattress.

    Includes:
    - Load-based pressure calculation
    - Mattress deformation effects
    - Time-dependent viscoelastic creep
    """

    def __init__(self, body: BodyModel, mattress: AdaptiveMattress):
        """
        Initialize pressure calculator.

        Args:
            body: BodyModel instance
            mattress: AdaptiveMattress instance
        """
        self.body = body
        self.mattress = mattress

        # Creep state (accumulated deformation over time)
        self.creep_factor = np.ones((mattress.n_cells_length, mattress.n_cells_width))
        self.time_under_load = np.zeros((mattress.n_cells_length, mattress.n_cells_width))

    def compute_pressure(self, dt=0):
        """
        Compute interface pressure distribution.

        Pressure = (Force / Area) × Creep Factor

        Args:
            dt: Time step for creep update (seconds)

        Returns:
            Pressure matrix in mmHg
        """
        # Get body weight distribution
        weight_matrix = self.body.get_pressure_weights()

        # Apply load to mattress and get deformation
        deformation = self.mattress.apply_load(weight_matrix)

        # Resample body properties to mattress grid
        contact_resampled = self._resample(self.body.contact_matrix)
        stiffness_resampled = self._resample(self.body.stiffness_matrix)
        weight_resampled = self._resample(weight_matrix)

        # Calculate contact area per cell (cm²)
        cell_area = self.mattress.cell_size**2
        effective_area = cell_area * contact_resampled  # Actual contact area
        effective_area = np.maximum(effective_area, 0.1)  # Prevent division by zero

        # Base pressure: P = F / A (Pa)
        force = weight_resampled * GRAVITY  # N
        pressure_pa = force / (effective_area * 1e-4)  # Pa (area in m²)

        # Convert to mmHg
        pressure_mmhg = pressure_pa * MMHG_PER_PASCAL

        # Apply tissue stiffness factor (stiffer tissue = higher pressure)
        pressure_mmhg *= (0.5 + 0.5 * stiffness_resampled)

        # Apply deformation effect (more deformation = lower pressure, better distribution)
        deformation_factor = 1.0 / (1.0 + deformation / self.mattress.cell_size)
        pressure_mmhg *= deformation_factor

        # Apply viscoelastic creep
        if dt > 0:
            self._update_creep(pressure_mmhg, dt)
        pressure_mmhg *= self.creep_factor

        # Add mattress cell base pressure contribution
        pressure_mmhg += self.mattress.cell_pressures * 0.1  # Small contribution

        return pressure_mmhg

    def _resample(self, matrix):
        """Resample matrix to mattress resolution."""
        return self.mattress._resample_to_mattress(matrix)

    def _update_creep(self, pressure, dt):
        """
        Update viscoelastic creep factor.

        Creep increases over time under sustained pressure,
        following: creep = 1 + (max_creep - 1) × (1 - exp(-t/τ))

        Args:
            pressure: Current pressure matrix
            dt: Time step in seconds
        """
        # Update time under load (only where pressure is significant)
        pressure_mask = pressure > 10  # mmHg threshold
        self.time_under_load[pressure_mask] += dt
        self.time_under_load[~pressure_mask] *= 0.9  # Decay when unloaded

        # Compute creep factor
        normalized_time = self.time_under_load / CREEP_TIME_CONSTANT
        self.creep_factor = 1.0 + (MAX_CREEP_FACTOR - 1.0) * (1 - np.exp(-normalized_time))

    def reset_creep(self):
        """Reset creep state (e.g., after repositioning)."""
        self.creep_factor = np.ones_like(self.creep_factor)
        self.time_under_load = np.zeros_like(self.time_under_load)


# =============================================================================
# BED SORE RISK MODEL
# =============================================================================

class BedSoreRiskModel:
    """
    Models pressure ulcer (bed sore) risk based on:
    - Pressure magnitude above capillary closing pressure
    - Duration of sustained pressure

    Uses cumulative damage approach: Risk = ∫(P - P_threshold)⁺ dt
    """

    def __init__(self, shape, threshold=CAPILLARY_CLOSING_PRESSURE):
        """
        Initialize risk model.

        Args:
            shape: Shape of pressure grid (rows, cols)
            threshold: Pressure threshold for damage (mmHg)
        """
        self.threshold = threshold
        self.shape = shape

        # Cumulative risk metrics
        self.cumulative_exposure = np.zeros(shape)  # mmHg × seconds
        self.time_above_threshold = np.zeros(shape)  # seconds
        self.risk_score = np.zeros(shape)  # 0-1 normalized risk

        # Risk model parameters (based on literature)
        # Higher pressure = exponentially higher risk
        self.pressure_exponent = 1.5
        # Time threshold for significant damage (2 hours typical)
        self.critical_time = 7200  # seconds

    def update(self, pressure, dt):
        """
        Update risk metrics based on current pressure.

        Args:
            pressure: Current pressure matrix (mmHg)
            dt: Time step (seconds)
        """
        # Calculate excess pressure above threshold
        excess_pressure = np.maximum(pressure - self.threshold, 0)

        # Update time above threshold
        above_threshold = pressure > self.threshold
        self.time_above_threshold[above_threshold] += dt

        # Accumulate weighted exposure (pressure^exponent × time)
        exposure_increment = (excess_pressure ** self.pressure_exponent) * dt
        self.cumulative_exposure += exposure_increment

        # Compute normalized risk score (0-1)
        # Risk increases sigmoidally with cumulative exposure
        exposure_normalized = self.cumulative_exposure / (self.threshold * self.critical_time)
        self.risk_score = 1 - np.exp(-exposure_normalized)

    def get_risk_zones(self):
        """
        Identify risk zones by severity.

        Returns:
            Dictionary with 'low', 'moderate', 'high', 'critical' zone masks
        """
        return {
            'low': self.risk_score < 0.25,
            'moderate': (self.risk_score >= 0.25) & (self.risk_score < 0.5),
            'high': (self.risk_score >= 0.5) & (self.risk_score < 0.75),
            'critical': self.risk_score >= 0.75
        }

    def reset(self):
        """Reset all risk metrics."""
        self.cumulative_exposure = np.zeros(self.shape)
        self.time_above_threshold = np.zeros(self.shape)
        self.risk_score = np.zeros(self.shape)


# =============================================================================
# OPTIMIZATION ALGORITHM
# =============================================================================

class PressureOptimizer:
    """
    Optimization algorithm to minimize pressure ulcer risk.

    Strategies:
    1. Mattress pressure adjustment (zone-based)
    2. Patient micro-repositioning
    3. Alternating pressure cycling

    Objective: Minimize peak pressure and cumulative risk
    """

    def __init__(self, body: BodyModel, mattress: AdaptiveMattress,
                 pressure_calc: PressureCalculator, risk_model: BedSoreRiskModel):
        """
        Initialize optimizer.

        Args:
            body: BodyModel instance
            mattress: AdaptiveMattress instance
            pressure_calc: PressureCalculator instance
            risk_model: BedSoreRiskModel instance
        """
        self.body = body
        self.mattress = mattress
        self.pressure_calc = pressure_calc
        self.risk_model = risk_model

        # Optimization parameters
        self.reposition_interval = 1800  # 30 minutes
        self.last_reposition_time = 0
        self.pressure_cycle_period = 300  # 5 minutes for alternating pressure

        # Zone definitions (for targeted pressure control)
        self._define_zones()

    def _define_zones(self):
        """Define mattress zones for pressure control."""
        n_rows = self.mattress.n_cells_length
        n_cols = self.mattress.n_cells_width

        # Create zone masks
        self.zones = {
            'head': np.zeros((n_rows, n_cols), dtype=bool),
            'shoulders': np.zeros((n_rows, n_cols), dtype=bool),
            'torso': np.zeros((n_rows, n_cols), dtype=bool),
            'pelvis': np.zeros((n_rows, n_cols), dtype=bool),
            'legs': np.zeros((n_rows, n_cols), dtype=bool),
            'heels': np.zeros((n_rows, n_cols), dtype=bool),
        }

        # Define zone boundaries (proportional)
        zone_boundaries = [
            ('head', 0, 0.12),
            ('shoulders', 0.12, 0.22),
            ('torso', 0.22, 0.45),
            ('pelvis', 0.45, 0.55),
            ('legs', 0.55, 0.95),
            ('heels', 0.95, 1.0),
        ]

        for zone_name, start_frac, end_frac in zone_boundaries:
            start_row = int(start_frac * n_rows)
            end_row = int(end_frac * n_rows)
            self.zones[zone_name][start_row:end_row, :] = True

    def optimize_step(self, current_time, current_pressure):
        """
        Perform one optimization step.

        Args:
            current_time: Simulation time in seconds
            current_pressure: Current pressure distribution

        Returns:
            Dictionary with optimization actions taken
        """
        actions = {}

        # 1. Adjust mattress pressures based on current loading
        self._adjust_zone_pressures(current_pressure)
        actions['pressure_adjusted'] = True

        # 2. Apply alternating pressure cycling
        cycle_phase = (current_time % self.pressure_cycle_period) / self.pressure_cycle_period
        self._apply_pressure_cycling(cycle_phase)
        actions['cycle_phase'] = cycle_phase

        # 3. Check if repositioning is needed
        if current_time - self.last_reposition_time >= self.reposition_interval:
            if self._check_reposition_needed(current_pressure):
                shift = self._compute_optimal_shift(current_pressure)
                self.body.apply_micro_shift(shift[0], shift[1])
                self.pressure_calc.reset_creep()
                self.last_reposition_time = current_time
                actions['repositioned'] = True
                actions['shift'] = shift

        return actions

    def _adjust_zone_pressures(self, current_pressure):
        """Adjust mattress zone pressures to reduce peak pressure."""
        for zone_name, zone_mask in self.zones.items():
            # Get pressure in this zone
            zone_pressure = current_pressure[zone_mask]
            if len(zone_pressure) == 0:
                continue

            max_zone_pressure = zone_pressure.max()
            mean_zone_pressure = zone_pressure.mean()

            # If pressure is high, reduce mattress support (allow more sink-in)
            if max_zone_pressure > 40:  # mmHg
                target = 15  # Lower mattress pressure
            elif max_zone_pressure > 32:
                target = 18
            else:
                target = 22  # Normal support

            self.mattress.set_zone_pressure(zone_mask, target)

        self.mattress.adjust_to_targets(rate=0.05)

    def _apply_pressure_cycling(self, phase):
        """
        Apply alternating pressure pattern.

        Creates wave-like pressure distribution that shifts load periodically.
        """
        n_rows = self.mattress.n_cells_length

        # Create alternating high/low pressure rows
        row_indices = np.arange(n_rows)

        # Sinusoidal modulation
        modulation = 0.15 * np.sin(2 * np.pi * (row_indices / 10 + phase))

        # Apply modulation to each row
        for i, mod in enumerate(modulation):
            self.mattress.cell_pressures[i, :] *= (1 + mod)

    def _check_reposition_needed(self, current_pressure):
        """Check if patient repositioning is needed."""
        # Reposition if any area exceeds critical threshold
        max_pressure = current_pressure.max()
        high_risk_area = (self.risk_model.risk_score > 0.5).sum()

        return max_pressure > 45 or high_risk_area > 50

    def _compute_optimal_shift(self, current_pressure):
        """
        Compute optimal micro-shift direction.

        Shifts away from highest pressure areas.
        """
        n_rows, n_cols = current_pressure.shape

        # Find center of mass of high pressure region
        high_pressure_mask = current_pressure > 35
        if not high_pressure_mask.any():
            return (0, 0)

        rows, cols = np.where(high_pressure_mask)
        center_row = rows.mean()
        center_col = cols.mean()

        # Shift away from center of high pressure
        grid_center_row = n_rows / 2
        grid_center_col = n_cols / 2

        shift_y = 1 if center_row > grid_center_row else -1
        shift_x = 1 if center_col > grid_center_col else -1

        return (shift_x, shift_y)

    def run_parameter_optimization(self, cell_sizes=[3, 5, 7],
                                   reposition_intervals=[900, 1800, 3600]):
        """
        Find optimal parameters through simulation.

        Args:
            cell_sizes: List of cell sizes to test (cm)
            reposition_intervals: List of intervals to test (seconds)

        Returns:
            Dictionary with optimal parameters and scores
        """
        results = []

        for cell_size in cell_sizes:
            for interval in reposition_intervals:
                # This would run full simulation with parameters
                # Simplified scoring based on expected outcomes

                # Smaller cells = better resolution = lower score
                size_score = cell_size / 5

                # Shorter intervals = less risk = lower score
                interval_score = interval / 1800

                # Combined score (lower is better)
                total_score = 0.4 * size_score + 0.6 * interval_score

                results.append({
                    'cell_size': cell_size,
                    'reposition_interval': interval,
                    'score': total_score
                })

        # Find best parameters
        best = min(results, key=lambda x: x['score'])

        return {
            'optimal_cell_size': best['cell_size'],
            'optimal_reposition_interval': best['reposition_interval'],
            'all_results': results
        }


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class PressureUlcerSimulation:
    """
    Main simulation engine that ties all components together.

    Runs time-stepping simulation with:
    - Pressure computation each timestep
    - Risk accumulation
    - Optimization adjustments
    - Data recording for visualization
    """

    def __init__(self, body_mass=DEFAULT_BODY_MASS,
                 cell_size=DEFAULT_CELL_SIZE,
                 simulation_time=DEFAULT_SIMULATION_TIME,
                 timestep=DEFAULT_TIMESTEP):
        """
        Initialize simulation.

        Args:
            body_mass: Patient body mass (kg)
            cell_size: Mattress cell size (cm)
            simulation_time: Total simulation duration (seconds)
            timestep: Time step for updates (seconds)
        """
        self.simulation_time = simulation_time
        self.timestep = timestep
        self.current_time = 0

        # Initialize components
        grid_res = cell_size  # Match body grid to mattress cells

        self.body = BodyModel(body_mass=body_mass, grid_resolution=grid_res)
        self.mattress = AdaptiveMattress(cell_size=cell_size)
        self.pressure_calc = PressureCalculator(self.body, self.mattress)
        self.risk_model = BedSoreRiskModel(
            (self.mattress.n_cells_length, self.mattress.n_cells_width)
        )
        self.optimizer = PressureOptimizer(
            self.body, self.mattress, self.pressure_calc, self.risk_model
        )

        # Data recording
        self.pressure_history = []
        self.risk_history = []
        self.time_history = []
        self.peak_pressure_history = []
        self.mean_pressure_history = []

    def run(self, verbose=True):
        """
        Run the full simulation.

        Args:
            verbose: Print progress updates

        Returns:
            Dictionary with simulation results
        """
        n_steps = int(self.simulation_time / self.timestep)

        if verbose:
            print(f"Starting pressure ulcer simulation...")
            print(f"  Duration: {self.simulation_time/60:.1f} minutes")
            print(f"  Timestep: {self.timestep} seconds")
            print(f"  Total steps: {n_steps}")
            print()

        for step in range(n_steps):
            self.current_time = step * self.timestep

            # 1. Compute current pressure distribution
            pressure = self.pressure_calc.compute_pressure(dt=self.timestep)

            # 2. Update bed sore risk
            self.risk_model.update(pressure, self.timestep)

            # 3. Run optimization
            self.optimizer.optimize_step(self.current_time, pressure)

            # 4. Simulate mattress airflow
            self.mattress.simulate_airflow(self.timestep)

            # 5. Record data
            self.pressure_history.append(pressure.copy())
            self.risk_history.append(self.risk_model.risk_score.copy())
            self.time_history.append(self.current_time)
            self.peak_pressure_history.append(pressure.max())
            self.mean_pressure_history.append(pressure.mean())

            # Progress update
            if verbose and (step + 1) % (n_steps // 10) == 0:
                progress = (step + 1) / n_steps * 100
                print(f"  Progress: {progress:.0f}% | "
                      f"Peak pressure: {pressure.max():.1f} mmHg | "
                      f"Max risk: {self.risk_model.risk_score.max():.3f}")

        if verbose:
            print("\nSimulation complete!")
            self._print_summary()

        return self._compile_results()

    def _print_summary(self):
        """Print simulation summary statistics."""
        final_risk = self.risk_history[-1]
        final_pressure = self.pressure_history[-1]

        print("\n" + "="*50)
        print("SIMULATION SUMMARY")
        print("="*50)
        print(f"\nPressure Statistics:")
        print(f"  Final peak pressure: {final_pressure.max():.1f} mmHg")
        print(f"  Final mean pressure: {final_pressure.mean():.1f} mmHg")
        print(f"  Time-averaged peak: {np.mean(self.peak_pressure_history):.1f} mmHg")

        print(f"\nRisk Assessment:")
        print(f"  Maximum risk score: {final_risk.max():.4f}")
        print(f"  Mean risk score: {final_risk.mean():.4f}")
        print(f"  High risk cells (>50%): {(final_risk > 0.5).sum()}")
        print(f"  Critical risk cells (>75%): {(final_risk > 0.75).sum()}")

        # Identify high-risk regions
        zones = self.risk_model.get_risk_zones()
        high_risk_area = zones['high'].sum() + zones['critical'].sum()
        print(f"  Total high-risk area: {high_risk_area} cells")

        print("="*50)

    def _compile_results(self):
        """Compile simulation results into dictionary."""
        return {
            'pressure_history': np.array(self.pressure_history),
            'risk_history': np.array(self.risk_history),
            'time_history': np.array(self.time_history),
            'peak_pressure_history': np.array(self.peak_pressure_history),
            'mean_pressure_history': np.array(self.mean_pressure_history),
            'final_pressure': self.pressure_history[-1],
            'final_risk': self.risk_history[-1],
            'mattress_shape': (self.mattress.n_cells_length, self.mattress.n_cells_width),
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

class SimulationVisualizer:
    """
    Visualization tools for simulation results.

    Outputs:
    - Animated pressure map over time
    - Final pressure distribution
    - Cumulative bed sore risk heatmap
    """

    def __init__(self, results):
        """
        Initialize visualizer with simulation results.

        Args:
            results: Dictionary from PressureUlcerSimulation.run()
        """
        self.results = results

        # Custom colormaps
        self.pressure_cmap = self._create_pressure_colormap()
        self.risk_cmap = self._create_risk_colormap()

    def _create_pressure_colormap(self):
        """Create custom colormap for pressure (green -> yellow -> red)."""
        colors = ['#2ecc71', '#f1c40f', '#e74c3c', '#8e44ad']
        return LinearSegmentedColormap.from_list('pressure', colors, N=256)

    def _create_risk_colormap(self):
        """Create custom colormap for risk (blue -> green -> yellow -> red)."""
        colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#c0392b']
        return LinearSegmentedColormap.from_list('risk', colors, N=256)

    def create_animation(self, interval=100, save_path=None):
        """
        Create animated pressure map.

        Args:
            interval: Milliseconds between frames
            save_path: Path to save animation (optional)

        Returns:
            Matplotlib FuncAnimation object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        # Initialize images
        pressure_data = self.results['pressure_history']
        risk_data = self.results['risk_history']

        vmax_pressure = np.percentile(pressure_data, 99)

        im_pressure = axes[0].imshow(
            pressure_data[0], cmap=self.pressure_cmap,
            aspect='auto', vmin=0, vmax=max(vmax_pressure, 50)
        )
        axes[0].set_title('Pressure Distribution (mmHg)')
        axes[0].set_xlabel('Width (cells)')
        axes[0].set_ylabel('Length (cells) - Head to Feet')
        plt.colorbar(im_pressure, ax=axes[0], label='Pressure (mmHg)')

        # Add threshold line in colorbar
        threshold_line = axes[0].axhline(y=-1, color='white', linewidth=0,
                                         label=f'Threshold: {CAPILLARY_CLOSING_PRESSURE} mmHg')

        im_risk = axes[1].imshow(
            risk_data[0], cmap=self.risk_cmap,
            aspect='auto', vmin=0, vmax=1
        )
        axes[1].set_title('Cumulative Risk Score')
        axes[1].set_xlabel('Width (cells)')
        axes[1].set_ylabel('Length (cells) - Head to Feet')
        plt.colorbar(im_risk, ax=axes[1], label='Risk Score (0-1)')

        time_text = fig.suptitle('Time: 0:00', fontsize=14, fontweight='bold')

        def update(frame):
            im_pressure.set_array(pressure_data[frame])
            im_risk.set_array(risk_data[frame])

            time_sec = self.results['time_history'][frame]
            minutes = int(time_sec // 60)
            seconds = int(time_sec % 60)
            time_text.set_text(f'Time: {minutes}:{seconds:02d} | '
                              f'Peak: {pressure_data[frame].max():.1f} mmHg | '
                              f'Max Risk: {risk_data[frame].max():.3f}')

            return im_pressure, im_risk, time_text

        anim = FuncAnimation(fig, update, frames=len(pressure_data),
                            interval=interval, blit=False)

        plt.tight_layout()

        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=10)
            print("Animation saved!")

        return anim

    def plot_final_state(self, save_path=None):
        """
        Plot final pressure distribution and risk heatmap.

        Args:
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Final pressure map
        final_pressure = self.results['final_pressure']
        im1 = axes[0, 0].imshow(final_pressure, cmap=self.pressure_cmap,
                                 aspect='auto', vmin=0,
                                 vmax=max(final_pressure.max(), 50))
        axes[0, 0].set_title('Final Pressure Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Width (cells)')
        axes[0, 0].set_ylabel('Length (cells)\n← Head    Feet →')
        plt.colorbar(im1, ax=axes[0, 0], label='Pressure (mmHg)')

        # Add contour for threshold
        axes[0, 0].contour(final_pressure, levels=[CAPILLARY_CLOSING_PRESSURE],
                          colors='white', linewidths=2, linestyles='--')

        # Final risk heatmap
        final_risk = self.results['final_risk']
        im2 = axes[0, 1].imshow(final_risk, cmap=self.risk_cmap,
                                 aspect='auto', vmin=0, vmax=1)
        axes[0, 1].set_title('Cumulative Bed Sore Risk', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Width (cells)')
        axes[0, 1].set_ylabel('Length (cells)\n← Head    Feet →')
        plt.colorbar(im2, ax=axes[0, 1], label='Risk Score (0-1)')

        # Time series - Peak pressure
        times_min = self.results['time_history'] / 60
        axes[1, 0].plot(times_min, self.results['peak_pressure_history'],
                       'b-', linewidth=2, label='Peak Pressure')
        axes[1, 0].plot(times_min, self.results['mean_pressure_history'],
                       'g-', linewidth=2, alpha=0.7, label='Mean Pressure')
        axes[1, 0].axhline(y=CAPILLARY_CLOSING_PRESSURE, color='r',
                          linestyle='--', linewidth=2, label=f'Damage Threshold ({CAPILLARY_CLOSING_PRESSURE} mmHg)')
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('Pressure (mmHg)')
        axes[1, 0].set_title('Pressure Over Time', fontsize=12, fontweight='bold')
        axes[1, 0].legend(loc='upper right')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, times_min[-1])

        # Risk histogram
        risk_flat = final_risk.flatten()
        axes[1, 1].hist(risk_flat, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='High Risk (0.5)')
        axes[1, 1].axvline(x=0.75, color='red', linestyle='--', linewidth=2, label='Critical Risk (0.75)')
        axes[1, 1].set_xlabel('Risk Score')
        axes[1, 1].set_ylabel('Number of Cells')
        axes[1, 1].set_title('Risk Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)

        # Add annotations for body regions
        self._add_body_region_annotations(axes[0, 0])
        self._add_body_region_annotations(axes[0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return fig

    def _add_body_region_annotations(self, ax):
        """Add body region labels to a plot."""
        shape = self.results['mattress_shape']
        n_rows = shape[0]

        # Region positions (approximate)
        regions = [
            ('Head', 0.06),
            ('Shoulders', 0.17),
            ('Torso', 0.33),
            ('Pelvis', 0.50),
            ('Thighs', 0.70),
            ('Calves', 0.88),
            ('Heels', 0.97),
        ]

        for name, y_frac in regions:
            y_pos = y_frac * n_rows
            ax.annotate(name, xy=(0, y_pos), xytext=(-5, y_pos),
                       fontsize=8, color='white', fontweight='bold',
                       ha='right', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    def plot_risk_zones(self, save_path=None):
        """
        Plot risk zones with anatomical overlay.

        Args:
            save_path: Path to save figure (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 12))

        final_risk = self.results['final_risk']

        # Create categorical risk map
        risk_categories = np.zeros_like(final_risk)
        risk_categories[final_risk < 0.25] = 0  # Low
        risk_categories[(final_risk >= 0.25) & (final_risk < 0.5)] = 1  # Moderate
        risk_categories[(final_risk >= 0.5) & (final_risk < 0.75)] = 2  # High
        risk_categories[final_risk >= 0.75] = 3  # Critical

        # Custom discrete colormap
        colors = ['#27ae60', '#f39c12', '#e74c3c', '#8e44ad']
        cmap = LinearSegmentedColormap.from_list('risk_zones', colors, N=4)

        im = ax.imshow(risk_categories, cmap=cmap, aspect='auto', vmin=0, vmax=3)

        # Add colorbar with labels
        cbar = plt.colorbar(im, ax=ax, ticks=[0.375, 1.125, 1.875, 2.625])
        cbar.ax.set_yticklabels(['Low\n(<25%)', 'Moderate\n(25-50%)',
                                  'High\n(50-75%)', 'Critical\n(>75%)'])
        cbar.set_label('Risk Level', fontsize=12)

        ax.set_title('Bed Sore Risk Zones\n(Supine Position)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Width (cells)')
        ax.set_ylabel('Length (cells) - Head to Feet')

        # Add body region annotations
        self._add_body_region_annotations(ax)

        # Add statistics text
        stats_text = (
            f"Risk Statistics:\n"
            f"Low risk: {(risk_categories == 0).sum()} cells\n"
            f"Moderate: {(risk_categories == 1).sum()} cells\n"
            f"High: {(risk_categories == 2).sum()} cells\n"
            f"Critical: {(risk_categories == 3).sum()} cells"
        )
        ax.text(1.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the complete simulation and visualization."""

    print("="*60)
    print("PRESSURE ULCER PREVENTION SIMULATION")
    print("="*60)
    print("\nSimulating supine patient on adaptive air mattress...")
    print()

    # Configuration
    config = {
        'body_mass': 75,           # kg (patient weight)
        'cell_size': 5,            # cm (mattress cell size)
        'simulation_time': 3600,   # seconds (1 hour)
        'timestep': 15,            # seconds
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Run simulation
    sim = PressureUlcerSimulation(**config)
    results = sim.run(verbose=True)

    # Run parameter optimization
    print("\n" + "="*50)
    print("PARAMETER OPTIMIZATION")
    print("="*50)
    opt_results = sim.optimizer.run_parameter_optimization()
    print(f"\nOptimal cell size: {opt_results['optimal_cell_size']} cm")
    print(f"Optimal repositioning interval: {opt_results['optimal_reposition_interval']/60:.0f} minutes")

    # Create visualizations
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)

    viz = SimulationVisualizer(results)

    # Plot final state
    print("\n1. Creating final state visualization...")
    fig1 = viz.plot_final_state(save_path='pressure_simulation_results.png')

    # Plot risk zones
    print("2. Creating risk zone map...")
    fig2 = viz.plot_risk_zones(save_path='risk_zones.png')

    # Create animation (saved as GIF)
    print("3. Creating animated pressure map...")
    try:
        anim = viz.create_animation(interval=100, save_path='pressure_animation.gif')
        print("   Animation saved to pressure_animation.gif")
    except Exception as e:
        print(f"   Animation creation skipped: {e}")

    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print("\nOutput files:")
    print("  - pressure_simulation_results.png (final state)")
    print("  - risk_zones.png (risk zone map)")
    print("  - pressure_animation.gif (animated pressure map)")

    # Display plots (uncomment for interactive use)
    # plt.show()

    return results, sim


if __name__ == "__main__":
    results, simulation = main()
