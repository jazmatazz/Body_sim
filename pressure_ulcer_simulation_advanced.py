#!/Users/jaspermetz/Documents/Body_Sim/.venv/bin/python3
"""
Advanced Pressure Ulcer Prevention Simulation
==============================================
A comprehensive biomedical simulation with full 3D tissue modeling,
thermal effects, shear stress, and realistic physics.

ENHANCEMENTS OVER BASIC VERSION:
1. 3D tissue depth modeling with 4 layers (skin, fat, muscle, bone)
2. Heterogeneous tissue properties with spatial variation
3. Realistic pneumatic airflow with pressure-volume dynamics
4. Thermal model with metabolic heat, conduction, and moisture
5. Nonlinear Hertzian contact mechanics
6. Shear stress from friction and body positioning
7. Comprehensive multi-factor risk model

Physical Basis:
- Tissue mechanics: Ogden hyperelastic model (simplified)
- Heat transfer: Pennes bioheat equation
- Contact: Hertzian contact theory
- Shear: Coulomb friction model
- Risk: Multi-factorial damage accumulation

Author: Biomedical Simulation Engineer
Dependencies: NumPy, Matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# General
GRAVITY = 9.81  # m/s²
MMHG_PER_PASCAL = 0.00750062

# Thermal constants
BODY_CORE_TEMP = 37.0  # °C
AMBIENT_TEMP = 22.0  # °C
SKIN_BLOOD_PERFUSION = 0.0005  # m³/s/m³ (baseline)
BLOOD_DENSITY = 1060  # kg/m³
BLOOD_SPECIFIC_HEAT = 3770  # J/(kg·K)
METABOLIC_HEAT_RATE = 1000  # W/m³ (basal metabolic rate per volume)

# Tissue layer properties (thickness in mm, modulus in kPa)
TISSUE_LAYERS = {
    'skin': {
        'thickness': 2.0,  # mm
        'elastic_modulus': 100.0,  # kPa (Young's modulus)
        'poisson_ratio': 0.49,
        'thermal_conductivity': 0.37,  # W/(m·K)
        'specific_heat': 3500,  # J/(kg·K)
        'density': 1100,  # kg/m³
        'damage_threshold': 32,  # mmHg
        'friction_coef': 0.5,
    },
    'fat': {
        'thickness': 15.0,
        'elastic_modulus': 2.0,
        'poisson_ratio': 0.49,
        'thermal_conductivity': 0.21,
        'specific_heat': 2300,
        'density': 900,
        'damage_threshold': 40,
        'friction_coef': 0.3,
    },
    'muscle': {
        'thickness': 30.0,
        'elastic_modulus': 10.0,
        'poisson_ratio': 0.45,
        'thermal_conductivity': 0.42,
        'specific_heat': 3600,
        'density': 1050,
        'damage_threshold': 60,
        'friction_coef': 0.4,
    },
    'bone': {
        'thickness': 10.0,
        'elastic_modulus': 15000.0,
        'poisson_ratio': 0.30,
        'thermal_conductivity': 0.58,
        'specific_heat': 1300,
        'density': 1900,
        'damage_threshold': 200,
        'friction_coef': 0.6,
    },
}

# Moisture parameters
SWEAT_RATE_BASE = 0.5  # g/(m²·min) baseline
SWEAT_RATE_TEMP_COEF = 0.3  # increase per °C above 34°C
MOISTURE_EVAP_RATE = 0.1  # fraction per minute
MOISTURE_DAMAGE_FACTOR = 1.5  # risk multiplier when moist

# Capillary closing pressure
CAPILLARY_CLOSING_PRESSURE = 32  # mmHg

# Shear damage threshold
SHEAR_DAMAGE_THRESHOLD = 10  # kPa

# =============================================================================
# QUADRIPLEGIC PATIENT PARAMETERS
# =============================================================================

QUADRIPLEGIC_FACTORS = {
    # Muscle atrophy: less padding over bony prominences
    'muscle_atrophy_factor': 0.4,  # 40% of normal muscle mass

    # Fat redistribution (often increased subcutaneous fat but less protective)
    'fat_effectiveness': 0.7,  # Fat provides 70% of normal protection

    # Impaired circulation (autonomic dysfunction)
    'baseline_perfusion_factor': 0.7,  # 70% of normal blood flow

    # Increased moisture (incontinence, inability to adjust)
    'baseline_moisture': 0.25,  # Start with 25% moisture saturation
    'moisture_accumulation_rate': 2.0,  # 2x faster moisture buildup

    # No protective sensation (can't feel pain to trigger movement)
    'sensation_factor': 0.0,  # No protective response

    # Skin fragility (reduced collagen, thinner dermis)
    'skin_fragility': 1.5,  # 50% more susceptible to damage

    # Spasticity effects (involuntary muscle contractions can cause shear)
    'spasticity_shear_factor': 1.3,  # 30% additional shear stress

    # Reduced tissue oxygenation
    'tissue_oxygenation': 0.75,  # 75% of normal

    # Damage rate multiplier (all factors combined)
    'damage_rate_multiplier': 1.5,  # Damage accumulates 1.5x faster
}

# Lateral rotation bed parameters
LATERAL_ROTATION_PARAMS = {
    'max_angle': 30,  # Maximum tilt angle (degrees)
    'rotation_period': 600,  # Full cycle time (seconds) - 10 minutes
    'pause_at_extremes': 60,  # Pause at max tilt (seconds)
    'transition_time': 120,  # Time to rotate from one side to other (seconds)
}

# =============================================================================
# MOISTURE MANAGEMENT INTERVENTIONS
# =============================================================================

MOISTURE_MANAGEMENT = {
    'none': {
        'name': 'No Moisture Management',
        'baseline_moisture_mult': 1.0,      # No reduction in baseline
        'accumulation_rate_mult': 1.0,      # Normal accumulation
        'evaporation_rate_mult': 1.0,       # Normal evaporation
        'damage_factor_mult': 1.0,          # Normal damage from moisture
    },
    'basic': {
        'name': 'Basic (Absorbent Pads)',
        'baseline_moisture_mult': 0.7,      # 30% reduction in baseline
        'accumulation_rate_mult': 0.8,      # 20% slower accumulation
        'evaporation_rate_mult': 1.2,       # 20% faster wicking
        'damage_factor_mult': 0.85,         # 15% less damage
    },
    'moderate': {
        'name': 'Moderate (Pads + Barrier Cream)',
        'baseline_moisture_mult': 0.5,      # 50% reduction in baseline
        'accumulation_rate_mult': 0.6,      # 40% slower accumulation
        'evaporation_rate_mult': 1.5,       # 50% faster wicking
        'damage_factor_mult': 0.7,          # 30% less damage
    },
    'comprehensive': {
        'name': 'Comprehensive (Catheter + Barriers + Pads)',
        'baseline_moisture_mult': 0.2,      # 80% reduction (catheter handles urine)
        'accumulation_rate_mult': 0.3,      # 70% slower (only sweat remains)
        'evaporation_rate_mult': 2.0,       # Fast wicking materials
        'damage_factor_mult': 0.5,          # 50% less damage (skin barriers)
    },
    'optimal': {
        'name': 'Optimal (Full Protocol)',
        'baseline_moisture_mult': 0.1,      # 90% reduction
        'accumulation_rate_mult': 0.2,      # 80% slower
        'evaporation_rate_mult': 2.5,       # Aggressive wicking
        'damage_factor_mult': 0.3,          # 70% less damage
    },
}


# =============================================================================
# 3D TISSUE MODEL
# =============================================================================

class TissueLayer:
    """
    Represents a single tissue layer with mechanical and thermal properties.
    """

    def __init__(self, name, props, grid_shape):
        self.name = name
        self.thickness = props['thickness'] / 1000  # Convert to meters
        self.E = props['elastic_modulus'] * 1000  # Convert to Pa
        self.nu = props['poisson_ratio']
        self.k_thermal = props['thermal_conductivity']
        self.cp = props['specific_heat']
        self.rho = props['density']
        self.damage_threshold = props['damage_threshold']
        self.friction_coef = props['friction_coef']

        # State matrices
        self.stress = np.zeros(grid_shape)  # Normal stress (Pa)
        self.shear_stress = np.zeros((grid_shape[0], grid_shape[1], 2))  # Shear (x,y)
        self.strain = np.zeros(grid_shape)
        self.temperature = np.ones(grid_shape) * BODY_CORE_TEMP
        self.damage = np.zeros(grid_shape)

    def compute_effective_modulus(self):
        """Compute plane strain modulus."""
        return self.E / (1 - self.nu**2)


class Tissue3DModel:
    """
    Full 3D tissue model with multiple layers and heterogeneous properties.

    Layers (from surface to deep):
    1. Skin (epidermis + dermis)
    2. Subcutaneous fat
    3. Muscle
    4. Bone (where applicable)
    """

    def __init__(self, grid_shape, cell_size):
        """
        Initialize 3D tissue model.

        Args:
            grid_shape: (rows, cols) of the surface grid
            cell_size: Size of each cell in cm
        """
        self.grid_shape = grid_shape
        self.cell_size = cell_size / 100  # Convert to meters
        self.cell_area = self.cell_size ** 2

        # Create tissue layers
        self.layers = {}
        for name, props in TISSUE_LAYERS.items():
            self.layers[name] = TissueLayer(name, props, grid_shape)

        # Heterogeneous property modifiers (spatial variation)
        self._initialize_heterogeneity()

        # Total tissue response
        self.surface_pressure = np.zeros(grid_shape)
        self.surface_shear = np.zeros((grid_shape[0], grid_shape[1], 2))
        self.deep_pressure = np.zeros(grid_shape)  # Pressure at bone level

    def _initialize_heterogeneity(self):
        """
        Create spatial variation in tissue properties.

        Models anatomical differences:
        - Thinner fat over bony prominences
        - Thicker muscle in gluteal region
        - Varying skin thickness
        """
        rows, cols = self.grid_shape

        # Create heterogeneity maps (multipliers for base properties)
        self.fat_thickness_map = np.ones(self.grid_shape)
        self.muscle_thickness_map = np.ones(self.grid_shape)
        self.skin_stiffness_map = np.ones(self.grid_shape)

        # Bony prominences: reduced fat, higher pressure transmission
        # Sacrum region (approximately 45-55% of length)
        sacrum_start = int(0.45 * rows)
        sacrum_end = int(0.55 * rows)
        center_start = int(0.35 * cols)
        center_end = int(0.65 * cols)

        self.fat_thickness_map[sacrum_start:sacrum_end, center_start:center_end] *= 0.3

        # Heels (last 5% of length)
        heel_start = int(0.95 * rows)
        self.fat_thickness_map[heel_start:, :] *= 0.2

        # Shoulder blades
        shoulder_start = int(0.12 * rows)
        shoulder_end = int(0.22 * rows)
        left_blade = slice(int(0.15 * cols), int(0.35 * cols))
        right_blade = slice(int(0.65 * cols), int(0.85 * cols))

        self.fat_thickness_map[shoulder_start:shoulder_end, left_blade] *= 0.5
        self.fat_thickness_map[shoulder_start:shoulder_end, right_blade] *= 0.5

        # Occiput (head - first 10%)
        self.fat_thickness_map[:int(0.10 * rows), :] *= 0.4

        # Gluteal region: more muscle
        gluteal_start = int(0.40 * rows)
        gluteal_end = int(0.55 * rows)
        self.muscle_thickness_map[gluteal_start:gluteal_end, :] *= 1.5

        # Apply heterogeneity to layers
        self.layers['fat'].thickness *= self.fat_thickness_map.mean()
        self.layers['muscle'].thickness *= self.muscle_thickness_map.mean()

    def compute_layered_stress(self, surface_force):
        """
        Compute stress distribution through tissue layers.

        Uses simplified stress transmission through layered medium.
        Stress spreads and attenuates with depth.

        Args:
            surface_force: Force matrix at surface (N per cell)

        Returns:
            Dictionary with stress at each layer
        """
        # Surface pressure
        self.surface_pressure = surface_force / self.cell_area
        surface_pressure_mmhg = self.surface_pressure * MMHG_PER_PASCAL

        # Stress transmission through layers
        # Using simplified cone model: stress spreads at 30° angle
        spread_angle = np.radians(30)

        current_pressure = self.surface_pressure.copy()
        cumulative_depth = 0

        for layer_name in ['skin', 'fat', 'muscle', 'bone']:
            layer = self.layers[layer_name]

            # Layer thickness (with heterogeneity for fat)
            if layer_name == 'fat':
                effective_thickness = layer.thickness * self.fat_thickness_map
            elif layer_name == 'muscle':
                effective_thickness = layer.thickness * self.muscle_thickness_map
            else:
                effective_thickness = layer.thickness

            # Mean thickness for spread calculation
            mean_thickness = np.mean(effective_thickness) if isinstance(effective_thickness, np.ndarray) else effective_thickness

            # Spread radius at this depth
            spread_radius = cumulative_depth * np.tan(spread_angle)
            spread_cells = max(1, int(spread_radius / self.cell_size))

            # Apply Gaussian smoothing to simulate stress spread
            if spread_cells > 0:
                current_pressure = self._gaussian_spread(current_pressure, spread_cells)

            # Stress at this layer (accounting for area increase)
            # Area ratio: (r + spread)² / r²
            if cumulative_depth > 0:
                area_factor = 1 / (1 + spread_radius / self.cell_size) ** 2
            else:
                area_factor = 1.0

            layer.stress = current_pressure * area_factor

            # Update depth
            cumulative_depth += mean_thickness

        # Deep pressure (at bone level)
        self.deep_pressure = self.layers['bone'].stress.copy()

        return {name: layer.stress for name, layer in self.layers.items()}

    def _gaussian_spread(self, matrix, sigma):
        """Apply Gaussian spreading to simulate stress distribution."""
        if sigma < 1:
            return matrix

        # Create Gaussian kernel
        size = min(sigma * 3, min(matrix.shape) // 4)
        if size < 1:
            return matrix

        x = np.arange(-size, size + 1)
        kernel_1d = np.exp(-x**2 / (2 * sigma**2))
        kernel_1d /= kernel_1d.sum()

        # Apply separable convolution
        result = matrix.copy()

        # Horizontal pass
        for i in range(result.shape[0]):
            result[i, :] = np.convolve(result[i, :], kernel_1d, mode='same')

        # Vertical pass
        for j in range(result.shape[1]):
            result[:, j] = np.convolve(result[:, j], kernel_1d, mode='same')

        return result

    def get_effective_modulus(self):
        """
        Compute effective modulus of the layered tissue.

        Uses series spring model for layers in compression.
        """
        total_compliance = 0
        total_thickness = 0

        for layer in self.layers.values():
            E_eff = layer.compute_effective_modulus()
            total_compliance += layer.thickness / E_eff
            total_thickness += layer.thickness

        return total_thickness / total_compliance if total_compliance > 0 else 1e6

    def get_spatially_varying_modulus(self):
        """
        Get effective modulus matrix accounting for spatial heterogeneity.
        """
        # Base effective modulus
        E_base = self.get_effective_modulus()

        # Modify based on fat thickness (less fat = stiffer response)
        E_matrix = E_base / self.fat_thickness_map

        # Normalize
        E_matrix = np.clip(E_matrix, E_base * 0.5, E_base * 5)

        return E_matrix


# =============================================================================
# THERMAL AND MOISTURE MODEL
# =============================================================================

class ThermalMoistureModel:
    """
    Models heat transfer and moisture accumulation at the interface.

    Based on Pennes bioheat equation:
    ρc(∂T/∂t) = k∇²T + ρb·cb·ωb(Ta - T) + Qm

    Where:
    - ρc: tissue heat capacity
    - k: thermal conductivity
    - ρb·cb·ωb: blood perfusion term
    - Ta: arterial blood temperature
    - Qm: metabolic heat generation
    """

    def __init__(self, grid_shape, cell_size, moisture_management='none'):
        """
        Initialize thermal model.

        Args:
            grid_shape: (rows, cols) grid dimensions
            cell_size: Cell size in cm
            moisture_management: Level of moisture intervention
                                ('none', 'basic', 'moderate', 'comprehensive', 'optimal')
        """
        self.grid_shape = grid_shape
        self.cell_size = cell_size / 100  # meters
        self.moisture_management = moisture_management

        # Set moisture management multipliers
        if moisture_management in MOISTURE_MANAGEMENT:
            mm = MOISTURE_MANAGEMENT[moisture_management]
            self.moisture_baseline_mult = mm['baseline_moisture_mult']
            self.moisture_accum_mult = mm['accumulation_rate_mult']
            self.moisture_evap_mult = mm['evaporation_rate_mult']
            self.moisture_damage_mult = mm['damage_factor_mult']
        else:
            self.moisture_baseline_mult = 1.0
            self.moisture_accum_mult = 1.0
            self.moisture_evap_mult = 1.0
            self.moisture_damage_mult = 1.0

        # Temperature field (°C)
        self.temperature = np.ones(grid_shape) * BODY_CORE_TEMP
        self.skin_temperature = np.ones(grid_shape) * 34.0  # Typical skin temp

        # Moisture field (0-1 normalized saturation)
        self.moisture = np.zeros(grid_shape)

        # Interface temperature with mattress
        self.interface_temp = np.ones(grid_shape) * 30.0

        # Blood perfusion (reduced under pressure)
        self.perfusion = np.ones(grid_shape) * SKIN_BLOOD_PERFUSION

        # Heat accumulation
        self.heat_accumulation = np.zeros(grid_shape)

        # Effective thermal properties
        self.k_eff = 0.37  # W/(m·K) skin conductivity
        self.rho_c_eff = 1100 * 3500  # ρ·c for skin

    def update(self, pressure_mmhg, contact_mask, mattress_temp, dt):
        """
        Update thermal and moisture state.

        Args:
            pressure_mmhg: Current pressure distribution
            contact_mask: Boolean mask of contact areas
            mattress_temp: Mattress surface temperature
            dt: Time step in seconds
        """
        # 1. Update blood perfusion (reduced under pressure)
        self._update_perfusion(pressure_mmhg)

        # 2. Compute metabolic heat
        Q_metabolic = self._compute_metabolic_heat(pressure_mmhg)

        # 3. Compute heat conduction (simplified 2D)
        Q_conduction = self._compute_conduction()

        # 4. Compute blood perfusion heat exchange
        Q_perfusion = self._compute_perfusion_heat()

        # 5. Compute interface heat exchange
        Q_interface = self._compute_interface_heat(contact_mask, mattress_temp)

        # 6. Update temperature
        dT = (Q_metabolic + Q_conduction + Q_perfusion + Q_interface) * dt / self.rho_c_eff
        self.skin_temperature += dT

        # Physiological bounds
        self.skin_temperature = np.clip(self.skin_temperature, 20, 42)

        # 7. Update moisture (sweating)
        self._update_moisture(contact_mask, dt)

        # 8. Update interface temperature
        self.interface_temp = 0.7 * self.skin_temperature + 0.3 * mattress_temp

        # 9. Track heat accumulation
        self.heat_accumulation += np.maximum(self.skin_temperature - 34, 0) * dt

    def _update_perfusion(self, pressure_mmhg):
        """
        Update blood perfusion based on pressure.

        Perfusion decreases as pressure approaches capillary closing pressure.
        """
        # Sigmoid reduction in perfusion
        pressure_ratio = pressure_mmhg / CAPILLARY_CLOSING_PRESSURE
        perfusion_factor = 1 / (1 + np.exp(3 * (pressure_ratio - 1)))

        self.perfusion = SKIN_BLOOD_PERFUSION * perfusion_factor

    def _compute_metabolic_heat(self, pressure_mmhg):
        """
        Compute metabolic heat generation.

        Reduced under ischemic conditions (high pressure).
        """
        # Base metabolic heat
        Q_base = METABOLIC_HEAT_RATE

        # Reduction under ischemia
        ischemia_factor = self.perfusion / SKIN_BLOOD_PERFUSION

        return Q_base * ischemia_factor

    def _compute_conduction(self):
        """
        Compute heat conduction using 2D Laplacian.
        """
        # Discrete Laplacian
        laplacian = np.zeros_like(self.skin_temperature)
        laplacian[1:-1, 1:-1] = (
            self.skin_temperature[:-2, 1:-1] +
            self.skin_temperature[2:, 1:-1] +
            self.skin_temperature[1:-1, :-2] +
            self.skin_temperature[1:-1, 2:] -
            4 * self.skin_temperature[1:-1, 1:-1]
        ) / (self.cell_size ** 2)

        return self.k_eff * laplacian

    def _compute_perfusion_heat(self):
        """
        Compute heat exchange due to blood perfusion.
        """
        rho_b_cb = BLOOD_DENSITY * BLOOD_SPECIFIC_HEAT
        return rho_b_cb * self.perfusion * (BODY_CORE_TEMP - self.skin_temperature)

    def _compute_interface_heat(self, contact_mask, mattress_temp):
        """
        Compute heat exchange with mattress at interface.
        """
        # Contact conductance (W/(m²·K))
        h_contact = 50  # Typical value for skin-fabric interface

        Q_interface = np.zeros_like(self.skin_temperature)
        Q_interface[contact_mask] = h_contact * (mattress_temp - self.skin_temperature[contact_mask])

        return Q_interface

    def _update_moisture(self, contact_mask, dt):
        """
        Update moisture level from sweating and evaporation.
        Applies moisture management multipliers if set.
        """
        # Sweating rate increases with temperature
        temp_excess = np.maximum(self.skin_temperature - 34, 0)
        sweat_rate = SWEAT_RATE_BASE + SWEAT_RATE_TEMP_COEF * temp_excess

        # Increase moisture (normalized, assumes some saturation capacity)
        # Apply moisture accumulation multiplier from moisture management
        moisture_increase = sweat_rate * (dt / 60) / 100 * self.moisture_accum_mult
        self.moisture += moisture_increase

        # Evaporation (reduced in contact areas due to occlusion)
        evap_factor = np.ones_like(self.moisture)
        evap_factor[contact_mask] = 0.2  # 80% reduction in occluded areas

        # Apply evaporation multiplier from moisture management
        effective_evap = MOISTURE_EVAP_RATE * self.moisture_evap_mult
        self.moisture *= (1 - effective_evap * (dt / 60) * evap_factor)

        # Clamp to valid range
        self.moisture = np.clip(self.moisture, 0, 1)

    def get_damage_multiplier(self):
        """
        Get tissue damage multiplier based on thermal and moisture conditions.
        Applies moisture damage multiplier from moisture management.
        """
        # Temperature factor (increased damage at elevated temps)
        temp_factor = 1 + 0.1 * np.maximum(self.skin_temperature - 34, 0)

        # Moisture factor (reduced by moisture management)
        base_moisture_factor = 1 + (MOISTURE_DAMAGE_FACTOR - 1) * self.moisture
        moisture_factor = 1 + (base_moisture_factor - 1) * self.moisture_damage_mult

        # Ischemia factor (reduced perfusion = more damage)
        perfusion_ratio = self.perfusion / SKIN_BLOOD_PERFUSION
        ischemia_factor = 1 + 2 * (1 - perfusion_ratio)

        return temp_factor * moisture_factor * ischemia_factor


# =============================================================================
# SHEAR STRESS MODEL
# =============================================================================

class ShearStressModel:
    """
    Models shear stress from friction and body positioning.

    Shear stress arises from:
    1. Friction at skin-mattress interface
    2. Body weight components on inclined surfaces
    3. Sliding tendency during repositioning

    Uses Coulomb friction: τ = μ × P
    """

    def __init__(self, grid_shape, tissue_model):
        """
        Initialize shear stress model.

        Args:
            grid_shape: (rows, cols) grid dimensions
            tissue_model: Tissue3DModel instance
        """
        self.grid_shape = grid_shape
        self.tissue = tissue_model

        # Shear stress components (Pa)
        self.shear_x = np.zeros(grid_shape)  # Lateral shear
        self.shear_y = np.zeros(grid_shape)  # Longitudinal shear
        self.shear_magnitude = np.zeros(grid_shape)

        # Friction coefficient map
        self.friction_coef = self._create_friction_map()

        # Bed angle (head elevation)
        self.bed_head_angle = 0  # degrees

        # Body sliding tendency
        self.sliding_factor = np.zeros(grid_shape)

    def _create_friction_map(self):
        """
        Create spatially varying friction coefficient map.

        Higher friction at bony prominences due to skin tension.
        """
        base_friction = 0.5  # Skin-fabric friction

        friction_map = np.ones(self.grid_shape) * base_friction

        # Modify based on tissue heterogeneity
        # Lower fat = higher skin tension = higher friction
        friction_map *= (2 - self.tissue.fat_thickness_map)

        return np.clip(friction_map, 0.3, 0.8)

    def compute_shear(self, normal_pressure, bed_angle=0, moisture=None):
        """
        Compute shear stress distribution.

        Args:
            normal_pressure: Normal pressure in Pa
            bed_angle: Head of bed elevation in degrees
            moisture: Moisture level (0-1), affects friction

        Returns:
            Total shear stress magnitude in Pa
        """
        self.bed_head_angle = bed_angle

        # Adjust friction for moisture
        if moisture is not None:
            # Wet skin has higher friction initially, then lower when saturated
            moisture_friction_mod = 1 + 0.3 * moisture * (1 - moisture) * 4
            effective_friction = self.friction_coef * moisture_friction_mod
        else:
            effective_friction = self.friction_coef

        # 1. Gravity-induced shear from bed inclination
        if bed_angle > 0:
            angle_rad = np.radians(bed_angle)

            # Longitudinal shear (sliding down the bed)
            # Upper body experiences more shear
            rows = self.grid_shape[0]
            position_factor = np.linspace(1, 0, rows)[:, np.newaxis]
            position_factor = np.broadcast_to(position_factor, self.grid_shape)

            gravity_shear_y = normal_pressure * np.sin(angle_rad) * position_factor

            # Friction limits the actual shear
            max_shear = effective_friction * normal_pressure
            self.shear_y = np.minimum(gravity_shear_y, max_shear)
        else:
            self.shear_y = np.zeros(self.grid_shape)

        # 2. Lateral shear from body curvature
        # Body is curved, creating lateral shear at edges
        cols = self.grid_shape[1]
        center = cols / 2
        lateral_position = np.abs(np.arange(cols) - center) / center
        lateral_position = np.broadcast_to(lateral_position, self.grid_shape)

        # Shear proportional to curvature and pressure
        curvature_factor = 0.1  # Relates body curvature to shear
        self.shear_x = normal_pressure * curvature_factor * lateral_position
        self.shear_x = np.minimum(self.shear_x, effective_friction * normal_pressure * 0.5)

        # Total shear magnitude
        self.shear_magnitude = np.sqrt(self.shear_x**2 + self.shear_y**2)

        # Store in tissue layers
        for layer in self.tissue.layers.values():
            # Shear attenuates with depth
            depth_factor = 1.0 if layer.name == 'skin' else 0.5
            layer.shear_stress[:, :, 0] = self.shear_x * depth_factor
            layer.shear_stress[:, :, 1] = self.shear_y * depth_factor

        return self.shear_magnitude

    def get_combined_stress(self, normal_pressure):
        """
        Compute combined stress using von Mises criterion.

        σ_vm = √(σ² + 3τ²)

        Args:
            normal_pressure: Normal pressure in Pa

        Returns:
            Von Mises equivalent stress in Pa
        """
        von_mises = np.sqrt(normal_pressure**2 + 3 * self.shear_magnitude**2)
        return von_mises

    def get_shear_damage_factor(self):
        """
        Get tissue damage factor from shear stress.

        Shear stress compounds with normal pressure to accelerate damage.
        """
        # Normalize shear by threshold
        shear_kpa = self.shear_magnitude / 1000
        normalized_shear = shear_kpa / SHEAR_DAMAGE_THRESHOLD

        # Damage factor (exponential increase above threshold)
        damage_factor = 1 + np.maximum(normalized_shear - 1, 0) ** 1.5

        return damage_factor


# =============================================================================
# ADVANCED PNEUMATIC MATTRESS MODEL
# =============================================================================

class PneumaticMattressModel:
    """
    Realistic pneumatic air mattress with pressure-volume dynamics.

    Models:
    - Individual air cell volumes and pressures
    - Ideal gas law behavior
    - Flow between cells through orifices
    - Active pressure control
    - Thermal effects on pressure
    """

    def __init__(self, length=200, width=90, cell_size=5,
                 max_pressure=50, initial_pressure=20):
        """
        Initialize pneumatic mattress.

        Args:
            length: Mattress length in cm
            width: Mattress width in cm
            cell_size: Cell size in cm
            max_pressure: Maximum cell pressure in mmHg
            initial_pressure: Initial cell pressure in mmHg
        """
        self.length = length / 100  # meters
        self.width = width / 100
        self.cell_size_m = cell_size / 100

        self.n_rows = int(length / cell_size)
        self.n_cols = int(width / cell_size)
        self.shape = (self.n_rows, self.n_cols)

        # Convert pressures to Pascals (gauge pressure)
        self.max_pressure_pa = max_pressure / MMHG_PER_PASCAL
        self.initial_pressure_pa = initial_pressure / MMHG_PER_PASCAL

        # Cell physical properties
        self.cell_volume_unloaded = (self.cell_size_m ** 2) * 0.10  # 10cm height
        self.cell_max_compression = 0.7  # Max 70% volume reduction

        # State variables
        self.cell_pressure = np.ones(self.shape) * self.initial_pressure_pa
        self.cell_volume = np.ones(self.shape) * self.cell_volume_unloaded
        self.cell_temperature = np.ones(self.shape) * (AMBIENT_TEMP + 273.15)  # Kelvin
        self.cell_moles = self._compute_moles()  # n from PV = nRT

        # Target pressures for control
        self.target_pressure = self.cell_pressure.copy()

        # Orifice flow parameters
        self.orifice_diameter = 0.002  # 2mm orifices between cells
        self.discharge_coef = 0.6

        # Surface properties
        self.surface_temp = np.ones(self.shape) * AMBIENT_TEMP
        self.surface_height = np.ones(self.shape) * self.cell_size_m * 0.10

    def _compute_moles(self):
        """Compute moles of air in each cell using ideal gas law."""
        R = 8.314  # J/(mol·K)
        # Add atmospheric pressure for absolute pressure
        P_abs = self.cell_pressure + 101325
        return (P_abs * self.cell_volume) / (R * self.cell_temperature)

    def apply_load(self, force_matrix):
        """
        Apply external load to mattress and compute deformation.

        Args:
            force_matrix: Force per cell in Newtons

        Returns:
            Surface deformation matrix in meters
        """
        # Resample force to mattress grid
        force_resampled = self._resample_to_grid(force_matrix)

        # Cell area
        cell_area = self.cell_size_m ** 2

        # Additional pressure from load
        load_pressure = force_resampled / cell_area

        # Total pressure (internal + load)
        total_pressure = self.cell_pressure + load_pressure

        # Volume change from compression (isothermal initially)
        # P1V1 = P2V2
        P1 = self.cell_pressure + 101325  # Absolute pressure before load
        P2 = total_pressure + 101325  # Absolute pressure with load

        new_volume = self.cell_volume * (P1 / P2)

        # Limit compression
        min_volume = self.cell_volume_unloaded * (1 - self.cell_max_compression)
        new_volume = np.maximum(new_volume, min_volume)

        # Volume change corresponds to height change
        volume_ratio = new_volume / self.cell_volume_unloaded
        self.surface_height = self.cell_size_m * 0.10 * volume_ratio

        # Deformation (positive = compression)
        deformation = (self.cell_size_m * 0.10) - self.surface_height

        # Update cell volume
        self.cell_volume = new_volume

        return deformation

    def simulate_airflow(self, dt):
        """
        Simulate air flow between cells and pressure equalization.

        Uses orifice flow equation:
        Q = Cd × A × √(2ΔP/ρ)

        Args:
            dt: Time step in seconds
        """
        # Air density at operating conditions
        rho_air = 1.2  # kg/m³

        # Orifice area
        orifice_area = np.pi * (self.orifice_diameter / 2) ** 2

        # Compute pressure gradients with neighbors
        pressure_diff = np.zeros((self.n_rows, self.n_cols, 4))  # Up, Down, Left, Right

        # Up neighbor
        pressure_diff[1:, :, 0] = self.cell_pressure[:-1, :] - self.cell_pressure[1:, :]
        # Down neighbor
        pressure_diff[:-1, :, 1] = self.cell_pressure[1:, :] - self.cell_pressure[:-1, :]
        # Left neighbor
        pressure_diff[:, 1:, 2] = self.cell_pressure[:, :-1] - self.cell_pressure[:, 1:]
        # Right neighbor
        pressure_diff[:, :-1, 3] = self.cell_pressure[:, 1:] - self.cell_pressure[:, :-1]

        # Volumetric flow rate for each direction
        # Q = Cd × A × √(2|ΔP|/ρ) × sign(ΔP)
        flow_rate = np.zeros_like(pressure_diff)
        nonzero = np.abs(pressure_diff) > 1  # Threshold for numerical stability

        flow_rate[nonzero] = (
            self.discharge_coef * orifice_area *
            np.sqrt(2 * np.abs(pressure_diff[nonzero]) / rho_air) *
            np.sign(pressure_diff[nonzero])
        )

        # Net flow into each cell
        net_flow = flow_rate.sum(axis=2)

        # Update moles (molar flow = volumetric flow × molar density)
        R = 8.314
        molar_density = (self.cell_pressure + 101325) / (R * self.cell_temperature)
        mole_change = net_flow * molar_density * dt

        self.cell_moles += mole_change
        self.cell_moles = np.maximum(self.cell_moles, 1e-10)  # Prevent negative

        # Recalculate pressure from new moles (at current volume and temp)
        self.cell_pressure = (self.cell_moles * R * self.cell_temperature) / self.cell_volume - 101325

        # Clamp pressures
        self.cell_pressure = np.clip(self.cell_pressure, 0, self.max_pressure_pa)

    def update_thermal(self, interface_temp, dt):
        """
        Update cell temperature based on interface temperature.

        Args:
            interface_temp: Temperature at skin-mattress interface (°C)
            dt: Time step in seconds
        """
        # Convert to Kelvin
        interface_K = interface_temp + 273.15

        # Thermal equilibration (simple first-order)
        tau = 300  # Time constant (5 minutes)

        self.cell_temperature += (interface_K - self.cell_temperature) * (dt / tau)

        # Temperature affects pressure (gay-lussac's law at constant volume)
        # P1/T1 = P2/T2
        # This is already handled through ideal gas law in pressure calculations

        # Update surface temperature
        self.surface_temp = interface_temp

    def set_zone_pressure(self, zone_mask, target_pressure_mmhg, rate=0.1):
        """
        Set target pressure for a specific zone.

        Args:
            zone_mask: Boolean mask for zone
            target_pressure_mmhg: Target pressure in mmHg
            rate: Adjustment rate (0-1)
        """
        target_pa = target_pressure_mmhg / MMHG_PER_PASCAL
        self.target_pressure[zone_mask] = target_pa

        # Adjust current pressure toward target
        diff = self.target_pressure - self.cell_pressure
        pressure_change = rate * diff

        # Modify moles to achieve pressure change
        R = 8.314
        mole_change = (pressure_change * self.cell_volume) / (R * self.cell_temperature)
        self.cell_moles += mole_change
        self.cell_moles = np.maximum(self.cell_moles, 1e-10)

        # Update pressure
        self.cell_pressure = (self.cell_moles * R * self.cell_temperature) / self.cell_volume - 101325
        self.cell_pressure = np.clip(self.cell_pressure, 0, self.max_pressure_pa)

    def _resample_to_grid(self, matrix):
        """Resample input matrix to mattress grid."""
        if matrix.shape == self.shape:
            return matrix

        # Simple resampling using block averaging
        result = np.zeros(self.shape)
        row_ratio = matrix.shape[0] / self.n_rows
        col_ratio = matrix.shape[1] / self.n_cols

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                r_start = int(i * row_ratio)
                r_end = max(int((i + 1) * row_ratio), r_start + 1)
                c_start = int(j * col_ratio)
                c_end = max(int((j + 1) * col_ratio), c_start + 1)

                r_end = min(r_end, matrix.shape[0])
                c_end = min(c_end, matrix.shape[1])

                if r_start < r_end and c_start < c_end:
                    result[i, j] = matrix[r_start:r_end, c_start:c_end].mean()

        return result

    def get_pressure_mmhg(self):
        """Get cell pressures in mmHg."""
        return self.cell_pressure * MMHG_PER_PASCAL


# =============================================================================
# NONLINEAR CONTACT MECHANICS
# =============================================================================

class HertzianContactModel:
    """
    Nonlinear contact mechanics using Hertzian contact theory.

    For contact between curved surfaces:
    P = (4/3) × E* × √R × δ^(3/2)

    Where:
    - E*: Combined elastic modulus
    - R: Effective radius of curvature
    - δ: Indentation depth

    Also includes:
    - Contact area calculation
    - Pressure distribution within contact zone
    - Nonlinear superposition of multiple contacts
    """

    def __init__(self, tissue_model, mattress):
        """
        Initialize contact model.

        Args:
            tissue_model: Tissue3DModel instance
            mattress: PneumaticMattressModel instance
        """
        self.tissue = tissue_model
        self.mattress = mattress

        self.grid_shape = mattress.shape
        self.cell_size = mattress.cell_size_m

        # Contact state
        self.contact_pressure = np.zeros(self.grid_shape)
        self.contact_area = np.zeros(self.grid_shape)
        self.indentation = np.zeros(self.grid_shape)

        # Body curvature radii (for Hertzian calculation)
        self._compute_body_curvature()

    def _compute_body_curvature(self):
        """
        Compute effective radius of curvature for body regions.

        Different body parts have different curvature.
        """
        self.curvature_radius = np.ones(self.grid_shape) * 0.15  # 15cm default

        rows, cols = self.grid_shape

        # Head (smaller radius)
        self.curvature_radius[:int(0.12 * rows), :] = 0.10

        # Shoulders (larger radius in center)
        shoulder_rows = slice(int(0.12 * rows), int(0.22 * rows))
        self.curvature_radius[shoulder_rows, :] = 0.08

        # Torso (larger radius)
        torso_rows = slice(int(0.22 * rows), int(0.45 * rows))
        self.curvature_radius[torso_rows, :] = 0.25

        # Pelvis/sacrum (very small radius - bony)
        pelvis_rows = slice(int(0.45 * rows), int(0.55 * rows))
        center = slice(int(0.35 * cols), int(0.65 * cols))
        self.curvature_radius[pelvis_rows, center] = 0.03

        # Thighs
        thigh_rows = slice(int(0.55 * rows), int(0.78 * rows))
        self.curvature_radius[thigh_rows, :] = 0.12

        # Calves
        calf_rows = slice(int(0.78 * rows), int(0.95 * rows))
        self.curvature_radius[calf_rows, :] = 0.06

        # Heels (very small radius)
        self.curvature_radius[int(0.95 * rows):, :] = 0.02

    def compute_contact_pressure(self, force_matrix):
        """
        Compute contact pressure using Hertzian contact theory.

        Args:
            force_matrix: Applied force per cell (N)

        Returns:
            Contact pressure matrix (Pa)
        """
        # Resample force to grid
        force = self.mattress._resample_to_grid(force_matrix)

        # Get effective moduli
        E_tissue = self.tissue.get_spatially_varying_modulus()
        E_mattress = 50000  # Pa (soft foam/air cell effective modulus)

        # Combined modulus: 1/E* = (1-ν1²)/E1 + (1-ν2²)/E2
        nu_tissue = 0.45
        nu_mattress = 0.49
        E_star = 1 / ((1 - nu_tissue**2) / E_tissue + (1 - nu_mattress**2) / E_mattress)

        # Hertzian contact for sphere on flat:
        # F = (4/3) × E* × √R × δ^(3/2)
        # Solving for δ: δ = (3F / (4 E* √R))^(2/3)

        R = self.curvature_radius

        # Indentation depth
        coef = 3 * force / (4 * E_star * np.sqrt(R) + 1e-10)
        self.indentation = np.power(np.maximum(coef, 0), 2/3)

        # Contact radius: a = √(R × δ)
        contact_radius = np.sqrt(R * self.indentation)
        self.contact_area = np.pi * contact_radius**2

        # Mean contact pressure: P_mean = F / A
        # But Hertzian gives pressure distribution: P(r) = P_max × √(1 - r²/a²)
        # P_max = (3/2) × P_mean

        P_mean = force / (self.contact_area + 1e-10)
        P_max = 1.5 * P_mean

        # Limit to physical values
        self.contact_pressure = np.clip(P_max, 0, 100000)  # Max 100 kPa

        return self.contact_pressure

    def compute_superimposed_pressure(self, force_matrix):
        """
        Compute pressure with nonlinear superposition of contact points.

        Neighboring contact zones interact, increasing local stress.

        Args:
            force_matrix: Applied force per cell (N)

        Returns:
            Total contact pressure (Pa) accounting for interactions
        """
        # Base Hertzian pressure
        base_pressure = self.compute_contact_pressure(force_matrix)

        # Stress interaction between neighboring loaded cells
        # Use weighted sum of nearby pressures
        interaction_kernel = np.array([
            [0.05, 0.15, 0.05],
            [0.15, 1.0, 0.15],
            [0.05, 0.15, 0.05]
        ])

        # Convolve to get interaction effects
        from scipy.signal import convolve2d
        try:
            interacted_pressure = convolve2d(base_pressure, interaction_kernel,
                                              mode='same', boundary='fill')
        except ImportError:
            # Fallback without scipy
            interacted_pressure = base_pressure * 1.3

        # Nonlinear combination (stress concentration)
        combined_pressure = np.sqrt(base_pressure**2 + 0.3 * interacted_pressure**2)

        return combined_pressure

    def get_pressure_mmhg(self):
        """Get contact pressure in mmHg."""
        return self.contact_pressure * MMHG_PER_PASCAL


# =============================================================================
# COMPREHENSIVE BODY MODEL
# =============================================================================

class AdvancedBodyModel:
    """
    Advanced body model with anatomical detail and mass distribution.
    """

    REGION_DEFINITIONS = {
        'head': {
            'mass_fraction': 0.08,
            'length_fraction': 0.12,
            'width_fraction': 0.35,
            'center_offset': 0,
        },
        'shoulders': {
            'mass_fraction': 0.12,
            'length_fraction': 0.08,
            'width_fraction': 0.95,
            'center_offset': 0,
        },
        'upper_torso': {
            'mass_fraction': 0.15,
            'length_fraction': 0.15,
            'width_fraction': 0.75,
            'center_offset': 0,
        },
        'lower_torso': {
            'mass_fraction': 0.18,
            'length_fraction': 0.10,
            'width_fraction': 0.70,
            'center_offset': 0,
        },
        'pelvis': {
            'mass_fraction': 0.15,
            'length_fraction': 0.10,
            'width_fraction': 0.80,
            'center_offset': 0,
        },
        'thighs': {
            'mass_fraction': 0.20,
            'length_fraction': 0.23,
            'width_fraction': 0.55,
            'center_offset': 0,
        },
        'calves': {
            'mass_fraction': 0.08,
            'length_fraction': 0.17,
            'width_fraction': 0.30,
            'center_offset': 0,
        },
        'heels': {
            'mass_fraction': 0.04,
            'length_fraction': 0.05,
            'width_fraction': 0.20,
            'center_offset': 0,
        },
    }

    def __init__(self, body_mass=70, body_length=180, body_width=65,
                 grid_resolution=5, waist_hinge_angle=0, gender='male'):
        """
        Initialize advanced body model.

        Args:
            body_mass: Total mass in kg
            body_length: Length in cm
            body_width: Width in cm
            grid_resolution: Cells per cm
            waist_hinge_angle: Angle (degrees) the bed hinges at the waist,
                               raising the upper body (0 = flat)
            gender: Body gender for SMPL model ('male', 'female', or 'neutral')
        """
        self.body_mass = body_mass
        self.body_length = body_length
        self.body_width = body_width
        self.waist_hinge_angle = waist_hinge_angle
        self.gender = gender.lower()

        self.n_rows = int(body_length / grid_resolution)
        self.n_cols = int(body_width / grid_resolution)
        self.grid_shape = (self.n_rows, self.n_cols)
        self.cell_size = grid_resolution  # cm

        # Waist hinge row (pelvis/lower_torso boundary, ~45% from head)
        self.hinge_row = int(0.45 * self.n_rows)

        # Mass distribution
        self.mass_matrix = np.zeros(self.grid_shape)
        self.region_map = np.zeros(self.grid_shape, dtype=int)

        # Build body
        self._build_body()

        # Current position offset
        self.offset_x = 0
        self.offset_y = 0

    def _build_body(self):
        """
        Construct mass distribution using the SMPL body model.

        Loads the SMPL neutral mesh, poses it supine with arms at sides,
        projects the triangulated surface onto the simulation grid, and
        stores the full mesh for 3D visualisation.
        """
        import smplx
        import torch
        import math as _math

        nr, nc = self.n_rows, self.n_cols
        cell = self.cell_size

        # ---- Load SMPL model ----
        smpl_dir = '/Users/jaspermetz/Documents/Body_Sim/smpl'
        model = smplx.create(smpl_dir, model_type='smpl', gender=self.gender)

        # ---- Pose: supine with arms at sides ----
        body_pose = torch.zeros(1, 69)
        # SMPL joints: 16=L_shoulder, 17=R_shoulder
        body_pose[0, 45 + 2] = -1.5   # L_shoulder (joint 16) Z-rot (arm down to side)
        body_pose[0, 48 + 2] = 1.5    # R_shoulder (joint 17) Z-rot (arm down to side)

        output = model(body_pose=body_pose)
        verts = output.vertices.detach().numpy()[0]   # (6890, 3)
        faces = model.faces                            # (13776, 3)

        # ---- Rotate from standing to supine (lying on back) ----
        angle = -_math.pi / 2
        Rx = np.array([
            [1, 0, 0],
            [0, _math.cos(angle), -_math.sin(angle)],
            [0, _math.sin(angle),  _math.cos(angle)],
        ])
        v = verts @ Rx.T
        v *= 100  # metres → cm

        # After rotation:  X = width,  Y = depth (height off bed),  Z = body length
        # Scale to match configured body_length
        raw_length = v[:, 2].max() - v[:, 2].min()
        scale = self.body_length / raw_length
        v *= scale

        # Flip Z so head is at Z=0, feet at Z=body_length
        # (matches grid convention: row 0 = head)
        v[:, 2] = v[:, 2].max() - v[:, 2] + v[:, 2].min()
        v[:, 2] -= v[:, 2].min()   # head at z=0

        v[:, 1] -= v[:, 1].min()   # back at y=0

        # Centre X within the grid
        v[:, 0] -= v[:, 0].mean()
        v[:, 0] += 40.0  # centre body midline at x=40

        # ---- Load PhysioNet PMD supine pressure template ----
        # Grand-average of 52 supine pressure maps (13 subjects × 4 postures)
        # from the PhysioNet Pressure Map Dataset (PMD) v1.0.0.
        # Original grid: 64 rows × 32 cols (Vista Medical FSA SoftFlex 2048).
        from scipy.ndimage import zoom
        import os

        pmd_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'pmd_data', 'supine_pressure_template.npy')
        pmd_template = np.load(pmd_path)  # (64, 32), raw sensor [0-1000]

        # Resample PMD template to simulation grid (nr × nc)
        zoom_r = nr / pmd_template.shape[0]
        zoom_c = nc / pmd_template.shape[1]
        pmd_resampled = zoom(pmd_template, (zoom_r, zoom_c), order=3)
        pmd_resampled = np.maximum(pmd_resampled, 0.0)

        # Use PMD distribution for mass matrix
        self.mass_matrix = pmd_resampled.copy()

        # ---- Build depth map and region map from SMPL mesh ----
        depth_map = np.zeros((nr, nc))
        for fi in range(len(faces)):
            tri = v[faces[fi]]
            cx = tri[:, 0].mean()
            cy = tri[:, 1].mean()
            cz = tri[:, 2].mean()
            row = int(cz / cell)
            col = int(cx / cell)
            if 0 <= row < nr and 0 <= col < nc:
                depth_map[row, col] = max(depth_map[row, col], cy)
                frac = cz / self.body_length
                self.region_map[row, col] = self._region_id_for_frac(frac)

        # ---- Depth profile per row (max depth across columns) ----
        self.depth_profile = np.zeros(nr)
        for i in range(nr):
            if depth_map[i].max() > 0:
                self.depth_profile[i] = depth_map[i].max()
        self.arm_depth_profile = self.depth_profile.copy()

        # ---- Store SMPL mesh for 3D visualisation ----
        self.smpl_vertices = v          # (6890, 3) in cm
        self.smpl_faces = faces         # (13776, 3)

        # ---- Normalise total mass ----
        total = self.mass_matrix.sum()
        if total > 0:
            self.mass_matrix *= self.body_mass / total

        # Enforce symmetry
        self._enforce_symmetry()

    @staticmethod
    def _region_id_for_frac(f):
        """Map fractional position to region ID."""
        if f < 0.12:
            return 1   # head
        elif f < 0.20:
            return 2   # shoulders
        elif f < 0.35:
            return 3   # upper torso
        elif f < 0.45:
            return 4   # lower torso
        elif f < 0.55:
            return 5   # pelvis
        elif f < 0.78:
            return 6   # thighs
        elif f < 0.95:
            return 7   # calves
        else:
            return 8   # heels/feet

    def _enforce_symmetry(self):
        """Enforce left-right symmetry."""
        mid = self.n_cols // 2
        left = self.mass_matrix[:, :mid]
        right = self.mass_matrix[:, -mid:][:, ::-1]
        avg = (left + right) / 2
        self.mass_matrix[:, :mid] = avg
        self.mass_matrix[:, -mid:] = avg[:, ::-1]

    def get_force_distribution(self):
        """
        Get force distribution (N) from mass, accounting for waist hinge.

        When the bed is hinged at the waist, the upper body is inclined.
        This reduces the normal force component from upper-body cells by
        cos(angle) and concentrates the excess force at the hinge region
        (sacrum/buttocks), modelling the weight bearing shift that occurs
        when a patient sits up.
        """
        force = self.mass_matrix * GRAVITY

        if self.waist_hinge_angle > 0:
            angle_rad = np.radians(self.waist_hinge_angle)
            hinge = self.hinge_row

            # Upper body normal force reduced by cos(angle)
            upper_force_original = force[:hinge, :].copy()
            normal_component = upper_force_original * np.cos(angle_rad)
            excess_force = (upper_force_original - normal_component).sum()

            force[:hinge, :] = normal_component

            # Redistribute excess force to the hinge region (sacrum/buttocks).
            # Spread over a small band around the hinge point.
            band_start = max(hinge - 2, 0)
            band_end = min(hinge + 3, self.n_rows)
            hinge_band = slice(band_start, band_end)

            hinge_area = force[hinge_band, :]
            hinge_total = hinge_area.sum()
            if hinge_total > 0:
                # Distribute proportionally to existing weight in hinge region
                force[hinge_band, :] += excess_force * (hinge_area / hinge_total)
            else:
                # Uniform fallback
                force[hinge_band, :] += excess_force / max(hinge_area.size, 1)

        return force

    def apply_micro_shift(self, dx, dy):
        """Apply small positional shift."""
        dx = int(np.clip(dx, -3, 3))
        dy = int(np.clip(dy, -3, 3))

        self.mass_matrix = np.roll(self.mass_matrix, dy, axis=0)
        self.mass_matrix = np.roll(self.mass_matrix, dx, axis=1)

        self.offset_x += dx
        self.offset_y += dy


# =============================================================================
# LATERAL ROTATION BED MODEL
# =============================================================================

class LateralRotationModel:
    """
    Models a lateral rotation bed that physically tilts the patient side-to-side.

    This is different from micro-repositioning (which requires patient movement).
    The bed itself rotates, redistributing pressure without patient participation.

    Key features:
    - Continuous or intermittent rotation
    - Adjustable angle (typically 20-45°)
    - Adjustable cycle time (typically 2-20 minutes)
    - Pressure redistribution calculation
    """

    def __init__(self, grid_shape, max_angle=30, rotation_period=600,
                 pause_duration=60, transition_time=120):
        """
        Initialize lateral rotation bed.

        Args:
            grid_shape: Shape of pressure grid (rows, cols)
            max_angle: Maximum tilt angle in degrees (typically 20-45°)
            rotation_period: Full cycle time in seconds (left→right→left)
            pause_duration: Time to pause at maximum tilt (seconds)
            transition_time: Time to rotate from one side to other (seconds)
        """
        self.grid_shape = grid_shape
        self.max_angle = max_angle
        self.rotation_period = rotation_period
        self.pause_duration = pause_duration
        self.transition_time = transition_time

        # Current state
        self.current_angle = 0.0  # degrees, positive = right, negative = left
        self.rotation_direction = 1  # 1 = rotating right, -1 = rotating left

        # Pressure redistribution factor per degree of tilt
        # Research shows ~3-5% pressure reduction per 10° of tilt
        self.pressure_reduction_per_degree = 0.004  # 0.4% per degree

        # Weight shift factor (how much weight shifts to lower side)
        self.weight_shift_per_degree = 0.015  # 1.5% weight shift per degree

        # Track time in current position
        self.time_in_position = 0

        # History for analysis
        self.angle_history = []
        self.time_history = []

    def update(self, current_time):
        """
        Update rotation angle based on current time.

        Uses sinusoidal motion for smooth, continuous rotation.

        Args:
            current_time: Current simulation time (seconds)

        Returns:
            Current tilt angle in degrees
        """
        # Calculate phase in rotation cycle (0 to 1)
        phase = (current_time % self.rotation_period) / self.rotation_period

        # Sinusoidal motion: angle = max_angle * sin(2π * phase)
        # This gives smooth acceleration/deceleration at extremes
        self.current_angle = self.max_angle * np.sin(2 * np.pi * phase)

        # Record history
        self.angle_history.append(self.current_angle)
        self.time_history.append(current_time)

        return self.current_angle

    def get_pressure_redistribution(self, base_pressure):
        """
        Calculate pressure redistribution due to current tilt angle.

        When tilted:
        - Pressure decreases on upper side (less weight)
        - Pressure increases on lower side (more weight)
        - Net effect: redistributes load, allows tissue recovery on upper side

        Args:
            base_pressure: Original pressure distribution (2D array)

        Returns:
            Modified pressure distribution
        """
        if abs(self.current_angle) < 0.1:
            return base_pressure

        # Use the input shape, not self.grid_shape (they may differ)
        input_shape = base_pressure.shape
        input_rows, input_cols = input_shape

        # Create lateral gradient based on tilt angle
        # Positive angle = tilted right = left side up, right side down
        tilt_rad = np.radians(self.current_angle)

        # Weight shift factor across the bed width
        shift_factor = self.weight_shift_per_degree * self.current_angle

        # Create gradient across columns (left to right)
        col_positions = np.linspace(-1, 1, input_cols)
        weight_gradient = 1.0 + shift_factor * col_positions

        # Apply gradient to pressure
        redistribution = np.ones(input_shape)
        for i in range(input_rows):
            redistribution[i, :] = weight_gradient

        # Apply redistribution
        modified_pressure = base_pressure * redistribution

        # Ensure no negative pressures
        modified_pressure = np.maximum(modified_pressure, 0)

        return modified_pressure

    def get_tissue_relief_map(self):
        """
        Get map showing which areas are getting pressure relief.

        Returns:
            2D array where:
            - Positive values = pressure relief (good)
            - Negative values = increased pressure
            - Zero = no change
        """
        relief = np.zeros(self.grid_shape)

        if abs(self.current_angle) < 0.1:
            return relief

        # Calculate relief based on position and tilt
        rows, cols = self.grid_shape
        col_positions = np.linspace(-1, 1, cols)

        # Relief proportional to position and opposite to tilt direction
        relief_gradient = -self.current_angle * self.pressure_reduction_per_degree * col_positions

        for i in range(rows):
            relief[i, :] = relief_gradient

        return relief

    def get_cumulative_relief_time(self, position_col, total_time):
        """
        Calculate how much time a given position has had pressure relief.

        Args:
            position_col: Column position (0 = left edge, max = right edge)
            total_time: Total simulation time

        Returns:
            Fraction of time with >20% pressure relief
        """
        if len(self.angle_history) == 0:
            return 0

        # Normalize column position to -1 to 1
        cols = self.grid_shape[1]
        norm_pos = 2 * position_col / (cols - 1) - 1

        # Count time steps where this position had significant relief
        relief_count = 0
        for angle in self.angle_history:
            # Position has relief when it's on the "up" side of tilt
            # Left side (negative norm_pos) has relief when angle is positive
            # Right side (positive norm_pos) has relief when angle is negative
            relief = -angle * self.pressure_reduction_per_degree * norm_pos
            if relief > 0.002:  # >0.2% relief threshold
                relief_count += 1

        return relief_count / len(self.angle_history) if self.angle_history else 0

    def get_status(self):
        """Get current rotation status."""
        if abs(self.current_angle) < 1:
            position = "CENTER"
        elif self.current_angle > 0:
            position = f"RIGHT {abs(self.current_angle):.1f}°"
        else:
            position = f"LEFT {abs(self.current_angle):.1f}°"

        return {
            'angle': self.current_angle,
            'position': position,
            'max_angle': self.max_angle,
            'period': self.rotation_period,
        }


# =============================================================================
# MULTI-FACTOR RISK MODEL
# =============================================================================

class ComprehensiveRiskModel:
    """
    Multi-factorial pressure ulcer risk model.

    Risk factors incorporated:
    1. Pressure magnitude and duration
    2. Shear stress
    3. Temperature (ischemia effects)
    4. Moisture
    5. Tissue vulnerability (depth, bony prominences)

    Based on cumulative damage theory:
    D(t) = ∫[f(P, τ, T, M, V)] dt

    For quadriplegic patients:
    - All damage rates are increased by damage_rate_multiplier (2.5x)
    - Skin fragility factor applied (1.5x)
    - Synergy factors increased (compounding vulnerabilities)
    """

    def __init__(self, grid_shape, quadriplegic_mode=False):
        """
        Initialize comprehensive risk model.

        Args:
            grid_shape: (rows, cols) grid dimensions
            quadriplegic_mode: If True, applies increased risk factors
        """
        self.grid_shape = grid_shape
        self.quadriplegic_mode = quadriplegic_mode

        # Component exposures
        self.pressure_exposure = np.zeros(grid_shape)  # mmHg·seconds
        self.shear_exposure = np.zeros(grid_shape)  # kPa·seconds
        self.thermal_exposure = np.zeros(grid_shape)  # °C·seconds above threshold
        self.moisture_exposure = np.zeros(grid_shape)  # saturation·seconds

        # Time tracking
        self.time_above_pressure_threshold = np.zeros(grid_shape)
        self.time_above_shear_threshold = np.zeros(grid_shape)

        # Cumulative damage
        self.damage_index = np.zeros(grid_shape)  # 0-1 normalized

        # Risk thresholds (lower for quadriplegic due to tissue fragility)
        if quadriplegic_mode:
            self.pressure_threshold = CAPILLARY_CLOSING_PRESSURE * 0.85  # Lower threshold
            self.shear_threshold = SHEAR_DAMAGE_THRESHOLD * 0.85
            self.temp_threshold = 37.0  # °C
        else:
            self.pressure_threshold = CAPILLARY_CLOSING_PRESSURE  # mmHg
            self.shear_threshold = SHEAR_DAMAGE_THRESHOLD  # kPa
            self.temp_threshold = 37.0  # °C (elevated interface temp)

        # Damage rate constants (increased for quadriplegic)
        if quadriplegic_mode:
            quad_mult = QUADRIPLEGIC_FACTORS['damage_rate_multiplier']
            self.pressure_rate = 1e-5 * quad_mult
            self.shear_rate = 5e-5 * quad_mult * QUADRIPLEGIC_FACTORS['spasticity_shear_factor']
            self.thermal_rate = 1e-4 * quad_mult
            self.moisture_rate = 2e-5 * quad_mult * QUADRIPLEGIC_FACTORS['moisture_accumulation_rate']
        else:
            self.pressure_rate = 1e-5  # damage per mmHg·second above threshold
            self.shear_rate = 5e-5  # damage per kPa·second above threshold
            self.thermal_rate = 1e-4  # damage per °C·second above threshold
            self.moisture_rate = 2e-5  # damage per saturation·second

        # Synergy factors (stronger for quadriplegic - compounding vulnerabilities)
        if quadriplegic_mode:
            self.pressure_shear_synergy = 2.0  # Higher synergy
            self.moisture_pressure_synergy = 1.8
            self.skin_fragility = QUADRIPLEGIC_FACTORS['skin_fragility']
        else:
            self.pressure_shear_synergy = 1.5  # Combined effect multiplier
            self.moisture_pressure_synergy = 1.3
            self.skin_fragility = 1.0

    def update(self, pressure_mmhg, shear_pa, temperature, moisture,
               tissue_vulnerability, dt):
        """
        Update risk metrics with new measurements.

        Args:
            pressure_mmhg: Pressure distribution (mmHg)
            shear_pa: Shear stress (Pa)
            temperature: Interface temperature (°C)
            moisture: Moisture saturation (0-1)
            tissue_vulnerability: Tissue damage susceptibility map
            dt: Time step (seconds)
        """
        # 1. Pressure damage component
        excess_pressure = np.maximum(pressure_mmhg - self.pressure_threshold, 0)
        self.pressure_exposure += excess_pressure * dt
        self.time_above_pressure_threshold += (pressure_mmhg > self.pressure_threshold) * dt

        pressure_damage = self.pressure_rate * excess_pressure * dt

        # 2. Shear damage component
        shear_kpa = shear_pa / 1000
        excess_shear = np.maximum(shear_kpa - self.shear_threshold, 0)
        self.shear_exposure += excess_shear * dt
        self.time_above_shear_threshold += (shear_kpa > self.shear_threshold) * dt

        shear_damage = self.shear_rate * excess_shear * dt

        # 3. Thermal damage component
        excess_temp = np.maximum(temperature - self.temp_threshold, 0)
        self.thermal_exposure += excess_temp * dt

        thermal_damage = self.thermal_rate * excess_temp * dt

        # 4. Moisture damage component
        self.moisture_exposure += moisture * dt

        moisture_damage = self.moisture_rate * moisture * dt

        # 5. Synergistic effects
        # Pressure + shear synergy
        has_both_pressure_shear = (excess_pressure > 0) & (excess_shear > 0)
        synergy_ps = np.where(has_both_pressure_shear,
                              self.pressure_shear_synergy, 1.0)

        # Moisture + pressure synergy
        has_moisture_pressure = (moisture > 0.3) & (excess_pressure > 0)
        synergy_mp = np.where(has_moisture_pressure,
                              self.moisture_pressure_synergy, 1.0)

        # 6. Apply tissue vulnerability and skin fragility (quadriplegic factor)
        vulnerability_factor = tissue_vulnerability * self.skin_fragility

        # 7. Total damage increment
        base_damage = pressure_damage + shear_damage + thermal_damage + moisture_damage
        total_damage = base_damage * synergy_ps * synergy_mp * vulnerability_factor

        # 8. Accumulate damage (with nonlinear acceleration at high damage)
        # Quadriplegic patients have faster damage acceleration due to impaired healing
        if self.quadriplegic_mode:
            damage_acceleration = 1 + 3 * self.damage_index  # Faster acceleration
        else:
            damage_acceleration = 1 + 2 * self.damage_index  # Damage begets damage
        self.damage_index += total_damage * damage_acceleration

        # 9. Clamp to 0-1 range
        self.damage_index = np.clip(self.damage_index, 0, 1)

    def get_risk_categories(self):
        """
        Categorize risk levels.

        Returns:
            Dictionary of risk category masks
        """
        return {
            'minimal': self.damage_index < 0.1,
            'low': (self.damage_index >= 0.1) & (self.damage_index < 0.25),
            'moderate': (self.damage_index >= 0.25) & (self.damage_index < 0.5),
            'high': (self.damage_index >= 0.5) & (self.damage_index < 0.75),
            'critical': self.damage_index >= 0.75
        }

    def get_primary_risk_factor(self):
        """
        Identify the primary contributing risk factor at each location.

        Returns:
            Matrix of factor indices (0=pressure, 1=shear, 2=thermal, 3=moisture)
        """
        # Normalize exposures
        factors = np.stack([
            self.pressure_exposure / (self.pressure_exposure.max() + 1e-10),
            self.shear_exposure / (self.shear_exposure.max() + 1e-10),
            self.thermal_exposure / (self.thermal_exposure.max() + 1e-10),
            self.moisture_exposure / (self.moisture_exposure.max() + 1e-10)
        ], axis=0)

        return np.argmax(factors, axis=0)

    def reset(self):
        """Reset all accumulated damage."""
        self.pressure_exposure = np.zeros(self.grid_shape)
        self.shear_exposure = np.zeros(self.grid_shape)
        self.thermal_exposure = np.zeros(self.grid_shape)
        self.moisture_exposure = np.zeros(self.grid_shape)
        self.time_above_pressure_threshold = np.zeros(self.grid_shape)
        self.time_above_shear_threshold = np.zeros(self.grid_shape)
        self.damage_index = np.zeros(self.grid_shape)


# =============================================================================
# ADVANCED OPTIMIZER
# =============================================================================

class AdvancedOptimizer:
    """
    Multi-objective optimization for pressure ulcer prevention.

    Objectives:
    1. Minimize peak pressure
    2. Minimize shear stress
    3. Maintain thermal comfort
    4. Control moisture
    5. Minimize cumulative damage

    Control variables:
    - Zone-based mattress pressure
    - Alternating pressure cycling
    - Micro-repositioning schedule
    - Bed angle
    """

    def __init__(self, body, mattress, tissue, thermal, shear, risk,
                 enable_zone_control=True, enable_alternating=True,
                 enable_repositioning=True, reposition_interval=1800,
                 pressure_cycle_period=300):
        """
        Initialize optimizer with all subsystem references.

        Args:
            enable_zone_control: Adaptively adjust zone pressures
            enable_alternating: Alternating pressure wave cycling
            enable_repositioning: Periodic micro-repositioning shifts
            reposition_interval: Seconds between repositioning (default 1800)
            pressure_cycle_period: Alternating cycle period in seconds
        """
        self.body = body
        self.mattress = mattress
        self.tissue = tissue
        self.thermal = thermal
        self.shear = shear
        self.risk = risk

        self.enable_zone_control = enable_zone_control
        self.enable_alternating = enable_alternating
        self.enable_repositioning = enable_repositioning

        # Control parameters
        self.reposition_interval = reposition_interval
        self.pressure_cycle_period = pressure_cycle_period
        self.last_reposition = 0

        # Zone definitions
        self._define_zones()

        # Optimal parameters (will be tuned)
        self.optimal_zone_pressures = {}

    def _define_zones(self):
        """Define mattress control zones."""
        n_rows, n_cols = self.mattress.shape

        self.zones = {
            'head': (slice(0, int(0.12*n_rows)), slice(None)),
            'shoulders': (slice(int(0.12*n_rows), int(0.22*n_rows)), slice(None)),
            'upper_back': (slice(int(0.22*n_rows), int(0.35*n_rows)), slice(None)),
            'lower_back': (slice(int(0.35*n_rows), int(0.45*n_rows)), slice(None)),
            'sacrum': (slice(int(0.45*n_rows), int(0.55*n_rows)), slice(int(0.25*n_cols), int(0.75*n_cols))),
            'buttocks': (slice(int(0.45*n_rows), int(0.55*n_rows)), slice(None)),
            'thighs': (slice(int(0.55*n_rows), int(0.78*n_rows)), slice(None)),
            'calves': (slice(int(0.78*n_rows), int(0.95*n_rows)), slice(None)),
            'heels': (slice(int(0.95*n_rows), None), slice(None)),
        }

    def optimize_step(self, current_time, pressure_mmhg, shear_stress, temperature):
        """
        Perform optimization step.

        Args:
            current_time: Simulation time (seconds)
            pressure_mmhg: Current pressure distribution
            shear_stress: Current shear stress
            temperature: Current interface temperature

        Returns:
            Actions taken
        """
        actions = {}

        # 1. Adaptive zone pressure control
        if self.enable_zone_control:
            self._optimize_zone_pressures(pressure_mmhg)
            actions['zone_pressures_adjusted'] = True

        # 2. Alternating pressure cycling
        if self.enable_alternating:
            cycle_phase = (current_time % self.pressure_cycle_period) / self.pressure_cycle_period
            self._apply_alternating_pressure(cycle_phase)
            actions['cycle_phase'] = cycle_phase

        # 3. Check repositioning need
        if self.enable_repositioning:
            if current_time - self.last_reposition >= self.reposition_interval:
                if self._should_reposition(pressure_mmhg, shear_stress):
                    shift = self._compute_optimal_shift()
                    self.body.apply_micro_shift(shift[0], shift[1])
                    self.last_reposition = current_time
                    actions['repositioned'] = True
                    actions['shift'] = shift

        # 4. Bed angle optimization (reduce shear if too high)
        if shear_stress.max() > 15000:  # 15 kPa threshold
            actions['recommend_lower_bed_angle'] = True

        return actions

    def _optimize_zone_pressures(self, pressure_mmhg):
        """Adaptively adjust zone pressures."""
        for zone_name, (row_slice, col_slice) in self.zones.items():
            zone_mask = np.zeros(self.mattress.shape, dtype=bool)
            zone_mask[row_slice, col_slice] = True

            zone_pressure = pressure_mmhg[zone_mask]
            if len(zone_pressure) == 0:
                continue

            max_p = zone_pressure.max()
            mean_p = zone_pressure.mean()

            # Target mattress pressure based on measured interface pressure
            if max_p > 50:
                target = 12  # Very low to allow more sink-in
            elif max_p > 40:
                target = 15
            elif max_p > 32:
                target = 18
            else:
                target = 22

            self.mattress.set_zone_pressure(zone_mask, target, rate=0.08)
            self.optimal_zone_pressures[zone_name] = target

    def _apply_alternating_pressure(self, phase):
        """Apply alternating pressure pattern."""
        n_rows = self.mattress.n_rows

        # Create wave pattern
        for i in range(n_rows):
            row_phase = (i / 8 + phase) % 1.0
            modulation = 0.12 * np.sin(2 * np.pi * row_phase)
            self.mattress.cell_pressure[i, :] *= (1 + modulation)

    def _should_reposition(self, pressure, shear):
        """Determine if repositioning is needed."""
        high_pressure = (pressure > 45).sum() > 20
        high_shear = (shear > 12000).sum() > 20
        high_risk = (self.risk.damage_index > 0.4).sum() > 30

        return high_pressure or high_shear or high_risk

    def _compute_optimal_shift(self):
        """Compute optimal micro-shift direction."""
        damage = self.risk.damage_index

        # Find centroid of high-damage area
        high_damage_mask = damage > 0.3
        if not high_damage_mask.any():
            return (0, 0)

        rows, cols = np.where(high_damage_mask)
        centroid_row = rows.mean()
        centroid_col = cols.mean()

        center_row = damage.shape[0] / 2
        center_col = damage.shape[1] / 2

        # Shift away from damage centroid
        dy = 1 if centroid_row < center_row else -1
        dx = 1 if centroid_col < center_col else -1

        return (dx, dy)

    def run_parameter_study(self, cell_sizes=[3, 5, 7],
                           reposition_intervals=[900, 1800, 3600],
                           cycle_periods=[180, 300, 600]):
        """
        Run parameter study to find optimal settings.

        Returns:
            Optimal parameters and comparative results
        """
        results = []

        for cell_size in cell_sizes:
            for interval in reposition_intervals:
                for cycle in cycle_periods:
                    # Score based on physics (actual simulation would be better)
                    # Smaller cells = better pressure distribution
                    cell_score = cell_size / 5

                    # More frequent repositioning = lower risk
                    interval_score = interval / 1800

                    # Moderate cycle period often optimal
                    cycle_score = abs(cycle - 300) / 300 * 0.5 + 0.5

                    total = 0.4 * cell_score + 0.4 * interval_score + 0.2 * cycle_score

                    results.append({
                        'cell_size': cell_size,
                        'reposition_interval': interval,
                        'cycle_period': cycle,
                        'score': total
                    })

        best = min(results, key=lambda x: x['score'])

        return {
            'optimal_cell_size': best['cell_size'],
            'optimal_reposition_interval': best['reposition_interval'],
            'optimal_cycle_period': best['cycle_period'],
            'all_results': results
        }


# =============================================================================
# MAIN SIMULATION ENGINE
# =============================================================================

class AdvancedPressureUlcerSimulation:
    """
    Comprehensive pressure ulcer simulation with all advanced features.

    For quadriplegic patients:
    - Micro-repositioning is DISABLED (patient cannot move)
    - Lateral rotation bed is ENABLED (bed physically tilts)
    - Risk factors are INCREASED (atrophy, moisture, fragility)
    - Tissue vulnerability is HIGHER
    """

    def __init__(self, body_mass=70, cell_size=5, simulation_time=3600,
                 timestep=15, waist_hinge_angle=0,
                 enable_zone_control=True, enable_alternating=True,
                 enable_repositioning=True, reposition_interval=1800,
                 pressure_cycle_period=300,
                 quadriplegic_mode=False,
                 enable_lateral_rotation=False,
                 lateral_rotation_angle=25,
                 lateral_rotation_period=600,
                 moisture_management='none',
                 gender='male'):
        """
        Initialize advanced simulation.

        Args:
            body_mass: Patient mass (kg)
            cell_size: Mattress cell size (cm)
            simulation_time: Total time (seconds)
            timestep: Update interval (seconds)
            waist_hinge_angle: Hinge angle at waist (degrees)
            enable_zone_control: Adaptive zone pressure control
            enable_alternating: Alternating pressure wave cycling
            enable_repositioning: Periodic micro-repositioning (disabled for quadriplegic)
            reposition_interval: Seconds between repositioning
            pressure_cycle_period: Alternating cycle period (seconds)
            quadriplegic_mode: If True, applies quadriplegic patient factors
            enable_lateral_rotation: If True, bed rotates side-to-side
            lateral_rotation_angle: Maximum tilt angle (degrees)
            lateral_rotation_period: Full rotation cycle time (seconds)
            moisture_management: Level of moisture intervention
                                ('none', 'basic', 'moderate', 'comprehensive', 'optimal')
            gender: Body gender for SMPL model ('male', 'female', or 'neutral')
        """
        self.gender = gender
        self.quadriplegic_mode = quadriplegic_mode
        self.moisture_management = moisture_management
        self.enable_zone_control = enable_zone_control
        self.enable_alternating = enable_alternating

        # CRITICAL: Quadriplegic patients CANNOT reposition themselves
        if quadriplegic_mode:
            self.enable_repositioning = False  # Force disable
            print("  [QUADRIPLEGIC MODE] Micro-repositioning DISABLED (patient cannot move)")
        else:
            self.enable_repositioning = enable_repositioning

        self.enable_lateral_rotation = enable_lateral_rotation
        self.lateral_rotation_angle = lateral_rotation_angle
        self.lateral_rotation_period = lateral_rotation_period

        self.reposition_interval = reposition_interval
        self.pressure_cycle_period = pressure_cycle_period
        self.simulation_time = simulation_time
        self.timestep = timestep
        self.waist_hinge_angle = waist_hinge_angle
        self.current_time = 0

        # Initialize body model
        self.body = AdvancedBodyModel(
            body_mass=body_mass,
            grid_resolution=cell_size,
            waist_hinge_angle=waist_hinge_angle,
            gender=gender
        )

        # Initialize mattress
        self.mattress = PneumaticMattressModel(cell_size=cell_size)

        # Initialize 3D tissue model
        self.tissue = Tissue3DModel(self.mattress.shape, cell_size)

        # Apply quadriplegic tissue modifications
        if quadriplegic_mode:
            self._apply_quadriplegic_tissue_changes()

        # Initialize thermal model with moisture management
        self.thermal = ThermalMoistureModel(self.mattress.shape, cell_size,
                                            moisture_management=moisture_management)

        # Apply quadriplegic baseline moisture (incontinence) with moisture management adjustment
        if quadriplegic_mode:
            base_moisture = QUADRIPLEGIC_FACTORS['baseline_moisture']
            adjusted_moisture = base_moisture * self.thermal.moisture_baseline_mult
            self.thermal.moisture = np.ones(self.mattress.shape) * adjusted_moisture
            mm_name = MOISTURE_MANAGEMENT.get(moisture_management, {}).get('name', moisture_management)
            print(f"  [QUADRIPLEGIC MODE] Baseline moisture: {base_moisture*100:.0f}% -> {adjusted_moisture*100:.0f}% (with {mm_name})")

        # Initialize shear model
        self.shear = ShearStressModel(self.mattress.shape, self.tissue)

        # Initialize contact model
        self.contact = HertzianContactModel(self.tissue, self.mattress)

        # Initialize risk model (with quadriplegic factors)
        self.risk = ComprehensiveRiskModel(self.mattress.shape,
                                           quadriplegic_mode=quadriplegic_mode)

        # Initialize lateral rotation bed (if enabled)
        if enable_lateral_rotation:
            self.lateral_rotation = LateralRotationModel(
                self.mattress.shape,
                max_angle=lateral_rotation_angle,
                rotation_period=lateral_rotation_period
            )
            print(f"  [LATERAL ROTATION] Enabled: ±{lateral_rotation_angle}° every {lateral_rotation_period}s")
        else:
            self.lateral_rotation = None

        # Initialize optimizer
        self.optimizer = AdvancedOptimizer(
            self.body, self.mattress, self.tissue,
            self.thermal, self.shear, self.risk,
            enable_zone_control=self.enable_zone_control,
            enable_alternating=self.enable_alternating,
            enable_repositioning=self.enable_repositioning,
            reposition_interval=self.reposition_interval,
            pressure_cycle_period=self.pressure_cycle_period,
        )

        # Create tissue vulnerability map
        self._create_vulnerability_map()

        # Data recording
        self.pressure_history = []
        self.shear_history = []
        self.temperature_history = []
        self.moisture_history = []
        self.risk_history = []
        self.time_history = []
        self.stats_history = []

    def _apply_quadriplegic_tissue_changes(self):
        """Apply tissue modifications for quadriplegic patient."""
        print("  [QUADRIPLEGIC MODE] Applying tissue modifications:")

        # Reduce muscle thickness (atrophy)
        atrophy = QUADRIPLEGIC_FACTORS['muscle_atrophy_factor']
        self.tissue.layers['muscle'].thickness *= atrophy
        print(f"    - Muscle thickness: {atrophy*100:.0f}% of normal (atrophy)")

        # Reduce fat effectiveness
        fat_eff = QUADRIPLEGIC_FACTORS['fat_effectiveness']
        self.tissue.fat_thickness_map *= fat_eff
        print(f"    - Fat cushioning: {fat_eff*100:.0f}% effective")

        # Reduce skin damage threshold (fragility)
        fragility = QUADRIPLEGIC_FACTORS['skin_fragility']
        self.tissue.layers['skin'].damage_threshold /= fragility
        print(f"    - Skin damage threshold: {100/fragility:.0f}% of normal")

    def _create_vulnerability_map(self):
        """Create tissue vulnerability map based on anatomy."""
        self.vulnerability = np.ones(self.mattress.shape)

        # Higher vulnerability over bony prominences (less fat)
        self.vulnerability *= (2 - self.tissue.fat_thickness_map)

        # Normalize
        self.vulnerability = np.clip(self.vulnerability, 0.5, 3.0)

    def run(self, verbose=True):
        """Run the full simulation."""
        n_steps = int(self.simulation_time / self.timestep)

        if verbose:
            print("="*60)
            print("ADVANCED PRESSURE ULCER SIMULATION")
            print("="*60)
            print(f"\nDuration: {self.simulation_time/60:.1f} minutes")
            print(f"Timestep: {self.timestep} seconds")
            print(f"Waist hinge angle: {self.waist_hinge_angle}°")
            print(f"Grid size: {self.mattress.shape}")
            if self.quadriplegic_mode:
                print(f"\n[QUADRIPLEGIC MODE ACTIVE]")
                print(f"  - Damage rate: {QUADRIPLEGIC_FACTORS['damage_rate_multiplier']}x faster")
                print(f"  - Skin fragility: {QUADRIPLEGIC_FACTORS['skin_fragility']}x damage rate")
                print(f"  - Moisture management: {self.moisture_management}")
                if self.lateral_rotation:
                    print(f"  - Lateral rotation: ±{self.lateral_rotation_angle}° every {self.lateral_rotation_period}s")
            print()

        for step in range(n_steps):
            self.current_time = step * self.timestep

            # 1. Run optimizer FIRST so interventions affect this step
            if step > 0:
                actions = self.optimizer.optimize_step(
                    self.current_time,
                    self.pressure_history[-1],
                    self.shear_history[-1],
                    self.thermal.skin_temperature
                )

            # 2. Get body force distribution
            force = self.body.get_force_distribution()

            # 3. Apply load to mattress
            deformation = self.mattress.apply_load(force)

            # 4. Reduce effective force by mattress support (zone/alternating
            #    pressure adjustments feed back into tissue loading here).
            #    mattress.cell_pressure is in Pa; convert to N over cell area.
            mattress_cell_area = self.mattress.cell_size_m ** 2
            support_on_mattress = self.mattress.cell_pressure * mattress_cell_area
            # Resample mattress support to body grid
            body_shape = force.shape
            support = np.zeros(body_shape)
            mr, mc = support_on_mattress.shape
            br, bc = body_shape
            for i in range(br):
                for j in range(bc):
                    mi = min(int(i * mr / br), mr - 1)
                    mj = min(int(j * mc / bc), mc - 1)
                    support[i, j] = support_on_mattress[mi, mj]
            effective_force = np.maximum(force - support, 0.0)

            # 5. Compute contact pressure (Hertzian) using effective force
            contact_pressure = self.contact.compute_superimposed_pressure(
                effective_force)
            pressure_mmhg = contact_pressure * MMHG_PER_PASCAL

            # 5b. Apply lateral rotation pressure redistribution (if enabled)
            if self.lateral_rotation is not None:
                self.lateral_rotation.update(self.current_time)
                pressure_mmhg = self.lateral_rotation.get_pressure_redistribution(pressure_mmhg)

            # 6. Compute stress through tissue layers
            self.tissue.compute_layered_stress(effective_force)

            # 7. Compute shear stress
            shear_stress = self.shear.compute_shear(
                contact_pressure,
                bed_angle=self.waist_hinge_angle,
                moisture=self.thermal.moisture
            )

            # 8. Update thermal and moisture state
            contact_mask = pressure_mmhg > 5
            self.thermal.update(
                pressure_mmhg,
                contact_mask,
                self.mattress.surface_temp.mean(),
                self.timestep
            )

            # 9. Update mattress thermal state
            self.mattress.update_thermal(self.thermal.interface_temp, self.timestep)

            # 10. Simulate mattress airflow
            self.mattress.simulate_airflow(self.timestep)

            # 11. Get damage multipliers
            thermal_multiplier = self.thermal.get_damage_multiplier()
            shear_multiplier = self.shear.get_shear_damage_factor()
            combined_vulnerability = self.vulnerability * thermal_multiplier * shear_multiplier

            # 12. Update risk model
            self.risk.update(
                pressure_mmhg,
                shear_stress,
                self.thermal.skin_temperature,
                self.thermal.moisture,
                combined_vulnerability,
                self.timestep
            )

            # 12. Record data
            self.pressure_history.append(pressure_mmhg.copy())
            self.shear_history.append(shear_stress.copy())
            self.temperature_history.append(self.thermal.skin_temperature.copy())
            self.moisture_history.append(self.thermal.moisture.copy())
            self.risk_history.append(self.risk.damage_index.copy())
            self.time_history.append(self.current_time)

            self.stats_history.append({
                'peak_pressure': pressure_mmhg.max(),
                'mean_pressure': pressure_mmhg.mean(),
                'peak_shear': shear_stress.max() / 1000,  # kPa
                'max_temp': self.thermal.skin_temperature.max(),
                'max_moisture': self.thermal.moisture.max(),
                'max_risk': self.risk.damage_index.max(),
            })

            # Progress
            if verbose and (step + 1) % max(n_steps // 10, 1) == 0:
                pct = (step + 1) / n_steps * 100
                stats = self.stats_history[-1]
                print(f"  {pct:3.0f}% | P: {stats['peak_pressure']:.1f} mmHg | "
                      f"τ: {stats['peak_shear']:.1f} kPa | "
                      f"T: {stats['max_temp']:.1f}°C | "
                      f"Risk: {stats['max_risk']:.3f}")

        if verbose:
            self._print_summary()

        return self._compile_results()

    def _print_summary(self):
        """Print comprehensive summary."""
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)

        final_stats = self.stats_history[-1]

        print("\nPressure Analysis:")
        print(f"  Peak interface pressure: {final_stats['peak_pressure']:.1f} mmHg")
        print(f"  Mean interface pressure: {final_stats['mean_pressure']:.1f} mmHg")
        print(f"  Time-averaged peak: {np.mean([s['peak_pressure'] for s in self.stats_history]):.1f} mmHg")

        print("\nShear Stress Analysis:")
        print(f"  Peak shear stress: {final_stats['peak_shear']:.2f} kPa")
        print(f"  Cells above shear threshold: {(self.shear_history[-1] > SHEAR_DAMAGE_THRESHOLD*1000).sum()}")

        print("\nThermal Analysis:")
        print(f"  Maximum skin temperature: {final_stats['max_temp']:.1f}°C")
        print(f"  Maximum moisture level: {final_stats['max_moisture']:.2f}")

        print("\nRisk Assessment:")
        risk_cats = self.risk.get_risk_categories()
        print(f"  Maximum damage index: {final_stats['max_risk']:.4f}")
        print(f"  Minimal risk cells: {risk_cats['minimal'].sum()}")
        print(f"  Low risk cells: {risk_cats['low'].sum()}")
        print(f"  Moderate risk cells: {risk_cats['moderate'].sum()}")
        print(f"  High risk cells: {risk_cats['high'].sum()}")
        print(f"  Critical risk cells: {risk_cats['critical'].sum()}")

        # Identify primary risk factors
        primary = self.risk.get_primary_risk_factor()
        factor_names = ['Pressure', 'Shear', 'Temperature', 'Moisture']
        for i, name in enumerate(factor_names):
            count = (primary == i).sum()
            print(f"  Primary factor '{name}': {count} cells")

        print("="*60)

    def _compile_results(self):
        """Compile results dictionary."""
        return {
            'pressure_history': np.array(self.pressure_history),
            'shear_history': np.array(self.shear_history),
            'temperature_history': np.array(self.temperature_history),
            'moisture_history': np.array(self.moisture_history),
            'risk_history': np.array(self.risk_history),
            'time_history': np.array(self.time_history),
            'stats_history': self.stats_history,
            'final_pressure': self.pressure_history[-1],
            'final_shear': self.shear_history[-1],
            'final_temperature': self.temperature_history[-1],
            'final_moisture': self.moisture_history[-1],
            'final_risk': self.risk_history[-1],
            'grid_shape': self.mattress.shape,
            'risk_categories': self.risk.get_risk_categories(),
            'primary_risk_factors': self.risk.get_primary_risk_factor(),
            # Body and bed geometry for visualization
            'body_mass_matrix': self.body.mass_matrix.copy(),
            'body_depth_profile': getattr(self.body, 'depth_profile',
                                          np.zeros(self.body.n_rows)),
            'arm_depth_profile': getattr(self.body, 'arm_depth_profile',
                                         np.zeros(self.body.n_rows)),
            'region_map': self.body.region_map.copy(),
            'hinge_row': self.body.hinge_row,
            'waist_hinge_angle': self.waist_hinge_angle,
            'cell_size_cm': self.body.cell_size,
            'smpl_vertices': getattr(self.body, 'smpl_vertices', None),
            'smpl_faces': getattr(self.body, 'smpl_faces', None),
        }


# =============================================================================
# ADVANCED VISUALIZATION
# =============================================================================

class AdvancedVisualizer:
    """
    Comprehensive visualization for advanced simulation.
    """

    def __init__(self, results):
        """Initialize with simulation results."""
        self.results = results

        # Colormaps
        self.pressure_cmap = LinearSegmentedColormap.from_list(
            'pressure', ['#2ecc71', '#f1c40f', '#e74c3c', '#8e44ad'], N=256
        )
        self.shear_cmap = LinearSegmentedColormap.from_list(
            'shear', ['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], N=256
        )
        self.temp_cmap = LinearSegmentedColormap.from_list(
            'temp', ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c'], N=256
        )
        self.moisture_cmap = LinearSegmentedColormap.from_list(
            'moisture', ['#ecf0f1', '#3498db', '#2980b9'], N=256
        )
        self.risk_cmap = LinearSegmentedColormap.from_list(
            'risk', ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'], N=256
        )

    def create_comprehensive_plot(self, save_path=None):
        """Create comprehensive 6-panel visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Pressure map
        im1 = axes[0, 0].imshow(
            self.results['final_pressure'],
            cmap=self.pressure_cmap,
            aspect='auto',
            vmin=0,
            vmax=60
        )
        axes[0, 0].set_title('Interface Pressure (mmHg)', fontweight='bold')
        axes[0, 0].contour(self.results['final_pressure'],
                          levels=[CAPILLARY_CLOSING_PRESSURE],
                          colors='white', linewidths=2, linestyles='--')
        plt.colorbar(im1, ax=axes[0, 0])

        # 2. Shear stress map
        shear_kpa = self.results['final_shear'] / 1000
        im2 = axes[0, 1].imshow(
            shear_kpa,
            cmap=self.shear_cmap,
            aspect='auto',
            vmin=0,
            vmax=15
        )
        axes[0, 1].set_title('Shear Stress (kPa)', fontweight='bold')
        axes[0, 1].contour(shear_kpa, levels=[SHEAR_DAMAGE_THRESHOLD],
                          colors='white', linewidths=2, linestyles='--')
        plt.colorbar(im2, ax=axes[0, 1])

        # 3. Temperature map
        im3 = axes[0, 2].imshow(
            self.results['final_temperature'],
            cmap=self.temp_cmap,
            aspect='auto',
            vmin=30,
            vmax=40
        )
        axes[0, 2].set_title('Skin Temperature (°C)', fontweight='bold')
        plt.colorbar(im3, ax=axes[0, 2])

        # 4. Moisture map
        im4 = axes[1, 0].imshow(
            self.results['final_moisture'],
            cmap=self.moisture_cmap,
            aspect='auto',
            vmin=0,
            vmax=0.5
        )
        axes[1, 0].set_title('Moisture Saturation', fontweight='bold')
        plt.colorbar(im4, ax=axes[1, 0])

        # 5. Risk map
        im5 = axes[1, 1].imshow(
            self.results['final_risk'],
            cmap=self.risk_cmap,
            aspect='auto',
            vmin=0,
            vmax=1
        )
        axes[1, 1].set_title('Cumulative Damage Index', fontweight='bold')
        plt.colorbar(im5, ax=axes[1, 1])

        # 6. Primary risk factor map
        factor_cmap = LinearSegmentedColormap.from_list(
            'factors', ['#e74c3c', '#9b59b6', '#e67e22', '#3498db'], N=4
        )
        im6 = axes[1, 2].imshow(
            self.results['primary_risk_factors'],
            cmap=factor_cmap,
            aspect='auto',
            vmin=0,
            vmax=3
        )
        axes[1, 2].set_title('Primary Risk Factor', fontweight='bold')
        cbar = plt.colorbar(im6, ax=axes[1, 2], ticks=[0.375, 1.125, 1.875, 2.625])
        cbar.ax.set_yticklabels(['Pressure', 'Shear', 'Temp', 'Moisture'])

        # Add body region labels to all plots
        for ax in axes.flat:
            ax.set_xlabel('Width (cells)')
            ax.set_ylabel('Length (Head → Feet)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def create_time_series_plot(self, save_path=None):
        """Create time series of key metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        times_min = self.results['time_history'] / 60
        stats = self.results['stats_history']

        # Pressure over time
        axes[0, 0].plot(times_min, [s['peak_pressure'] for s in stats],
                       'b-', linewidth=2, label='Peak')
        axes[0, 0].plot(times_min, [s['mean_pressure'] for s in stats],
                       'g-', linewidth=2, alpha=0.7, label='Mean')
        axes[0, 0].axhline(CAPILLARY_CLOSING_PRESSURE, color='r',
                          linestyle='--', label='Threshold')
        axes[0, 0].set_xlabel('Time (min)')
        axes[0, 0].set_ylabel('Pressure (mmHg)')
        axes[0, 0].set_title('Interface Pressure Over Time', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Shear over time
        axes[0, 1].plot(times_min, [s['peak_shear'] for s in stats],
                       'purple', linewidth=2)
        axes[0, 1].axhline(SHEAR_DAMAGE_THRESHOLD, color='r',
                          linestyle='--', label='Threshold')
        axes[0, 1].set_xlabel('Time (min)')
        axes[0, 1].set_ylabel('Shear Stress (kPa)')
        axes[0, 1].set_title('Peak Shear Stress Over Time', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Temperature and moisture
        ax2 = axes[1, 0].twinx()
        l1, = axes[1, 0].plot(times_min, [s['max_temp'] for s in stats],
                              'r-', linewidth=2, label='Temperature')
        l2, = ax2.plot(times_min, [s['max_moisture'] for s in stats],
                       'b-', linewidth=2, label='Moisture')
        axes[1, 0].set_xlabel('Time (min)')
        axes[1, 0].set_ylabel('Temperature (°C)', color='r')
        ax2.set_ylabel('Moisture', color='b')
        axes[1, 0].set_title('Thermal & Moisture Over Time', fontweight='bold')
        axes[1, 0].legend(handles=[l1, l2], loc='upper right')
        axes[1, 0].grid(True, alpha=0.3)

        # Risk accumulation
        axes[1, 1].plot(times_min, [s['max_risk'] for s in stats],
                       'darkred', linewidth=2)
        axes[1, 1].fill_between(times_min, 0, [s['max_risk'] for s in stats],
                                alpha=0.3, color='red')
        axes[1, 1].set_xlabel('Time (min)')
        axes[1, 1].set_ylabel('Damage Index')
        axes[1, 1].set_title('Maximum Risk Accumulation', fontweight='bold')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def create_body_silhouette(self, save_path=None):
        """Create an interactive 3D Plotly visualization of the body shape
        with anatomical labels so the user can rotate and verify accuracy."""
        import plotly.graph_objects as go

        body_mass = self.results['body_mass_matrix']
        depth_profile = self.results.get('body_depth_profile', None)
        arm_depth_profile = self.results.get('arm_depth_profile', None)
        region_map = self.results.get('region_map', None)
        cell_cm = self.results.get('cell_size_cm', 5)
        nr, nc = body_mass.shape

        body_length_cm = nr * cell_cm
        body_width_cm = nc * cell_cm
        VISUAL_BODY_SCALE = 1.8

        if depth_profile is None:
            depth_profile = np.ones(nr) * 5.0
        if arm_depth_profile is None:
            arm_depth_profile = np.ones(nr) * 3.0

        dp_torso = np.array(depth_profile) * VISUAL_BODY_SCALE
        dp_arm = np.array(arm_depth_profile) * VISUAL_BODY_SCALE
        mass_norm = body_mass / body_mass.max() if body_mass.max() > 0 else body_mass
        body_mask = mass_norm > 0.001

        # Helper: pick depth for a group based on whether it's arm or torso
        def _group_depth(row, g_left, g_right):
            """Return scaled depth for this group (arm vs torso)."""
            mid_col = (g_left + g_right) // 2
            if region_map is not None and region_map[row, mid_col] == 10:
                return dp_arm[row]
            return dp_torso[row]

        # Build 3D surface: for each body cell, height = elliptical
        # cross-section based on depth profile.  We super-sample columns
        # to get a smooth rounded surface.
        n_theta = 20  # half-circle resolution for cross-section
        theta = np.linspace(0, np.pi, n_theta)  # 0 to pi = top half dome

        # Collect all surface vertices
        xs_all = []  # width (cm)
        ys_all = []  # length (cm)
        zs_all = []  # height (cm)
        colors_all = []  # mass intensity

        for i in range(nr):
            if not np.any(body_mask[i, :]):
                continue

            cols_active = np.where(body_mask[i, :])[0]
            y_cm = (i + 0.5) * cell_cm  # position along body

            # Find left/right extents and whether legs are split
            groups = []
            # Group consecutive active columns
            start = cols_active[0]
            for k in range(1, len(cols_active)):
                if cols_active[k] - cols_active[k - 1] > 1:
                    groups.append((start, cols_active[k - 1]))
                    start = cols_active[k]
            groups.append((start, cols_active[-1]))

            for g_left, g_right in groups:
                cx = ((g_left + g_right) / 2.0 + 0.5) * cell_cm
                hw = ((g_right - g_left) / 2.0 + 0.5) * cell_cm
                h = _group_depth(i, g_left, g_right)
                if hw < 0.5 or h < 0.5:
                    continue

                # Elliptical cross section at this row
                for t in theta:
                    x = cx + hw * np.cos(t)
                    z = h * np.sin(t)
                    xs_all.append(x)
                    ys_all.append(y_cm)
                    zs_all.append(z)
                    # Color intensity from mass at nearest col
                    j_near = int(np.clip(x / cell_cm, 0, nc - 1))
                    colors_all.append(mass_norm[i, j_near])

        # Also build a proper mesh surface for smooth rendering
        # For each row, create the cross-section arc and stack them
        fig = go.Figure()

        # Build mesh row-by-row with consistent theta sampling
        # We'll create a surface for each contiguous body segment
        for i in range(nr):
            if not np.any(body_mask[i, :]):
                continue
            cols_active = np.where(body_mask[i, :])[0]
            y_cm = (i + 0.5) * cell_cm

            # Group consecutive columns (handles split legs)
            groups = []
            start = cols_active[0]
            for k in range(1, len(cols_active)):
                if cols_active[k] - cols_active[k - 1] > 1:
                    groups.append((start, cols_active[k - 1]))
                    start = cols_active[k]
            groups.append((start, cols_active[-1]))

            for g_left, g_right in groups:
                cx = ((g_left + g_right) / 2.0 + 0.5) * cell_cm
                hw = ((g_right - g_left) / 2.0 + 0.5) * cell_cm
                h = _group_depth(i, g_left, g_right)
                if hw < 0.5 or h < 0.5:
                    continue

                arc_x = cx + hw * np.cos(theta)
                arc_z = h * np.sin(theta)

                fig.add_trace(go.Scatter3d(
                    x=arc_x, y=np.full(n_theta, y_cm), z=arc_z,
                    mode='lines',
                    line=dict(color='#d4956b', width=3),
                    showlegend=False, hoverinfo='skip'
                ))

        # Add filled body surface using Mesh3d
        # Build triangulated surface: connect arcs of adjacent rows
        all_verts_x = []
        all_verts_y = []
        all_verts_z = []
        all_intensity = []
        tri_i = []
        tri_j = []
        tri_k = []

        # Store arcs per row for meshing
        row_arcs = {}  # row_index -> list of (cx, hw, h, group_id)
        group_counter = 0
        row_group_map = {}  # (row, group_idx) -> global group_id

        for i in range(nr):
            if not np.any(body_mask[i, :]):
                continue
            cols_active = np.where(body_mask[i, :])[0]
            groups = []
            start = cols_active[0]
            for k_idx in range(1, len(cols_active)):
                if cols_active[k_idx] - cols_active[k_idx - 1] > 1:
                    groups.append((start, cols_active[k_idx - 1]))
                    start = cols_active[k_idx]
            groups.append((start, cols_active[-1]))

            row_arcs[i] = []
            for gi, (g_left, g_right) in enumerate(groups):
                cx = ((g_left + g_right) / 2.0 + 0.5) * cell_cm
                hw = ((g_right - g_left) / 2.0 + 0.5) * cell_cm
                h = _group_depth(i, g_left, g_right)
                if hw < 0.5 or h < 0.5:
                    continue
                row_arcs[i].append((cx, hw, h))
                row_group_map[(i, len(row_arcs[i]) - 1)] = group_counter
                group_counter += 1

        # Now build triangulated mesh between consecutive rows
        sorted_rows = sorted(row_arcs.keys())
        vert_offset = 0

        for idx in range(len(sorted_rows) - 1):
            r1 = sorted_rows[idx]
            r2 = sorted_rows[idx + 1]
            y1 = (r1 + 0.5) * cell_cm
            y2 = (r2 + 0.5) * cell_cm

            arcs1 = row_arcs[r1]
            arcs2 = row_arcs[r2]

            # Match groups: pair arcs by proximity of center x
            for a1 in arcs1:
                cx1, hw1, h1 = a1
                # Find closest arc in next row
                best = None
                best_dist = 1e9
                for a2 in arcs2:
                    d = abs(a2[0] - cx1)
                    if d < best_dist:
                        best_dist = d
                        best = a2
                if best is None or best_dist > 20:
                    continue
                cx2, hw2, h2 = best

                # Build arc vertices for both rows
                arc1_x = cx1 + hw1 * np.cos(theta)
                arc1_z = h1 * np.sin(theta)
                arc2_x = cx2 + hw2 * np.cos(theta)
                arc2_z = h2 * np.sin(theta)

                # Mass intensity at row midpoints
                j1_center = int(np.clip(cx1 / cell_cm, 0, nc - 1))
                j2_center = int(np.clip(cx2 / cell_cm, 0, nc - 1))
                int1 = mass_norm[r1, j1_center]
                int2 = mass_norm[r2, j2_center]

                base = len(all_verts_x)
                for ti in range(n_theta):
                    all_verts_x.append(arc1_x[ti])
                    all_verts_y.append(y1)
                    all_verts_z.append(arc1_z[ti])
                    all_intensity.append(int1)
                for ti in range(n_theta):
                    all_verts_x.append(arc2_x[ti])
                    all_verts_y.append(y2)
                    all_verts_z.append(arc2_z[ti])
                    all_intensity.append(int2)

                # Triangulate strip between the two arcs
                for ti in range(n_theta - 1):
                    v0 = base + ti
                    v1 = base + ti + 1
                    v2 = base + n_theta + ti
                    v3 = base + n_theta + ti + 1
                    tri_i.extend([v0, v0])
                    tri_j.extend([v1, v2])
                    tri_k.extend([v2, v3])

        # Add mesh surface
        fig.add_trace(go.Mesh3d(
            x=all_verts_x, y=all_verts_y, z=all_verts_z,
            i=tri_i, j=tri_j, k=tri_k,
            intensity=all_intensity,
            colorscale=[
                [0.0, '#f5dcc8'],   # light skin
                [0.3, '#e8beac'],   # medium skin
                [0.6, '#d4956b'],   # tan
                [1.0, '#a0522d'],   # darker where denser
            ],
            opacity=0.95,
            flatshading=False,
            showscale=False,
            lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3,
                          roughness=0.5, fresnel=0.2),
            lightposition=dict(x=100, y=0, z=200),
            hovertemplate='Width: %{x:.0f}cm<br>Length: %{y:.0f}cm<br>'
                          'Height: %{z:.1f}cm<extra></extra>',
        ))

        # Add anatomical labels as 3D text annotations
        labels = [
            (0.03, 'Head'),
            (0.12, 'Neck'),
            (0.19, 'Shoulders'),
            (0.27, 'Chest'),
            (0.37, 'Abdomen'),
            (0.42, 'Waist'),
            (0.49, 'Hips'),
            (0.53, 'Buttocks'),
            (0.62, 'Thighs'),
            (0.72, 'Knees'),
            (0.82, 'Calves'),
            (0.94, 'Feet'),
        ]

        for frac, label in labels:
            row_idx = min(int(frac * nr), nr - 1)
            y_cm = (row_idx + 0.5) * cell_cm
            h = dp_torso[row_idx]
            cols_active = np.where(body_mask[row_idx, :])[0]
            if len(cols_active) > 0:
                right_x = (cols_active[-1] + 1.5) * cell_cm + 3
            else:
                right_x = body_width_cm * 0.8
            fig.add_trace(go.Scatter3d(
                x=[right_x], y=[y_cm], z=[h + 2],
                mode='text',
                text=[label],
                textfont=dict(size=11, color='white', family='Arial Black'),
                showlegend=False, hoverinfo='skip'
            ))

        # Layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Width (cm)', range=[-5, body_width_cm + 15],
                           backgroundcolor='#1a1a2e', gridcolor='#333',
                           color='white'),
                yaxis=dict(title='Body Length (cm)', range=[-5, body_length_cm + 5],
                           backgroundcolor='#1a1a2e', gridcolor='#333',
                           color='white'),
                zaxis=dict(title='Height (cm)', range=[-2, 30],
                           backgroundcolor='#1a1a2e', gridcolor='#333',
                           color='white'),
                bgcolor='#1a1a2e',
                camera=dict(
                    eye=dict(x=1.5, y=-0.8, z=0.8),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='data',
            ),
            paper_bgcolor='#1a1a2e',
            title=dict(text='3D Body Shape — Rotate to Inspect',
                       font=dict(color='white', size=18)),
            margin=dict(l=0, r=0, t=50, b=0),
            width=1100, height=900,
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Saved: {save_path}")

        return fig

    def create_body_bed_view(self, save_path=None):
        """Create side-view and 3D visualization of the body on the hinged bed."""
        fig = plt.figure(figsize=(18, 8))

        n_rows, n_cols = self.results['grid_shape']  # mattress grid
        angle_deg = self.results['waist_hinge_angle']
        angle_rad = np.radians(angle_deg)
        body_mass_raw = self.results['body_mass_matrix']
        pressure = self.results['final_pressure']
        cell_size = self.results['cell_size_cm']

        # Resample body mass to mattress grid (they may differ in size)
        body_mass = np.zeros((n_rows, n_cols))
        bm_rows, bm_cols = body_mass_raw.shape
        for i in range(n_rows):
            for j in range(n_cols):
                bi = int(i * bm_rows / n_rows)
                bj = int(j * bm_cols / n_cols)
                bi = min(bi, bm_rows - 1)
                bj = min(bj, bm_cols - 1)
                body_mass[i, j] = body_mass_raw[bi, bj]

        # Hinge row in mattress grid coordinates
        hinge_row = int(0.45 * n_rows)

        # ---------------------------------------------------------------
        # Compute bed surface coordinates for each row (side view frame)
        # Head = row 0 (raised), feet = row n_rows-1 (flat).
        # Hinge at (hinge_x_pos, 0); inclined section rises to the left.
        # ---------------------------------------------------------------
        hinge_x_pos = hinge_row * cell_size
        bed_x = np.zeros(n_rows)
        bed_y = np.zeros(n_rows)
        for i in range(n_rows):
            if i <= hinge_row:
                d = (hinge_row - i) * cell_size
                bed_x[i] = hinge_x_pos - d * np.cos(angle_rad)
                bed_y[i] = d * np.sin(angle_rad)
            else:
                bed_x[i] = hinge_x_pos + (i - hinge_row) * cell_size
                bed_y[i] = 0.0

        # Body side profile (use anatomical depth profile if available)
        # Scale up for visual clarity (real depth ~12cm is too thin on 200cm bed)
        VISUAL_BODY_SCALE = 1.8
        depth_profile = self.results.get('body_depth_profile', None)
        if depth_profile is not None and len(depth_profile) > 0:
            body_h = np.interp(
                np.linspace(0, 1, n_rows),
                np.linspace(0, 1, len(depth_profile)),
                depth_profile) * VISUAL_BODY_SCALE
        else:
            body_profile = body_mass.sum(axis=1)
            max_h = 20.0
            body_h = body_profile / (body_profile.max() + 1e-10) * max_h

        # Body top surface (perpendicular to bed surface at each row)
        body_top_x = np.zeros(n_rows)
        body_top_y = np.zeros(n_rows)
        for i in range(n_rows):
            if i <= hinge_row:
                body_top_x[i] = bed_x[i] + body_h[i] * np.sin(angle_rad)
                body_top_y[i] = bed_y[i] + body_h[i] * np.cos(angle_rad)
            else:
                body_top_x[i] = bed_x[i]
                body_top_y[i] = body_h[i]

        # ======================= Side View =============================
        ax1 = fig.add_subplot(121)

        # Mattress block
        bed_thick = 8  # cm visual thickness
        bed_bot_x = np.zeros(n_rows)
        bed_bot_y = np.zeros(n_rows)
        for i in range(n_rows):
            if i <= hinge_row:
                bed_bot_x[i] = bed_x[i] - bed_thick * np.sin(angle_rad)
                bed_bot_y[i] = bed_y[i] - bed_thick * np.cos(angle_rad)
            else:
                bed_bot_x[i] = bed_x[i]
                bed_bot_y[i] = -bed_thick

        mattress_px = np.concatenate([bed_x, bed_bot_x[::-1]])
        mattress_py = np.concatenate([bed_y, bed_bot_y[::-1]])
        ax1.fill(mattress_px, mattress_py, color='#5d6d7e', alpha=0.7,
                 label='Mattress')
        ax1.plot(bed_x, bed_y, 'k-', linewidth=2.5)
        ax1.plot(bed_bot_x, bed_bot_y, color='#4a5568', linewidth=1.5)

        # Body strips coloured by pressure with skin-tone base
        avg_p = pressure.mean(axis=1)
        has_body = body_h > 0.3
        for i in range(n_rows - 1):
            if body_h[i] < 0.3 and body_h[i + 1] < 0.3:
                continue
            sx = [bed_x[i], bed_x[i + 1], body_top_x[i + 1], body_top_x[i]]
            sy = [bed_y[i], bed_y[i + 1], body_top_y[i + 1], body_top_y[i]]
            # Blend skin tone with pressure colour
            p_frac = min(avg_p[i] / 60.0, 1.0)
            if p_frac < 0.05:
                colour = '#e8beac'  # skin tone for low pressure
            else:
                colour = self.pressure_cmap(p_frac)
            ax1.fill(sx, sy, color=colour, edgecolor='none')

        # Bold body outline
        if has_body.any():
            ax1.plot(body_top_x[has_body], body_top_y[has_body],
                     color='#4a2c17', linewidth=2.5)
            # Draw bed-contact line under body too
            ax1.plot(bed_x[has_body], bed_y[has_body],
                     color='#4a2c17', linewidth=1.0, alpha=0.5)

        # Hinge marker and angle arc
        ax1.plot(hinge_x_pos, 0, 'ro', markersize=10, zorder=5,
                 label=f'Hinge ({angle_deg}°)')
        arc_r = 18
        arc_angles = np.linspace(np.pi, np.pi - angle_rad, 30)
        ax1.plot(hinge_x_pos + arc_r * np.cos(arc_angles),
                 arc_r * np.sin(arc_angles), 'r-', linewidth=1.5)
        ax1.text(hinge_x_pos - arc_r - 5, 4, f'{angle_deg}°',
                 color='red', fontsize=11, fontweight='bold')

        # Head / Feet labels
        first_body = np.argmax(has_body) if has_body.any() else 0
        last_body = (n_rows - 1 - np.argmax(has_body[::-1])
                     if has_body.any() else n_rows - 1)
        ax1.annotate('Head',
                     xy=(bed_x[first_body], body_top_y[first_body] + 3),
                     fontsize=11, ha='center', fontweight='bold')
        ax1.annotate('Feet',
                     xy=(bed_x[last_body], body_top_y[last_body] + 3),
                     fontsize=11, ha='center', fontweight='bold')

        ax1.set_xlabel('Position (cm)', fontsize=11)
        ax1.set_ylabel('Height (cm)', fontsize=11)
        ax1.set_title('Side View: Patient on Hinged Bed',
                       fontweight='bold', fontsize=13)
        ax1.legend(loc='upper right')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.2)

        sm = plt.cm.ScalarMappable(cmap=self.pressure_cmap,
                                   norm=plt.Normalize(0, 60))
        sm.set_array([])
        plt.colorbar(sm, ax=ax1, shrink=0.6, label='Avg Pressure (mmHg)')

        # ======================= 3D View ===============================
        ax3d = fig.add_subplot(122, projection='3d')

        col_cm = np.arange(n_cols) * cell_size
        C, _ = np.meshgrid(col_cm, np.arange(n_rows))

        X3 = C.astype(float)
        Y3 = np.zeros_like(X3)
        Z3 = np.zeros_like(X3)
        for i in range(n_rows):
            Y3[i, :] = bed_x[i]
            Z3[i, :] = bed_y[i]

        # Body height per cell (anatomical shape)
        no_body = body_mass.max(axis=1) < 0.01
        body_h_2d = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            row_max = body_mass[i, :].max()
            if row_max > 0.01:
                body_h_2d[i, :] = body_h[i] * (body_mass[i, :] / row_max)

        # Skin-tone base; overlay pressure as tint
        from matplotlib.colors import to_rgba
        skin = np.array(to_rgba('#e8beac'))
        p_norm = np.clip(pressure / 60.0, 0, 1)
        fcolors = np.zeros((*pressure.shape, 4))
        for i in range(n_rows):
            for j in range(n_cols):
                if body_mass[i, j] > 0.01:
                    pf = p_norm[i, j]
                    overlay = np.array(self.pressure_cmap(pf))
                    # Blend: low pressure → skin tone, high → pressure colour
                    fcolors[i, j] = skin * (1 - pf) + overlay * pf
                    fcolors[i, j, 3] = 1.0
                else:
                    fcolors[i, j] = [0, 0, 0, 0]

        # Bed surface (darker)
        ax3d.plot_surface(X3, Y3, Z3, color='#5d6d7e', alpha=0.4,
                          edgecolor='#7f8c8d', linewidth=0.3)

        # Body surface raised by anatomical depth
        Z3_body = Z3.copy()
        for i in range(n_rows):
            cos_a = np.cos(angle_rad) if i <= hinge_row else 1.0
            Z3_body[i, :] += body_h_2d[i, :] * cos_a

        ax3d.plot_surface(X3, Y3, Z3_body, facecolors=fcolors,
                          alpha=1.0, shade=True)

        ax3d.set_xlabel('Width (cm)')
        ax3d.set_ylabel('Length (cm)')
        ax3d.set_zlabel('Height (cm)')
        ax3d.set_title('3D View: Pressure on Hinged Bed',
                        fontweight='bold', fontsize=13)
        ax3d.view_init(elev=25, azim=-55)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def create_interactive_3d(self, save_path=None):
        """
        Create interactive 3D visualization using Plotly with three panels:
        1) Anatomical body shape  2) Pressure distribution  3) Damage risk
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        n_rows, n_cols = self.results['grid_shape']
        angle_deg = self.results['waist_hinge_angle']
        angle_rad = np.radians(angle_deg)
        body_mass_raw = self.results['body_mass_matrix']
        pressure = self.results['final_pressure']
        risk = self.results['final_risk']
        cell_size = self.results['cell_size_cm']

        bm_rows, bm_cols = body_mass_raw.shape
        body_mass = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                bi = min(int(i * bm_rows / n_rows), bm_rows - 1)
                bj = min(int(j * bm_cols / n_cols), bm_cols - 1)
                body_mass[i, j] = body_mass_raw[bi, bj]

        hinge_row = int(0.45 * n_rows)
        col_cm = np.arange(n_cols) * cell_size

        bed_y = np.zeros(n_rows)
        bed_z = np.zeros(n_rows)
        for i in range(n_rows):
            if i <= hinge_row:
                d = (hinge_row - i) * cell_size
                bed_y[i] = hinge_row * cell_size - d * np.cos(angle_rad)
                bed_z[i] = d * np.sin(angle_rad)
            else:
                bed_y[i] = hinge_row * cell_size + (i - hinge_row) * cell_size

        # Shift so head (row 0) is at Y=0, feet go positive
        bed_y_offset = bed_y[0]
        bed_y -= bed_y_offset

        X_bed = np.tile(col_cm, (n_rows, 1))
        Y_bed = np.tile(bed_y, (n_cols, 1)).T
        Z_bed = np.tile(bed_z, (n_cols, 1)).T

        # ---- Colorscales ----
        skin_pressure_scale = [
            [0.0, '#e8beac'], [0.15, '#f5d5c8'], [0.3, '#f7dc6f'],
            [0.5, '#f0932b'], [0.75, '#e74c3c'], [1.0, '#8e44ad'],
        ]
        skin_risk_scale = [
            [0.0, '#e8beac'], [0.2, '#abebc6'], [0.4, '#f7dc6f'],
            [0.6, '#e67e22'], [0.8, '#e74c3c'], [1.0, '#8e44ad'],
        ]

        # ---- SMPL mesh: transform vertices onto hinged bed ----
        smpl_verts = self.results.get('smpl_vertices', None)
        smpl_faces = self.results.get('smpl_faces', None)

        if smpl_verts is not None and smpl_faces is not None:
            sv = smpl_verts.copy()
            # sv[:,0]=X(width), sv[:,1]=Y(depth/height off bed),
            # sv[:,2]=Z(body length, 0=head, max=feet)
            hinge_z_cm = hinge_row * cell_size  # ~80cm from head

            # Centre body X on the mattress grid
            body_center_x = (sv[:, 0].max() + sv[:, 0].min()) / 2.0
            sv[:, 0] += 40.0 - body_center_x  # centre body midline at x=40

            vx = sv[:, 0].copy()
            vy = np.zeros(len(sv))
            vz = np.zeros(len(sv))

            body_len = sv[:, 2].max()

            # ---- Compute per-slice minimum depth (contact surface) ----
            # Bin vertices by position along body, find the lowest Y
            # (closest to bed) per bin so each cross-section sits on bed.
            # sv[:,2]: Z=0 is FEET, Z=body_len is HEAD (after flip).
            # Bed: row 0 is head (raised), row n_rows-1 is feet (flat).
            # Convert: bed_row_pos = body_len - sv[:,2]  (0=head, body_len=feet)
            n_bins = n_rows
            min_depth_per_bin = np.full(n_bins, np.inf)
            for vi in range(len(sv)):
                bed_pos = body_len - sv[vi, 2]  # 0=head, body_len=feet
                b = int(np.clip(bed_pos / cell_size, 0, n_bins - 1))
                if sv[vi, 1] < min_depth_per_bin[b]:
                    min_depth_per_bin[b] = sv[vi, 1]

            # Fill any empty bins
            for b in range(n_bins):
                if np.isinf(min_depth_per_bin[b]):
                    min_depth_per_bin[b] = 0.0

            for vi in range(len(sv)):
                bed_pos = body_len - sv[vi, 2]  # 0=head, body_len=feet

                # Depth above the contact surface at this slice
                b = int(np.clip(bed_pos / cell_size, 0, n_bins - 1))
                contact_depth = sv[vi, 1] - min_depth_per_bin[b]

                # Map to bed row (fractional) and interpolate
                row_frac = bed_pos / cell_size
                row_lo = int(np.clip(row_frac, 0, n_rows - 2))
                row_hi = min(row_lo + 1, n_rows - 1)
                frac = np.clip(row_frac - row_lo, 0, 1)

                # Interpolate bed surface Y and Z at this body position
                surf_y = bed_y[row_lo] * (1 - frac) + bed_y[row_hi] * frac
                surf_z = bed_z[row_lo] * (1 - frac) + bed_z[row_hi] * frac

                # Compute the bed surface normal direction (for depth offset)
                dy = bed_y[row_hi] - bed_y[row_lo]
                dz = bed_z[row_hi] - bed_z[row_lo]
                seg_len = np.sqrt(dy**2 + dz**2)
                if seg_len > 0:
                    ny = -dz / seg_len
                    nz = dy / seg_len
                else:
                    ny, nz = 0, 1

                # Place vertex on bed surface + contact_depth along normal
                vy[vi] = surf_y + contact_depth * ny
                vz[vi] = surf_z + contact_depth * nz

            # surf_y already uses offset-corrected bed_y, no extra shift needed

            # Helper: map a 2D grid onto per-vertex values
            def _grid_to_verts(grid_data):
                vals = np.zeros(len(sv))
                for vi in range(len(sv)):
                    row = int(np.clip((body_len - sv[vi, 2]) / cell_size, 0, n_rows - 1))
                    col = int(np.clip(vx[vi] / cell_size, 0, n_cols - 1))
                    vals[vi] = grid_data[row, col]
                return vals

            vert_pressure = _grid_to_verts(pressure)
            vert_risk = _grid_to_verts(risk)

            fi = smpl_faces[:, 0]
            fj = smpl_faces[:, 1]
            fk = smpl_faces[:, 2]
            has_mesh = True
        else:
            has_mesh = False

        # ---- 2-panel figure ----
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=['Pressure Distribution', 'Damage Risk Index'],
            horizontal_spacing=0.05,
        )

        # --- Panel 1: Pressure ---
        fig.add_trace(go.Surface(
            x=X_bed, y=Y_bed, z=Z_bed,
            surfacecolor=np.zeros_like(Z_bed),
            colorscale=[[0, '#5d6d7e'], [1, '#5d6d7e']],
            showscale=False, opacity=0.6,
        ), row=1, col=1)

        if has_mesh:
            fig.add_trace(go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=fi, j=fj, k=fk,
                intensity=vert_pressure,
                colorscale=skin_pressure_scale,
                cmin=0, cmax=60,
                colorbar=dict(title='mmHg', x=0.44, len=0.8),
                opacity=1.0,
                flatshading=False,
                lighting=dict(ambient=0.5, diffuse=0.6, specular=0.2,
                              roughness=0.5, fresnel=0.1),
                lightposition=dict(x=100, y=-50, z=200),
            ), row=1, col=1)

        # --- Panel 2: Risk ---
        fig.add_trace(go.Surface(
            x=X_bed, y=Y_bed, z=Z_bed,
            surfacecolor=np.zeros_like(Z_bed),
            colorscale=[[0, '#5d6d7e'], [1, '#5d6d7e']],
            showscale=False, opacity=0.6,
        ), row=1, col=2)

        if has_mesh:
            fig.add_trace(go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=fi, j=fj, k=fk,
                intensity=vert_risk,
                colorscale=skin_risk_scale,
                cmin=0, cmax=1,
                colorbar=dict(title='Damage', x=1.0, len=0.8),
                opacity=1.0,
                flatshading=False,
                lighting=dict(ambient=0.5, diffuse=0.6, specular=0.2,
                              roughness=0.5, fresnel=0.1),
                lightposition=dict(x=100, y=-50, z=200),
            ), row=1, col=2)

        # Hinge lines
        hinge_y_pos = bed_y[hinge_row]
        hinge_z_pos = bed_z[hinge_row]
        for col_idx in [1, 2]:
            fig.add_trace(go.Scatter3d(
                x=[col_cm[0] - 2, col_cm[-1] + 2],
                y=[hinge_y_pos, hinge_y_pos],
                z=[hinge_z_pos + 0.5, hinge_z_pos + 0.5],
                mode='lines+text',
                line=dict(color='red', width=6),
                text=[f'Hinge ({angle_deg}°)', ''],
                textposition='top center',
                textfont=dict(size=12, color='red'),
                showlegend=False,
            ), row=1, col=col_idx)

        scene_cfg = dict(
            xaxis_title='Width (cm)',
            yaxis_title='Length (cm)',
            zaxis_title='Height (cm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.8, z=0.8)),
        )
        # ---- Time slider via restyle (reliable for 3D subplots) ----
        pressure_hist = self.results.get('pressure_history', None)
        risk_hist = self.results.get('risk_history', None)
        time_hist = self.results.get('time_history', None)

        slider_steps = []
        if (has_mesh and pressure_hist is not None
                and risk_hist is not None and len(pressure_hist) > 1):
            n_total = len(pressure_hist)
            n_frames = min(30, n_total)
            frame_indices = np.linspace(0, n_total - 1, n_frames, dtype=int)

            for idx in frame_indices:
                t_sec = time_hist[idx]
                mins = int(t_sec // 60)
                secs = int(t_sec % 60)
                label = f'{mins}:{secs:02d}'

                vp = _grid_to_verts(pressure_hist[idx]).tolist()
                vr = _grid_to_verts(risk_hist[idx]).tolist()

                # restyle traces 1 (pressure body) and 3 (risk body)
                slider_steps.append(dict(
                    args=[
                        {'intensity': [vp, vr]},  # new data
                        [1, 3],                    # trace indices
                    ],
                    label=label,
                    method='restyle',
                ))

        fig.update_layout(
            scene=scene_cfg, scene2=scene_cfg,
            title=dict(text=f'Interactive 3D — Waist Hinge {angle_deg}°',
                       x=0.5, font=dict(size=18)),
            height=750, width=1500,
            margin=dict(l=10, r=10, t=60, b=100),
        )

        if slider_steps:
            fig.update_layout(
                sliders=[dict(
                    active=0,
                    currentvalue=dict(
                        prefix='Time: ',
                        font=dict(size=16),
                        visible=True,
                    ),
                    pad=dict(t=50, b=10),
                    len=0.9,
                    x=0.05,
                    xanchor='left',
                    steps=slider_steps,
                )],
            )

        if save_path:
            fig.write_html(save_path, include_plotlyjs=True)
            print(f"Saved: {save_path}")

        return fig

    def create_animation(self, interval=100, save_path=None):
        """Create animated visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        pressure_data = self.results['pressure_history']
        shear_data = self.results['shear_history'] / 1000  # kPa
        risk_data = self.results['risk_history']

        # Initialize images
        im1 = axes[0].imshow(pressure_data[0], cmap=self.pressure_cmap,
                             aspect='auto', vmin=0, vmax=60)
        axes[0].set_title('Pressure (mmHg)')
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(shear_data[0], cmap=self.shear_cmap,
                             aspect='auto', vmin=0, vmax=15)
        axes[1].set_title('Shear (kPa)')
        plt.colorbar(im2, ax=axes[1])

        im3 = axes[2].imshow(risk_data[0], cmap=self.risk_cmap,
                             aspect='auto', vmin=0, vmax=1)
        axes[2].set_title('Risk')
        plt.colorbar(im3, ax=axes[2])

        time_text = fig.suptitle('Time: 0:00', fontsize=12, fontweight='bold')

        def update(frame):
            im1.set_array(pressure_data[frame])
            im2.set_array(shear_data[frame])
            im3.set_array(risk_data[frame])

            t = self.results['time_history'][frame]
            mins = int(t // 60)
            secs = int(t % 60)
            time_text.set_text(f'Time: {mins}:{secs:02d}')

            return im1, im2, im3, time_text

        anim = FuncAnimation(fig, update, frames=len(pressure_data),
                            interval=interval, blit=False)

        plt.tight_layout()

        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=10)
            print("Animation saved!")

        return anim


# =============================================================================
# REPOSITIONING STRATEGY COMPARISON
# =============================================================================

STRATEGIES = {
    'No Intervention': dict(
        enable_zone_control=False, enable_alternating=False,
        enable_repositioning=False),
    'Zone Control Only': dict(
        enable_zone_control=True, enable_alternating=False,
        enable_repositioning=False),
    'Alternating Pressure Only': dict(
        enable_zone_control=False, enable_alternating=True,
        enable_repositioning=False, pressure_cycle_period=300),
    'Repositioning q30 min': dict(
        enable_zone_control=False, enable_alternating=False,
        enable_repositioning=True, reposition_interval=1800),
    'Repositioning q15 min': dict(
        enable_zone_control=False, enable_alternating=False,
        enable_repositioning=True, reposition_interval=900),
    'Alternating + Reposition q30': dict(
        enable_zone_control=False, enable_alternating=True,
        enable_repositioning=True, reposition_interval=1800,
        pressure_cycle_period=300),
    'Full (Zone+Alt+Repos q15)': dict(
        enable_zone_control=True, enable_alternating=True,
        enable_repositioning=True, reposition_interval=900,
        pressure_cycle_period=300),
}


def run_strategy_comparison(body_mass=75, cell_size=5, simulation_time=3600,
                            timestep=15, waist_hinge_angle=25):
    """
    Run the simulation under each repositioning strategy and collect results.
    """
    all_results = {}
    base_cfg = dict(body_mass=body_mass, cell_size=cell_size,
                    simulation_time=simulation_time, timestep=timestep,
                    waist_hinge_angle=waist_hinge_angle)

    for name, flags in STRATEGIES.items():
        print(f"\n  Running: {name} ...")
        cfg = {**base_cfg, **flags}
        sim = AdvancedPressureUlcerSimulation(**cfg)
        results = sim.run(verbose=False)

        stats = results['stats_history']
        risk_cats = results['risk_categories']
        all_results[name] = {
            'peak_pressure': results['final_pressure'].max(),
            'mean_pressure': results['final_pressure'][
                results['final_pressure'] > 0].mean(),
            'peak_shear': results['final_shear'].max() / 1000,  # kPa
            'max_temp': results['final_temperature'].max(),
            'max_damage': results['final_risk'].max(),
            'mean_damage': results['final_risk'][
                results['final_risk'] > 0].mean(),
            'critical_cells': int(risk_cats['critical'].sum()),
            'high_cells': int(risk_cats['high'].sum()),
            'moderate_cells': int(risk_cats['moderate'].sum()),
            'pressure_history': [s['peak_pressure'] for s in stats],
            'risk_history': [s['max_risk'] for s in stats],
            'time_min': np.array(results['time_history']) / 60,
        }

    return all_results


def create_strategy_comparison_html(all_results, save_path='strategy_comparison.html'):
    """Create interactive Plotly comparison of repositioning strategies."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = list(all_results.keys())
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71',
              '#9b59b6', '#1abc9c']

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Peak Pressure Over Time',
            'Max Damage Index Over Time',
            'Final Risk Metrics Comparison',
            'Final Pressure & Shear Comparison',
        ],
        vertical_spacing=0.12, horizontal_spacing=0.10,
    )

    # --- Row 1, Col 1: Peak pressure over time ---
    for i, name in enumerate(names):
        r = all_results[name]
        fig.add_trace(go.Scatter(
            x=r['time_min'], y=r['pressure_history'],
            mode='lines', name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            legendgroup=name, showlegend=True,
        ), row=1, col=1)
    fig.add_hline(y=32, line_dash='dot', line_color='red',
                  annotation_text='Capillary closing (32 mmHg)',
                  row=1, col=1)

    # --- Row 1, Col 2: Risk over time ---
    for i, name in enumerate(names):
        r = all_results[name]
        fig.add_trace(go.Scatter(
            x=r['time_min'], y=r['risk_history'],
            mode='lines', name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            legendgroup=name, showlegend=False,
        ), row=1, col=2)

    # --- Row 2, Col 1: Bar chart — critical / high / moderate cells ---
    for i, name in enumerate(names):
        r = all_results[name]
        fig.add_trace(go.Bar(
            x=['Critical', 'High', 'Moderate'],
            y=[r['critical_cells'], r['high_cells'], r['moderate_cells']],
            name=name,
            marker_color=colors[i % len(colors)],
            legendgroup=name, showlegend=False,
        ), row=2, col=1)

    # --- Row 2, Col 2: Peak pressure & shear grouped bars ---
    for i, name in enumerate(names):
        r = all_results[name]
        fig.add_trace(go.Bar(
            x=['Peak Pressure (mmHg)', 'Mean Damage Index'],
            y=[r['peak_pressure'], r['mean_damage'] * 100],
            name=name,
            marker_color=colors[i % len(colors)],
            legendgroup=name, showlegend=False,
        ), row=2, col=2)

    fig.update_layout(
        title=dict(text='Repositioning Strategy Comparison — '
                        'Pressure Ulcer Prevention',
                   x=0.5, font=dict(size=18)),
        height=850, width=1400,
        barmode='group',
        legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center'),
        margin=dict(l=60, r=30, t=80, b=120),
    )
    fig.update_yaxes(title_text='Peak Pressure (mmHg)', row=1, col=1)
    fig.update_yaxes(title_text='Max Damage Index', row=1, col=2)
    fig.update_yaxes(title_text='Number of Cells', row=2, col=1)
    fig.update_xaxes(title_text='Time (min)', row=1, col=1)
    fig.update_xaxes(title_text='Time (min)', row=1, col=2)

    fig.write_html(save_path, include_plotlyjs=True)
    print(f"Saved: {save_path}")
    return fig


def print_strategy_analysis(all_results):
    """Print a ranked analysis of strategies."""
    print("\n" + "=" * 70)
    print("REPOSITIONING STRATEGY ANALYSIS")
    print("=" * 70)

    # Rank by mean damage (lower is better)
    ranked = sorted(all_results.items(), key=lambda x: x[1]['mean_damage'])

    print(f"\n{'Strategy':<35} {'Peak P':>8} {'Mean Dmg':>9} "
          f"{'Critical':>9} {'High':>6} {'Mod':>5}")
    print("-" * 75)
    for name, r in ranked:
        print(f"{name:<35} {r['peak_pressure']:>7.1f}  "
              f"{r['mean_damage']:>8.4f}  "
              f"{r['critical_cells']:>8}  {r['high_cells']:>5}  "
              f"{r['moderate_cells']:>4}")

    best_name, best = ranked[0]
    worst_name, worst = ranked[-1]

    print(f"\n{'='*70}")
    print(f"OPTIMAL STRATEGY:  {best_name}")
    print(f"{'='*70}")
    print(f"  Peak pressure:   {best['peak_pressure']:.1f} mmHg")
    print(f"  Mean damage:     {best['mean_damage']:.4f}")
    print(f"  Critical cells:  {best['critical_cells']}")
    print(f"  High-risk cells: {best['high_cells']}")

    reduction = ((worst['mean_damage'] - best['mean_damage'])
                 / (worst['mean_damage'] + 1e-10)) * 100
    print(f"\n  Damage reduction vs worst ({worst_name}): {reduction:.1f}%")
    print(f"  Critical-cell reduction: "
          f"{worst['critical_cells'] - best['critical_cells']} fewer cells")
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function for advanced simulation."""

    print("="*70)
    print("ADVANCED PRESSURE ULCER PREVENTION SIMULATION")
    print("With 3D Tissue, Thermal, Shear, Waist Hinge, and Multi-Factor Risk Modeling")
    print("="*70)

    # Configuration
    config = {
        'body_mass': 75,
        'cell_size': 5,
        'simulation_time': 3600,  # 1 hour
        'timestep': 15,
        'waist_hinge_angle': 25,  # 25° hinge at waist (patient sitting up)
        # Quadriplegic patient settings
        'quadriplegic_mode': True,
        'enable_lateral_rotation': True,
        'lateral_rotation_angle': 30,
        'lateral_rotation_period': 300,  # 5 min cycle
        'moisture_management': 'comprehensive',  # Catheter + barriers + pads
        'gender': 'neutral',  # 'male', 'female', or 'neutral'
    }

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # Run simulation
    sim = AdvancedPressureUlcerSimulation(**config)
    results = sim.run(verbose=True)

    # ------------------------------------------------------------------
    # REPOSITIONING STRATEGY COMPARISON
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARING REPOSITIONING STRATEGIES")
    print("=" * 60)

    strategy_results = run_strategy_comparison(
        body_mass=config['body_mass'],
        cell_size=config['cell_size'],
        simulation_time=config['simulation_time'],
        timestep=config['timestep'],
        waist_hinge_angle=config['waist_hinge_angle'],
    )

    print_strategy_analysis(strategy_results)

    # ------------------------------------------------------------------
    # VISUALIZATIONS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    viz = AdvancedVisualizer(results)

    print("\n1. Creating interactive 3D view...")
    viz.create_interactive_3d('body_bed_3d_v2.html')

    print("2. Creating strategy comparison (interactive)...")
    create_strategy_comparison_html(strategy_results,
                                    'strategy_comparison.html')

    print("3. Creating body & bed view...")
    viz.create_body_bed_view('body_bed_view.png')

    print("4. Creating comprehensive state plot...")
    viz.create_comprehensive_plot('advanced_simulation_results.png')

    print("5. Creating time series plot...")
    viz.create_time_series_plot('advanced_time_series.png')

    print("6. Creating animation...")
    try:
        viz.create_animation(interval=100, save_path='advanced_animation.gif')
    except Exception as e:
        print(f"   Animation skipped: {e}")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    print("  - body_bed_3d.html            (interactive 3D)")
    print("  - strategy_comparison.html    (interactive strategy comparison)")
    print("  - body_bed_view.png")
    print("  - advanced_simulation_results.png")
    print("  - advanced_time_series.png")
    print("  - advanced_animation.gif")

    return results, sim


if __name__ == "__main__":
    results, simulation = main()
