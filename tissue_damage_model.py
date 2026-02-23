#!/usr/bin/env python3
"""
Comprehensive Tissue Damage Model for Pressure Ulcer Simulation
================================================================

This module implements an evidence-based tissue damage model incorporating
8 key factors from peer-reviewed literature:

1. Deep Tissue Injury (DTI) - Pressure at bone-muscle interface
2. Reactive Hyperemia - Blood reperfusion and tissue recovery
3. Temperature Effects - Microclimate impact on tissue vulnerability
4. Moisture Effects - Skin maceration from sweat/incontinence
5. Patient Micro-movements - Small movements even in paralyzed patients
6. Body Position - Supine, lateral, prone with different pressure points
7. Tissue Stiffness Variation - Age, edema, cachexia effects
8. Time-Dependent Tolerance - Cumulative damage from repeated insults

References:
-----------
Deep Tissue Injury:
    - Gefen A. (2009). The importance of internal strain as opposed to
      interface pressure in the prevention of pressure ulcers.
      J Tissue Viability, 18(2), 31-35.
    - Loerakker S, et al. (2010). Temporal effects of mechanical loading
      on deformation-induced damage in skeletal muscle tissue.
      Ann Biomed Eng, 38(8), 2577-2587.

Reactive Hyperemia:
    - Tzen YT, et al. (2017). Trends of reactive hyperaemia responses to
      repetitive loading on skin tissue of rats.
      J Tissue Viability, 26(3), 180-188.
    - Soetens JT, et al. (2019). Reactive hyperemia at different depths
      in sacral tissue. Clinical Biomechanics, 67, 1-8.

Temperature/Microclimate:
    - Kottner J, et al. (2018). Microclimate: A critical review in the
      context of pressure ulcer prevention.
      Clinical Biomechanics, 59, 62-70.
    - Lachenbruch C. (2005). Skin cooling surfaces: estimating the
      importance of limiting skin temperature.
      Ostomy Wound Manage, 51(2), 70-79.

Moisture:
    - Beeckman D, et al. (2014). Incontinence-associated dermatitis: a
      review of the evidence. J Wound Ostomy Continence Nurs, 41(3), 312.
    - Gefen A. (2014). Modeling the effects of moisture-related skin-
      support friction on the risk for superficial pressure ulcers.
      Int Wound J, 11(6), 653-663.

Micro-movements:
    - Stinson MD, et al. (2013). Spinal cord injury and pressure ulcer
      prevention: Using functional activity in pressure relief.
      Nursing Research and Practice, 2013, 860396.
    - Call E, et al. (2022). Continuous pressure monitoring of inpatient
      spinal cord injured patients. Spinal Cord, 60, 920-927.

Body Positions:
    - Moore Z, et al. (2011). The 30 degree tilt position vs the 90
      degree lateral and supine positions. J Clin Nurs, 20(17-18), 2564.
    - Colin D, et al. (1996). Comparison of 90 degree and 30 degree
      laterally inclined positions. Lancet, 347(9013), 1371.

Tissue Stiffness:
    - Gefen A, et al. (2005). In vivo muscle stiffening under bone
      compression promotes deep pressure sores.
      J Biomech Eng, 127(3), 512-524.
    - Oomens CW, et al. (2010). Pressure induced deep tissue injury
      explained. Ann Biomed Eng, 38(3), 719-729.

Cumulative Damage:
    - Gefen A, Weihs D. (2016). Cytoskeleton and plasma-membrane damage
      resulting from exposure to sustained deformations.
      J Biomech, 49(12), 2355-2361.
    - Coleman S, et al. (2014). Patient risk factors for pressure ulcer
      development. Int J Nursing Studies, 51(7), 974-1003.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from enum import Enum


# =============================================================================
# CONSTANTS FROM PEER-REVIEWED LITERATURE
# =============================================================================

# Pressure thresholds (Landis 1930, Kosiak 1959)
CAPILLARY_CLOSING_PRESSURE = 32.0  # mmHg
SHEAR_DAMAGE_THRESHOLD = 10.0  # mmHg equivalent

# Deep Tissue Injury (Gefen 2009, Loerakker 2010)
# Muscle is 3-10x more sensitive to pressure than skin
MUSCLE_SENSITIVITY_FACTOR = 4.0  # Conservative estimate
# Direct deformation damage occurs within minutes at high strain
DTI_STRAIN_THRESHOLD = 0.5  # 50% strain causes rapid damage
DTI_TIME_TO_DAMAGE_MIN = 10  # Minutes for visible DTI at high strain

# Reactive Hyperemia (Tzen 2017, Soetens 2019)
# Recovery rate when pressure relieved (proportion per hour)
HYPEREMIA_RECOVERY_BASE = 0.3  # 30% recovery per hour at baseline
# Recovery diminishes with repeated loading cycles
HYPEREMIA_FATIGUE_FACTOR = 0.05  # 5% less recovery per cycle

# Temperature Effects (Kottner 2018, Lachenbruch 2005)
# Each 1°C rise increases metabolic demand ~10% and reduces tolerance
TEMP_BASELINE = 33.0  # Normal skin surface temperature (°C)
TEMP_DAMAGE_MULTIPLIER = 0.1  # 10% more damage per °C above baseline
TEMP_RISE_RATE = 1.5  # °C per hour under occlusion
TEMP_MAX_RISE = 3.0  # Maximum temperature rise observed

# Moisture Effects (Beeckman 2014, Gefen 2014)
# Wet skin has 4.99x higher risk of pressure ulcers
MOISTURE_RISK_MULTIPLIER = 4.99  # From Beeckman meta-analysis
# Friction coefficient increases 2-4x when wet
FRICTION_WET_MULTIPLIER = 3.0
# Maceration weakens skin after prolonged exposure
MACERATION_TIME_HOURS = 2.0  # Time for significant maceration

# Micro-movements (Stinson 2013, Call 2022)
# Most micro-movements provide <25% pressure relief
MICROMOVEMENT_PRESSURE_RELIEF = 0.25  # 25% max
# Frequency of spontaneous movements (per hour)
MICROMOVEMENT_FREQUENCY_ABLE = 20  # Able-bodied
MICROMOVEMENT_FREQUENCY_SCI = 2  # Spinal cord injury

# Body Position pressure multipliers (Moore 2011, Colin 1996)
# Relative risk compared to optimal support surface
POSITION_PRESSURE_FACTORS = {
    'supine': {
        'sacrum': 1.0,
        'heels': 1.0,
        'scapulae': 0.7,
        'occiput': 0.5,
    },
    'lateral_30': {  # 30° tilt - recommended position
        'sacrum': 0.3,
        'trochanter': 0.6,
        'heels': 0.5,
        'shoulder': 0.7,
    },
    'lateral_90': {  # 90° side-lying - NOT recommended
        'trochanter': 2.5,  # Very high risk
        'shoulder': 1.5,
        'ankle': 1.2,
    },
    'prone': {
        'anterior_iliac': 0.8,
        'chest': 0.6,
        'face': 0.4,
    },
    'semi_fowler': {  # 30-45° head elevation
        'sacrum': 1.3,  # Increased due to shear
        'ischium': 1.5,
        'heels': 0.8,
    },
    'high_fowler': {  # >60° head elevation
        'sacrum': 1.8,  # High shear forces
        'ischium': 2.0,
        'heels': 0.6,
    },
}

# Tissue Stiffness (Gefen 2005, Oomens 2010)
# Stiffness multipliers relative to healthy young adult
TISSUE_STIFFNESS_FACTORS = {
    'healthy_young': 1.0,
    'healthy_elderly': 1.3,  # Age-related stiffening
    'diabetes': 1.5,  # Glycation stiffens tissue
    'edema': 0.7,  # Fluid makes tissue more deformable but fragile
    'cachexia': 0.5,  # Muscle wasting reduces support
    'scar_tissue': 2.0,  # Previous ulcer site
    'spasticity': 1.4,  # Muscle tone increases pressure
}

# Cumulative Damage (Gefen & Weihs 2016, Coleman 2014)
# Tissue tolerance decreases with repeated insults
TOLERANCE_DECAY_PER_INSULT = 0.1  # 10% reduction per major insult
# Minimum tolerance (tissue can't get infinitely weak)
MIN_TOLERANCE_FACTOR = 0.3  # 30% of original tolerance
# Recovery of tolerance (very slow - weeks)
TOLERANCE_RECOVERY_PER_DAY = 0.02  # 2% per day

# Braden Scale risk multipliers (Bergstrom 1987, Dweekat 2023)
BRADEN_RISK_MULTIPLIERS = {
    'very_high': {'score_range': (6, 9), 'multiplier': 2.0},
    'high': {'score_range': (10, 12), 'multiplier': 1.5},
    'moderate': {'score_range': (13, 14), 'multiplier': 1.2},
    'at_risk': {'score_range': (15, 18), 'multiplier': 1.0},
    'no_risk': {'score_range': (19, 23), 'multiplier': 0.5},
}


# =============================================================================
# ENUMERATIONS
# =============================================================================

class BodyPosition(Enum):
    """Patient body positions."""
    SUPINE = 'supine'
    LATERAL_30 = 'lateral_30'
    LATERAL_90 = 'lateral_90'
    PRONE = 'prone'
    SEMI_FOWLER = 'semi_fowler'
    HIGH_FOWLER = 'high_fowler'


class PatientCondition(Enum):
    """Patient tissue condition categories."""
    HEALTHY_YOUNG = 'healthy_young'
    HEALTHY_ELDERLY = 'healthy_elderly'
    DIABETES = 'diabetes'
    EDEMA = 'edema'
    CACHEXIA = 'cachexia'
    SCAR_TISSUE = 'scar_tissue'
    SPASTICITY = 'spasticity'


class MoistureLevel(Enum):
    """Skin moisture levels (Braden subscale)."""
    RARELY_MOIST = 1  # Skin usually dry
    OCCASIONALLY_MOIST = 2  # Requires linen change once/day
    OFTEN_MOIST = 3  # Linen must be changed frequently
    CONSTANTLY_MOIST = 4  # Almost constant moisture


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PatientProfile:
    """
    Patient characteristics affecting pressure ulcer risk.
    Based on Braden Scale and clinical risk factors.
    """
    braden_score: int = 8  # 6-23, lower = higher risk
    age: int = 75
    weight_kg: float = 75.0
    height_cm: float = 175.0

    # Braden subscales
    sensory_perception: int = 1  # 1-4 (1 = completely limited)
    moisture: int = 2  # 1-4 (4 = constantly moist)
    activity: int = 1  # 1-4 (1 = bedfast)
    mobility: int = 1  # 1-4 (1 = completely immobile)
    nutrition: int = 2  # 1-4 (1 = very poor)
    friction_shear: int = 1  # 1-3 (1 = problem)

    # Clinical conditions
    conditions: List[PatientCondition] = field(default_factory=lambda: [PatientCondition.SPASTICITY])
    has_incontinence: bool = True
    has_diabetes: bool = False
    previous_ulcer_sites: List[str] = field(default_factory=list)  # Region names

    # Mobility
    can_reposition: bool = False
    spinal_cord_injury: bool = True

    def get_braden_multiplier(self) -> float:
        """Get damage multiplier based on Braden score."""
        for level, data in BRADEN_RISK_MULTIPLIERS.items():
            if data['score_range'][0] <= self.braden_score <= data['score_range'][1]:
                return data['multiplier']
        return 1.0

    def get_tissue_stiffness_factor(self) -> float:
        """Combined tissue stiffness from all conditions."""
        factor = 1.0
        for condition in self.conditions:
            factor *= TISSUE_STIFFNESS_FACTORS.get(condition.value, 1.0)
        if self.has_diabetes and PatientCondition.DIABETES not in self.conditions:
            factor *= TISSUE_STIFFNESS_FACTORS['diabetes']
        return factor

    def get_micromovement_frequency(self) -> float:
        """Movements per hour based on mobility status."""
        if self.spinal_cord_injury:
            return MICROMOVEMENT_FREQUENCY_SCI
        elif self.mobility <= 2:  # Very/completely immobile
            return MICROMOVEMENT_FREQUENCY_SCI * 1.5
        else:
            return MICROMOVEMENT_FREQUENCY_ABLE


@dataclass
class TissueState:
    """
    Tracks tissue state at each location over time.
    Implements cumulative damage and recovery mechanisms.
    """
    # Shape: (rows, cols)
    shape: Tuple[int, int]

    # Primary damage tracking (PTI - mmHg·hours)
    surface_pti: np.ndarray = None  # Skin/surface damage
    deep_pti: np.ndarray = None  # Deep tissue (muscle) damage

    # Temperature tracking
    local_temperature: np.ndarray = None  # °C above baseline

    # Moisture tracking
    moisture_duration: np.ndarray = None  # Hours of exposure

    # Cumulative tolerance reduction
    tolerance_factor: np.ndarray = None  # 0.3 to 1.0
    damage_cycles: np.ndarray = None  # Number of damage-recovery cycles

    # Ischemia-reperfusion tracking
    ischemia_duration: np.ndarray = None  # Continuous hours under pressure
    last_relief_time: np.ndarray = None  # Time since last relief (hours)

    def __post_init__(self):
        """Initialize arrays."""
        if self.surface_pti is None:
            self.surface_pti = np.zeros(self.shape)
        if self.deep_pti is None:
            self.deep_pti = np.zeros(self.shape)
        if self.local_temperature is None:
            self.local_temperature = np.zeros(self.shape)
        if self.moisture_duration is None:
            self.moisture_duration = np.zeros(self.shape)
        if self.tolerance_factor is None:
            self.tolerance_factor = np.ones(self.shape)
        if self.damage_cycles is None:
            self.damage_cycles = np.zeros(self.shape)
        if self.ischemia_duration is None:
            self.ischemia_duration = np.zeros(self.shape)
        if self.last_relief_time is None:
            self.last_relief_time = np.zeros(self.shape)

    def copy(self) -> 'TissueState':
        """Create a deep copy."""
        return TissueState(
            shape=self.shape,
            surface_pti=self.surface_pti.copy(),
            deep_pti=self.deep_pti.copy(),
            local_temperature=self.local_temperature.copy(),
            moisture_duration=self.moisture_duration.copy(),
            tolerance_factor=self.tolerance_factor.copy(),
            damage_cycles=self.damage_cycles.copy(),
            ischemia_duration=self.ischemia_duration.copy(),
            last_relief_time=self.last_relief_time.copy(),
        )


# =============================================================================
# TISSUE DAMAGE MODEL
# =============================================================================

class ComprehensiveTissueDamageModel:
    """
    Evidence-based tissue damage model incorporating all 8 factors.

    Usage:
        model = ComprehensiveTissueDamageModel(patient, position)
        for each timestep:
            damage = model.calculate_damage(pressure_map, shear_map, dt_hours)
    """

    def __init__(self, patient: PatientProfile,
                 position: BodyPosition = BodyPosition.SUPINE,
                 grid_shape: Tuple[int, int] = (40, 18)):
        """
        Initialize damage model.

        Args:
            patient: Patient risk profile
            position: Current body position
            grid_shape: (rows, cols) of pressure map
        """
        self.patient = patient
        self.position = position
        self.grid_shape = grid_shape

        # Initialize tissue state
        self.state = TissueState(shape=grid_shape)

        # Track simulation time
        self.total_time_hours = 0.0

        # Position-specific pressure factors
        self._update_position_factors()

    def _update_position_factors(self):
        """Update pressure factors based on body position."""
        self.position_factors = np.ones(self.grid_shape)

        # Get position-specific factors
        pos_data = POSITION_PRESSURE_FACTORS.get(self.position.value, {})

        # Map anatomical regions to grid locations
        # Assuming standard supine orientation: head at top (row 0)
        rows, cols = self.grid_shape

        # Define region boundaries (normalized 0-1)
        regions = {
            'occiput': (0.0, 0.08, 0.3, 0.7),  # Head
            'scapulae': (0.10, 0.25, 0.2, 0.8),  # Upper back
            'sacrum': (0.40, 0.55, 0.35, 0.65),  # Lower back
            'heels': (0.92, 1.0, 0.2, 0.8),  # Feet
            'trochanter': (0.45, 0.55, 0.1, 0.3),  # Hip (lateral)
            'shoulder': (0.15, 0.25, 0.0, 0.2),  # Shoulder (lateral)
        }

        for region, factor in pos_data.items():
            if region in regions:
                r0, r1, c0, c1 = regions[region]
                row_start = int(r0 * rows)
                row_end = int(r1 * rows)
                col_start = int(c0 * cols)
                col_end = int(c1 * cols)
                self.position_factors[row_start:row_end, col_start:col_end] = factor

    def set_position(self, position: BodyPosition):
        """Change patient position."""
        self.position = position
        self._update_position_factors()

    def calculate_damage(self, pressure_map: np.ndarray,
                        shear_map: np.ndarray,
                        dt_hours: float,
                        apply_micromovement: bool = True) -> Dict[str, np.ndarray]:
        """
        Calculate comprehensive tissue damage for one timestep.

        Args:
            pressure_map: Interface pressure (mmHg)
            shear_map: Shear stress (mmHg equivalent)
            dt_hours: Time step in hours
            apply_micromovement: Whether to apply random micro-movements

        Returns:
            Dictionary with damage components and total
        """
        self.total_time_hours += dt_hours

        # Ensure maps match grid shape
        if pressure_map.shape != self.grid_shape:
            from scipy.ndimage import zoom
            scale = (self.grid_shape[0] / pressure_map.shape[0],
                    self.grid_shape[1] / pressure_map.shape[1])
            pressure_map = zoom(pressure_map, scale, order=1)
            shear_map = zoom(shear_map, scale, order=1)

        # Apply position factors
        effective_pressure = pressure_map * self.position_factors

        # Apply micro-movements (random temporary relief)
        if apply_micromovement:
            effective_pressure = self._apply_micromovements(effective_pressure, dt_hours)

        # 1. Calculate surface PTI (skin damage)
        surface_damage = self._calc_surface_damage(effective_pressure, shear_map, dt_hours)

        # 2. Calculate deep tissue injury
        dti_damage = self._calc_deep_tissue_damage(effective_pressure, dt_hours)

        # 3. Apply temperature effects
        temp_multiplier = self._update_temperature(effective_pressure, dt_hours)

        # 4. Apply moisture effects
        moisture_multiplier = self._update_moisture(dt_hours)

        # 5. Apply tissue stiffness factor
        stiffness_factor = self.patient.get_tissue_stiffness_factor()

        # 6. Apply Braden risk multiplier
        braden_multiplier = self.patient.get_braden_multiplier()

        # 7. Apply tolerance reduction (cumulative damage effect)
        tolerance = self.state.tolerance_factor

        # 8. Handle reactive hyperemia (recovery)
        recovery = self._calc_recovery(effective_pressure, dt_hours)

        # Combined damage calculation
        combined_multiplier = (temp_multiplier * moisture_multiplier *
                              stiffness_factor * braden_multiplier / tolerance)

        # Update state
        total_surface = surface_damage * combined_multiplier
        total_deep = dti_damage * combined_multiplier

        self.state.surface_pti += total_surface
        self.state.deep_pti += total_deep

        # Apply recovery
        self.state.surface_pti = np.maximum(0, self.state.surface_pti - recovery)
        self.state.deep_pti = np.maximum(0, self.state.deep_pti - recovery * 0.5)  # Deep heals slower

        # Update damage cycles and tolerance
        self._update_tolerance(effective_pressure, dt_hours)

        # Combined PTI (surface + deep, with deep weighted more heavily)
        combined_pti = self.state.surface_pti + self.state.deep_pti * MUSCLE_SENSITIVITY_FACTOR

        return {
            'surface_pti': self.state.surface_pti.copy(),
            'deep_pti': self.state.deep_pti.copy(),
            'combined_pti': combined_pti,
            'temperature': self.state.local_temperature.copy(),
            'moisture_duration': self.state.moisture_duration.copy(),
            'tolerance_factor': self.state.tolerance_factor.copy(),
            'temp_multiplier': temp_multiplier,
            'moisture_multiplier': moisture_multiplier,
        }

    def _apply_micromovements(self, pressure: np.ndarray, dt_hours: float) -> np.ndarray:
        """Apply random micro-movements for temporary pressure relief."""
        freq = self.patient.get_micromovement_frequency()

        # Probability of movement in this timestep
        p_movement = 1 - np.exp(-freq * dt_hours)

        if np.random.random() < p_movement:
            # Random cells get temporary relief
            relief_mask = np.random.random(self.grid_shape) < 0.3  # 30% of cells
            relief_amount = np.random.uniform(0.1, MICROMOVEMENT_PRESSURE_RELIEF)
            pressure = pressure.copy()
            pressure[relief_mask] *= (1 - relief_amount)

        return pressure

    def _calc_surface_damage(self, pressure: np.ndarray, shear: np.ndarray,
                             dt_hours: float) -> np.ndarray:
        """Calculate surface (skin) PTI damage."""
        # Excess pressure above threshold
        excess_pressure = np.maximum(0, pressure - CAPILLARY_CLOSING_PRESSURE)
        excess_shear = np.maximum(0, shear - SHEAR_DAMAGE_THRESHOLD)

        # Shear amplification (1.0 to 2.0×)
        shear_factor = 1 + np.clip(shear / SHEAR_DAMAGE_THRESHOLD, 0, 1)

        # Synergy when both present (Gefen 2014)
        has_both = (excess_pressure > 0) & (excess_shear > 0)
        synergy = np.where(has_both, 1.5, 1.0)

        # PTI in mmHg·hours
        pti = (excess_pressure * shear_factor + excess_shear * 0.5) * synergy * dt_hours

        return pti

    def _calc_deep_tissue_damage(self, pressure: np.ndarray, dt_hours: float) -> np.ndarray:
        """
        Calculate deep tissue injury (DTI) at bone-muscle interface.

        Based on Gefen 2009: Internal strain is more important than
        interface pressure for DTI. Muscle damage occurs within minutes
        at high strain levels.
        """
        # Estimate internal strain from interface pressure
        # Higher pressure = higher strain at bony prominences
        # Using simplified relationship from FEM studies
        estimated_strain = pressure / 200  # Normalize: 100 mmHg ≈ 50% strain

        # Regions over bony prominences have higher strain concentration
        strain_concentration = self._get_strain_concentration_map()
        effective_strain = estimated_strain * strain_concentration

        # Damage increases nonlinearly above strain threshold
        excess_strain = np.maximum(0, effective_strain - DTI_STRAIN_THRESHOLD)

        # Time factor: damage accelerates with duration
        # (Loerakker 2010: damage visible within minutes at high strain)
        time_factor = 1 + self.state.ischemia_duration * 0.5

        # DTI contribution to PTI
        dti_pti = excess_strain * pressure * dt_hours * time_factor

        return dti_pti

    def _get_strain_concentration_map(self) -> np.ndarray:
        """Map of strain concentration factors at bony prominences."""
        concentration = np.ones(self.grid_shape)
        rows, cols = self.grid_shape

        # Bony prominences have higher strain concentration (1.5-3×)
        prominences = {
            'sacrum': ((0.40, 0.55), (0.35, 0.65), 2.5),
            'ischium': ((0.50, 0.60), (0.25, 0.40), 2.0),
            'ischium_r': ((0.50, 0.60), (0.60, 0.75), 2.0),
            'trochanter_l': ((0.45, 0.55), (0.10, 0.25), 2.0),
            'trochanter_r': ((0.45, 0.55), (0.75, 0.90), 2.0),
            'heels': ((0.92, 1.0), (0.20, 0.80), 2.0),
            'scapulae_l': ((0.12, 0.22), (0.15, 0.35), 1.5),
            'scapulae_r': ((0.12, 0.22), (0.65, 0.85), 1.5),
            'occiput': ((0.0, 0.08), (0.30, 0.70), 1.5),
        }

        for region, ((r0, r1), (c0, c1), factor) in prominences.items():
            row_slice = slice(int(r0 * rows), int(r1 * rows))
            col_slice = slice(int(c0 * cols), int(c1 * cols))
            concentration[row_slice, col_slice] = factor

        return concentration

    def _update_temperature(self, pressure: np.ndarray, dt_hours: float) -> np.ndarray:
        """
        Update local temperature and return damage multiplier.

        Based on Kottner 2018: Temperature rises ~3°C under sustained
        pressure due to occlusion and metabolic heat.
        """
        # Temperature rises where pressure causes occlusion
        under_pressure = pressure > CAPILLARY_CLOSING_PRESSURE

        # Increase temperature (capped at max rise)
        self.state.local_temperature[under_pressure] += TEMP_RISE_RATE * dt_hours
        self.state.local_temperature = np.clip(
            self.state.local_temperature, 0, TEMP_MAX_RISE
        )

        # Temperature decreases when pressure relieved
        relieved = ~under_pressure
        self.state.local_temperature[relieved] *= (1 - 0.5 * dt_hours)  # Cool down

        # Damage multiplier: 10% increase per °C (Lachenbruch 2005)
        multiplier = 1 + self.state.local_temperature * TEMP_DAMAGE_MULTIPLIER

        return multiplier

    def _update_moisture(self, dt_hours: float) -> np.ndarray:
        """
        Update moisture duration and return damage multiplier.

        Based on Beeckman 2014: Incontinence increases PU risk 4.99×.
        Maceration weakens skin over time.
        """
        if self.patient.has_incontinence or self.patient.moisture >= 3:
            # High-risk moisture areas (perineal, sacral)
            rows, cols = self.grid_shape
            moist_region = np.zeros(self.grid_shape, dtype=bool)
            moist_region[int(0.35*rows):int(0.65*rows), int(0.25*cols):int(0.75*cols)] = True

            self.state.moisture_duration[moist_region] += dt_hours
        else:
            # Gradual drying
            self.state.moisture_duration *= (1 - 0.2 * dt_hours)

        # Maceration develops over time
        maceration_factor = np.clip(
            self.state.moisture_duration / MACERATION_TIME_HOURS, 0, 1
        )

        # Moisture multiplier (up to 4.99× based on literature)
        moisture_level = self.patient.moisture / 4.0  # Normalize Braden subscale
        base_multiplier = 1 + (MOISTURE_RISK_MULTIPLIER - 1) * moisture_level

        # Additional maceration effect
        multiplier = base_multiplier * (1 + maceration_factor)

        return multiplier

    def _calc_recovery(self, pressure: np.ndarray, dt_hours: float) -> np.ndarray:
        """
        Calculate tissue recovery via reactive hyperemia.

        Based on Tzen 2017: Recovery occurs when pressure relieved,
        but diminishes with repeated loading cycles.
        """
        # Identify cells currently under pressure vs relieved
        under_pressure = pressure > CAPILLARY_CLOSING_PRESSURE
        relieved = ~under_pressure

        # Update ischemia duration
        self.state.ischemia_duration[under_pressure] += dt_hours
        self.state.ischemia_duration[relieved] = 0

        # Recovery rate diminishes with cycles (Tzen 2017)
        base_recovery = HYPEREMIA_RECOVERY_BASE * dt_hours
        fatigue = HYPEREMIA_FATIGUE_FACTOR * self.state.damage_cycles
        effective_recovery_rate = np.maximum(0.1, base_recovery - fatigue)

        # Only recover where currently relieved
        recovery = np.zeros(self.grid_shape)
        recovery[relieved] = effective_recovery_rate[relieved] * CAPILLARY_CLOSING_PRESSURE

        return recovery

    def _update_tolerance(self, pressure: np.ndarray, dt_hours: float):
        """
        Update tissue tolerance based on cumulative damage.

        Based on Gefen & Weihs 2016: Repeated insults reduce tissue
        tolerance, leading to faster damage accumulation.
        """
        # Check for damage threshold crossings (new "insults")
        significant_damage = self.state.surface_pti > 5.0  # mmHg·h threshold

        # Count new damage cycles
        under_pressure = pressure > CAPILLARY_CLOSING_PRESSURE
        new_cycle = significant_damage & ~under_pressure
        self.state.damage_cycles[new_cycle] += 1

        # Reduce tolerance with each cycle
        tolerance_reduction = self.state.damage_cycles * TOLERANCE_DECAY_PER_INSULT
        self.state.tolerance_factor = np.maximum(
            MIN_TOLERANCE_FACTOR,
            1.0 - tolerance_reduction
        )

        # Mark previous ulcer sites with reduced tolerance
        for site in self.patient.previous_ulcer_sites:
            # Would need site-to-grid mapping
            pass

    def get_summary(self) -> Dict:
        """Get summary statistics of current tissue state."""
        combined_pti = (self.state.surface_pti +
                       self.state.deep_pti * MUSCLE_SENSITIVITY_FACTOR)

        # Critical PTI threshold for HAPI (mmHg·hours)
        # Based on Reswick-Rogers adjusted for patient risk
        critical_pti = 20.0 / self.patient.get_braden_multiplier()

        return {
            'total_time_hours': self.total_time_hours,
            'max_surface_pti': float(self.state.surface_pti.max()),
            'max_deep_pti': float(self.state.deep_pti.max()),
            'max_combined_pti': float(combined_pti.max()),
            'cells_at_risk': int((combined_pti >= critical_pti).sum()),
            'critical_pti_threshold': critical_pti,
            'mean_temperature_rise': float(self.state.local_temperature.mean()),
            'mean_tolerance': float(self.state.tolerance_factor.mean()),
            'total_damage_cycles': int(self.state.damage_cycles.sum()),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_default_patient(patient_type: str = 'quadriplegic') -> PatientProfile:
    """Create common patient profiles."""

    profiles = {
        'quadriplegic': PatientProfile(
            braden_score=8,
            age=45,
            sensory_perception=1,
            moisture=2,
            activity=1,
            mobility=1,
            nutrition=2,
            friction_shear=1,
            conditions=[PatientCondition.SPASTICITY],
            spinal_cord_injury=True,
            has_incontinence=True,
        ),
        'elderly_nursing_home': PatientProfile(
            braden_score=12,
            age=82,
            sensory_perception=2,
            moisture=2,
            activity=2,
            mobility=2,
            nutrition=2,
            friction_shear=2,
            conditions=[PatientCondition.HEALTHY_ELDERLY],
            spinal_cord_injury=False,
            has_incontinence=True,
        ),
        'icu_patient': PatientProfile(
            braden_score=9,
            age=65,
            sensory_perception=1,
            moisture=3,
            activity=1,
            mobility=1,
            nutrition=1,
            friction_shear=1,
            conditions=[PatientCondition.EDEMA],
            spinal_cord_injury=False,
            has_incontinence=True,
        ),
        'diabetic': PatientProfile(
            braden_score=14,
            age=70,
            sensory_perception=2,
            moisture=2,
            activity=3,
            mobility=3,
            nutrition=2,
            friction_shear=2,
            conditions=[PatientCondition.DIABETES],
            has_diabetes=True,
            spinal_cord_injury=False,
            has_incontinence=False,
        ),
        'healthy_control': PatientProfile(
            braden_score=23,
            age=30,
            sensory_perception=4,
            moisture=4,
            activity=4,
            mobility=4,
            nutrition=4,
            friction_shear=3,
            conditions=[PatientCondition.HEALTHY_YOUNG],
            spinal_cord_injury=False,
            has_incontinence=False,
        ),
    }

    return profiles.get(patient_type, profiles['quadriplegic'])


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    print("Comprehensive Tissue Damage Model")
    print("=" * 60)

    # Create test patient
    patient = create_default_patient('quadriplegic')
    print(f"\nPatient: Quadriplegic, Braden {patient.braden_score}")
    print(f"  Risk multiplier: {patient.get_braden_multiplier():.1f}×")
    print(f"  Stiffness factor: {patient.get_tissue_stiffness_factor():.2f}")
    print(f"  Micro-movement freq: {patient.get_micromovement_frequency()}/hour")

    # Create model
    model = ComprehensiveTissueDamageModel(patient, BodyPosition.SUPINE)

    # Simulate pressure map (simplified)
    rows, cols = 40, 18
    pressure = np.zeros((rows, cols))
    shear = np.zeros((rows, cols))

    # Sacral pressure (high)
    pressure[16:22, 6:12] = 80  # mmHg
    shear[16:22, 6:12] = 15

    # Heel pressure
    pressure[36:40, 4:14] = 60
    shear[36:40, 4:14] = 8

    print(f"\nSimulating 2 hours with pressure peaks at {pressure.max():.0f} mmHg...")

    # Run simulation
    dt = 5/60  # 5-minute steps
    for step in range(24):  # 2 hours
        result = model.calculate_damage(pressure, shear, dt)

    summary = model.get_summary()
    print(f"\nResults after {summary['total_time_hours']:.1f} hours:")
    print(f"  Max surface PTI: {summary['max_surface_pti']:.1f} mmHg·h")
    print(f"  Max deep PTI: {summary['max_deep_pti']:.1f} mmHg·h")
    print(f"  Max combined PTI: {summary['max_combined_pti']:.1f} mmHg·h")
    print(f"  Cells at risk: {summary['cells_at_risk']}")
    print(f"  Critical threshold: {summary['critical_pti_threshold']:.1f} mmHg·h")
    print(f"  Mean temp rise: {summary['mean_temperature_rise']:.2f}°C")
    print(f"  Mean tolerance: {summary['mean_tolerance']:.2f}")
