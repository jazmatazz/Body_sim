#!/usr/bin/env python3
"""
Literature-Based Pressure Ulcer Model Upgrades
===============================================

Implements scientifically-grounded improvements based on peer-reviewed research:

1. Dual-mechanism damage (deformation + ischemia) - Gefen et al.
2. Deep tissue injury modeling - Loerakker et al.
3. Oxygen transport with perfusion coupling
4. Reperfusion injury (ROS burst)
5. Cell population dynamics (ODE system) - Zeevi et al. 2024
6. Nonlinear pressure-time tradeoff
7. Strain-based damage thresholds (Boltzmann sigmoid)

Each class is annotated with:
- [PHYSICAL]: Physically justified by literature
- [HEURISTIC]: Reasonable approximation, needs calibration
- [SIMPLIFIED]: Captures key physics, ignores secondary effects

References:
- PLOS Comp Bio 2015: Agent-based PU model
- J Biomech 2008: Strain-time thresholds
- Math Biosci Eng 2024: Cell population ODEs
"""

import numpy as np
from typing import Tuple, Optional, Dict


# =============================================================================
# CONSTANTS FROM LITERATURE
# =============================================================================

# Capillary closing pressure [PHYSICAL]
CAPILLARY_CLOSING_PRESSURE_MMHG = 32.0

# Strain thresholds from Gefen et al. 2008 [PHYSICAL]
STRAIN_THRESHOLD_1H = 0.65    # 65% strain tolerable for 1 hour
STRAIN_THRESHOLD_4H = 0.40    # 40% strain tolerable for 4+ hours
STRAIN_THRESHOLD_TRANSITION_TIME = 90 * 60  # 90 minutes (seconds)

# Oxygen thresholds [PHYSICAL]
O2_NORMAL = 0.95              # Normal tissue oxygen saturation
O2_HYPOXIA = 0.50             # Hypoxia threshold
O2_ANOXIA = 0.10              # Severe anoxia

# Deep tissue stress concentration factors [PHYSICAL: from FE studies]
BONE_STRESS_CONCENTRATION = {
    'sacrum': 2.5,
    'heel': 2.2,
    'scapula': 1.8,
    'occiput': 1.6,
    'greater_trochanter': 2.0,
    'ischial_tuberosity': 2.3,
}

# Damage timescales [PHYSICAL]
DEFORMATION_DAMAGE_TIMESCALE = 300      # seconds (5 min for visible effect)
ISCHEMIC_DAMAGE_TIMESCALE = 7200        # seconds (2 hours)
REPERFUSION_DAMAGE_TIMESCALE = 60       # seconds (rapid)


# =============================================================================
# DUAL-MECHANISM DAMAGE MODEL
# =============================================================================

class DualMechanismDamageModel:
    """
    Two-pathway damage accumulation based on literature.

    Pathway 1: Direct deformation damage
    - Timescale: minutes
    - Mechanism: membrane stretch, organelle distortion
    - Threshold: strain-dependent (Boltzmann sigmoid)
    - Reference: Gefen et al. 2008

    Pathway 2: Ischemic damage
    - Timescale: hours
    - Mechanism: ATP depletion, metabolic failure
    - Threshold: oxygen-dependent
    - Reference: PUABM (Loerakker 2015)

    [PHYSICAL]: Dual pathways well-established
    [HEURISTIC]: Rate constants need calibration
    """

    def __init__(self, grid_shape: Tuple[int, int]):
        self.grid_shape = grid_shape

        # Damage accumulators
        self.deformation_damage = np.zeros(grid_shape)
        self.ischemic_damage = np.zeros(grid_shape)
        self.reperfusion_damage = np.zeros(grid_shape)

        # State trackers
        self.time_under_strain = np.zeros(grid_shape)
        self.oxygen_deficit_integral = np.zeros(grid_shape)
        self.xanthine_oxidase = np.zeros(grid_shape)  # For reperfusion injury

        # Previous state (for reperfusion detection)
        self.previous_oxygen = np.ones(grid_shape) * O2_NORMAL

        # Total simulation time
        self.total_time = 0

    def compute_strain_threshold(self, time_under_load: np.ndarray) -> np.ndarray:
        """
        Time-dependent strain threshold (Boltzmann sigmoid).

        [PHYSICAL]: Based on Gefen et al. experimental data
        E_c(t) = E_4h + (E_1h - E_4h) / (1 + exp(k*(t - t_half)))

        Cells can tolerate:
        - 65% strain for 1 hour
        - 40% strain for 4+ hours
        - Transition occurs around 90 minutes
        """
        k = 0.002  # Steepness parameter (1/seconds)
        t_half = STRAIN_THRESHOLD_TRANSITION_TIME

        threshold = (
            STRAIN_THRESHOLD_4H +
            (STRAIN_THRESHOLD_1H - STRAIN_THRESHOLD_4H) /
            (1 + np.exp(k * (time_under_load - t_half)))
        )

        return threshold

    def update_deformation_damage(self, strain: np.ndarray, dt: float):
        """
        Update damage from direct mechanical deformation.

        [PHYSICAL]: Strain threshold concept
        [HEURISTIC]: Damage rate constant

        Args:
            strain: Engineering strain (0-1, can exceed 1)
            dt: Time step in seconds
        """
        # Update time under significant strain
        significant_strain = strain > 0.2  # 20% threshold for tracking
        self.time_under_strain[significant_strain] += dt
        self.time_under_strain[~significant_strain] *= 0.95  # Decay when unloaded

        # Get time-dependent threshold
        current_threshold = self.compute_strain_threshold(self.time_under_strain)

        # Excess strain above threshold
        excess_strain = np.maximum(strain - current_threshold, 0)

        # Damage rate increases nonlinearly with excess strain
        # [HEURISTIC]: Power law relationship
        damage_rate = 0.001 * (excess_strain ** 1.5)  # per second

        # Accumulate damage
        self.deformation_damage += damage_rate * dt

        # Cap at 1.0 (complete damage)
        self.deformation_damage = np.clip(self.deformation_damage, 0, 1)

    def update_ischemic_damage(self, oxygen_level: np.ndarray, dt: float):
        """
        Update damage from ischemia (oxygen deprivation).

        [PHYSICAL]: Oxygen threshold for cell viability
        [HEURISTIC]: Integration approach and rate constants

        Args:
            oxygen_level: Normalized oxygen (0-1)
            dt: Time step in seconds
        """
        # Oxygen deficit (how far below normal)
        deficit = np.maximum(O2_HYPOXIA - oxygen_level, 0)

        # Accumulate oxygen deficit over time
        self.oxygen_deficit_integral += deficit * dt

        # Ischemic damage requires sustained deficit
        # [PHYSICAL]: Takes hours to develop
        critical_deficit_threshold = 1800  # 30 minutes of hypoxia (sec * fraction)

        # Damage rate proportional to deficit, only after threshold
        active_ischemia = self.oxygen_deficit_integral > critical_deficit_threshold

        damage_rate = np.zeros_like(oxygen_level)
        damage_rate[active_ischemia] = (
            0.0001 * deficit[active_ischemia] *
            (self.oxygen_deficit_integral[active_ischemia] / critical_deficit_threshold)
        )

        self.ischemic_damage += damage_rate * dt
        self.ischemic_damage = np.clip(self.ischemic_damage, 0, 1)

        # Xanthine oxidase accumulates during ischemia
        # [PHYSICAL]: XO conversion during hypoxia
        xo_accumulation = np.zeros_like(oxygen_level)
        xo_accumulation[oxygen_level < O2_HYPOXIA] = 0.001
        self.xanthine_oxidase += xo_accumulation * dt
        self.xanthine_oxidase = np.clip(self.xanthine_oxidase, 0, 1)

    def update_reperfusion_damage(self, oxygen_level: np.ndarray, dt: float):
        """
        Update damage from reperfusion injury (ROS burst).

        [PHYSICAL]: XO + O2 -> ROS -> tissue damage
        [SIMPLIFIED]: Doesn't model full ROS chemistry

        Key insight from PUABM:
        "Abrupt pressure release can be worst-case scenario"

        Args:
            oxygen_level: Current oxygen level
            dt: Time step
        """
        # Detect reperfusion: oxygen increasing after deficit
        oxygen_increase = oxygen_level - self.previous_oxygen
        reperfusion_mask = (
            (oxygen_increase > 0.05) &      # Oxygen rapidly increasing
            (self.xanthine_oxidase > 0.1) & # XO accumulated
            (self.previous_oxygen < O2_HYPOXIA)  # Was hypoxic
        )

        if reperfusion_mask.any():
            # ROS production proportional to XO and available O2
            ros_production = (
                self.xanthine_oxidase[reperfusion_mask] *
                oxygen_level[reperfusion_mask] *
                oxygen_increase[reperfusion_mask]
            )

            # ROS causes rapid damage
            # [PHYSICAL]: ROS damage is fast
            ros_damage_rate = 0.1  # per second per unit ROS

            self.reperfusion_damage[reperfusion_mask] += (
                ros_damage_rate * ros_production * dt
            )

            # XO consumed in reaction
            self.xanthine_oxidase[reperfusion_mask] *= 0.8

        # Store for next iteration
        self.previous_oxygen = oxygen_level.copy()

        self.reperfusion_damage = np.clip(self.reperfusion_damage, 0, 1)

    def update(self, strain: np.ndarray, oxygen_level: np.ndarray, dt: float):
        """
        Update all damage pathways.

        Args:
            strain: Engineering strain field
            oxygen_level: Oxygen saturation field
            dt: Time step in seconds
        """
        self.total_time += dt

        self.update_deformation_damage(strain, dt)
        self.update_ischemic_damage(oxygen_level, dt)
        self.update_reperfusion_damage(oxygen_level, dt)

    def get_total_damage(self) -> np.ndarray:
        """
        Combine damage pathways with synergistic effects.

        [PHYSICAL]: Damage pathways interact
        [HEURISTIC]: Synergy factors

        Returns:
            Total damage index (0-1)
        """
        # Synergistic interaction: deformation sensitizes to ischemia
        synergy_factor = 1 + 0.5 * self.deformation_damage

        total = (
            self.deformation_damage +
            self.ischemic_damage * synergy_factor +
            self.reperfusion_damage
        )

        return np.clip(total, 0, 1)

    def get_damage_breakdown(self) -> Dict[str, np.ndarray]:
        """Get individual damage components."""
        return {
            'deformation': self.deformation_damage.copy(),
            'ischemic': self.ischemic_damage.copy(),
            'reperfusion': self.reperfusion_damage.copy(),
            'total': self.get_total_damage(),
        }


# =============================================================================
# DEEP TISSUE INJURY MODEL
# =============================================================================

class DeepTissueInjuryModel:
    """
    Models stress concentration at bone-tissue interface.

    [PHYSICAL]: DTI initiates at depth, not surface
    - Bony prominences cause geometric stress concentration
    - Muscle incompressibility amplifies deep stress
    - FE studies show 2-3x concentration factors

    Reference: Gefen 2018, Loerakker 2015
    """

    def __init__(self, grid_shape: Tuple[int, int]):
        self.grid_shape = grid_shape

        # Stress at different depths
        self.surface_stress = np.zeros(grid_shape)
        self.subcutaneous_stress = np.zeros(grid_shape)
        self.muscle_stress = np.zeros(grid_shape)
        self.bone_interface_stress = np.zeros(grid_shape)

        # Build anatomical maps
        self.stress_concentration_map = self._build_concentration_map()
        self.fat_thickness_map = self._build_fat_thickness_map()

    def _build_concentration_map(self) -> np.ndarray:
        """
        Build stress concentration factor map.

        [PHYSICAL]: FE models show 2-3x concentration over bone
        [HEURISTIC]: Spatial distribution simplified
        """
        rows, cols = self.grid_shape
        concentration = np.ones(self.grid_shape) * 1.0  # Baseline

        # Sacrum region (45-55% length, center)
        r1, r2 = int(0.45 * rows), int(0.55 * rows)
        c1, c2 = int(0.30 * cols), int(0.70 * cols)
        concentration[r1:r2, c1:c2] = BONE_STRESS_CONCENTRATION['sacrum']

        # Heels (95-100% length)
        concentration[int(0.95 * rows):, :] = BONE_STRESS_CONCENTRATION['heel']

        # Scapulae (12-22% length, lateral)
        r1, r2 = int(0.12 * rows), int(0.22 * rows)
        concentration[r1:r2, :int(0.25 * cols)] = BONE_STRESS_CONCENTRATION['scapula']
        concentration[r1:r2, int(0.75 * cols):] = BONE_STRESS_CONCENTRATION['scapula']

        # Occiput (0-8% length, center)
        concentration[:int(0.08 * rows), int(0.35 * cols):int(0.65 * cols)] = \
            BONE_STRESS_CONCENTRATION['occiput']

        # Apply Gaussian smoothing for realistic transitions
        concentration = self._smooth_field(concentration, sigma=2)

        return concentration

    def _build_fat_thickness_map(self) -> np.ndarray:
        """
        Build subcutaneous fat thickness map (mm).

        [PHYSICAL]: Less fat over bony prominences
        [HEURISTIC]: Values are population averages
        """
        rows, cols = self.grid_shape
        fat = np.ones(self.grid_shape) * 15.0  # 15mm baseline

        # Sacrum: very thin
        r1, r2 = int(0.45 * rows), int(0.55 * rows)
        c1, c2 = int(0.30 * cols), int(0.70 * cols)
        fat[r1:r2, c1:c2] = 5.0

        # Heels: minimal fat
        fat[int(0.95 * rows):, :] = 4.0

        # Scapulae: moderate
        r1, r2 = int(0.12 * rows), int(0.22 * rows)
        fat[r1:r2, :int(0.30 * cols)] = 10.0
        fat[r1:r2, int(0.70 * cols):] = 10.0

        # Smooth transitions
        fat = self._smooth_field(fat, sigma=3)

        return fat

    def _smooth_field(self, field: np.ndarray, sigma: int) -> np.ndarray:
        """Apply Gaussian smoothing."""
        if sigma < 1:
            return field

        result = field.copy()
        kernel_size = sigma * 3

        # Simple box filter approximation
        for _ in range(sigma):
            temp = result.copy()
            temp[1:-1, 1:-1] = (
                result[:-2, 1:-1] + result[2:, 1:-1] +
                result[1:-1, :-2] + result[1:-1, 2:] +
                result[1:-1, 1:-1]
            ) / 5
            result = temp

        return result

    def compute_depth_resolved_stress(self, surface_pressure: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute stress at each tissue depth.

        [PHYSICAL]: Stress transmission through layered medium
        Key insight: stress INCREASES with depth over bony prominences

        Args:
            surface_pressure: Interface pressure (mmHg)

        Returns:
            Dictionary of stress at each depth
        """
        self.surface_stress = surface_pressure.copy()

        # Fat layer attenuation
        # [PHYSICAL]: Fat absorbs/distributes load
        fat_reference = 15.0  # mm
        fat_attenuation = np.exp(-(self.fat_thickness_map - fat_reference) / 10)

        # Subcutaneous stress (partially attenuated)
        self.subcutaneous_stress = surface_pressure * (0.8 + 0.2 * fat_attenuation)

        # Muscle stress (further attenuated in soft areas, concentrated over bone)
        self.muscle_stress = surface_pressure * (0.6 + 0.4 * fat_attenuation)

        # Bone interface stress (HIGHEST over bony prominences)
        # [PHYSICAL]: Geometric focusing at bone
        self.bone_interface_stress = (
            surface_pressure *
            self.stress_concentration_map *
            fat_attenuation
        )

        return {
            'surface': self.surface_stress,
            'subcutaneous': self.subcutaneous_stress,
            'muscle': self.muscle_stress,
            'bone_interface': self.bone_interface_stress,
        }

    def get_dti_initiation_risk(self) -> np.ndarray:
        """
        Get DTI initiation risk based on deep stress.

        [PHYSICAL]: DTI starts at bone interface
        """
        # Higher threshold for deep tissue (muscle initially more tolerant)
        deep_threshold = 60  # mmHg

        excess = np.maximum(self.bone_interface_stress - deep_threshold, 0)
        risk = 1 - np.exp(-excess / 50)

        return risk


# =============================================================================
# OXYGEN TRANSPORT MODEL
# =============================================================================

class OxygenTransportModel:
    """
    Models oxygen diffusion, consumption, and perfusion.

    [PHYSICAL]: Based on Fick's diffusion + Michaelis-Menten consumption
    [SIMPLIFIED]: 2D approximation of 3D transport

    Equation:
    ∂O₂/∂t = D∇²O₂ - V_max·O₂/(K_m + O₂) + Q_perfusion

    Reference: Multiscale PU model (2019)
    """

    def __init__(self, grid_shape: Tuple[int, int], cell_size_cm: float):
        self.grid_shape = grid_shape
        self.cell_size = cell_size_cm / 100  # meters

        # Oxygen field (normalized 0-1)
        self.oxygen = np.ones(grid_shape) * O2_NORMAL

        # Physical parameters [PHYSICAL]
        self.D_O2 = 2.0e-9           # m²/s diffusion in tissue
        self.V_max = 0.005           # 1/s max consumption rate
        self.K_m = 0.1               # Michaelis constant
        self.Q_baseline = 0.01       # 1/s baseline perfusion supply

        # Derived
        self.dt_stable = 0.5 * self.cell_size**2 / self.D_O2  # Stability limit

    def update(self, pressure_mmhg: np.ndarray, dt: float) -> np.ndarray:
        """
        Update oxygen field.

        Args:
            pressure_mmhg: Current pressure distribution
            dt: Time step (seconds)

        Returns:
            Updated oxygen field
        """
        # Use substeps if dt is large for stability
        n_substeps = max(1, int(dt / self.dt_stable) + 1)
        sub_dt = dt / n_substeps

        for _ in range(n_substeps):
            # 1. Diffusion term: D∇²O₂
            laplacian = self._compute_laplacian(self.oxygen)
            diffusion = self.D_O2 * laplacian / (self.cell_size ** 2)

            # 2. Consumption term: Michaelis-Menten kinetics [PHYSICAL]
            consumption = self.V_max * self.oxygen / (self.K_m + self.oxygen)

            # 3. Perfusion supply (pressure-dependent) [PHYSICAL]
            perfusion_factor = self._compute_perfusion_factor(pressure_mmhg)
            supply = self.Q_baseline * perfusion_factor * (1 - self.oxygen)

            # 4. Update
            dO2_dt = diffusion - consumption + supply
            self.oxygen += dO2_dt * sub_dt

        # Physical bounds
        self.oxygen = np.clip(self.oxygen, 0, 1)

        return self.oxygen

    def _compute_perfusion_factor(self, pressure_mmhg: np.ndarray) -> np.ndarray:
        """
        Compute perfusion reduction factor.

        [PHYSICAL]: Capillary closing at ~32 mmHg
        Uses sigmoid for gradual occlusion
        """
        P_half = CAPILLARY_CLOSING_PRESSURE_MMHG
        k = 0.1  # Steepness

        # Sigmoid: 1 at low pressure, 0 at high pressure
        factor = 1 / (1 + np.exp(k * (pressure_mmhg - P_half)))

        return factor

    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute 2D discrete Laplacian."""
        laplacian = np.zeros_like(field)
        laplacian[1:-1, 1:-1] = (
            field[:-2, 1:-1] + field[2:, 1:-1] +
            field[1:-1, :-2] + field[1:-1, 2:] -
            4 * field[1:-1, 1:-1]
        )
        return laplacian

    def get_hypoxia_map(self) -> np.ndarray:
        """Get hypoxia severity (0 = normal, 1 = anoxic)."""
        severity = (O2_HYPOXIA - self.oxygen) / (O2_HYPOXIA - O2_ANOXIA)
        return np.clip(severity, 0, 1)

    def get_ischemic_region(self) -> np.ndarray:
        """Get boolean mask of ischemic regions."""
        return self.oxygen < O2_HYPOXIA


# =============================================================================
# CELL POPULATION DYNAMICS (from 2024 paper)
# =============================================================================

class CellPopulationModel:
    """
    Cell population dynamics under mechanical and biochemical stress.

    ODE System from Zeevi et al. (2024):
    Ṅ = (r₀ - kₙ₁·I)·(1 - N/K₀)·H(p)·N
    İ = kₙ₂·I·(1 - I/K₁) + kₐ₀·D
    Ḋ = (1 - N/K₀)·kₐ₁ - kₐ₂·D

    Where:
    - N: cell population (normalized)
    - I: immune/neutrophil density
    - D: DAMP (damage-associated molecular pattern) concentration
    - H(p): Hill function for oxygen dependence

    [PHYSICAL]: Well-established population dynamics
    [HEURISTIC]: Parameter values need tissue-specific calibration
    """

    def __init__(self, grid_shape: Tuple[int, int]):
        self.grid_shape = grid_shape

        # State variables
        self.N = np.ones(grid_shape)      # Cell population (normalized to 1)
        self.I = np.zeros(grid_shape)     # Immune cell density
        self.D = np.zeros(grid_shape)     # DAMP concentration

        # Parameters [HEURISTIC: from paper, may need calibration]
        self.r0 = 0.0001       # Base reproduction rate (1/s) - slow
        self.K0 = 1.0          # Cell carrying capacity
        self.K1 = 2.0          # Immune carrying capacity (can overshoot)
        self.kn1 = 0.01        # Immune suppression of reproduction
        self.kn2 = 0.0005      # Immune proliferation rate
        self.ka0 = 0.001       # DAMP-induced immune recruitment
        self.ka1 = 0.0005      # DAMP production from damage
        self.ka2 = 0.0002      # DAMP clearance rate

        # Hill function parameters [PHYSICAL]
        self.hill_n = 2        # Cooperativity
        self.hill_K = 0.3      # Half-saturation

    def hill_function(self, oxygen: np.ndarray) -> np.ndarray:
        """
        Hill function for oxygen-dependent cell viability.

        H(p) = p^n / (K^n + p^n)

        [PHYSICAL]: Standard biochemical kinetics
        """
        return oxygen**self.hill_n / (self.hill_K**self.hill_n + oxygen**self.hill_n)

    def update(self, oxygen: np.ndarray, mechanical_damage: np.ndarray, dt: float):
        """
        Update cell population dynamics.

        Args:
            oxygen: Oxygen level field (0-1)
            mechanical_damage: Direct mechanical damage (0-1)
            dt: Time step (seconds)
        """
        # Oxygen-dependent viability
        H_p = self.hill_function(oxygen)

        # Damage fraction (affects DAMP production)
        damage_fraction = 1 - self.N / self.K0

        # Additional DAMP from mechanical damage
        mechanical_damp = 0.001 * mechanical_damage

        # Cell population dynamics
        dN_dt = (
            (self.r0 - self.kn1 * self.I) *
            damage_fraction *
            H_p * self.N
        )

        # Immune response dynamics
        dI_dt = (
            self.kn2 * self.I * (1 - self.I / self.K1) +
            self.ka0 * self.D
        )

        # DAMP dynamics
        dD_dt = (
            damage_fraction * self.ka1 +
            mechanical_damp -
            self.ka2 * self.D
        )

        # Euler update (could use RK4 for better accuracy)
        self.N += dN_dt * dt
        self.I += dI_dt * dt
        self.D += dD_dt * dt

        # Physical bounds
        self.N = np.clip(self.N, 0.01, 1)  # Cells can't go to zero (numerical)
        self.I = np.clip(self.I, 0, 5)      # Immune can spike
        self.D = np.clip(self.D, 0, 10)     # DAMP can accumulate

    def get_tissue_viability(self) -> np.ndarray:
        """
        Get tissue viability (cell population).

        Returns:
            Normalized viability (0-1)
        """
        return self.N.copy()

    def get_inflammation_level(self) -> np.ndarray:
        """
        Get inflammation level.

        Returns:
            Normalized inflammation (0 = none, 1 = severe)
        """
        return np.clip(self.I / 2, 0, 1)

    def get_damage_signals(self) -> np.ndarray:
        """
        Get DAMP concentration.

        Returns:
            Normalized DAMP level
        """
        return np.clip(self.D / 5, 0, 1)


# =============================================================================
# PRESSURE-TIME TRADEOFF MODEL
# =============================================================================

class PressureTimeTradeoffModel:
    """
    Nonlinear pressure-time damage relationship.

    [PHYSICAL]: Not a simple P×t integral
    - High pressure: damage in minutes (deformation)
    - Moderate pressure: damage in hours (ischemia)
    - Low pressure: damage in days or never

    Based on clinical observations and experimental data.
    """

    def __init__(self, grid_shape: Tuple[int, int]):
        self.grid_shape = grid_shape

        # Cumulative exposure at different pressure regimes
        self.exposure_low = np.zeros(grid_shape)      # <32 mmHg
        self.exposure_moderate = np.zeros(grid_shape) # 32-60 mmHg
        self.exposure_high = np.zeros(grid_shape)     # 60-100 mmHg
        self.exposure_extreme = np.zeros(grid_shape)  # >100 mmHg

        # Time above thresholds
        self.time_above_32 = np.zeros(grid_shape)
        self.time_above_60 = np.zeros(grid_shape)
        self.time_above_100 = np.zeros(grid_shape)

    def compute_damage_rate(self, pressure_mmhg: np.ndarray) -> np.ndarray:
        """
        Compute pressure-dependent damage rate.

        [PHYSICAL]: Different mechanisms dominate at different pressures
        [HEURISTIC]: Specific rate values need calibration

        Returns:
            Damage rate (1/second)
        """
        rate = np.zeros_like(pressure_mmhg)

        # Regime 1: Below capillary closing (<32 mmHg)
        # Minimal damage, only very long-term effects
        mask1 = pressure_mmhg < 32
        rate[mask1] = 1e-7 * pressure_mmhg[mask1] / 32

        # Regime 2: Partial ischemia (32-60 mmHg)
        # Slow damage, primarily ischemic
        mask2 = (pressure_mmhg >= 32) & (pressure_mmhg < 60)
        rate[mask2] = 1e-6 + 5e-6 * (pressure_mmhg[mask2] - 32) / 28

        # Regime 3: Significant ischemia + some deformation (60-100 mmHg)
        # Moderate damage, mixed mechanism
        mask3 = (pressure_mmhg >= 60) & (pressure_mmhg < 100)
        rate[mask3] = 6e-6 + 2e-5 * ((pressure_mmhg[mask3] - 60) / 40) ** 1.5

        # Regime 4: Deformation-dominated (>100 mmHg)
        # Rapid damage, direct cellular injury
        mask4 = pressure_mmhg >= 100
        rate[mask4] = 2.6e-5 + 1e-4 * ((pressure_mmhg[mask4] - 100) / 50) ** 2

        return rate

    def update(self, pressure_mmhg: np.ndarray, dt: float):
        """
        Update pressure-time exposure tracking.

        Args:
            pressure_mmhg: Current pressure field
            dt: Time step (seconds)
        """
        # Update time above thresholds
        self.time_above_32 += (pressure_mmhg >= 32) * dt
        self.time_above_60 += (pressure_mmhg >= 60) * dt
        self.time_above_100 += (pressure_mmhg >= 100) * dt

        # Update regime-specific exposures
        self.exposure_low += (pressure_mmhg < 32) * pressure_mmhg * dt
        self.exposure_moderate += ((pressure_mmhg >= 32) & (pressure_mmhg < 60)) * pressure_mmhg * dt
        self.exposure_high += ((pressure_mmhg >= 60) & (pressure_mmhg < 100)) * pressure_mmhg * dt
        self.exposure_extreme += (pressure_mmhg >= 100) * pressure_mmhg * dt

    def get_weighted_exposure(self) -> np.ndarray:
        """
        Get weighted cumulative exposure.

        Higher pressure regimes weighted more heavily.

        Returns:
            Weighted exposure index
        """
        return (
            0.01 * self.exposure_low +
            0.1 * self.exposure_moderate +
            0.5 * self.exposure_high +
            2.0 * self.exposure_extreme
        )

    def get_damage_index(self) -> np.ndarray:
        """
        Convert exposure to damage index.

        [HEURISTIC]: Calibrated so 1.0 ≈ visible ulcer formation

        Clinical reference: ~20 days at moderate pressure
        """
        weighted = self.get_weighted_exposure()

        # Normalize so clinically relevant timeframes give ~1.0
        # 20 days at 40 mmHg = 20*86400*40 = 69,120,000 mmHg·s
        normalization = 1e6

        return 1 - np.exp(-weighted / normalization)


# =============================================================================
# INTEGRATED LITERATURE-BASED MODEL
# =============================================================================

class LiteratureBasedPressureUlcerModel:
    """
    Integrated model combining all literature-based improvements.

    Components:
    1. Dual-mechanism damage (deformation + ischemia)
    2. Deep tissue injury modeling
    3. Oxygen transport
    4. Cell population dynamics
    5. Pressure-time tradeoff

    Usage:
        model = LiteratureBasedPressureUlcerModel(grid_shape, cell_size)
        for t in simulation:
            results = model.update(pressure, strain, dt)
    """

    def __init__(self, grid_shape: Tuple[int, int], cell_size_cm: float):
        self.grid_shape = grid_shape
        self.cell_size = cell_size_cm

        # Initialize all submodels
        self.damage_model = DualMechanismDamageModel(grid_shape)
        self.dti_model = DeepTissueInjuryModel(grid_shape)
        self.oxygen_model = OxygenTransportModel(grid_shape, cell_size_cm)
        self.cell_model = CellPopulationModel(grid_shape)
        self.pt_model = PressureTimeTradeoffModel(grid_shape)

        # Simulation time
        self.time = 0

    def update(self, pressure_mmhg: np.ndarray,
               strain: Optional[np.ndarray] = None,
               dt: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Update all model components.

        Args:
            pressure_mmhg: Interface pressure field
            strain: Engineering strain field (computed if not provided)
            dt: Time step (seconds)

        Returns:
            Dictionary of all computed fields
        """
        self.time += dt

        # Compute strain from pressure if not provided
        if strain is None:
            # Simplified: strain ≈ pressure / tissue_modulus
            # Assuming ~50 kPa effective modulus
            pressure_pa = pressure_mmhg / 0.00750062
            strain = pressure_pa / 50000
            strain = np.clip(strain, 0, 1.5)

        # 1. Compute depth-resolved stress
        depth_stress = self.dti_model.compute_depth_resolved_stress(pressure_mmhg)

        # 2. Update oxygen transport
        oxygen = self.oxygen_model.update(pressure_mmhg, dt)

        # 3. Update dual-mechanism damage
        self.damage_model.update(strain, oxygen, dt)

        # 4. Update cell population
        mech_damage = self.damage_model.deformation_damage
        self.cell_model.update(oxygen, mech_damage, dt)

        # 5. Update pressure-time tracking
        self.pt_model.update(pressure_mmhg, dt)

        # Compile results
        return {
            # Primary outputs
            'total_damage': self.damage_model.get_total_damage(),
            'tissue_viability': self.cell_model.get_tissue_viability(),
            'oxygen': oxygen,

            # Damage breakdown
            'deformation_damage': self.damage_model.deformation_damage,
            'ischemic_damage': self.damage_model.ischemic_damage,
            'reperfusion_damage': self.damage_model.reperfusion_damage,

            # Depth-resolved
            'surface_stress': depth_stress['surface'],
            'bone_interface_stress': depth_stress['bone_interface'],
            'dti_risk': self.dti_model.get_dti_initiation_risk(),

            # Cell dynamics
            'inflammation': self.cell_model.get_inflammation_level(),
            'damage_signals': self.cell_model.get_damage_signals(),

            # Exposure tracking
            'pressure_time_damage': self.pt_model.get_damage_index(),
            'time_above_threshold': self.pt_model.time_above_32,
        }

    def get_comprehensive_risk(self) -> np.ndarray:
        """
        Compute comprehensive risk score combining all factors.

        Returns:
            Risk score (0-1)
        """
        # Weight factors [HEURISTIC]
        w_damage = 0.4
        w_viability = 0.3
        w_dti = 0.2
        w_exposure = 0.1

        risk = (
            w_damage * self.damage_model.get_total_damage() +
            w_viability * (1 - self.cell_model.get_tissue_viability()) +
            w_dti * self.dti_model.get_dti_initiation_risk() +
            w_exposure * self.pt_model.get_damage_index()
        )

        return np.clip(risk, 0, 1)


# =============================================================================
# TESTING / DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("Literature-Based Pressure Ulcer Model Components")
    print("=" * 50)

    # Test grid
    grid_shape = (40, 18)
    cell_size = 5  # cm

    # Initialize integrated model
    model = LiteratureBasedPressureUlcerModel(grid_shape, cell_size)

    # Simulate 1 hour with constant 50 mmHg pressure over sacrum
    pressure = np.ones(grid_shape) * 20  # Baseline 20 mmHg
    pressure[18:22, 6:12] = 50  # Sacral region at 50 mmHg

    print(f"\nSimulating 1 hour with 50 mmHg sacral pressure...")
    print(f"Grid: {grid_shape}, Cell size: {cell_size} cm")

    dt = 60  # 1 minute timesteps
    for t in range(60):  # 60 minutes
        results = model.update(pressure, dt=dt)

        if (t + 1) % 15 == 0:
            print(f"\nTime: {t+1} min")
            print(f"  Max total damage: {results['total_damage'].max():.4f}")
            print(f"  Min tissue viability: {results['tissue_viability'].min():.4f}")
            print(f"  Min oxygen: {results['oxygen'].min():.3f}")
            print(f"  Max inflammation: {results['inflammation'].max():.3f}")
            print(f"  Max DTI risk: {results['dti_risk'].max():.3f}")

    print("\n" + "=" * 50)
    print("Final comprehensive risk:")
    risk = model.get_comprehensive_risk()
    print(f"  Max: {risk.max():.4f}")
    print(f"  Mean (loaded area): {risk[18:22, 6:12].mean():.4f}")

    print("\nDamage breakdown (sacral region):")
    breakdown = model.damage_model.get_damage_breakdown()
    for name, field in breakdown.items():
        print(f"  {name}: {field[18:22, 6:12].mean():.4f}")
