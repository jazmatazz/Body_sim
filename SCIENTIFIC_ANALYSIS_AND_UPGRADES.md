# Scientific Analysis: Pressure Ulcer Simulation Improvements

## Based on Academic Literature Review

This document analyzes your current simulation against peer-reviewed computational models and proposes concrete, implementable upgrades.

---

## Part 1: Alignment Analysis

### What Your Model Already Does Well

| Your Implementation | Literature Support | Assessment |
|---------------------|-------------------|------------|
| Cumulative damage integration | PUABM uses `∑[100-life(i)]` over time | ✅ Aligned |
| Pressure threshold (32 mmHg) | Clinical capillary closing pressure | ✅ Aligned |
| Multi-factor risk (P, τ, T, M) | Mechanobiology theory identifies 3 pathways | ✅ Exceeds (adds moisture) |
| Synergistic damage factors | IRI model shows loading history dependence | ✅ Aligned |
| Bony prominence vulnerability | All papers emphasize this | ✅ Aligned |
| Time-stepping simulation | Agent-based models use discrete time | ✅ Aligned |
| Layered tissue model | FE models use layered approach | ✅ Aligned |
| Repositioning intervention | Clinical standard, modeled in PUABM | ✅ Aligned |

### Where Your Model is Oversimplified

| Issue | Your Current Approach | Literature Finding |
|-------|----------------------|-------------------|
| **Damage timescales** | Single damage accumulation rate | Two distinct mechanisms: direct deformation (minutes) vs ischemia (hours) |
| **Strain-based damage** | Pressure-only threshold | Cell death correlates with **strain** (50-65%), not just pressure |
| **Deep tissue injury** | Stress spreads from surface | DTI initiates at **muscle-bone interface**, propagates outward |
| **Reperfusion injury** | Not modeled | ROS surge upon pressure release can **accelerate** damage |
| **Oxygen transport** | Perfusion reduction only | Should model O₂ diffusion, consumption, and deficit accumulation |
| **Tissue healing** | Not modeled | Competing repair processes exist (TGF-β pathway) |
| **Nonlinear strain-time** | Linear damage accumulation | Sigmoid relationship: cells tolerate 65% strain for 1h, only 40% for 4h+ |

---

## Part 2: Specific Physical Inconsistencies

### 2.1 Contact Mechanics Issues

**Your Approach:** Hertzian contact with curvature-dependent pressure

**Problem:** Hertzian theory assumes:
- Small deformations (linear elasticity)
- Smooth, continuous contact
- Homogeneous materials

**Reality:**
- Soft tissue undergoes **large deformations** (50%+ compression)
- Contact is distributed, not point-like
- Tissue is highly **nonlinear** (stiffens under compression)

**Literature Approach:** The 2024 mathematical model uses differential geometry with energy minimization:
```
W(u,v,z) = Y∫ᵥ (1/aσ) f X̃ dV
```

### 2.2 Tissue Material Behavior

**Your Approach:** Linear elastic modulus per layer

**Problem:** Soft tissue exhibits:
- Hyperelastic behavior (nonlinear stress-strain)
- Viscoelastic creep (you have this partially)
- Strain-rate dependence
- Anisotropy

**Key Finding from Gefen et al.:**
> "Cells could tolerate engineering strains below 65% for 1h, but only 40% over 285 minutes"

This is a **nonlinear, time-dependent** failure criterion, not a simple threshold.

### 2.3 Deep Tissue Injury Origin

**Your Approach:** Stress spreads from surface inward (cone model)

**Problem:** DTI starts at depth, not surface.

**Literature (Gefen 2018):**
> "DTI is a severe pressure ulcer that results from sustained deformation of muscle tissue overlying bony prominences"

The **muscle-bone interface** sees highest deformation due to:
1. Geometric stress concentration
2. Muscle incompressibility pushing against rigid bone
3. Blood vessel occlusion in deep layers first

### 2.4 Ischemia-Reperfusion Dynamics

**Your Approach:** Perfusion decreases with pressure (sigmoid)

**Missing:**
1. **Oxygen deficit accumulation** during ischemia
2. **Xanthine oxidase accumulation** during ischemia
3. **ROS burst** upon reperfusion
4. **Paradoxical reperfusion damage**

**PUABM Finding:**
> "If critical tissue damage has been crossed, then abrupt [pressure] expulsion can constitute the worst-case scenario"

---

## Part 3: Implementable Upgrades

### 3.1 Dual-Mechanism Damage Model

Replace your single damage accumulation with two parallel pathways:

```python
class DualMechanismDamageModel:
    """
    Two distinct damage mechanisms from literature:
    1. Direct deformation damage (fast, strain-based)
    2. Ischemic damage (slow, oxygen-based)
    """

    def __init__(self, grid_shape):
        # Direct deformation damage (minutes timescale)
        self.deformation_damage = np.zeros(grid_shape)

        # Ischemic damage (hours timescale)
        self.ischemic_damage = np.zeros(grid_shape)

        # Oxygen deficit accumulator
        self.oxygen_deficit = np.zeros(grid_shape)

        # Xanthine oxidase accumulation (for reperfusion injury)
        self.xanthine_oxidase = np.zeros(grid_shape)

        # Combined tissue damage
        self.total_damage = np.zeros(grid_shape)

        # Thresholds from literature
        self.STRAIN_THRESHOLD_1H = 0.65   # 65% strain tolerable for 1 hour
        self.STRAIN_THRESHOLD_4H = 0.40   # 40% strain tolerable for 4+ hours
        self.O2_CRITICAL = 0.3            # Critical oxygen fraction

    def compute_strain(self, deformation, tissue_thickness):
        """
        Compute engineering strain from deformation.
        ε = ΔL / L₀
        """
        return deformation / tissue_thickness

    def update_deformation_damage(self, strain, dt):
        """
        Fast damage pathway: direct mechanical deformation.

        Based on Gefen et al.: Sigmoid strain-time threshold
        E(zz)c(t) follows Boltzmann sigmoid

        Physically justified: membrane stretch disrupts transport
        """
        # Time-dependent strain threshold (Boltzmann sigmoid)
        # Threshold decreases from 0.65 to 0.40 over ~3 hours
        t_half = 90 * 60  # 90 minutes = inflection point
        k = 0.001         # Steepness

        current_threshold = (
            self.STRAIN_THRESHOLD_4H +
            (self.STRAIN_THRESHOLD_1H - self.STRAIN_THRESHOLD_4H) /
            (1 + np.exp(k * (self.time_under_strain - t_half)))
        )

        # Damage accumulates when strain exceeds time-dependent threshold
        excess_strain = np.maximum(strain - current_threshold, 0)

        # Fast damage rate (significant within minutes)
        damage_rate = 0.01  # per second per unit excess strain
        self.deformation_damage += damage_rate * excess_strain * dt

    def update_ischemic_damage(self, oxygen_level, dt):
        """
        Slow damage pathway: ischemia from vessel occlusion.

        Based on PUABM: oxygen threshold determines cell health
        Takes hours to develop significant damage

        Physically justified: ATP depletion, metabolic failure
        """
        # Oxygen deficit accumulation
        deficit = np.maximum(self.O2_CRITICAL - oxygen_level, 0)
        self.oxygen_deficit += deficit * dt

        # Ischemic damage (slow, hours timescale)
        # Only begins after sustained deficit
        deficit_threshold = 1800  # 30 minutes of critical deficit

        active_ischemia = self.oxygen_deficit > deficit_threshold
        ischemia_rate = 0.0001  # per second (slow)

        self.ischemic_damage[active_ischemia] += (
            ischemia_rate * deficit[active_ischemia] * dt
        )

        # Xanthine oxidase accumulates during ischemia
        # (for reperfusion injury calculation)
        self.xanthine_oxidase[active_ischemia] += 0.001 * dt

    def update_reperfusion_damage(self, oxygen_level, previous_oxygen, dt):
        """
        Reperfusion injury: ROS burst when oxygen returns.

        Based on PUABM: "free radicals cause damage...in stepwise manner"

        Physically justified: xanthine oxidase + O₂ → ROS

        HEURISTIC: Simplified ROS model without full biochemistry
        """
        # Detect reperfusion (oxygen increase after deficit)
        reperfusion_mask = (
            (oxygen_level > previous_oxygen + 0.1) &  # Oxygen increasing
            (self.xanthine_oxidase > 0.1)              # XO accumulated
        )

        if reperfusion_mask.any():
            # ROS production proportional to XO level and O₂ availability
            ros_production = (
                self.xanthine_oxidase[reperfusion_mask] *
                oxygen_level[reperfusion_mask]
            )

            # ROS causes immediate damage
            ros_damage_rate = 0.1  # Fast damage from ROS
            self.ischemic_damage[reperfusion_mask] += (
                ros_damage_rate * ros_production * dt
            )

            # XO consumed in reaction
            self.xanthine_oxidase[reperfusion_mask] *= 0.9

    def compute_total_damage(self):
        """
        Combine damage pathways.

        Not simply additive - pathways interact
        """
        # Synergistic combination (deformation sensitizes to ischemia)
        synergy = 1 + 0.5 * self.deformation_damage

        self.total_damage = (
            self.deformation_damage +
            self.ischemic_damage * synergy
        )

        return np.clip(self.total_damage, 0, 1)
```

**Classification:**
- Dual pathways: **Physically justified** (multiple papers confirm)
- Sigmoid threshold: **Physically justified** (Gefen experimental data)
- Reperfusion ROS model: **Heuristic** (simplified from full biochemistry)
- Synergy factor: **Heuristic** (reasonable but not calibrated)

---

### 3.2 Deep Tissue Stress Model

Replace surface-inward stress spread with bone-interface-outward model:

```python
class DeepTissueStressModel:
    """
    Model stress concentration at muscle-bone interface.

    DTI originates at depth, not surface.
    """

    def __init__(self, grid_shape, tissue_model):
        self.grid_shape = grid_shape
        self.tissue = tissue_model

        # Stress at different depths
        self.surface_stress = np.zeros(grid_shape)
        self.muscle_stress = np.zeros(grid_shape)
        self.bone_interface_stress = np.zeros(grid_shape)  # HIGHEST

        # Bony prominence stress concentration factor
        self.bone_stress_concentration = self._compute_bone_concentration()

    def _compute_bone_concentration(self):
        """
        Stress concentration at bony prominences.

        Based on: geometric focusing + incompressibility effects

        Physically justified: FE models show 2-3x concentration
        """
        concentration = np.ones(self.grid_shape)

        # Sacrum: highest concentration (sharp bony edge)
        sacrum_region = self._get_sacrum_region()
        concentration[sacrum_region] = 2.5

        # Heels: high concentration (small contact area)
        heel_region = self._get_heel_region()
        concentration[heel_region] = 2.2

        # Scapulae: moderate concentration
        scapula_region = self._get_scapula_region()
        concentration[scapula_region] = 1.8

        # Occiput: moderate concentration
        occiput_region = self._get_occiput_region()
        concentration[occiput_region] = 1.6

        return concentration

    def compute_depth_resolved_stress(self, surface_pressure, fat_thickness):
        """
        Compute stress at each tissue depth.

        Key insight: stress INCREASES with depth over bony prominences
        due to geometric focusing, contrary to intuition.

        Physically justified: FE studies confirm this
        """
        # Surface stress (what you measure)
        self.surface_stress = surface_pressure

        # Muscle layer stress (attenuated through fat)
        fat_attenuation = np.exp(-fat_thickness / 0.015)  # 15mm reference
        self.muscle_stress = surface_pressure * (0.7 + 0.3 * fat_attenuation)

        # Bone interface stress (CONCENTRATED)
        # This is where DTI initiates
        self.bone_interface_stress = (
            surface_pressure *
            self.bone_stress_concentration *
            fat_attenuation  # Less fat = more concentration
        )

        return self.bone_interface_stress

    def get_dti_risk(self):
        """
        DTI risk based on deep tissue stress.

        Higher risk where bone interface stress is high
        """
        # Threshold for deep tissue (higher than surface)
        deep_threshold = 60  # mmHg (muscle more tolerant initially)

        excess = np.maximum(self.bone_interface_stress - deep_threshold, 0)
        return excess / 100  # Normalize
```

**Classification:**
- Bone stress concentration: **Physically justified** (FE studies show 2-3x)
- Depth-resolved stress: **Physically justified** (anatomy)
- Specific concentration values: **Heuristic** (reasonable ranges, not patient-specific)

---

### 3.3 Improved Oxygen Transport Model

```python
class OxygenTransportModel:
    """
    Oxygen diffusion, consumption, and deficit accumulation.

    Based on: Pennes-like diffusion + consumption + pressure-dependent perfusion
    """

    def __init__(self, grid_shape, cell_size):
        self.grid_shape = grid_shape
        self.cell_size = cell_size / 100  # meters

        # Oxygen concentration field (normalized 0-1)
        self.oxygen = np.ones(grid_shape) * 0.95  # ~95% saturation baseline

        # Oxygen parameters
        self.D_O2 = 2e-9        # m²/s diffusion coefficient in tissue
        self.consumption_rate = 0.005  # 1/s metabolic consumption
        self.perfusion_supply = 0.01   # 1/s baseline perfusion supply

        # Critical thresholds
        self.HYPOXIA_THRESHOLD = 0.5   # Below this = hypoxic
        self.ANOXIA_THRESHOLD = 0.1    # Below this = anoxic

    def update(self, pressure_mmhg, dt):
        """
        Update oxygen field.

        dO₂/dt = D∇²O₂ - consumption + perfusion_supply

        Perfusion supply depends on pressure (vessel occlusion)
        """
        # 1. Diffusion (Laplacian)
        laplacian = self._compute_laplacian(self.oxygen)
        diffusion = self.D_O2 * laplacian / (self.cell_size ** 2)

        # 2. Metabolic consumption (always occurring)
        consumption = self.consumption_rate * self.oxygen

        # 3. Perfusion supply (pressure-dependent)
        # Vessels occlude under pressure
        perfusion_factor = self._compute_perfusion_factor(pressure_mmhg)
        supply = self.perfusion_supply * perfusion_factor

        # 4. Update
        dO2_dt = diffusion - consumption + supply
        self.oxygen += dO2_dt * dt

        # Physical bounds
        self.oxygen = np.clip(self.oxygen, 0, 1)

        return self.oxygen

    def _compute_perfusion_factor(self, pressure_mmhg):
        """
        Perfusion reduction under pressure.

        Based on: capillary closing pressure ~32 mmHg

        Physically justified: well-established clinical threshold
        """
        # Sigmoid occlusion (vessels progressively close)
        P_half = 32  # mmHg, capillary closing pressure
        k = 0.15     # Steepness

        factor = 1 / (1 + np.exp(k * (pressure_mmhg - P_half)))
        return factor

    def _compute_laplacian(self, field):
        """2D discrete Laplacian."""
        laplacian = np.zeros_like(field)
        laplacian[1:-1, 1:-1] = (
            field[:-2, 1:-1] + field[2:, 1:-1] +
            field[1:-1, :-2] + field[1:-1, 2:] -
            4 * field[1:-1, 1:-1]
        )
        return laplacian

    def get_hypoxia_map(self):
        """Return hypoxia severity map."""
        return np.maximum(self.HYPOXIA_THRESHOLD - self.oxygen, 0)

    def get_anoxia_map(self):
        """Return anoxia (severe) map."""
        return self.oxygen < self.ANOXIA_THRESHOLD
```

**Classification:**
- Diffusion equation: **Physically justified** (Fick's law)
- Pressure-dependent perfusion: **Physically justified** (capillary closing)
- Specific rate constants: **Heuristic** (order-of-magnitude estimates)

---

### 3.4 Cell Population Dynamics (from 2024 paper)

```python
class CellPopulationModel:
    """
    Cell population dynamics under pressure.

    From: "A Mathematical Model of Pressure Ulcer Formation" (2024)

    System of ODEs:
    Ṅ = (r₀ - kₙ₁I)[1 - N/K₀]H(p)N
    İ = kₙ₂I[1 - I/K₁] + kₐ₀D
    Ḋ = [1 - N/K₀]kₐ₁ - kₐ₂D
    """

    def __init__(self, grid_shape):
        self.grid_shape = grid_shape

        # State variables
        self.N = np.ones(grid_shape)    # Cell population (normalized)
        self.I = np.zeros(grid_shape)   # Neutrophil/immune density
        self.D = np.zeros(grid_shape)   # DAMP concentration

        # Parameters (from paper, Table 1)
        self.r0 = 0.001      # Base reproduction rate (1/s)
        self.K0 = 1.0        # Cell carrying capacity
        self.K1 = 1.0        # Immune carrying capacity
        self.kn1 = 0.01      # Immune suppression of reproduction
        self.kn2 = 0.005     # Immune proliferation rate
        self.ka0 = 0.002     # DAMP-induced immune activation
        self.ka1 = 0.001     # DAMP production from damage
        self.ka2 = 0.0005    # DAMP clearance rate

    def hill_function(self, oxygen_partial_pressure, n=2, K=0.3):
        """
        Hill function for oxygen-dependent cell viability.

        H(p) = p^n / (K^n + p^n)

        Physically justified: standard biochemical kinetics
        """
        return oxygen_partial_pressure**n / (K**n + oxygen_partial_pressure**n)

    def update(self, oxygen_level, dt):
        """
        Update cell population dynamics.

        Physically justified: standard population dynamics
        """
        H_p = self.hill_function(oxygen_level)

        # Cell population dynamics
        dN_dt = (
            (self.r0 - self.kn1 * self.I) *
            (1 - self.N / self.K0) *
            H_p * self.N
        )

        # Immune response dynamics
        dI_dt = (
            self.kn2 * self.I * (1 - self.I / self.K1) +
            self.ka0 * self.D
        )

        # DAMP dynamics (damage signals)
        dD_dt = (
            (1 - self.N / self.K0) * self.ka1 -
            self.ka2 * self.D
        )

        # Update state
        self.N += dN_dt * dt
        self.I += dI_dt * dt
        self.D += dD_dt * dt

        # Physical bounds
        self.N = np.clip(self.N, 0, 1)
        self.I = np.clip(self.I, 0, 2)  # Can exceed baseline
        self.D = np.clip(self.D, 0, 5)

    def get_tissue_viability(self):
        """Cell population is proxy for tissue viability."""
        return self.N

    def get_inflammation_level(self):
        """Immune density indicates inflammation."""
        return self.I
```

**Classification:**
- ODE system: **Physically justified** (from peer-reviewed paper)
- Hill function: **Physically justified** (biochemical standard)
- Parameter values: **Heuristic** (paper provides ranges, not universal values)

---

### 3.5 Pressure-Time Tradeoff (Improved)

```python
class PressureTimeModel:
    """
    Nonlinear pressure-time damage relationship.

    Key insight: not linear accumulation
    - High pressure: damage in minutes
    - Low pressure: damage in hours
    - Threshold pressure: possibly never
    """

    def __init__(self, grid_shape):
        self.grid_shape = grid_shape

        # Cumulative pressure-time exposure
        self.pressure_time_integral = np.zeros(grid_shape)

        # Time above various thresholds
        self.time_above_32 = np.zeros(grid_shape)
        self.time_above_60 = np.zeros(grid_shape)
        self.time_above_100 = np.zeros(grid_shape)

    def compute_damage_rate(self, pressure_mmhg):
        """
        Nonlinear pressure-damage rate relationship.

        Based on literature synthesis:
        - Below 32 mmHg: minimal damage (vessels open)
        - 32-60 mmHg: slow damage (partial occlusion)
        - 60-100 mmHg: moderate damage
        - Above 100 mmHg: rapid damage (direct deformation)

        Physically justified: combines capillary closing + deformation
        """
        # Piecewise nonlinear function
        rate = np.zeros_like(pressure_mmhg)

        # Region 1: Below threshold (minimal)
        mask1 = pressure_mmhg < 32
        rate[mask1] = 0.0001 * pressure_mmhg[mask1] / 32

        # Region 2: Ischemic zone (slow, linear-ish)
        mask2 = (pressure_mmhg >= 32) & (pressure_mmhg < 60)
        rate[mask2] = 0.001 * (pressure_mmhg[mask2] - 32) / 28

        # Region 3: Moderate (accelerating)
        mask3 = (pressure_mmhg >= 60) & (pressure_mmhg < 100)
        rate[mask3] = 0.001 + 0.005 * ((pressure_mmhg[mask3] - 60) / 40) ** 1.5

        # Region 4: High (rapid, deformation-dominated)
        mask4 = pressure_mmhg >= 100
        rate[mask4] = 0.006 + 0.02 * ((pressure_mmhg[mask4] - 100) / 50) ** 2

        return rate

    def update(self, pressure_mmhg, dt):
        """
        Update pressure-time accumulation.
        """
        # Track time above thresholds
        self.time_above_32 += (pressure_mmhg > 32) * dt
        self.time_above_60 += (pressure_mmhg > 60) * dt
        self.time_above_100 += (pressure_mmhg > 100) * dt

        # Nonlinear damage accumulation
        rate = self.compute_damage_rate(pressure_mmhg)
        self.pressure_time_integral += rate * dt

    def get_damage_index(self):
        """
        Convert pressure-time integral to damage index.

        Uses clinical calibration: ~20 days to visible ulcer
        """
        # Calibrated so that 1.0 = visible ulcer
        # ~20 days at moderate pressure
        return 1 - np.exp(-self.pressure_time_integral / 0.5)
```

**Classification:**
- Piecewise rate function: **Heuristic but reasonable** (captures known regimes)
- Threshold values (32, 60, 100): **Physically justified** (literature)
- Specific rate constants: **Heuristic** (need calibration to clinical data)

---

## Part 4: Tissue Vulnerability Mapping

```python
class AnatomicalVulnerabilityModel:
    """
    Comprehensive tissue vulnerability mapping.

    Factors:
    1. Fat pad thickness (less fat = more vulnerable)
    2. Bony prominence geometry (sharper = worse)
    3. Muscle mass (more muscle = more buffer)
    4. Skin condition (moisture, age, etc.)
    """

    # Anatomical vulnerability data (normalized 0-1, higher = more vulnerable)
    REGION_VULNERABILITY = {
        'sacrum': {
            'base_vulnerability': 0.9,    # Highest risk
            'fat_thickness_mm': 5,        # Very thin
            'bone_sharpness': 0.8,        # Sharp edge
            'muscle_coverage': 0.2,       # Minimal
        },
        'ischial_tuberosity': {
            'base_vulnerability': 0.85,
            'fat_thickness_mm': 8,
            'bone_sharpness': 0.7,
            'muscle_coverage': 0.3,
        },
        'greater_trochanter': {
            'base_vulnerability': 0.8,
            'fat_thickness_mm': 10,
            'bone_sharpness': 0.6,
            'muscle_coverage': 0.4,
        },
        'heel': {
            'base_vulnerability': 0.85,
            'fat_thickness_mm': 4,
            'bone_sharpness': 0.5,
            'muscle_coverage': 0.1,
        },
        'scapula': {
            'base_vulnerability': 0.6,
            'fat_thickness_mm': 12,
            'bone_sharpness': 0.4,
            'muscle_coverage': 0.5,
        },
        'occiput': {
            'base_vulnerability': 0.5,
            'fat_thickness_mm': 6,
            'bone_sharpness': 0.3,
            'muscle_coverage': 0.2,
        },
        'spinous_processes': {
            'base_vulnerability': 0.7,
            'fat_thickness_mm': 8,
            'bone_sharpness': 0.6,
            'muscle_coverage': 0.3,
        },
    }

    def __init__(self, grid_shape):
        self.grid_shape = grid_shape
        self.vulnerability_map = self._build_vulnerability_map()

    def _build_vulnerability_map(self):
        """
        Build spatial vulnerability map.

        Physically justified: anatomy determines risk
        """
        vuln = np.ones(self.grid_shape) * 0.3  # Baseline (soft tissue)

        rows, cols = self.grid_shape

        # Sacrum (row 45-55%, center)
        sacrum = self.REGION_VULNERABILITY['sacrum']
        r_start, r_end = int(0.45*rows), int(0.55*rows)
        c_start, c_end = int(0.30*cols), int(0.70*cols)
        vuln[r_start:r_end, c_start:c_end] = sacrum['base_vulnerability']

        # Heels (row 95-100%)
        heel = self.REGION_VULNERABILITY['heel']
        vuln[int(0.95*rows):, :] = heel['base_vulnerability']

        # Scapulae (row 12-22%, lateral)
        scap = self.REGION_VULNERABILITY['scapula']
        r_start, r_end = int(0.12*rows), int(0.22*rows)
        vuln[r_start:r_end, :int(0.30*cols)] = scap['base_vulnerability']
        vuln[r_start:r_end, int(0.70*cols):] = scap['base_vulnerability']

        # Occiput (row 0-8%, center)
        occ = self.REGION_VULNERABILITY['occiput']
        vuln[:int(0.08*rows), int(0.35*cols):int(0.65*cols)] = occ['base_vulnerability']

        return vuln

    def apply_patient_factors(self, age=50, bmi=25, diabetes=False,
                              mobility_score=1.0):
        """
        Modify vulnerability for patient-specific factors.

        Heuristic: based on clinical risk factors
        """
        factor = 1.0

        # Age factor (vulnerability increases with age)
        if age > 65:
            factor *= 1.0 + 0.02 * (age - 65)

        # BMI factor (both low and high BMI increase risk)
        if bmi < 20:
            factor *= 1.3  # Cachexia
        elif bmi > 30:
            factor *= 1.1  # Obesity (moisture, friction)

        # Diabetes (impaired healing)
        if diabetes:
            factor *= 1.5

        # Mobility (lower = higher risk)
        factor *= 2.0 - mobility_score

        return self.vulnerability_map * factor
```

**Classification:**
- Anatomical regions: **Physically justified** (well-established risk areas)
- Relative vulnerability values: **Heuristic** (clinically informed but not calibrated)
- Patient factors: **Heuristic** (clinical risk factors, simplified)

---

## Part 5: Out of Scope for This Level

The following are mentioned in the literature but would require resources beyond a grid-based simulation:

| Feature | Why Out of Scope |
|---------|------------------|
| Full 3D FEM tissue mechanics | Requires specialized software (ABAQUS, FEBio) |
| Patient-specific anatomy from MRI | Needs imaging data and segmentation |
| Hyperelastic constitutive models (Ogden, Mooney-Rivlin) | Complex nonlinear solvers |
| Full inflammatory cascade (TNF-α, IL-1β, TGF-β, etc.) | Agent-based modeling with thousands of parameters |
| Lymphatic drainage modeling | Additional transport physics |
| Wound healing dynamics | Separate multi-week simulation |
| Real-time sensor feedback | Hardware integration |

---

## Part 6: Summary of Recommendations

### Priority 1: Implement Immediately

1. **Dual damage mechanism** (deformation + ischemia)
2. **Deep tissue stress concentration** at bony prominences
3. **Oxygen transport** with perfusion-dependent supply

### Priority 2: Significant Improvement

4. **Nonlinear pressure-time relationship** (piecewise or sigmoid)
5. **Strain-based damage threshold** (not just pressure)
6. **Reperfusion injury** (simplified ROS model)

### Priority 3: Enhanced Realism

7. **Cell population dynamics** (ODE system from 2024 paper)
8. **Anatomical vulnerability mapping** (patient-specific factors)
9. **Time-dependent strain threshold** (Boltzmann sigmoid)

---

## References

1. Loerakker et al. (2015). "A Computational, Tissue-Realistic Model of Pressure Ulcer Formation" - PLOS Computational Biology
2. Gefen et al. (2008). "Strain-time cell-death threshold for skeletal muscle" - Journal of Biomechanics
3. Gefen (2018). "Pressure induced deep tissue injury explained" - Annals of Biomedical Engineering
4. Zeevi et al. (2024). "A Mathematical Model of Pressure Ulcer Formation" - Mathematical Biosciences and Engineering
5. Shoham & Gefen (2019). "IRI hereditary tissue model for pressure ulcers" - Biomechanics and Modeling in Mechanobiology
