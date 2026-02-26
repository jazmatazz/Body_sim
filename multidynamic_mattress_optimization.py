#!/usr/bin/env python3
"""
Multi-Dynamic Air Mattress Optimization
========================================
Optimizes air mattress configuration for pressure ulcer prevention:
- Cell grid sizes (finding optimal cell dimensions)
- Movement patterns (alternating, wave, zone-based, etc.)
- Cycle periods (inflation/deflation timing)

Goal: Find the optimal configuration that minimizes pressure ulcer risk
for quadriplegic patients without requiring nurse intervention.

Uses SMPL body model for anatomically accurate pressure distribution.
"""

import numpy as np
import math
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS - EVIDENCE-BASED VALUES
# =============================================================================
# See EVIDENCE_BASE.md for full citations and methodology

GRAVITY = 9.81  # m/s²
MMHG_PER_PASCAL = 0.00750062

# Capillary closing pressure thresholds
# Source: Landis EM. Heart 1930;15:209-228 (verified by Mahler et al. Clin Sci 1988)
# Note: NPIAP/EPUAP 2019 guidelines state this is NOT a universal threshold
CAPILLARY_CLOSING_PRESSURE_ARTERIOLAR = 32  # mmHg (range: 18-38, Mahler 1988: 37.7±3.7)
CAPILLARY_CLOSING_PRESSURE_VENULAR = 12     # mmHg (range: 6-18)
CAPILLARY_CLOSING_PRESSURE = 32  # Using arteriolar value as conservative threshold

# Body dimensions (standard adult)
BODY_LENGTH = 180  # cm
BODY_WIDTH = 45    # cm (shoulder width approximation for contact)

# Mattress dimensions
MATTRESS_LENGTH = 200  # cm
MATTRESS_WIDTH = 90    # cm

# Bed incline (head of bed elevation)
WAIST_HINGE_POSITION = 0.45  # 45% from head (waist/hip area)
DEFAULT_INCLINE_ANGLE = 30   # degrees - typical for quadriplegic patients

# Quadriplegic patient factors
# Sources: Kowalczyk et al. J Spinal Cord Med 2013 (PMID: 23809593)
QUADRIPLEGIC_FACTORS = {
    'muscle_atrophy_factor': 0.4,       # Reduced muscle mass in SCI
    'baseline_perfusion_factor': 0.7,   # Reduced tissue perfusion
    'skin_fragility': 1.3,              # Thinner skin at pressure points (sacrum: 2.1mm vs 3.2mm)
    'damage_rate_multiplier': 1.5,      # Faster damage accumulation
}

# Anatomical weight distribution when supine (% of body weight per region)
# Based on: Defloor T. J Clin Nurs 2000; Hobson DA. Prosthet Orthot Int 1992
# With 30° incline, weight shifts significantly to sacrum (Moore 2011)
# Tuned to produce clinically realistic pressures (100-300 mmHg at bony prominences)
ANATOMICAL_WEIGHT_DISTRIBUTION = {
    # Region: (z_start, z_end, base_weight_fraction, incline_multiplier)
    # incline_multiplier: how much weight increases with 30° incline
    'occiput': (0.00, 0.10, 0.04, 0.6),      # Head: reduced with incline
    'scapula': (0.10, 0.25, 0.08, 0.7),      # Upper back: reduced with incline
    'back': (0.25, 0.40, 0.12, 0.8),         # Mid back: slightly reduced
    'sacrum': (0.40, 0.55, 0.14, 1.6),       # Sacrum: highest with incline (CRITICAL)
    'buttock': (0.55, 0.65, 0.14, 1.3),      # Buttocks: increased with incline
    'thigh': (0.65, 0.80, 0.16, 1.0),        # Thighs: unchanged
    'calf': (0.80, 0.92, 0.14, 1.0),         # Calves: unchanged
    'heel': (0.92, 1.00, 0.18, 1.2),         # Heels: slight increase with incline
}

# =============================================================================
# EVIDENCE-BASED TISSUE THICKNESS VALUES (mm)
# =============================================================================
# Source: Kowalczyk et al. J Spinal Cord Med 2013;36(3):239-243. PMID: 23809593
# Measured by high-frequency ultrasound in n=32 SCI patients vs n=34 controls

TISSUE_THICKNESS_SCI = {
    # Skin thickness at bony prominences (SCI patients)
    'sacrum': 2.1,           # mm (SD: 0.9) - significantly thinner than controls
    'ischium': 2.2,          # mm (SD: 0.6)
    'trochanter': 1.8,       # mm (SD: 0.4)
    # Derived/estimated values
    'heel': 16.0,            # mm - heel pad (PMID: 3886923, slightly less than healthy 16.6mm)
    'scapula': 2.5,          # mm - estimated, limited data available
    'occiput': 3.0,          # mm - estimated, limited data available
}

TISSUE_THICKNESS_HEALTHY = {
    # Skin thickness at bony prominences (healthy controls)
    'sacrum': 3.2,           # mm (SD: 0.5)
    'ischium': 2.6,          # mm (SD: 0.5)
    'trochanter': 1.9,       # mm (SD: 0.5)
    # Additional values
    'heel': 16.6,            # mm - heel pad (PMID: 3886923)
    'scapula': 3.0,          # mm - estimated
    'occiput': 3.5,          # mm - estimated
}

# Bulk tissue thickness over ischium (total soft tissue, not just skin)
# Source: Sonenblum et al. J Tissue Viability 2018. PMID: 31353285
BULK_TISSUE_ISCHIUM_RANGE = (5.6, 32.1)  # mm, varies by individual/population

# Gluteal tissue thickness
# Source: Systematic review PMC10272691
GLUTEAL_MUSCLE_THICKNESS = 38.1    # mm (SD: 2.1) - ventrogluteal site
GLUTEAL_SUBCUTANEOUS = 19.9        # mm (SD: 2.5)

# =============================================================================
# STRAIN THRESHOLDS FOR TISSUE DAMAGE
# =============================================================================
# Source: Gefen et al. J Biomech 2008;41(9):2003-2012. PMID: 18501912

STRAIN_THRESHOLDS = {
    '1_hour': 0.65,          # 65% engineering strain - 95% cells survive
    '4_hours': 0.40,         # 40% strain - sustainable for ~285 minutes
    '6_hours': 0.35,         # 30-50% strain causes <50% cell death
}

# Peak strains measured during sitting (MRI + FE analysis)
# Source: Linder-Ganz et al. J Biomech 2007;40(7):1443-1454. PMID: 16920122
PEAK_SITTING_STRAIN_GLUTEUS = 0.74  # 74% ± 7% principal compressive strain
PEAK_SITTING_STRESS_GLUTEUS = 32.0  # kPa ± 9 kPa

# =============================================================================
# TISSUE DEFORMATION MODEL
# =============================================================================

class TissueDeformationModel:
    """
    Models realistic soft tissue deformation under pressure.

    Evidence-Based Model:
    - Tissue compresses non-linearly under load
    - Different body regions have different stiffness based on tissue composition
    - Tissue exhibits viscoelastic creep over time
    - Compressed tissue spreads, increasing contact area

    Key Citations:
    - Gefen A, et al. J Biomech 2008;41(9):2003-2012. PMID: 18501912
      (Strain-time cell-death thresholds: <65% for 1hr, <40% for 4+ hrs)
    - Ceelen KK, et al. J Biomech 2008;41(16):3399-3404. PMID: 19010470
      (Compression-damage relationship, shear strain as predictor)
    - Linder-Ganz E, et al. J Biomech 2007;40(7):1443-1454. PMID: 16920122
      (Peak gluteal strain during sitting: 74±7%)
    - Gefen A, Weihs D. Ann Biomed Eng 2016;44(4):1250-1261. PMID: 25480479
      (Two DTI mechanisms: direct deformation vs ischemic)

    Note: Stiffness values are modeled estimates informed by published FE studies.
    Direct measurement data is limited; values calibrated to match clinical observations.
    """

    # Tissue mechanical properties by body part
    # Stiffness in kPa (higher = stiffer = less compression)
    # Note: These are MODEL ESTIMATES calibrated to FE study results
    # Direct experimental measurements vary significantly (213±101% shear strain, PMID: 30201252)
    TISSUE_STIFFNESS = {
        # Bony prominences - very stiff (thin tissue over bone)
        'sacrum': 80.0,       # kPa - 2.1mm skin (PMID:23809593), minimal fat
        'coccyx': 100.0,      # kPa
        'ischial': 60.0,      # kPa - 2.2mm skin
        'heel': 50.0,         # kPa - 16.6mm fat pad (PMID:3886923)
        'malleolus': 90.0,    # kPa
        'scapula': 45.0,      # kPa
        'occiput': 70.0,      # kPa
        'spinous': 85.0,      # kPa
        'elbow': 65.0,        # kPa
        # Well-padded areas - softer (more fat/muscle)
        'buttock': 8.0,       # kPa - 38mm muscle + 20mm fat (PMC10272691)
        'thigh': 12.0,        # kPa
        'calf': 15.0,         # kPa
        'back': 20.0,         # kPa
        'shoulder': 25.0,     # kPa
    }

    # Maximum compression ratio by tissue type (0-1)
    # Based on strain tolerance data from PMID: 18501912
    # At 50% strain, cells begin to die within hours
    MAX_COMPRESSION_RATIO = {
        'sacrum': 0.85,       # Only 15% compression (thin tissue)
        'coccyx': 0.90,
        'ischial': 0.75,
        'heel': 0.60,         # Fat pad can compress significantly
        'malleolus': 0.85,
        'scapula': 0.70,
        'occiput': 0.80,
        'spinous': 0.90,
        'elbow': 0.80,
        'buttock': 0.35,      # Can compress to 35% (65% max strain)
        'thigh': 0.40,
        'calf': 0.45,
        'back': 0.50,
        'shoulder': 0.55,
    }

    # Creep time constants (seconds to reach 63% of final deformation)
    # Note: Limited direct evidence; estimated from viscoelastic tissue models
    CREEP_TIME_CONSTANT = {
        'sacrum': 120,        # Fast creep (thin tissue)
        'coccyx': 100,
        'ischial': 150,
        'heel': 180,
        'malleolus': 120,
        'scapula': 200,
        'occiput': 150,
        'spinous': 120,
        'elbow': 150,
        'buttock': 600,       # Slow creep (thick tissue)
        'thigh': 500,
        'calf': 400,
        'back': 350,
        'shoulder': 300,
    }

    def __init__(self):
        """Initialize deformation tracking arrays."""
        self.current_compression = None  # Current compression at each vertex
        self.creep_compression = None    # Additional creep deformation
        self.time_under_load = None      # Time each vertex has been loaded
        self.contact_area_multiplier = None  # How much contact area has spread

    def initialize(self, n_vertices: int):
        """Initialize arrays for tracking deformation state."""
        self.current_compression = np.zeros(n_vertices)
        self.creep_compression = np.zeros(n_vertices)
        self.time_under_load = np.zeros(n_vertices)
        self.contact_area_multiplier = np.ones(n_vertices)
        self.vertex_in_contact = np.zeros(n_vertices, dtype=bool)

    def calculate_compression(self, vertex_idx: int, body_part: str,
                             local_pressure_pa: float, dt: float) -> float:
        """
        Calculate tissue compression for a single vertex.

        Uses a non-linear stress-strain relationship:
        - Low pressure: tissue compresses easily (toe region)
        - Medium pressure: linear elastic region
        - High pressure: stiffening (tissue bottoms out)

        Args:
            vertex_idx: Index of the vertex
            body_part: Body part label for this vertex
            local_pressure_pa: Local pressure in Pascals
            dt: Time step in seconds

        Returns:
            Compression depth in cm
        """
        # Get tissue properties
        stiffness = self.TISSUE_STIFFNESS.get(body_part, 20.0)  # kPa
        max_ratio = self.MAX_COMPRESSION_RATIO.get(body_part, 0.5)
        creep_tau = self.CREEP_TIME_CONSTANT.get(body_part, 300)

        # Convert pressure to kPa
        pressure_kpa = local_pressure_pa / 1000.0

        # Non-linear compression model (Ogden-type hyperelastic simplified)
        # Compression increases with pressure but stiffens at high compression
        if pressure_kpa <= 0:
            instant_compression_ratio = 0.0
        else:
            # Normalized pressure (relative to stiffness)
            normalized_pressure = pressure_kpa / stiffness

            # Non-linear response: soft at low pressure, stiff at high
            # Using tanh-like saturation curve
            instant_compression_ratio = (1 - max_ratio) * (1 - np.exp(-normalized_pressure * 2))
            instant_compression_ratio = min(instant_compression_ratio, 1 - max_ratio)

        # Viscoelastic creep: tissue continues to deform under sustained load
        # Creep adds additional compression over time
        if pressure_kpa > 0:
            self.time_under_load[vertex_idx] += dt

            # Exponential creep approach to final value
            # Creep adds up to 30% more compression over time
            creep_factor = 1 - np.exp(-self.time_under_load[vertex_idx] / creep_tau)
            additional_creep = instant_compression_ratio * 0.3 * creep_factor

            self.creep_compression[vertex_idx] = additional_creep
        else:
            # Recovery when pressure removed (slower than loading)
            recovery_rate = 0.1  # 10% recovery per second
            self.time_under_load[vertex_idx] = max(0, self.time_under_load[vertex_idx] - dt * recovery_rate)
            self.creep_compression[vertex_idx] *= (1 - recovery_rate * dt)

        # Total compression ratio
        total_compression_ratio = instant_compression_ratio + self.creep_compression[vertex_idx]
        total_compression_ratio = min(total_compression_ratio, 1 - max_ratio)

        self.current_compression[vertex_idx] = total_compression_ratio

        # Update contact area multiplier
        # As tissue compresses, it spreads out, increasing contact area
        # Poisson's ratio effect: volume approximately conserved
        if total_compression_ratio > 0:
            # Area increases as 1/compression (volume conservation)
            # But limited by tissue constraints
            self.contact_area_multiplier[vertex_idx] = 1.0 + total_compression_ratio * 0.5
        else:
            self.contact_area_multiplier[vertex_idx] = 1.0

        return total_compression_ratio

    def get_deformed_position(self, original_y: float, compression_ratio: float,
                              tissue_thickness_mm: float) -> float:
        """
        Calculate the deformed Y position (height) of a vertex.

        Args:
            original_y: Original height above mattress surface (cm)
            compression_ratio: How much tissue has compressed (0-1)
            tissue_thickness_mm: Tissue thickness at this point (mm)

        Returns:
            New Y position after deformation (cm)
        """
        tissue_thickness_cm = tissue_thickness_mm / 10.0

        # Compression reduces height
        compression_depth = tissue_thickness_cm * compression_ratio

        # But can't compress more than original height above surface
        compression_depth = min(compression_depth, max(0, original_y))

        return original_y - compression_depth

    def calculate_effective_contact_pressure(self, base_pressure: float,
                                             compression_ratio: float,
                                             contact_area_mult: float) -> float:
        """
        Calculate effective pressure accounting for tissue deformation.

        When tissue compresses:
        1. Contact area increases (spreads out)
        2. But internal tissue stress concentrates near bone
        3. Net effect depends on tissue type

        Args:
            base_pressure: Pressure without deformation (mmHg)
            compression_ratio: Current compression (0-1)
            contact_area_mult: Contact area multiplier from spreading

        Returns:
            Effective interface pressure (mmHg)
        """
        # Contact area increase reduces surface pressure
        surface_pressure = base_pressure / contact_area_mult

        # But internal stress concentration increases damage potential
        # At high compression, tissue stress concentrates near bone
        if compression_ratio > 0.3:
            # Stress concentration factor increases with compression
            stress_concentration = 1.0 + (compression_ratio - 0.3) * 1.5
        else:
            stress_concentration = 1.0

        # Effective pressure for damage calculation
        effective_pressure = surface_pressure * stress_concentration

        return effective_pressure

    def get_deep_tissue_stress(self, surface_pressure: float,
                               compression_ratio: float,
                               tissue_thickness_mm: float) -> float:
        """
        Calculate stress at the deep tissue layer (muscle/bone interface).

        Deep tissue injury (DTI) occurs when stress at the bone-muscle
        interface exceeds tissue tolerance, even if surface pressure is low.

        Args:
            surface_pressure: Pressure at skin surface (mmHg)
            compression_ratio: Current compression (0-1)
            tissue_thickness_mm: Tissue thickness (mm)

        Returns:
            Estimated deep tissue stress (mmHg equivalent)
        """
        # Stress amplification at bone interface
        # Thin tissue = more amplification
        # High compression = more amplification

        if tissue_thickness_mm < 5:
            # Very thin tissue - stress transfers directly to bone
            depth_factor = 2.0
        elif tissue_thickness_mm < 10:
            depth_factor = 1.5
        elif tissue_thickness_mm < 20:
            depth_factor = 1.2
        else:
            # Thick tissue - stress dissipates
            depth_factor = 0.8

        # Compression increases deep stress (tissue being crushed against bone)
        compression_amplification = 1.0 + compression_ratio * 2.0

        deep_stress = surface_pressure * depth_factor * compression_amplification

        return deep_stress


# =============================================================================
# SMPL BODY MODEL FOR PRESSURE CALCULATION
# =============================================================================

class SMPLBodyPressureModel:
    """
    Uses SMPL body mesh to calculate anatomically accurate pressure distribution.

    The SMPL model provides:
    - Accurate body surface geometry
    - Vertex positions for contact detection
    - Body part segmentation for tissue thickness estimation

    Now includes tissue deformation modeling:
    - Soft tissue compression under load
    - Contact area spreading
    - Viscoelastic creep over time
    - Deep tissue stress calculation
    """

    # SMPL body part indices (approximate vertex ranges for key areas)
    # EVIDENCE-BASED tissue thickness values
    # Sources cited in EVIDENCE_BASE.md
    BODY_PART_TISSUE_THICKNESS = {
        # Bony prominences - MEASURED VALUES from ultrasound/MRI studies
        # Kowalczyk et al. J Spinal Cord Med 2013. PMID: 23809593 (SCI patients)
        'sacrum': 2.1,      # mm skin thickness (SD: 0.9) - SCI patients
        'coccyx': 2.0,      # mm - estimated similar to sacrum
        'ischial': 2.2,     # mm skin thickness (SD: 0.6) - SCI patients
        'trochanter': 1.8,  # mm skin thickness (SD: 0.4)
        # Heel pad: Prichasuk et al. PMID: 3886923
        'heel': 16.0,       # mm - heel pad (slightly less than healthy 16.6mm)
        'malleolus': 3.0,   # mm - estimated, limited data
        'scapula': 2.5,     # mm - estimated, limited direct data
        'occiput': 3.0,     # mm - estimated, limited direct data
        'spinous': 3.0,     # mm - estimated
        'elbow': 3.0,       # mm - estimated
        # Well-padded areas - from systematic review PMC10272691
        'buttock': 58.0,    # mm total (38mm muscle + 20mm subcutaneous)
        'thigh': 40.0,      # mm - estimated based on gluteal data
        'calf': 25.0,       # mm - estimated
        'back': 15.0,       # mm thoracic
        'shoulder': 12.0,   # mm deltoid
    }

    def __init__(self, body_mass: float = 75, incline_angle: float = DEFAULT_INCLINE_ANGLE):
        """
        Initialize SMPL body model for pressure calculations.

        Args:
            body_mass: Patient mass in kg
            incline_angle: Head of bed elevation in degrees
        """
        self.body_mass = body_mass
        self.incline_angle = incline_angle
        self.vertices = None
        self.original_vertices = None  # Store undeformed positions
        self.faces = None
        self.vertex_body_parts = None

        # Tissue deformation model
        self.deformation_model = TissueDeformationModel()

        self._load_smpl_model()

    def _load_smpl_model(self):
        """Load SMPL model and position on inclined bed surface."""
        try:
            import smplx
            import torch

            print("Loading SMPL body model for pressure simulation...")

            # Load SMPL model
            smpl_dir = '/Users/jaspermetz/Documents/Body_Sim/smpl'
            model = smplx.create(smpl_dir, model_type='smpl', gender='neutral')

            # Pose: supine with arms at sides
            body_pose = torch.zeros(1, 69)
            body_pose[0, 45 + 2] = -1.5   # L_shoulder Z-rot (arm down)
            body_pose[0, 48 + 2] = 1.5    # R_shoulder Z-rot (arm down)

            output = model(body_pose=body_pose)
            verts = output.vertices.detach().numpy()[0]  # (6890, 3)
            self.faces = model.faces

            # Rotate from standing to supine (lying on back)
            angle = -math.pi / 2
            Rx = np.array([
                [1, 0, 0],
                [0, math.cos(angle), -math.sin(angle)],
                [0, math.sin(angle), math.cos(angle)],
            ])
            v = verts @ Rx.T
            v *= 100  # metres -> cm

            # Scale to match body length
            raw_length = v[:, 2].max() - v[:, 2].min()
            scale = BODY_LENGTH / raw_length
            v *= scale

            # Position body: head at Z=0, feet at Z=body_length
            v[:, 2] -= v[:, 2].min()

            # Center X on mattress width
            v[:, 0] -= v[:, 0].mean()
            v[:, 0] += MATTRESS_WIDTH / 2

            # Y is height above mattress surface
            # Shift so back (lowest points) touch the surface
            v[:, 1] -= v[:, 1].min()

            # Apply incline transformation
            self._apply_incline(v)

            self.vertices = v

            # Assign body parts to vertices based on position
            self._assign_body_parts()

            print(f"  Loaded {len(self.vertices)} vertices")

        except Exception as e:
            print(f"Warning: Could not load SMPL model: {e}")
            print("  Falling back to simplified pressure model")
            self.vertices = None

    def _apply_incline(self, v: np.ndarray):
        """Apply bed incline transformation to vertices."""
        hinge_z = WAIST_HINGE_POSITION * BODY_LENGTH + 12  # Body starts 12cm from head
        angle_rad = np.radians(self.incline_angle)

        for i in range(len(v)):
            body_z = v[i, 2]
            body_y = v[i, 1]  # Height above surface

            if body_z < hinge_z:
                # Above hinge - on inclined section
                dist_from_hinge = hinge_z - body_z

                # Surface position at this point
                surf_z = hinge_z - dist_from_hinge * np.cos(angle_rad)
                surf_y_offset = dist_from_hinge * np.sin(angle_rad)

                # Normal direction (perpendicular to inclined surface)
                norm_z = -np.sin(angle_rad)
                norm_y = np.cos(angle_rad)

                # Position vertex on surface + height along normal
                v[i, 2] = surf_z + body_y * norm_z
                v[i, 1] = surf_y_offset + body_y * norm_y
            # Below hinge stays as-is (flat section)

    def _assign_body_parts(self):
        """Assign body part labels to vertices based on position."""
        if self.vertices is None:
            return

        n_verts = len(self.vertices)
        self.vertex_body_parts = np.empty(n_verts, dtype=object)
        self.vertex_tissue_thickness = np.zeros(n_verts)

        for i, v in enumerate(self.vertices):
            x, y, z = v  # x=width, y=height above bed, z=length along body

            # Normalize z position (0=head, 1=feet)
            z_norm = z / BODY_LENGTH

            # Normalize x position relative to body center
            x_norm = (x - MATTRESS_WIDTH/2) / (BODY_WIDTH/2)  # -1 to 1

            # Assign body part based on position
            # Consider all vertices - we'll filter by contact later
            # Head region (0-10%)
            if z_norm < 0.10:
                if abs(x_norm) < 0.4:
                    part = 'occiput'
                else:
                    part = 'shoulder'

            # Shoulder/scapula region (10-25%)
            elif z_norm < 0.25:
                if abs(x_norm) > 0.5:
                    part = 'scapula'
                elif abs(x_norm) < 0.2:
                    part = 'spinous'
                else:
                    part = 'back'

            # Upper back (25-40%)
            elif z_norm < 0.40:
                if abs(x_norm) < 0.2:
                    part = 'spinous'
                else:
                    part = 'back'

            # Sacrum/coccyx region (40-55%) - CRITICAL with incline
            # Wider zone to capture actual sacral prominence
            elif z_norm < 0.55:
                if abs(x_norm) < 0.45:  # Wider sacrum zone
                    part = 'sacrum' if z_norm < 0.52 else 'coccyx'
                else:
                    part = 'buttock'

            # Buttocks/ischial region (55-65%)
            elif z_norm < 0.65:
                if abs(x_norm) < 0.25:
                    part = 'coccyx'
                elif abs(x_norm) < 0.55:
                    part = 'ischial'  # Ischial tuberosities - bony
                else:
                    part = 'buttock'

            # Thighs (65-80%)
            elif z_norm < 0.80:
                part = 'thigh'

            # Calves (80-92%)
            elif z_norm < 0.92:
                part = 'calf'

            # Heels/ankles (92-100%)
            else:
                if abs(x_norm) > 0.4:
                    part = 'malleolus'
                else:
                    part = 'heel'

            self.vertex_body_parts[i] = part
            self.vertex_tissue_thickness[i] = self.BODY_PART_TISSUE_THICKNESS.get(part, 20.0)

    def calculate_pressure_map(self, rows: int, cols: int, dt: float = 1.0,
                                 return_deformation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate pressure and shear distribution maps from SMPL body geometry.

        Now includes tissue deformation modeling:
        - Soft tissue compresses under load
        - Contact area spreads as tissue deforms
        - Deep tissue stress calculated for DTI risk
        - Iterative pressure-deformation coupling

        Returns pressure map where high pressure occurs at:
        - Bony prominences (thin tissue)
        - Weight-bearing areas
        - Areas affected by incline (sacrum with 30° elevation)

        Args:
            rows, cols: Grid dimensions for mattress
            dt: Time step in seconds for creep calculation
            return_deformation: If True, return deformation data as third element

        Returns:
            Tuple of (pressure_map, shear_map) in mmHg
            If return_deformation=True: (pressure_map, shear_map, deformation_data)
        """
        if self.vertices is None:
            # Fallback to simplified model
            result = create_body_pressure_map_simplified(rows, cols, self.body_mass, self.incline_angle)
            if return_deformation:
                return result[0], result[1], {}
            return result

        # Initialize deformation model if needed
        n_verts = len(self.vertices)
        if self.deformation_model.current_compression is None:
            self.deformation_model.initialize(n_verts)

        # Store original vertices on first call
        if self.original_vertices is None:
            self.original_vertices = self.vertices.copy()

        pressure_map = np.zeros((rows, cols))
        shear_map = np.zeros((rows, cols))
        contact_count = np.zeros((rows, cols))
        tissue_thickness_map = np.zeros((rows, cols))
        compression_map = np.zeros((rows, cols))
        deep_tissue_stress_map = np.zeros((rows, cols))

        # Cell dimensions
        cell_length = MATTRESS_LENGTH / rows
        cell_width = MATTRESS_WIDTH / cols
        cell_area = (cell_length / 100) * (cell_width / 100)  # m²

        # Total body weight
        body_weight = self.body_mass * GRAVITY  # Newtons

        # Iterative pressure-deformation calculation
        # Tissue deformation affects contact area which affects pressure
        n_iterations = 3  # Balance between accuracy and speed

        for iteration in range(n_iterations):
            # Use deformed vertices for contact detection
            working_vertices = self.vertices.copy()

            # Apply current deformation to vertex positions
            for vi in range(n_verts):
                compression = self.deformation_model.current_compression[vi]
                if compression > 0:
                    tissue_thickness = self.vertex_tissue_thickness[vi]
                    # Reduce height (Y coordinate) based on compression
                    original_y = self.original_vertices[vi, 1]
                    working_vertices[vi, 1] = self.deformation_model.get_deformed_position(
                        original_y, compression, tissue_thickness
                    )

            # Contact threshold - vertices close to surface are in contact
            # As tissue compresses, more vertices come into contact
            base_contact_threshold = 8.0  # cm
            # Reduce threshold as deformation increases (tighter contact)
            avg_compression = np.mean(self.deformation_model.current_compression[
                self.deformation_model.current_compression > 0
            ]) if np.any(self.deformation_model.current_compression > 0) else 0
            contact_threshold = base_contact_threshold * (1 - 0.3 * avg_compression)

            # Reset maps for this iteration
            contact_count.fill(0)
            tissue_thickness_map.fill(0)
            vertex_to_cell = {}  # Track which vertices are in which cells

            # First pass: find minimum height in each grid cell
            min_height = np.full((rows, cols), np.inf)
            for i, v in enumerate(working_vertices):
                x, y, z = v
                if 0 <= x < MATTRESS_WIDTH and 0 <= z < MATTRESS_LENGTH:
                    col = int(x / cell_width)
                    row = int(z / cell_length)
                    if 0 <= row < rows and 0 <= col < cols:
                        min_height[row, col] = min(min_height[row, col], y)

            # Second pass: identify contact vertices and assign to cells
            for i, v in enumerate(working_vertices):
                x, y, z = v

                if 0 <= x < MATTRESS_WIDTH and 0 <= z < MATTRESS_LENGTH:
                    col = int(x / cell_width)
                    row = int(z / cell_length)
                    if 0 <= row < rows and 0 <= col < cols:
                        local_min = min_height[row, col]
                        if local_min < np.inf and y < local_min + contact_threshold:
                            contact_count[row, col] += 1
                            tissue_thickness_map[row, col] += self.vertex_tissue_thickness[i]

                            # Track vertex assignment to cell
                            cell_key = (row, col)
                            if cell_key not in vertex_to_cell:
                                vertex_to_cell[cell_key] = []
                            vertex_to_cell[cell_key].append(i)

                            # Mark vertex as in contact
                            self.deformation_model.vertex_in_contact[i] = True
                        else:
                            self.deformation_model.vertex_in_contact[i] = False

            # Normalize tissue thickness
            for i in range(rows):
                for j in range(cols):
                    if contact_count[i, j] > 0:
                        tissue_thickness_map[i, j] /= contact_count[i, j]
                    else:
                        tissue_thickness_map[i, j] = 20.0  # Default

            # Calculate contact area and weight distribution
            total_contact_vertices = np.sum(contact_count)
            if total_contact_vertices == 0:
                if return_deformation:
                    return pressure_map, shear_map, {}
                return pressure_map, shear_map

            # Incline effects on weight distribution
            angle_rad = np.radians(self.incline_angle)
            sin_angle = np.sin(angle_rad)
            weight_shift = sin_angle * 0.4

            # Calculate pressure for each cell
            for i in range(rows):
                for j in range(cols):
                    if contact_count[i, j] > 0:
                        # Get contact area multiplier from deformation
                        cell_vertices = vertex_to_cell.get((i, j), [])
                        if cell_vertices:
                            avg_contact_mult = np.mean([
                                self.deformation_model.contact_area_multiplier[vi]
                                for vi in cell_vertices
                            ])
                        else:
                            avg_contact_mult = 1.0

                        # Effective contact count increases with tissue spreading
                        effective_contact = contact_count[i, j] * avg_contact_mult

                        # Anatomical weight distribution (replaces vertex-based distribution)
                        row_pos = i / rows

                        # Find which anatomical region this cell is in
                        anatomical_weight = 0.05  # Default
                        incline_mult = 1.0
                        for region, (z_start, z_end, base_weight, inc_mult) in ANATOMICAL_WEIGHT_DISTRIBUTION.items():
                            if z_start <= row_pos < z_end:
                                # Weight fraction for this region, distributed across cells in region
                                region_rows = int((z_end - z_start) * rows)
                                anatomical_weight = base_weight / max(region_rows, 1)
                                incline_mult = inc_mult
                                break

                        # Apply incline effect (30° shifts weight to sacrum)
                        incline_factor = 1.0 + (incline_mult - 1.0) * (self.incline_angle / 30.0)

                        # Calculate local weight based on anatomical distribution
                        local_weight = body_weight * anatomical_weight * incline_factor  # N

                        # Tissue factor - thin tissue = higher pressure concentration
                        # Capped to avoid extreme values when combined with anatomical distribution
                        # Clinical range: sacrum sees ~1.5-2x higher pressure than padded areas
                        raw_tissue_factor = 25.0 / tissue_thickness_map[i, j]
                        tissue_factor = 1.0 + (raw_tissue_factor - 1.0) * 0.3  # Dampen effect

                        # Effective area increases with spreading
                        effective_area = cell_area * avg_contact_mult

                        base_pressure_pa = local_weight / effective_area
                        base_pressure_mmhg = base_pressure_pa * MMHG_PER_PASCAL

                        # Adjust for tissue thickness
                        adjusted_pressure = base_pressure_mmhg * tissue_factor

                        pressure_map[i, j] = adjusted_pressure

                        # Calculate shear stress
                        if row_pos < WAIST_HINGE_POSITION:
                            friction_coef = 0.5
                            shear = adjusted_pressure * sin_angle * friction_coef
                            shear *= (row_pos / WAIST_HINGE_POSITION)
                            shear_map[i, j] = shear

            # Update tissue deformation based on calculated pressures
            # This is the key feedback loop - pressure causes deformation
            for cell_key, vertices in vertex_to_cell.items():
                row, col = cell_key
                local_pressure_mmhg = pressure_map[row, col]
                local_pressure_pa = local_pressure_mmhg / MMHG_PER_PASCAL  # Convert back to Pa

                for vi in vertices:
                    body_part = self.vertex_body_parts[vi]
                    tissue_thickness = self.vertex_tissue_thickness[vi]

                    # Calculate compression for this vertex
                    compression = self.deformation_model.calculate_compression(
                        vi, body_part, local_pressure_pa, dt / n_iterations
                    )

                    # Calculate deep tissue stress
                    contact_mult = self.deformation_model.contact_area_multiplier[vi]
                    effective_pressure = self.deformation_model.calculate_effective_contact_pressure(
                        local_pressure_mmhg, compression, contact_mult
                    )
                    deep_stress = self.deformation_model.get_deep_tissue_stress(
                        effective_pressure, compression, tissue_thickness
                    )

                    # Accumulate for cell averages
                    compression_map[row, col] += compression
                    deep_tissue_stress_map[row, col] += deep_stress

            # Average compression and deep stress per cell
            for i in range(rows):
                for j in range(cols):
                    if contact_count[i, j] > 0:
                        compression_map[i, j] /= contact_count[i, j]
                        deep_tissue_stress_map[i, j] /= contact_count[i, j]

        # Apply quadriplegic factors
        pressure_map *= QUADRIPLEGIC_FACTORS['skin_fragility']
        shear_map *= QUADRIPLEGIC_FACTORS.get('spasticity_shear_factor', 1.3)

        if return_deformation:
            deformation_data = {
                'compression_map': compression_map,
                'deep_tissue_stress_map': deep_tissue_stress_map,
                'contact_count': contact_count,
                'tissue_thickness_map': tissue_thickness_map,
                'avg_compression': np.mean(self.deformation_model.current_compression[
                    self.deformation_model.vertex_in_contact
                ]) if np.any(self.deformation_model.vertex_in_contact) else 0,
                'max_compression': np.max(self.deformation_model.current_compression),
                'vertices_in_contact': int(np.sum(self.deformation_model.vertex_in_contact)),
            }
            return pressure_map, shear_map, deformation_data

        return pressure_map, shear_map


# Global SMPL body model instance (lazy loaded)
_smpl_body_model = None

def get_smpl_body_model(body_mass: float = 75, incline_angle: float = DEFAULT_INCLINE_ANGLE) -> SMPLBodyPressureModel:
    """Get or create the SMPL body pressure model."""
    global _smpl_body_model
    if _smpl_body_model is None:
        _smpl_body_model = SMPLBodyPressureModel(body_mass, incline_angle)
    return _smpl_body_model


def create_body_pressure_map(rows: int, cols: int, body_mass: float = 75,
                              incline_angle: float = DEFAULT_INCLINE_ANGLE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create pressure and shear distribution maps using SMPL body model.

    This is the main interface function that uses the SMPL body for
    anatomically accurate pressure calculations.
    """
    model = get_smpl_body_model(body_mass, incline_angle)
    return model.calculate_pressure_map(rows, cols)


def create_body_pressure_map_simplified(rows: int, cols: int, body_mass: float = 75,
                              incline_angle: float = DEFAULT_INCLINE_ANGLE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create pressure and shear distribution maps for supine body with bed incline.

    The 30-degree head-of-bed elevation causes:
    1. Weight shift toward sacrum/buttocks (increased pressure)
    2. Shear stress from gravity pulling body downward
    3. Reduced pressure at head/shoulders

    Key pressure points (bony prominences):
    - Occiput (back of head): reduced with incline
    - Scapulae (shoulder blades): reduced with incline
    - Sacrum/coccyx: HIGHEST pressure - INCREASED with incline
    - Heels: high concentrated pressure

    Args:
        rows, cols: Grid dimensions
        body_mass: Patient mass in kg
        incline_angle: Head of bed elevation in degrees

    Returns:
        Tuple of (pressure_map, shear_map) in mmHg
    """
    pressure_map = np.zeros((rows, cols))
    shear_map = np.zeros((rows, cols))

    # Normalize positions (0-1 range)
    row_positions = np.linspace(0, 1, rows)
    col_positions = np.linspace(0, 1, cols)

    # Body contact area (centered, ~50% of mattress width)
    body_left = int(cols * 0.25)
    body_right = int(cols * 0.75)

    # Base body weight distribution (kg/m² -> pressure)
    body_weight = body_mass * GRAVITY  # N
    mattress_area = (MATTRESS_LENGTH / 100) * (MATTRESS_WIDTH / 100)  # m²
    base_pressure = (body_weight / mattress_area) * MMHG_PER_PASCAL  # mmHg

    # Incline effects
    angle_rad = np.radians(incline_angle)
    sin_angle = np.sin(angle_rad)
    cos_angle = np.cos(angle_rad)

    # Weight shift factor: how much weight transfers to lower body
    # At 30 degrees, significant weight shifts to sacrum
    weight_shift_factor = sin_angle * 0.4  # 40% of sin component shifts down

    for i, row_pos in enumerate(row_positions):
        for j, col_pos in enumerate(col_positions):
            # Check if within body contact zone
            if body_left <= j <= body_right:
                # Base contact pressure
                pressure = base_pressure * 0.3

                # Determine if above or below hinge
                above_hinge = row_pos < WAIST_HINGE_POSITION

                # Weight redistribution due to incline
                if above_hinge:
                    # Above hinge (head/shoulders): weight reduced
                    incline_pressure_mod = 1.0 - weight_shift_factor * (WAIST_HINGE_POSITION - row_pos) / WAIST_HINGE_POSITION
                else:
                    # Below hinge (sacrum/legs): weight increased near hinge
                    distance_from_hinge = (row_pos - WAIST_HINGE_POSITION) / (1 - WAIST_HINGE_POSITION)
                    # Peak pressure increase at sacrum (just below hinge)
                    if distance_from_hinge < 0.3:  # Sacrum region
                        incline_pressure_mod = 1.0 + weight_shift_factor * 2.5 * (1 - distance_from_hinge / 0.3)
                    else:
                        incline_pressure_mod = 1.0

                # Head region (0-10% of length)
                if row_pos < 0.10:
                    if 0.40 <= col_pos <= 0.60:
                        pressure += 25 * incline_pressure_mod

                # Shoulder/scapulae region (10-25%)
                elif 0.10 <= row_pos < 0.25:
                    if 0.25 <= col_pos <= 0.38 or 0.62 <= col_pos <= 0.75:
                        pressure += 35 * incline_pressure_mod
                    elif 0.38 < col_pos < 0.62:
                        pressure += 15 * incline_pressure_mod

                # Upper back (25-40%)
                elif 0.25 <= row_pos < 0.40:
                    pressure += 18 * incline_pressure_mod

                # Sacrum/coccyx region (40-55%) - CRITICAL - INCREASED with incline
                elif 0.40 <= row_pos < 0.55:
                    if 0.35 <= col_pos <= 0.65:
                        # Sacrum - highest pressure, significantly increased with incline
                        pressure += 50 * incline_pressure_mod
                    else:
                        pressure += 20 * incline_pressure_mod

                # Buttocks (55-65%)
                elif 0.55 <= row_pos < 0.65:
                    if 0.30 <= col_pos <= 0.45 or 0.55 <= col_pos <= 0.70:
                        pressure += 30 * incline_pressure_mod

                # Thighs (65-80%)
                elif 0.65 <= row_pos < 0.80:
                    if 0.30 <= col_pos <= 0.42 or 0.58 <= col_pos <= 0.70:
                        pressure += 15

                # Calves (80-92%)
                elif 0.80 <= row_pos < 0.92:
                    if 0.32 <= col_pos <= 0.40 or 0.60 <= col_pos <= 0.68:
                        pressure += 12

                # Heels (92-100%) - HIGH pressure
                elif row_pos >= 0.92:
                    if 0.32 <= col_pos <= 0.38 or 0.62 <= col_pos <= 0.68:
                        pressure += 40

                pressure_map[i, j] = pressure

                # Calculate shear stress (from gravity pulling body down incline)
                # Shear = normal_force * sin(angle) * friction_coefficient
                # Highest shear at sacrum where body tends to slide
                if above_hinge and pressure > 0:
                    # Above hinge: body slides down toward sacrum
                    friction_coef = 0.5  # Skin-mattress friction
                    shear = pressure * sin_angle * friction_coef
                    # Shear increases closer to hinge (more sliding tendency)
                    shear *= (row_pos / WAIST_HINGE_POSITION)
                    shear_map[i, j] = shear

    # Apply quadriplegic factors
    pressure_map *= QUADRIPLEGIC_FACTORS['skin_fragility']
    # Quadriplegic patients have increased shear due to spasticity
    shear_map *= QUADRIPLEGIC_FACTORS.get('spasticity_shear_factor', 1.3)

    return pressure_map, shear_map


# =============================================================================
# MOVEMENT PATTERNS
# =============================================================================

@dataclass
class MovementPattern:
    """Defines a cell inflation/deflation pattern."""
    name: str
    description: str

    def get_cell_state(self, row: int, col: int, rows: int, cols: int,
                       phase: float) -> float:
        """
        Get inflation state for a cell at given phase.

        Args:
            row, col: Cell position
            rows, cols: Grid dimensions
            phase: Current cycle phase (0-1)

        Returns:
            Inflation level (0 = fully deflated, 1 = fully inflated)
        """
        raise NotImplementedError


class AlternatingCheckerboard(MovementPattern):
    """Classic checkerboard alternating pattern - every other cell."""

    def __init__(self):
        super().__init__(
            name="Alternating Checkerboard",
            description="Every other cell alternates (checkerboard pattern)"
        )

    def get_cell_state(self, row, col, rows, cols, phase):
        is_group_a = (row + col) % 2 == 0
        if is_group_a:
            return 0.5 + 0.5 * np.cos(2 * np.pi * phase)
        else:
            return 0.5 - 0.5 * np.cos(2 * np.pi * phase)


class WaveHorizontal(MovementPattern):
    """Horizontal wave - rows inflate/deflate sequentially."""

    def __init__(self, wave_width: int = 3):
        super().__init__(
            name=f"Horizontal Wave (width={wave_width})",
            description="Wave moves head to foot"
        )
        self.wave_width = wave_width

    def get_cell_state(self, row, col, rows, cols, phase):
        # Wave position (0 to rows)
        wave_center = phase * (rows + self.wave_width)
        distance = abs(row - wave_center)

        if distance < self.wave_width:
            # In the deflation wave
            return 0.3 + 0.7 * (distance / self.wave_width)
        return 1.0


class WaveVertical(MovementPattern):
    """Vertical wave - columns inflate/deflate side to side."""

    def __init__(self, wave_width: int = 2):
        super().__init__(
            name=f"Vertical Wave (width={wave_width})",
            description="Wave moves left to right to left"
        )
        self.wave_width = wave_width

    def get_cell_state(self, row, col, rows, cols, phase):
        # Oscillate left to right
        if phase < 0.5:
            wave_center = phase * 2 * cols
        else:
            wave_center = (1 - phase) * 2 * cols

        distance = abs(col - wave_center)

        if distance < self.wave_width:
            return 0.3 + 0.7 * (distance / self.wave_width)
        return 1.0


class ZoneBased(MovementPattern):
    """
    Zone-based pattern - different zones have different cycles.
    Targets high-risk areas (sacrum, heels) more aggressively.
    """

    def __init__(self):
        super().__init__(
            name="Zone-Based Adaptive",
            description="High-risk zones cycle faster"
        )

    def get_cell_state(self, row, col, rows, cols, phase):
        row_pos = row / rows
        col_pos = col / cols

        # Determine zone and cycle multiplier
        if 0.40 <= row_pos < 0.55 and 0.35 <= col_pos <= 0.65:
            # Sacrum zone - fastest cycle (2x speed)
            local_phase = (phase * 2) % 1.0
            return 0.5 + 0.5 * np.cos(2 * np.pi * local_phase)
        elif row_pos >= 0.92 and (0.32 <= col_pos <= 0.40 or 0.58 <= col_pos <= 0.68):
            # Heel zones - fast cycle (1.5x speed)
            local_phase = (phase * 1.5) % 1.0
            return 0.5 + 0.5 * np.cos(2 * np.pi * local_phase)
        elif 0.10 <= row_pos < 0.25:
            # Scapulae zone - moderate cycle
            local_phase = (phase * 1.2) % 1.0
            return 0.5 + 0.5 * np.cos(2 * np.pi * local_phase)
        else:
            # Normal zones - standard cycle
            return 0.5 + 0.5 * np.cos(2 * np.pi * phase)


class CircularWave(MovementPattern):
    """Circular wave emanating from center outward."""

    def __init__(self, wave_width: float = 0.15):
        super().__init__(
            name="Circular Wave",
            description="Wave expands from center outward"
        )
        self.wave_width = wave_width

    def get_cell_state(self, row, col, rows, cols, phase):
        # Normalized distance from center
        center_row, center_col = rows / 2, cols / 2
        dist = np.sqrt(((row - center_row) / rows) ** 2 +
                       ((col - center_col) / cols) ** 2)

        # Wave expands outward
        wave_radius = phase * 0.8  # Max radius 0.8

        distance_from_wave = abs(dist - wave_radius)

        if distance_from_wave < self.wave_width:
            return 0.3 + 0.7 * (distance_from_wave / self.wave_width)
        return 1.0


class DiagonalWave(MovementPattern):
    """Diagonal wave pattern."""

    def __init__(self):
        super().__init__(
            name="Diagonal Wave",
            description="Wave moves diagonally across mattress"
        )

    def get_cell_state(self, row, col, rows, cols, phase):
        # Diagonal position (0-1 based on row+col)
        diag_pos = (row / rows + col / cols) / 2

        # Wave position
        wave_pos = phase
        distance = abs(diag_pos - wave_pos)

        # Wrap around
        distance = min(distance, 1 - distance)

        wave_width = 0.15
        if distance < wave_width:
            return 0.3 + 0.7 * (distance / wave_width)
        return 1.0


class RowGroups(MovementPattern):
    """Alternating row groups (2-3 rows at a time)."""

    def __init__(self, group_size: int = 2):
        super().__init__(
            name=f"Row Groups (size={group_size})",
            description=f"Groups of {group_size} rows alternate"
        )
        self.group_size = group_size

    def get_cell_state(self, row, col, rows, cols, phase):
        group = (row // self.group_size) % 2
        if group == 0:
            return 0.5 + 0.5 * np.cos(2 * np.pi * phase)
        else:
            return 0.5 - 0.5 * np.cos(2 * np.pi * phase)


class MultiFrequencyZone(MovementPattern):
    """
    Different body zones have different frequencies.
    High-risk areas cycle faster for more frequent relief.
    """

    def __init__(self):
        super().__init__(
            name="Multi-Frequency Zones",
            description="Each body zone has optimized frequency"
        )

        # Zone definitions: (row_start, row_end, col_start, col_end, frequency_mult)
        self.zones = [
            # Sacrum - highest risk, fastest cycle
            (0.40, 0.55, 0.30, 0.70, 3.0),
            # Heels - high risk
            (0.90, 1.00, 0.25, 0.45, 2.5),
            (0.90, 1.00, 0.55, 0.75, 2.5),
            # Scapulae
            (0.10, 0.25, 0.20, 0.40, 2.0),
            (0.10, 0.25, 0.60, 0.80, 2.0),
            # Occiput
            (0.00, 0.10, 0.35, 0.65, 1.5),
        ]

    def get_cell_state(self, row, col, rows, cols, phase):
        row_pos = row / rows
        col_pos = col / cols

        # Find which zone this cell belongs to
        freq_mult = 1.0  # Default frequency

        for r_start, r_end, c_start, c_end, mult in self.zones:
            if r_start <= row_pos < r_end and c_start <= col_pos <= c_end:
                freq_mult = mult
                break

        local_phase = (phase * freq_mult) % 1.0

        # Alternating pattern within zone
        is_group_a = (row + col) % 2 == 0
        if is_group_a:
            return 0.5 + 0.5 * np.cos(2 * np.pi * local_phase)
        else:
            return 0.5 - 0.5 * np.cos(2 * np.pi * local_phase)


class SequentialRows(MovementPattern):
    """Each row deflates one at a time in sequence."""

    def __init__(self):
        super().__init__(
            name="Sequential Rows",
            description="One row deflates at a time, moving down the body"
        )

    def get_cell_state(self, row, col, rows, cols, phase):
        # Which row should be deflated at this phase
        active_row = int(phase * rows) % rows

        if row == active_row:
            return 0.3  # Deflated
        elif abs(row - active_row) == 1 or abs(row - active_row) == rows - 1:
            return 0.6  # Partially deflated (transition)
        return 1.0  # Fully inflated


# All available patterns
MOVEMENT_PATTERNS = {
    'alternating': AlternatingCheckerboard(),
    'wave_horizontal': WaveHorizontal(wave_width=3),
    'wave_vertical': WaveVertical(wave_width=2),
    'zone_based': ZoneBased(),
    'circular': CircularWave(),
    'diagonal': DiagonalWave(),
    'row_groups_1': RowGroups(group_size=1),
    'row_groups_2': RowGroups(group_size=2),
    'row_groups_3': RowGroups(group_size=3),
    'multi_frequency': MultiFrequencyZone(),
    'sequential_rows': SequentialRows(),
}


# =============================================================================
# AIR MATTRESS SIMULATION
# =============================================================================

class MultiDynamicAirMattress:
    """
    Multi-dynamic air mattress with configurable cell grid and movement patterns.
    """

    def __init__(self, cell_size_cm: float = 5,
                 movement_pattern: MovementPattern = None,
                 cycle_period_sec: float = 300,
                 min_cell_pressure: float = 5,
                 max_cell_pressure: float = 40):
        """
        Initialize mattress.

        Args:
            cell_size_cm: Size of each air cell in cm
            movement_pattern: Pattern for inflation/deflation
            cycle_period_sec: Time for one complete cycle in seconds
            min_cell_pressure: Minimum cell pressure when deflated (mmHg)
            max_cell_pressure: Maximum cell pressure when inflated (mmHg)
        """
        self.cell_size = cell_size_cm
        self.rows = int(MATTRESS_LENGTH / cell_size_cm)
        self.cols = int(MATTRESS_WIDTH / cell_size_cm)
        self.shape = (self.rows, self.cols)

        self.pattern = movement_pattern or AlternatingCheckerboard()
        self.cycle_period = cycle_period_sec

        self.min_pressure = min_cell_pressure
        self.max_pressure = max_cell_pressure

        # Cell state (inflation level 0-1)
        self.cell_state = np.ones(self.shape)

        # Current pressures
        self.cell_pressure = np.ones(self.shape) * max_cell_pressure

    def update(self, time_sec: float):
        """Update cell states based on current time."""
        phase = (time_sec % self.cycle_period) / self.cycle_period

        for i in range(self.rows):
            for j in range(self.cols):
                self.cell_state[i, j] = self.pattern.get_cell_state(
                    i, j, self.rows, self.cols, phase
                )

        # Convert state to pressure
        pressure_range = self.max_pressure - self.min_pressure
        self.cell_pressure = self.min_pressure + self.cell_state * pressure_range

    def get_support_pressure(self) -> np.ndarray:
        """Get current support pressure provided by each cell."""
        return self.cell_pressure

    def get_effective_interface_pressure(self, body_pressure: np.ndarray) -> np.ndarray:
        """
        Calculate effective interface pressure between body and mattress.

        Evidence-Based Model:
        Clinical trials show variable APM effectiveness:
        - E²MAO RCT (PMID:28598759): 83% PU risk reduction vs foam
        - Belgian RCT (PMID:31234104): Static air (5.2%) outperformed APM (11.7%)
        - PRESSURE 2: No significant difference vs high-spec foam
        - Systematic review (PMID:35229980): Insufficient evidence for APM superiority

        Model Assumptions:
        - Deflated cells provide 20-35% relief at bony prominences (conservative)
        - Soft tissue areas: 35-50% relief (based on immersion/envelopment)
        - Inflated cells take redistributed load (weight conservation)

        Note: NPIAP/EPUAP 2019 guidelines state interface pressure alone is
        insufficient; shear and tissue deformation also critical for DTI risk.

        Args:
            body_pressure: Body weight distribution pressure map

        Returns:
            Effective interface pressure at each cell
        """
        if body_pressure.shape != self.shape:
            body_pressure = self._resample(body_pressure, self.shape)

        effective_pressure = np.zeros(self.shape)
        pressure_relieved = np.zeros(self.shape)  # Track how much pressure is "saved"

        # First pass: calculate relief for deflated cells
        for i in range(self.rows):
            for j in range(self.cols):
                local_body_pressure = body_pressure[i, j]
                cell_inflation = self.cell_state[i, j]

                row_pos = i / self.rows
                # Sacrum (40-60%) and heels (90%+) are high pressure zones
                is_bony_prominence = (0.40 <= row_pos < 0.60) or (row_pos >= 0.90)

                if cell_inflation < 0.5:
                    # Deflated cell - provides relief based on clinical APM research
                    # Good APMs achieve 30-50% relief; bony areas get less benefit
                    deflation_level = (0.5 - cell_inflation) / 0.2  # 0 to 1

                    if is_bony_prominence:
                        # Bony areas: 20-35% relief (bone limits how much tissue can sink)
                        relief_pct = 0.20 + 0.15 * deflation_level
                    else:
                        # Soft tissue: 35-50% relief (tissue can immerse into deflated cell)
                        relief_pct = 0.35 + 0.15 * deflation_level

                    pressure_relieved[i, j] = local_body_pressure * relief_pct
                    effective_pressure[i, j] = local_body_pressure * (1 - relief_pct)

                elif cell_inflation < 0.8:
                    # Partially deflated - some relief (10-20%)
                    partial = (0.8 - cell_inflation) / 0.3
                    relief_pct = 0.10 + 0.10 * partial
                    pressure_relieved[i, j] = local_body_pressure * relief_pct
                    effective_pressure[i, j] = local_body_pressure * (1 - relief_pct)
                else:
                    # Fully inflated - no relief, will receive transferred load
                    effective_pressure[i, j] = local_body_pressure

        # Second pass: redistribute relieved pressure to inflated neighbors
        # The relieved pressure MUST go somewhere - body weight is conserved
        for i in range(self.rows):
            for j in range(self.cols):
                if self.cell_state[i, j] >= 0.8:
                    # This is an inflated cell - receives load from ALL nearby deflated cells
                    total_transfer = 0

                    # Check all cells in a wider radius (2 cells) for pressure transfer
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.rows and 0 <= nj < self.cols:
                                if pressure_relieved[ni, nj] > 0:
                                    # Distance-based transfer weight
                                    dist = abs(di) + abs(dj)
                                    if dist == 1:
                                        weight = 1.0  # Direct neighbors get full share
                                    elif dist == 2:
                                        weight = 0.5  # Diagonal or 2-away
                                    else:
                                        weight = 0.25  # Further away

                                    # Count total inflated cells that will share this load
                                    total_weight = 0
                                    for di2 in range(-2, 3):
                                        for dj2 in range(-2, 3):
                                            ni2, nj2 = ni + di2, nj + dj2
                                            if 0 <= ni2 < self.rows and 0 <= nj2 < self.cols:
                                                if self.cell_state[ni2, nj2] >= 0.8:
                                                    d2 = abs(di2) + abs(dj2)
                                                    if d2 == 1:
                                                        total_weight += 1.0
                                                    elif d2 == 2:
                                                        total_weight += 0.5
                                                    elif d2 > 0:
                                                        total_weight += 0.25

                                    if total_weight > 0:
                                        # Transfer proportional share of relieved pressure
                                        share = pressure_relieved[ni, nj] * weight / total_weight
                                        total_transfer += share

                    effective_pressure[i, j] += total_transfer

        # Light smoothing for tissue load distribution
        kernel_size = 3
        smoothed = np.zeros_like(effective_pressure)
        for i in range(self.rows):
            for j in range(self.cols):
                count = 0
                total = 0
                for di in range(-kernel_size//2, kernel_size//2 + 1):
                    for dj in range(-kernel_size//2, kernel_size//2 + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.rows and 0 <= nj < self.cols:
                            weight = 1.0 / (1 + abs(di) + abs(dj))
                            total += effective_pressure[ni, nj] * weight
                            count += weight
                smoothed[i, j] = total / count if count > 0 else effective_pressure[i, j]

        return smoothed

    def _get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Get valid neighbor indices."""
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.rows and 0 <= nj < self.cols:
                neighbors.append((ni, nj))
        return neighbors

    def _resample(self, arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resample array to target shape."""
        from scipy.ndimage import zoom

        zoom_factors = (target_shape[0] / arr.shape[0],
                        target_shape[1] / arr.shape[1])
        return zoom(arr, zoom_factors, order=1)


# =============================================================================
# PRESSURE ULCER RISK MODEL
# =============================================================================

class PressureUlcerRiskModel:
    """
    Models cumulative tissue damage from pressure AND shear over time.

    Based on research evidence:
    - Interface pressure magnitude matters
    - Shear stress compounds pressure damage (multiplicative effect)
    - Duration of CONTINUOUS pressure is critical
    - Intermittent relief prevents damage accumulation
    - Brief relief periods allow tissue reperfusion
    - Recovery is faster with frequent relief cycles

    The 30-degree bed incline significantly increases:
    - Sacral pressure (weight shift)
    - Shear stress (gravity pulling body down)
    """

    def __init__(self, grid_shape: Tuple[int, int]):
        self.shape = grid_shape

        # Cumulative damage (0 = healthy, 1 = ulcer formed)
        self.damage = np.zeros(grid_shape)

        # Continuous time under ischemic pressure (resets on relief)
        self.continuous_ischemia_time = np.zeros(grid_shape)

        # Total ischemic exposure (doesn't reset)
        self.total_ischemia_time = np.zeros(grid_shape)

        # Time in relief state
        self.relief_time = np.zeros(grid_shape)

        # Track if tissue is currently recovering
        self.recovering = np.zeros(grid_shape, dtype=bool)

        # Damage parameters
        self.ischemia_threshold = CAPILLARY_CLOSING_PRESSURE  # 32 mmHg

        # Shear damage threshold (kPa equivalent)
        self.shear_threshold = 10  # mmHg equivalent

        # Damage rate: Based on research
        # - 2 hours continuous pressure -> Stage 1 ulcer risk
        # - But intermittent relief dramatically reduces risk
        self.base_damage_rate = 0.25 / 3600  # 25% damage per hour continuous

        # Recovery rate when pressure relieved
        # Tissue recovers ~3x faster than it damages (with good perfusion)
        self.recovery_rate = 0.75 / 3600  # 75% recovery per hour

        # Minimum relief time for effective recovery (15 seconds)
        self.min_relief_time = 15

        # Apply quadriplegic damage multiplier
        self.base_damage_rate *= QUADRIPLEGIC_FACTORS['damage_rate_multiplier']
        # Quadriplegic patients recover slower due to poor perfusion
        self.recovery_rate *= QUADRIPLEGIC_FACTORS['baseline_perfusion_factor']

    def update(self, interface_pressure: np.ndarray, shear_stress: np.ndarray, dt: float):
        """
        Update damage model for one timestep.

        Args:
            interface_pressure: Current interface pressure (mmHg)
            shear_stress: Current shear stress (mmHg equivalent)
            dt: Time step in seconds
        """
        # Combined stress using von Mises-like criterion
        # Shear amplifies pressure damage
        shear_factor = 1 + np.clip(shear_stress / self.shear_threshold, 0, 2)
        effective_pressure = interface_pressure * shear_factor

        # Identify ischemic cells (effective pressure above capillary closing)
        ischemic = effective_pressure > self.ischemia_threshold
        relieved = ~ischemic

        # Update timing
        self.continuous_ischemia_time[ischemic] += dt
        self.continuous_ischemia_time[relieved] = 0

        self.total_ischemia_time[ischemic] += dt

        self.relief_time[relieved] += dt
        self.relief_time[ischemic] = 0

        # Calculate damage for ischemic cells
        # Damage depends on:
        # 1. Effective pressure magnitude above threshold
        # 2. Continuous duration (exponential increase after 30 min)

        excess_pressure = np.maximum(effective_pressure - self.ischemia_threshold, 0)
        pressure_factor = 1 + (excess_pressure / self.ischemia_threshold) * 0.5

        # Shear adds additional damage component
        shear_damage_factor = 1 + np.clip(shear_stress / (self.shear_threshold * 2), 0, 1)

        # Time factor: damage accelerates after sustained pressure
        # - First 15 min: linear damage
        # - 15-60 min: accelerating damage
        # - >60 min: high damage rate
        continuous_min = self.continuous_ischemia_time / 60
        time_factor = np.ones_like(continuous_min)
        time_factor = np.where(continuous_min > 15, 1 + (continuous_min - 15) / 45, time_factor)
        time_factor = np.where(continuous_min > 60, 2 + (continuous_min - 60) / 30, time_factor)
        time_factor = np.minimum(time_factor, 4)  # Cap at 4x

        damage_increment = self.base_damage_rate * pressure_factor * shear_damage_factor * time_factor * dt
        self.damage[ischemic] += damage_increment[ischemic]

        # Recovery for relieved cells
        # Recovery is effective only after minimum relief time
        effective_relief = relieved & (self.relief_time > self.min_relief_time)

        # Recovery rate increases with relief duration (up to a point)
        relief_effectiveness = np.minimum(self.relief_time / 60, 2)  # Max at 2 min relief
        recovery = self.recovery_rate * relief_effectiveness * dt

        self.damage[effective_relief] = np.maximum(
            self.damage[effective_relief] - recovery[effective_relief], 0
        )

        # Cap damage at 1.0
        self.damage = np.clip(self.damage, 0, 1)

    def get_risk_score(self) -> float:
        """
        Get overall risk score (0-100).

        Score interpretation:
        - 0-20: Excellent - minimal risk
        - 20-40: Good - low risk
        - 40-60: Moderate - some risk areas
        - 60-80: High - significant risk
        - 80-100: Critical - ulcer likely
        """
        peak_damage = self.damage.max()
        mean_damage = self.damage[self.damage > 0.01].mean() if np.any(self.damage > 0.01) else 0

        # Cells in different damage categories
        total_cells = self.damage.size
        cells_minor = np.sum((self.damage > 0.05) & (self.damage <= 0.2)) / total_cells
        cells_moderate = np.sum((self.damage > 0.2) & (self.damage <= 0.4)) / total_cells
        cells_high = np.sum((self.damage > 0.4) & (self.damage <= 0.6)) / total_cells
        cells_critical = np.sum(self.damage > 0.6) / total_cells

        # Weighted score
        score = (
            peak_damage * 25 +           # Peak damage: 25%
            mean_damage * 25 +            # Mean damage: 25%
            cells_minor * 10 +            # Minor damage spread: 10%
            cells_moderate * 15 +         # Moderate damage: 15%
            cells_high * 25 +             # High damage: 25%
            cells_critical * 50           # Critical cells heavily weighted
        ) * 100

        return min(score, 100)

    def get_metrics(self) -> Dict:
        """Get detailed damage metrics."""
        return {
            'peak_damage': float(self.damage.max()),
            'mean_damage': float(self.damage[self.damage > 0].mean()) if np.any(self.damage > 0) else 0,
            'cells_at_risk': int(np.sum(self.damage > 0.1)),
            'cells_moderate': int(np.sum(self.damage > 0.3)),
            'cells_high': int(np.sum(self.damage > 0.5)),
            'cells_critical': int(np.sum(self.damage > 0.7)),
            'risk_score': self.get_risk_score(),
            'max_continuous_ischemia_time': float(self.continuous_ischemia_time.max()),
            'total_ischemia_time': float(self.total_ischemia_time.max()),
        }


# =============================================================================
# OPTIMIZATION SIMULATION
# =============================================================================

def run_single_configuration(cell_size: float,
                              pattern_name: str,
                              cycle_period: float,
                              simulation_time: float = 7200,  # 2 hours
                              timestep: float = 10,  # 10 second steps
                              body_mass: float = 75,
                              incline_angle: float = DEFAULT_INCLINE_ANGLE,
                              verbose: bool = False,
                              track_deformation: bool = False) -> Dict:
    """
    Run simulation for a single mattress configuration.

    Now includes tissue deformation tracking:
    - Tissue compresses over time under sustained pressure
    - Creep behavior adds to compression with duration
    - Contact area changes affect pressure distribution
    - Deep tissue stress tracked for DTI risk

    Args:
        cell_size: Cell size in cm
        pattern_name: Name of movement pattern
        cycle_period: Cycle period in seconds
        simulation_time: Total simulation time in seconds
        timestep: Simulation timestep in seconds
        body_mass: Patient body mass in kg
        incline_angle: Head of bed elevation in degrees
        verbose: Print progress
        track_deformation: Track detailed deformation metrics (slower)

    Returns:
        Dictionary of results and metrics
    """
    # Create mattress
    pattern = MOVEMENT_PATTERNS.get(pattern_name, AlternatingCheckerboard())
    mattress = MultiDynamicAirMattress(
        cell_size_cm=cell_size,
        movement_pattern=pattern,
        cycle_period_sec=cycle_period
    )

    # Create SMPL body model for this simulation
    # Each simulation needs its own model to track deformation state independently
    body_model = SMPLBodyPressureModel(body_mass, incline_angle)

    # Create risk model
    risk_model = PressureUlcerRiskModel(mattress.shape)

    # Run simulation
    times = np.arange(0, simulation_time, timestep)

    pressure_history = []
    damage_history = []
    deformation_history = [] if track_deformation else None

    for t in times:
        # Update mattress state
        mattress.update(t)

        # Get body pressure map with deformation effects
        # Pass timestep for creep calculation
        if track_deformation:
            body_pressure, body_shear, deform_data = body_model.calculate_pressure_map(
                mattress.rows, mattress.cols, dt=timestep, return_deformation=True
            )
        else:
            body_pressure, body_shear = body_model.calculate_pressure_map(
                mattress.rows, mattress.cols, dt=timestep
            )

        # Get effective interface pressure (mattress modulates body pressure)
        interface_pressure = mattress.get_effective_interface_pressure(body_pressure)

        # Shear is reduced when cells deflate (less friction surface)
        effective_shear = body_shear * mattress.cell_state

        # Update risk model with pressure AND shear
        risk_model.update(interface_pressure, effective_shear, timestep)

        # Record history
        pressure_history.append({
            'time': t,
            'peak_pressure': float(interface_pressure.max()),
            'mean_pressure': float(interface_pressure[interface_pressure > 0].mean())
                             if np.any(interface_pressure > 0) else 0,
            'cells_above_threshold': int(np.sum(interface_pressure > CAPILLARY_CLOSING_PRESSURE)),
        })

        damage_history.append({
            'time': t,
            **risk_model.get_metrics()
        })

        if track_deformation and deform_data:
            deformation_history.append({
                'time': t,
                'avg_compression': deform_data.get('avg_compression', 0),
                'max_compression': deform_data.get('max_compression', 0),
                'vertices_in_contact': deform_data.get('vertices_in_contact', 0),
                'peak_deep_tissue_stress': float(deform_data.get('deep_tissue_stress_map', np.array([0])).max()),
            })

    # Final metrics
    final_metrics = risk_model.get_metrics()

    # Get final pressure map
    final_body_pressure, _ = body_model.calculate_pressure_map(mattress.rows, mattress.cols)
    final_pressure_map = mattress.get_effective_interface_pressure(final_body_pressure)

    result = {
        'config': {
            'cell_size': cell_size,
            'pattern': pattern_name,
            'pattern_description': pattern.description,
            'cycle_period': cycle_period,
            'grid_shape': mattress.shape,
            'total_cells': mattress.rows * mattress.cols,
        },
        'final_metrics': final_metrics,
        'pressure_history': pressure_history,
        'damage_history': damage_history,
        'final_damage_map': risk_model.damage.copy(),
        'final_pressure_map': final_pressure_map,
    }

    if track_deformation:
        result['deformation_history'] = deformation_history
        # Get final deformation state
        if body_model.deformation_model.current_compression is not None:
            result['final_deformation'] = {
                'max_compression': float(body_model.deformation_model.current_compression.max()),
                'avg_compression': float(np.mean(body_model.deformation_model.current_compression[
                    body_model.deformation_model.vertex_in_contact
                ])) if np.any(body_model.deformation_model.vertex_in_contact) else 0,
                'total_creep': float(np.sum(body_model.deformation_model.creep_compression)),
            }

    return result


def run_optimization(cell_sizes: List[float] = None,
                     patterns: List[str] = None,
                     cycle_periods: List[float] = None,
                     simulation_time: float = 7200,
                     body_mass: float = 75,
                     incline_angle: float = DEFAULT_INCLINE_ANGLE,
                     verbose: bool = True) -> Dict:
    """
    Run optimization across all configuration combinations.

    Args:
        cell_sizes: List of cell sizes to test (cm)
        patterns: List of pattern names to test
        cycle_periods: List of cycle periods to test (seconds)
        simulation_time: Simulation duration (seconds)
        body_mass: Patient body mass (kg)
        incline_angle: Head of bed elevation in degrees
        verbose: Print progress

    Returns:
        Dictionary with all results and rankings
    """
    # Defaults
    if cell_sizes is None:
        cell_sizes = [3, 5, 7, 10, 15]
    if patterns is None:
        patterns = list(MOVEMENT_PATTERNS.keys())
    if cycle_periods is None:
        cycle_periods = [60, 180, 300, 600]  # 1, 3, 5, 10 minutes

    all_results = []
    total_configs = len(cell_sizes) * len(patterns) * len(cycle_periods)

    if verbose:
        print(f"\n{'='*70}")
        print("MULTI-DYNAMIC AIR MATTRESS OPTIMIZATION")
        print(f"{'='*70}")
        print(f"\nTesting {total_configs} configurations:")
        print(f"  Cell sizes: {cell_sizes} cm")
        print(f"  Patterns: {len(patterns)} types")
        print(f"  Cycle periods: {[p/60 for p in cycle_periods]} minutes")
        print(f"  Simulation time: {simulation_time/60:.0f} minutes")
        print(f"  Body mass: {body_mass} kg")
        print(f"  Bed incline: {incline_angle}° (head of bed elevation)")
        print(f"\n{'='*70}\n")

    config_num = 0
    for cell_size in cell_sizes:
        for pattern in patterns:
            for cycle in cycle_periods:
                config_num += 1
                if verbose:
                    print(f"[{config_num}/{total_configs}] "
                          f"Cell: {cell_size}cm, Pattern: {pattern}, "
                          f"Cycle: {cycle/60:.1f}min...", end=" ")

                result = run_single_configuration(
                    cell_size=cell_size,
                    pattern_name=pattern,
                    cycle_period=cycle,
                    simulation_time=simulation_time,
                    body_mass=body_mass,
                    incline_angle=incline_angle
                )

                all_results.append(result)

                if verbose:
                    score = result['final_metrics']['risk_score']
                    peak = result['final_metrics']['peak_damage']
                    print(f"Risk: {score:.1f}, Peak Damage: {peak:.3f}")

    # Rank results
    all_results.sort(key=lambda x: x['final_metrics']['risk_score'])

    # Find optimal configuration
    optimal = all_results[0]

    if verbose:
        print(f"\n{'='*70}")
        print("OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        print(f"\n🏆 OPTIMAL CONFIGURATION:")
        print(f"   Cell Size: {optimal['config']['cell_size']} cm")
        print(f"   Pattern: {optimal['config']['pattern']}")
        print(f"   Description: {optimal['config']['pattern_description']}")
        print(f"   Cycle Period: {optimal['config']['cycle_period']/60:.1f} minutes")
        print(f"   Grid: {optimal['config']['grid_shape'][0]} x {optimal['config']['grid_shape'][1]} cells")
        print(f"\n   Final Risk Score: {optimal['final_metrics']['risk_score']:.2f}")
        print(f"   Peak Damage: {optimal['final_metrics']['peak_damage']:.4f}")
        print(f"   Cells at Risk: {optimal['final_metrics']['cells_at_risk']}")

        print(f"\n📊 TOP 5 CONFIGURATIONS:")
        for i, r in enumerate(all_results[:5]):
            print(f"   {i+1}. Cell={r['config']['cell_size']}cm, "
                  f"Pattern={r['config']['pattern']}, "
                  f"Cycle={r['config']['cycle_period']/60:.0f}min "
                  f"→ Risk: {r['final_metrics']['risk_score']:.2f}")

        print(f"\n❌ WORST 3 CONFIGURATIONS:")
        for i, r in enumerate(all_results[-3:]):
            print(f"   {len(all_results)-2+i}. Cell={r['config']['cell_size']}cm, "
                  f"Pattern={r['config']['pattern']}, "
                  f"Cycle={r['config']['cycle_period']/60:.0f}min "
                  f"→ Risk: {r['final_metrics']['risk_score']:.2f}")

    return {
        'all_results': all_results,
        'optimal': optimal,
        'rankings': [(r['config'], r['final_metrics']['risk_score'])
                     for r in all_results],
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_optimization_visualization(results: Dict,
                                       save_path: str = 'mattress_optimization.html'):
    """Create interactive visualization of optimization results."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    all_results = results['all_results']
    optimal = results['optimal']

    # Extract data for plotting
    cell_sizes = sorted(set(r['config']['cell_size'] for r in all_results))
    patterns = sorted(set(r['config']['pattern'] for r in all_results))
    cycle_periods = sorted(set(r['config']['cycle_period'] for r in all_results))

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Risk Score by Cell Size',
            'Risk Score by Pattern',
            'Risk Score by Cycle Period',
            'Cell Size vs Cycle Period Heatmap',
            'Optimal Configuration - Damage Map',
            'Optimal Configuration - Pressure Relief Timeline',
        ],
        specs=[
            [{'type': 'box'}, {'type': 'box'}],
            [{'type': 'box'}, {'type': 'heatmap'}],
            [{'type': 'heatmap'}, {'type': 'scatter'}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # 1. Risk by Cell Size (box plot)
    for cs in cell_sizes:
        scores = [r['final_metrics']['risk_score']
                  for r in all_results if r['config']['cell_size'] == cs]
        fig.add_trace(go.Box(y=scores, name=f'{cs}cm', boxmean=True), row=1, col=1)

    # 2. Risk by Pattern (box plot)
    for p in patterns:
        scores = [r['final_metrics']['risk_score']
                  for r in all_results if r['config']['pattern'] == p]
        fig.add_trace(go.Box(y=scores, name=p[:15], boxmean=True), row=1, col=2)

    # 3. Risk by Cycle Period (box plot)
    for cp in cycle_periods:
        scores = [r['final_metrics']['risk_score']
                  for r in all_results if r['config']['cycle_period'] == cp]
        fig.add_trace(go.Box(y=scores, name=f'{cp/60:.0f}min', boxmean=True), row=2, col=1)

    # 4. Heatmap: Cell Size vs Cycle Period (average across patterns)
    heatmap_data = np.zeros((len(cell_sizes), len(cycle_periods)))
    for i, cs in enumerate(cell_sizes):
        for j, cp in enumerate(cycle_periods):
            scores = [r['final_metrics']['risk_score']
                      for r in all_results
                      if r['config']['cell_size'] == cs and r['config']['cycle_period'] == cp]
            heatmap_data[i, j] = np.mean(scores) if scores else 0

    fig.add_trace(go.Heatmap(
        z=heatmap_data,
        x=[f'{cp/60:.0f}min' for cp in cycle_periods],
        y=[f'{cs}cm' for cs in cell_sizes],
        colorscale='RdYlGn_r',
        colorbar=dict(title='Risk Score', x=1.02),
    ), row=2, col=2)

    # 5. Optimal damage map
    damage_map = optimal['final_damage_map']
    fig.add_trace(go.Heatmap(
        z=damage_map,
        colorscale='Reds',
        colorbar=dict(title='Damage', x=0.45, len=0.3, y=0.15),
    ), row=3, col=1)

    # 6. Pressure timeline
    history = optimal['pressure_history']
    times = [h['time']/60 for h in history]
    peak_pressures = [h['peak_pressure'] for h in history]

    fig.add_trace(go.Scatter(
        x=times, y=peak_pressures,
        mode='lines', name='Peak Pressure',
        line=dict(color='blue', width=2),
    ), row=3, col=2)

    fig.add_hline(y=CAPILLARY_CLOSING_PRESSURE, line_dash='dash',
                  line_color='red', row=3, col=2,
                  annotation_text=f'Ischemia Threshold ({CAPILLARY_CLOSING_PRESSURE} mmHg)')

    # Layout
    fig.update_layout(
        title=dict(
            text=f'<b>Multi-Dynamic Air Mattress Optimization Results</b><br>'
                 f'<sup>Optimal: {optimal["config"]["cell_size"]}cm cells, '
                 f'{optimal["config"]["pattern"]}, '
                 f'{optimal["config"]["cycle_period"]/60:.0f}min cycle</sup>',
            x=0.5,
            font=dict(size=18),
        ),
        height=1000,
        width=1400,
        showlegend=False,
    )

    fig.update_xaxes(title_text='Cell Size', row=1, col=1)
    fig.update_xaxes(title_text='Pattern', row=1, col=2)
    fig.update_xaxes(title_text='Cycle Period', row=2, col=1)
    fig.update_xaxes(title_text='Cycle Period', row=2, col=2)
    fig.update_xaxes(title_text='Column', row=3, col=1)
    fig.update_xaxes(title_text='Time (minutes)', row=3, col=2)

    fig.update_yaxes(title_text='Risk Score', row=1, col=1)
    fig.update_yaxes(title_text='Risk Score', row=1, col=2)
    fig.update_yaxes(title_text='Risk Score', row=2, col=1)
    fig.update_yaxes(title_text='Cell Size', row=2, col=2)
    fig.update_yaxes(title_text='Row (Head→Feet)', row=3, col=1)
    fig.update_yaxes(title_text='Pressure (mmHg)', row=3, col=2)

    fig.write_html(save_path, include_plotlyjs=True, full_html=True)
    print(f"\nSaved visualization: {save_path}")

    return fig


def create_pattern_comparison_visualization(results: Dict,
                                             save_path: str = 'pattern_comparison.html'):
    """Create detailed comparison of movement patterns."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    all_results = results['all_results']
    patterns = sorted(set(r['config']['pattern'] for r in all_results))

    # Get best result for each pattern
    pattern_best = {}
    for pattern in patterns:
        pattern_results = [r for r in all_results if r['config']['pattern'] == pattern]
        pattern_results.sort(key=lambda x: x['final_metrics']['risk_score'])
        pattern_best[pattern] = pattern_results[0]

    # Create figure with pattern visualizations
    n_patterns = len(patterns)
    cols = 4
    rows = (n_patterns + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[MOVEMENT_PATTERNS[p].name for p in patterns],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    # Visualize each pattern at a specific phase
    for idx, pattern_name in enumerate(patterns):
        pattern = MOVEMENT_PATTERNS[pattern_name]

        # Create pattern visualization grid
        test_rows, test_cols = 20, 10
        pattern_grid = np.zeros((test_rows, test_cols))

        phase = 0.25  # Show at 25% through cycle
        for i in range(test_rows):
            for j in range(test_cols):
                pattern_grid[i, j] = pattern.get_cell_state(
                    i, j, test_rows, test_cols, phase
                )

        row = idx // cols + 1
        col = idx % cols + 1

        fig.add_trace(go.Heatmap(
            z=pattern_grid,
            colorscale='RdYlGn',
            showscale=(idx == 0),
            colorbar=dict(title='Inflation', x=1.02) if idx == 0 else None,
        ), row=row, col=col)

    fig.update_layout(
        title=dict(
            text='<b>Movement Pattern Comparison</b><br>'
                 '<sup>Green = Inflated (support), Red = Deflated (relief)</sup>',
            x=0.5,
            font=dict(size=18),
        ),
        height=300 * rows,
        width=1400,
    )

    fig.write_html(save_path, include_plotlyjs=True, full_html=True)
    print(f"Saved: {save_path}")

    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-DYNAMIC AIR MATTRESS OPTIMIZATION FOR PRESSURE ULCER PREVENTION")
    print("="*70)
    print("\nQuadriplegic patient configuration:")
    print(f"  - 30° head-of-bed incline (for respiratory/feeding)")
    print(f"  - Weight shift toward sacrum modeled")
    print(f"  - Shear stress from gravity included")
    print(f"  - Quadriplegic tissue factors applied")

    # Run optimization with various configurations
    results = run_optimization(
        cell_sizes=[3, 5, 7, 10, 15],
        patterns=list(MOVEMENT_PATTERNS.keys()),
        cycle_periods=[60, 180, 300, 600],  # 1, 3, 5, 10 minutes
        simulation_time=7200,  # 2 hours
        body_mass=75,
        incline_angle=30,  # 30 degree head-of-bed elevation
        verbose=True
    )

    # Create visualizations
    print("\nGenerating visualizations...")
    create_optimization_visualization(results, 'mattress_optimization.html')
    create_pattern_comparison_visualization(results, 'pattern_comparison.html')

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print("\nOutput files:")
    print("  - mattress_optimization.html (main results)")
    print("  - pattern_comparison.html (pattern visualizations)")
