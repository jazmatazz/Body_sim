#!/usr/bin/env python3
"""
Predicted Time-to-Damage Analysis
==================================

Estimates time until tissue damage occurs for each mattress configuration
using literature-based pressure-time injury thresholds.

Models used:
- Reswick & Rogers (1976): Inverse pressure-time relationship
- Gefen (2008): Deep tissue injury thresholds
- Linder-Ganz et al. (2006): Cell death thresholds

Key thresholds:
- 32 mmHg (capillary closing): Ischemia begins
- 70 mmHg sustained: Damage in ~2 hours
- 150+ mmHg: Damage in <1 hour
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import zoom
from fpdf import FPDF
import tempfile
import os
from multidynamic_mattress_optimization import (
    SMPLBodyPressureModel, MultiDynamicAirMattress,
    MOVEMENT_PATTERNS
)
from evolved_pattern import EvolvedOptimalPattern
from all_mattress_pti import RealisticMattressState


# =============================================================================
# REALISTIC PRESSURE REDISTRIBUTION FACTORS
# =============================================================================
# Based on clinical pressure mapping studies (Defloor 2000, Reenalda 2009)
FOAM_REDISTRIBUTION = 0.15      # Foam reduces peak pressure by ~85%
APM_INFLATED_REDISTRIBUTION = 0.15  # Inflated APM cells similar to foam
APM_DEFLATED_RELIEF = 0.85      # Deflated cells provide additional 85% relief

# 30-degree lateral wedge pressure factor
# At 30° tilt, sacral pressure reduces to ~50% of supine (Defloor 2000, Colin 1996)
# This avoids full trochanter loading while still providing sacral relief
WEDGE_30_DEG_SACRAL_FACTOR = 0.50


# =============================================================================
# STII - STRAIN-TIME INJURY INDEX (Linder-Ganz/Gefen 2007)
# =============================================================================
# Sigmoid strain-time relationship:
#   - Tissue tolerates 65% strain for ~1 hour
#   - Tolerance drops to 40% strain by 4 hours
#   - Transition centered at ~90 minutes
#
# STII models cumulative strain exposure relative to time-varying threshold

# Strain tolerance parameters
STII_STRAIN_THRESHOLD_1H = 0.65   # 65% strain tolerable for 1 hour
STII_STRAIN_THRESHOLD_4H = 0.40   # 40% strain tolerable for 4+ hours
STII_TRANSITION_TIME = 5400.0     # 90 min = inflection point (seconds)
STII_STEEPNESS = 0.002            # Sigmoid steepness parameter

# Three-phase damage response
STII_LAG_PHASE_HOURS = 0.5        # Initial 30-min lag (cellular adaptation)
STII_RAPID_PHASE_HOURS = 2.0      # Rapid escalation phase
STII_PLATEAU_PHASE_HOURS = 6.0    # Damage plateaus after this

# Pressure-to-strain conversion (Gefen approximation)
# Interface pressure (mmHg) → tissue strain (%)
PRESSURE_TO_STRAIN_FACTOR = 0.008  # 1 mmHg ≈ 0.8% strain at bone-muscle interface


def calculate_strain_threshold(time_seconds: float) -> float:
    """
    Calculate time-varying strain threshold using sigmoid model.

    Based on Linder-Ganz & Gefen (2007): tissue tolerance decreases
    sigmoidally from 65% strain (1h) to 40% strain (4h+).

    Args:
        time_seconds: Cumulative loading time in seconds

    Returns:
        Maximum tolerable strain (0-1) at this time point
    """
    # Sigmoid transition from high to low tolerance
    sigmoid = 1 / (1 + np.exp(STII_STEEPNESS * (time_seconds - STII_TRANSITION_TIME)))

    # Interpolate between 1h and 4h thresholds
    threshold = STII_STRAIN_THRESHOLD_4H + \
                (STII_STRAIN_THRESHOLD_1H - STII_STRAIN_THRESHOLD_4H) * sigmoid

    return threshold


def calculate_stii(pressure_mmHg: float, time_seconds: float,
                   cumulative_strain_exposure: float = 0.0) -> dict:
    """
    Calculate Strain-Time Injury Index (Linder-Ganz/Gefen model).

    STII measures how close current strain is to the time-varying threshold.
    STII = actual_strain / threshold_strain

    STII > 1.0 indicates injury threshold exceeded.

    Args:
        pressure_mmHg: Current interface pressure
        time_seconds: Current time in simulation
        cumulative_strain_exposure: Previous cumulative STII (for tracking)

    Returns:
        Dict with STII metrics
    """
    # Convert pressure to tissue strain
    actual_strain = pressure_mmHg * PRESSURE_TO_STRAIN_FACTOR

    # Get time-varying threshold
    threshold_strain = calculate_strain_threshold(time_seconds)

    # Calculate instantaneous STII
    instant_stii = actual_strain / threshold_strain if threshold_strain > 0 else 0

    # Apply three-phase damage response
    time_hours = time_seconds / 3600

    if time_hours < STII_LAG_PHASE_HOURS:
        # Lag phase: cellular adaptation, reduced damage rate
        phase_factor = 0.3
    elif time_hours < STII_RAPID_PHASE_HOURS:
        # Rapid phase: accelerated damage
        phase_factor = 1.5
    else:
        # Plateau phase: damage rate stabilizes
        phase_factor = 1.0

    # Weighted STII accumulation
    weighted_stii = instant_stii * phase_factor

    return {
        'instant_stii': instant_stii,
        'weighted_stii': weighted_stii,
        'actual_strain': actual_strain,
        'threshold_strain': threshold_strain,
        'phase_factor': phase_factor,
        'threshold_exceeded': instant_stii > 1.0,
    }


# =============================================================================
# PERFUSION RECOVERY MODEL
# =============================================================================
# Based on reactive hyperemia studies (Bader 1990, Herrman 1999, Makhsous 2007)
#
# Key findings:
# - Blood flow doesn't instantly return when pressure is removed
# - Recovery time depends on duration and severity of ischemia
# - Tissue is more vulnerable during incomplete recovery
# - Reactive hyperemia (excess blood flow) occurs after ischemia

# Recovery time constants (minutes)
PERFUSION_RECOVERY_RATE_BASE = 0.1      # Base recovery rate per minute (10% per min)
PERFUSION_RECOVERY_RATE_FAST = 0.3      # Fast recovery for short ischemia
PERFUSION_RECOVERY_RATE_SLOW = 0.03    # Slow recovery for prolonged ischemia

# Ischemia duration thresholds (minutes)
SHORT_ISCHEMIA_THRESHOLD = 10.0         # <10 min = fast recovery
MEDIUM_ISCHEMIA_THRESHOLD = 30.0        # 10-30 min = normal recovery
PROLONGED_ISCHEMIA_THRESHOLD = 60.0     # >60 min = slow recovery

# Vulnerability during incomplete recovery
RECOVERY_VULNERABILITY_FACTOR = 1.5     # 50% more vulnerable during recovery
INCOMPLETE_RECOVERY_THRESHOLD = 0.8     # <80% perfused = incomplete recovery


class PerfusionState:
    """
    Tracks tissue perfusion state with realistic recovery dynamics.

    Perfusion level: 0.0 = fully ischemic, 1.0 = fully perfused

    Based on:
    - Bader (1990): Reactive hyperemia measurements
    - Herrman (1999): Tissue oxygenation recovery
    - Makhsous (2007): Perfusion recovery after pressure relief
    """

    def __init__(self):
        self.perfusion = 1.0            # Current perfusion level (0-1)
        self.ischemia_duration = 0.0    # Cumulative ischemia time (minutes)
        self.recovery_duration = 0.0    # Time spent recovering (minutes)
        self.is_ischemic = False        # Currently under damaging pressure
        self.peak_ischemia = 0.0        # Track longest ischemic episode

    def update(self, pressure_mmHg: float, dt_minutes: float,
               threshold_mmHg: float = 32.0) -> float:
        """
        Update perfusion state based on current pressure.

        Args:
            pressure_mmHg: Current interface pressure
            dt_minutes: Time step in minutes
            threshold_mmHg: Pressure above which ischemia occurs

        Returns:
            Current perfusion level (0-1)
        """
        if pressure_mmHg >= threshold_mmHg:
            # ISCHEMIA: Pressure above threshold
            self.is_ischemic = True
            self.ischemia_duration += dt_minutes
            self.recovery_duration = 0.0  # Reset recovery counter

            # Perfusion decreases based on pressure magnitude
            # Higher pressure = faster perfusion loss
            pressure_ratio = pressure_mmHg / threshold_mmHg
            perfusion_loss_rate = 0.1 * pressure_ratio  # Per minute
            self.perfusion = max(0.0, self.perfusion - perfusion_loss_rate * dt_minutes)

            # Track peak ischemia duration for this episode
            self.peak_ischemia = max(self.peak_ischemia, self.ischemia_duration)

        else:
            # RECOVERY: Pressure below threshold
            if self.is_ischemic:
                # Just transitioned to recovery - record ischemia duration
                self.peak_ischemia = self.ischemia_duration
                self.is_ischemic = False
                self.ischemia_duration = 0.0

            self.recovery_duration += dt_minutes

            # Recovery rate depends on how long tissue was ischemic
            if self.peak_ischemia < SHORT_ISCHEMIA_THRESHOLD:
                recovery_rate = PERFUSION_RECOVERY_RATE_FAST
            elif self.peak_ischemia < PROLONGED_ISCHEMIA_THRESHOLD:
                recovery_rate = PERFUSION_RECOVERY_RATE_BASE
            else:
                recovery_rate = PERFUSION_RECOVERY_RATE_SLOW

            # Perfusion recovers toward 1.0
            recovery_increment = recovery_rate * dt_minutes
            self.perfusion = min(1.0, self.perfusion + recovery_increment)

            # Reset peak ischemia tracking once fully recovered
            if self.perfusion >= 0.99:
                self.peak_ischemia = 0.0

        return self.perfusion

    def get_vulnerability_factor(self) -> float:
        """
        Get damage vulnerability multiplier based on perfusion state.

        Tissue is more vulnerable when:
        1. Currently ischemic (perfusion low)
        2. Recovering but not yet fully perfused

        Returns:
            Vulnerability multiplier (1.0 = normal, >1.0 = increased vulnerability)
        """
        if self.perfusion >= INCOMPLETE_RECOVERY_THRESHOLD:
            return 1.0  # Fully or nearly fully perfused

        # Vulnerability increases as perfusion decreases
        # At perfusion = 0: vulnerability = RECOVERY_VULNERABILITY_FACTOR
        # At perfusion = INCOMPLETE_RECOVERY_THRESHOLD: vulnerability = 1.0
        vulnerability = 1.0 + (1.0 - self.perfusion / INCOMPLETE_RECOVERY_THRESHOLD) * \
                        (RECOVERY_VULNERABILITY_FACTOR - 1.0)

        return vulnerability


# =============================================================================
# DAMAGE THRESHOLD MODELS
# =============================================================================

def reswick_rogers_threshold(pressure_mmHg: float) -> float:
    """
    Reswick-Rogers (1976) pressure-time threshold.

    Returns safe exposure time in hours for given pressure.
    Based on clinical observations of pressure ulcer formation.

    Args:
        pressure_mmHg: Interface pressure in mmHg

    Returns:
        Safe exposure time in hours (inf if below threshold)
    """
    if pressure_mmHg < 32:
        return float('inf')  # Below capillary closing pressure
    elif pressure_mmHg < 70:
        # Linear interpolation: 32 mmHg -> 8h, 70 mmHg -> 2h
        return 8.0 - (pressure_mmHg - 32) * (6.0 / 38)
    elif pressure_mmHg < 150:
        # 70 mmHg -> 2h, 150 mmHg -> 0.5h
        return 2.0 - (pressure_mmHg - 70) * (1.5 / 80)
    elif pressure_mmHg < 300:
        # 150 mmHg -> 0.5h, 300 mmHg -> 0.17h (10 min)
        return 0.5 - (pressure_mmHg - 150) * (0.33 / 150)
    else:
        return 0.17  # ~10 minutes at extreme pressure


def gefen_deep_tissue_threshold(pressure_mmHg: float) -> float:
    """
    Gefen (2008) deep tissue injury threshold.

    Deep tissue injury occurs faster than surface damage because
    stress concentrates at bone-muscle interface.

    Args:
        pressure_mmHg: Interface pressure in mmHg

    Returns:
        Time to deep tissue injury in hours
    """
    if pressure_mmHg < 32:
        return float('inf')

    # Convert to internal stress (roughly 2x interface pressure at bone)
    internal_stress_kPa = pressure_mmHg * 0.1333 * 2.0

    # Gefen model: cell death at ~10-20 kPa sustained stress
    # Time decreases exponentially with stress
    if internal_stress_kPa < 10:
        return float('inf')
    elif internal_stress_kPa < 20:
        # 10 kPa -> 4h, 20 kPa -> 1h
        return 4.0 * np.exp(-0.139 * (internal_stress_kPa - 10))
    else:
        # Rapid damage above 20 kPa
        return 1.0 * np.exp(-0.05 * (internal_stress_kPa - 20))


def compute_damage_accumulation(pressure_history: list, dt_hours: float,
                                 threshold_func=reswick_rogers_threshold,
                                 use_perfusion_recovery: bool = True) -> dict:
    """
    Compute cumulative damage using multiple models:
    1. Miner's rule with perfusion recovery (traditional)
    2. STII - Strain-Time Injury Index (Linder-Ganz/Gefen 2007)

    Damage fraction D = Σ (t_i / T_safe_i) × vulnerability_i
    where:
    - t_i is time at pressure_i
    - T_safe_i is safe duration at that pressure
    - vulnerability_i accounts for incomplete perfusion recovery

    Damage occurs when D >= 1.0

    Args:
        pressure_history: List of pressure values over time (mmHg)
        dt_hours: Time step in hours
        threshold_func: Function returning safe time for given pressure
        use_perfusion_recovery: If True, apply perfusion recovery modeling

    Returns:
        Dict with damage metrics including STII and perfusion history
    """
    cumulative_damage = 0.0
    cumulative_stii = 0.0
    damage_history = []
    stii_history = []
    perfusion_history = []
    vulnerability_history = []
    time_to_damage = None
    time_to_stii_damage = None

    # Initialize perfusion state tracker
    perfusion_state = PerfusionState() if use_perfusion_recovery else None
    dt_minutes = dt_hours * 60
    dt_seconds = dt_hours * 3600

    for i, pressure in enumerate(pressure_history):
        time_seconds = i * dt_seconds

        # Update perfusion state
        if perfusion_state:
            perfusion = perfusion_state.update(pressure, dt_minutes)
            vulnerability = perfusion_state.get_vulnerability_factor()
        else:
            perfusion = 1.0 if pressure < 32 else 0.5
            vulnerability = 1.0

        perfusion_history.append(perfusion)
        vulnerability_history.append(vulnerability)

        # Calculate STII for this time point
        stii_result = calculate_stii(pressure, time_seconds, cumulative_stii)
        cumulative_stii += stii_result['weighted_stii'] * dt_hours
        stii_history.append(cumulative_stii)

        if time_to_stii_damage is None and cumulative_stii >= 1.0:
            time_to_stii_damage = i * dt_hours

        # Calculate traditional damage with vulnerability multiplier
        safe_time = threshold_func(pressure)
        if safe_time > 0 and safe_time != float('inf'):
            # Base damage increment
            damage_increment = dt_hours / safe_time

            # Apply vulnerability factor (incomplete perfusion = more damage)
            if use_perfusion_recovery:
                damage_increment *= vulnerability

            cumulative_damage += damage_increment

        damage_history.append(cumulative_damage)

        if time_to_damage is None and cumulative_damage >= 1.0:
            time_to_damage = i * dt_hours

    return {
        # Traditional damage metrics
        'cumulative_damage': cumulative_damage,
        'damage_history': damage_history,
        'time_to_damage_hours': time_to_damage,
        'final_damage_fraction': cumulative_damage,
        # STII metrics (Linder-Ganz/Gefen 2007)
        'cumulative_stii': cumulative_stii,
        'stii_history': stii_history,
        'time_to_stii_damage_hours': time_to_stii_damage,
        # Perfusion metrics
        'perfusion_history': perfusion_history,
        'vulnerability_history': vulnerability_history,
        'min_perfusion': min(perfusion_history) if perfusion_history else 1.0,
        'avg_perfusion': np.mean(perfusion_history) if perfusion_history else 1.0,
    }


def create_time_to_damage_analysis():
    """Create time-to-damage prediction for all mattress configurations."""

    print("Generating Time-to-Damage Prediction...")
    print("=" * 60)
    print("Using damage accumulation model (Miner's rule) with:")
    print("  - Reswick-Rogers pressure-time threshold")
    print("  - Gefen deep tissue injury model")
    print("  - Perfusion recovery dynamics")
    print("=" * 60)

    # Generate body pressure map
    model = SMPLBodyPressureModel(75, 30)
    body_pressure, body_shear = model.calculate_pressure_map(40, 18)

    print(f"\nBase body pressure: Peak={body_pressure.max():.1f} mmHg")

    # All configurations
    configs = {
        'Manual Repositioning (2h)': {'type': 'foam_manual'},  # Foam with manual repositioning
    }

    for name, pattern in MOVEMENT_PATTERNS.items():
        display_name = pattern.name if hasattr(pattern, 'name') else name.replace('_', ' ').title()
        configs[display_name] = {'type': 'apm', 'pattern': pattern, 'key': name}

    configs['Evolved Optimal'] = {'type': 'apm', 'pattern': EvolvedOptimalPattern(), 'key': 'evolved_optimal'}

    print(f"\nAnalyzing {len(configs)} configurations:")
    for name in configs:
        print(f"  - {name}")

    # Simulate 8 hours (typical overnight duration)
    total_time_hours = 8.0
    total_time_sec = total_time_hours * 3600
    n_samples = 480  # Every minute
    time_points_sec = np.linspace(0, total_time_sec, n_samples)
    dt_hours = (time_points_sec[1] - time_points_sec[0]) / 3600

    # Sacrum location (highest risk)
    h, w = body_pressure.shape
    sacrum_row, sacrum_col = int(0.4 * h), int(0.5 * w)

    # Collect pressure history for each config
    all_results = {}

    for config_name, config in configs.items():
        print(f"\n  Processing: {config_name}")

        if config['type'] == 'apm':
            pattern = config['pattern']
            mattress = MultiDynamicAirMattress(cell_size_cm=5, movement_pattern=pattern)
            realistic_state = RealisticMattressState(mattress.rows, mattress.cols)

            scale_y = mattress.rows / body_pressure.shape[0]
            scale_x = mattress.cols / body_pressure.shape[1]
            body_resampled = zoom(body_pressure, (scale_y, scale_x), order=1)

        pressure_history = []

        for time_sec in time_points_sec:
            if config['type'] == 'foam_manual':
                # Foam mattress with manual repositioning every 2 hours
                # Pattern: supine (0-2h) -> left 30° wedge (2-4h) -> supine (4-6h) -> right 30° wedge (6-8h)
                time_hours = time_sec / 3600
                cycle_position = int(time_hours / 2) % 4  # 0=supine, 1=left, 2=supine, 3=right

                if cycle_position in [0, 2]:  # Supine (flat on back)
                    # Sacrum under full pressure
                    effective_pressure = body_pressure * FOAM_REDISTRIBUTION
                else:  # 30° lateral wedge (left or right side)
                    # Sacrum partially relieved at 30° tilt
                    # Avoids full trochanter loading while providing sacral relief
                    effective_pressure = body_pressure * FOAM_REDISTRIBUTION * WEDGE_30_DEG_SACRAL_FACTOR
            elif config['type'] == 'foam':
                # Foam mattress without repositioning (static)
                effective_pressure = body_pressure * FOAM_REDISTRIBUTION
            else:
                mattress.update(time_sec)
                target_state = mattress.cell_state.copy()
                actual_state = realistic_state.update(target_state, time_sec)

                # Start with inflated APM baseline (similar to foam)
                effective_pressure = body_resampled * APM_INFLATED_REDISTRIBUTION
                for row in range(mattress.rows):
                    for col in range(mattress.cols):
                        inflation = actual_state[row, col]
                        if inflation < 0.5:
                            # Deflated cells provide additional pressure relief
                            relief_factor = 1 - inflation
                            pressure_relieved = effective_pressure[row, col] * relief_factor * APM_DEFLATED_RELIEF
                            effective_pressure[row, col] -= pressure_relieved

                # Resample back
                scale_y_back = body_pressure.shape[0] / effective_pressure.shape[0]
                scale_x_back = body_pressure.shape[1] / effective_pressure.shape[1]
                effective_pressure = zoom(effective_pressure, (scale_y_back, scale_x_back), order=1)

            # Get sacrum pressure (peak in 3x3 area)
            r_start = max(0, sacrum_row - 1)
            r_end = min(h, sacrum_row + 2)
            c_start = max(0, sacrum_col - 1)
            c_end = min(w, sacrum_col + 2)
            sacrum_pressure = effective_pressure[r_start:r_end, c_start:c_end].max()
            pressure_history.append(sacrum_pressure)

        # Compute damage with both models
        reswick_damage = compute_damage_accumulation(
            pressure_history, dt_hours, reswick_rogers_threshold
        )
        gefen_damage = compute_damage_accumulation(
            pressure_history, dt_hours, gefen_deep_tissue_threshold
        )

        all_results[config_name] = {
            'pressure_history': pressure_history,
            'avg_pressure': np.mean(pressure_history),
            'peak_pressure': np.max(pressure_history),
            'reswick': reswick_damage,
            'gefen': gefen_damage,
            'time_hours': [t / 3600 for t in time_points_sec],
        }

        # Report
        rr_ttd = reswick_damage['time_to_damage_hours']
        gf_ttd = gefen_damage['time_to_damage_hours']
        print(f"    Avg Pressure: {np.mean(pressure_history):.1f} mmHg")
        print(f"    Time to Surface Damage (R-R): {rr_ttd:.1f}h" if rr_ttd else "    Time to Surface Damage (R-R): >8h")
        print(f"    Time to Deep Tissue Injury (Gefen): {gf_ttd:.1f}h" if gf_ttd else "    Time to Deep Tissue Injury (Gefen): >8h")

    # Create visualization
    create_damage_visualization(all_results, total_time_hours)

    return all_results


def create_damage_visualization(all_results: dict, total_time_hours: float):
    """Create graphs and HTML tables with all damage prediction results."""

    config_names = list(all_results.keys())

    # Extract metrics
    avg_pressures = [all_results[name]['avg_pressure'] for name in config_names]

    reswick_ttd = []
    gefen_ttd = []
    for name in config_names:
        r_ttd = all_results[name]['reswick']['time_to_damage_hours']
        g_ttd = all_results[name]['gefen']['time_to_damage_hours']
        reswick_ttd.append(r_ttd if r_ttd is not None else total_time_hours)
        gefen_ttd.append(g_ttd if g_ttd is not None else total_time_hours)

    reswick_damage = [all_results[name]['reswick']['final_damage_fraction'] for name in config_names]
    stii_values = [all_results[name]['reswick'].get('cumulative_stii', 0) for name in config_names]

    # Colors
    colors = ['#2ecc71' if name == 'Evolved Optimal' else '#e74c3c' if 'Manual Repositioning' in name else '#3498db' for name in config_names]

    # Calculate percent changes
    baseline_pressure = avg_pressures[0]
    baseline_dti = gefen_ttd[0]
    baseline_stii = stii_values[0]
    baseline_damage = reswick_damage[0]

    pct_pressure = [((p - baseline_pressure) / baseline_pressure * 100) if baseline_pressure > 0 else 0 for p in avg_pressures]
    pct_dti = [((d - baseline_dti) / baseline_dti * 100) if baseline_dti > 0 else 0 for d in gefen_ttd]
    pct_stii = [((s - baseline_stii) / baseline_stii * 100) if baseline_stii > 0 else 0 for s in stii_values]
    damage_reduction = [(1 - d / baseline_damage) * 100 for d in reswick_damage]
    dti_efficiency = [(d / baseline_dti * 100) if baseline_dti > 0 else 100 for d in gefen_ttd]

    # Create Plotly figure with 2x3 bar charts (all metrics)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            '<b>Average Pressure (mmHg)</b>',
            '<b>Time to Surface Damage (hours)</b>',
            '<b>Time to Deep Tissue Injury (hours)</b>',
            '<b>STII (Strain-Time Injury Index)</b>',
            '<b>Damage Fraction</b>',
            '<b>Damage Reduction (%)</b>',
        ],
        vertical_spacing=0.25,
        horizontal_spacing=0.08,
    )

    # Row 1, Col 1: Average Pressure
    fig.add_trace(
        go.Bar(x=config_names, y=avg_pressures, marker_color=colors,
               text=[f'{p:.1f}' for p in avg_pressures],
               textposition='outside'),
        row=1, col=1
    )

    # Row 1, Col 2: Surface damage time
    fig.add_trace(
        go.Bar(x=config_names, y=reswick_ttd, marker_color=colors,
               text=[f'{t:.1f}h' if t < total_time_hours else '>8h' for t in reswick_ttd],
               textposition='outside'),
        row=1, col=2
    )

    # Row 1, Col 3: DTI time
    fig.add_trace(
        go.Bar(x=config_names, y=gefen_ttd, marker_color=colors,
               text=[f'{t:.1f}h' if t < total_time_hours else '>8h' for t in gefen_ttd],
               textposition='outside'),
        row=1, col=3
    )

    # Row 2, Col 1: STII
    fig.add_trace(
        go.Bar(x=config_names, y=stii_values, marker_color=colors,
               text=[f'{s:.1f}' for s in stii_values],
               textposition='outside'),
        row=2, col=1
    )

    # Row 2, Col 2: Damage Fraction
    fig.add_trace(
        go.Bar(x=config_names, y=reswick_damage, marker_color=colors,
               text=[f'{d:.2f}' for d in reswick_damage],
               textposition='outside'),
        row=2, col=2
    )

    # Row 2, Col 3: Damage reduction %
    fig.add_trace(
        go.Bar(x=config_names, y=damage_reduction, marker_color=colors,
               text=[f'{d:.0f}%' for d in damage_reduction],
               textposition='outside'),
        row=2, col=3
    )

    fig.update_layout(
        title=dict(
            text='<b>Complete Simulation Results: All Metrics</b><br><sup>8-hour simulation | 13 configurations | STII model (Linder-Ganz & Gefen 2007)</sup>',
            x=0.5, font=dict(size=18)
        ),
        height=900, width=1500, showlegend=False,
    )
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="mmHg", row=1, col=1)
    fig.update_yaxes(title_text="Hours", row=1, col=2)
    fig.update_yaxes(title_text="Hours", row=1, col=3)
    fig.update_yaxes(title_text="STII", row=2, col=1)
    fig.update_yaxes(title_text="Fraction", row=2, col=2)
    fig.update_yaxes(title_text="%", row=2, col=3)

    # Generate Plotly HTML
    plotly_html = fig.to_html(include_plotlyjs=True, full_html=False)

    # Create HTML with graphs + tables
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Simulation Results - All Data</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 40px auto; padding: 20px; }}
        h1 {{ color: #2c3e50; text-align: center; }}
        h2 {{ color: #2c3e50; text-align: center; font-weight: normal; font-size: 14px; }}
        h3 {{ color: #2c3e50; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #2c3e50; color: white; }}
        .evolved {{ background-color: #c8f7c5; }}
        .baseline {{ background-color: #f5b7b1; }}
        .best-header {{ background-color: #2ecc71; color: white; }}
    </style>
</head>
<body>
    {plotly_html}

    <h3>Table 1: Raw Simulation Data</h3>
    <table>
        <tr>
            <th>Configuration</th>
            <th>Avg Pressure (mmHg)</th>
            <th>Surface Damage (hours)</th>
            <th>Time to DTI (hours)</th>
            <th>STII</th>
            <th>Damage Fraction</th>
        </tr>
"""

    for i, name in enumerate(config_names):
        r = all_results[name]
        row_class = 'evolved' if name == 'Evolved Optimal' else 'baseline' if 'Manual Repositioning' in name else ''
        r_ttd = r['reswick']['time_to_damage_hours']
        g_ttd = r['gefen']['time_to_damage_hours']
        r_str = f"{r_ttd:.1f}" if r_ttd else ">8.0"
        g_str = f"{g_ttd:.1f}" if g_ttd else ">8.0"

        html += f'        <tr class="{row_class}"><td>{name}</td><td>{r["avg_pressure"]:.2f}</td><td>{r_str}</td><td>{g_str}</td><td>{stii_values[i]:.2f}</td><td>{reswick_damage[i]:.2f}</td></tr>\n'

    html += """    </table>

    <h3>Table 2: Percent Change from Baseline (Manual Repositioning)</h3>
    <table>
        <tr>
            <th>Configuration</th>
            <th>Pressure Change (%)</th>
            <th>DTI Change (%)</th>
            <th>STII Change (%)</th>
            <th>Damage Reduction (%)</th>
        </tr>
"""

    for i, name in enumerate(config_names):
        row_class = 'evolved' if name == 'Evolved Optimal' else 'baseline' if 'Manual Repositioning' in name else ''
        pct_p = 'Baseline' if i == 0 else f'{pct_pressure[i]:+.1f}'
        pct_d = 'Baseline' if i == 0 else f'{pct_dti[i]:+.1f}'
        pct_s = 'Baseline' if i == 0 else f'{pct_stii[i]:+.1f}'

        html += f'        <tr class="{row_class}"><td>{name}</td><td>{pct_p}</td><td>{pct_d}</td><td>{pct_s}</td><td>{damage_reduction[i]:.1f}</td></tr>\n'

    html += f"""    </table>

    <h3 style="color: #2ecc71;">Best Configuration: Evolved Optimal</h3>
    <table style="width: 50%;">
        <tr class="best-header"><th>Metric</th><th>Value</th></tr>
        <tr class="evolved"><td>Pressure Reduction</td><td>{pct_pressure[-1]:+.1f}%</td></tr>
        <tr class="evolved"><td>DTI Improvement</td><td>{pct_dti[-1]:+.1f}%</td></tr>
        <tr class="evolved"><td>STII Reduction</td><td>{pct_stii[-1]:+.1f}%</td></tr>
        <tr class="evolved"><td>Damage Reduction</td><td>{damage_reduction[-1]:.1f}%</td></tr>
        <tr class="evolved"><td>DTI Efficiency</td><td>{dti_efficiency[-1]:.0f}%</td></tr>
    </table>
</body>
</html>
"""

    with open('sim_results.html', 'w') as f:
        f.write(html)

    # Export PDF with charts and tables
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        chart_path = tmp.name
        fig.write_image(chart_path, width=1500, height=900, scale=2)

    pdf = FPDF(orientation='L', unit='mm', format='letter')
    pdf.set_auto_page_break(auto=True, margin=15)

    # Page 1: Charts
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 8, 'Tissue Damage Analysis', align='C', ln=True)
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 5, '8-hour simulation | STII model (Linder-Ganz & Gefen 2007)', align='C', ln=True)
    pdf.ln(3)
    pdf.image(chart_path, x=15, w=240)

    # Page 2: Tables
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Table 1: Raw Simulation Data', ln=True)

    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    col1 = [50, 25, 25, 25, 20, 25]
    headers1 = ['Configuration', 'Pressure', 'Surface (h)', 'DTI (h)', 'STII', 'Damage']
    for i, h in enumerate(headers1):
        pdf.cell(col1[i], 7, h, border=1, fill=True, align='C')
    pdf.ln()

    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(0, 0, 0)
    for i, name in enumerate(config_names):
        r = all_results[name]
        if name == 'Evolved Optimal':
            pdf.set_fill_color(200, 247, 197)
            fill = True
        elif 'Manual Repositioning' in name:
            pdf.set_fill_color(245, 183, 177)
            fill = True
        else:
            fill = False

        r_ttd = r['reswick']['time_to_damage_hours']
        g_ttd = r['gefen']['time_to_damage_hours']
        r_str = f"{r_ttd:.1f}" if r_ttd else ">8.0"
        g_str = f"{g_ttd:.1f}" if g_ttd else ">8.0"

        pdf.cell(col1[0], 6, name, border=1, fill=fill)
        pdf.cell(col1[1], 6, f'{r["avg_pressure"]:.2f}', border=1, fill=fill, align='C')
        pdf.cell(col1[2], 6, r_str, border=1, fill=fill, align='C')
        pdf.cell(col1[3], 6, g_str, border=1, fill=fill, align='C')
        pdf.cell(col1[4], 6, f'{stii_values[i]:.2f}', border=1, fill=fill, align='C')
        pdf.cell(col1[5], 6, f'{reswick_damage[i]:.2f}', border=1, fill=fill, align='C')
        pdf.ln()

    pdf.ln(10)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Table 2: Percent Change from Baseline', ln=True)

    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    col2 = [50, 30, 30, 30, 30]
    headers2 = ['Configuration', 'Pressure %', 'DTI %', 'STII %', 'Damage Red %']
    for i, h in enumerate(headers2):
        pdf.cell(col2[i], 7, h, border=1, fill=True, align='C')
    pdf.ln()

    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(0, 0, 0)
    for i, name in enumerate(config_names):
        if name == 'Evolved Optimal':
            pdf.set_fill_color(200, 247, 197)
            fill = True
        elif 'Manual Repositioning' in name:
            pdf.set_fill_color(245, 183, 177)
            fill = True
        else:
            fill = False

        pct_p = 'Baseline' if i == 0 else f'{pct_pressure[i]:+.1f}'
        pct_d = 'Baseline' if i == 0 else f'{pct_dti[i]:+.1f}'
        pct_s = 'Baseline' if i == 0 else f'{pct_stii[i]:+.1f}'

        pdf.cell(col2[0], 6, name, border=1, fill=fill)
        pdf.cell(col2[1], 6, pct_p, border=1, fill=fill, align='C')
        pdf.cell(col2[2], 6, pct_d, border=1, fill=fill, align='C')
        pdf.cell(col2[3], 6, pct_s, border=1, fill=fill, align='C')
        pdf.cell(col2[4], 6, f'{damage_reduction[i]:.1f}', border=1, fill=fill, align='C')
        pdf.ln()

    pdf.ln(10)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(46, 204, 113)
    pdf.cell(0, 8, 'Best Configuration: Evolved Optimal', ln=True)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(46, 204, 113)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(50, 7, 'Metric', border=1, fill=True, align='C')
    pdf.cell(40, 7, 'Value', border=1, fill=True, align='C')
    pdf.ln()

    pdf.set_font('Helvetica', '', 9)
    pdf.set_fill_color(200, 247, 197)
    pdf.set_text_color(0, 0, 0)
    best_metrics = [
        ('Pressure Reduction', f'{pct_pressure[-1]:+.1f}%'),
        ('DTI Improvement', f'{pct_dti[-1]:+.1f}%'),
        ('STII Reduction', f'{pct_stii[-1]:+.1f}%'),
        ('Damage Reduction', f'{damage_reduction[-1]:.1f}%'),
        ('DTI Efficiency', f'{dti_efficiency[-1]:.0f}%'),
    ]
    for metric, value in best_metrics:
        pdf.cell(50, 6, metric, border=1, fill=True)
        pdf.cell(40, 6, value, border=1, fill=True, align='C')
        pdf.ln()

    pdf.output('sim_results.pdf')
    os.unlink(chart_path)

    print(f"\nSaved: sim_results.html")
    print(f"Saved: sim_results.pdf")

    # Print summary table
    print("\n" + "=" * 120)
    print("TIME-TO-DAMAGE PREDICTION SUMMARY (8 HOURS) - STII FORMAL INDEX")
    print("=" * 120)
    print(f"{'Configuration':<28} {'Avg Press':>9} {'Surface':>10} {'DTI':>10} {'STII':>8} {'Damage':>10} {'Min Perf':>10}")
    print(f"{'':28} {'(mmHg)':>9} {'(hours)':>10} {'(hours)':>10} {'':>8} {'Fraction':>10} {'':>10}")
    print("-" * 120)

    foam_reswick = all_results['Manual Repositioning (2h)']['reswick']['time_to_damage_hours']
    foam_gefen = all_results['Manual Repositioning (2h)']['gefen']['time_to_damage_hours']

    for name in config_names:
        r = all_results[name]
        r_ttd = r['reswick']['time_to_damage_hours']
        g_ttd = r['gefen']['time_to_damage_hours']

        r_str = f"{r_ttd:.1f}" if r_ttd else ">8.0"
        g_str = f"{g_ttd:.1f}" if g_ttd else ">8.0"

        stii = r['reswick'].get('cumulative_stii', 0)
        min_perf = r['reswick'].get('min_perfusion', 1.0)

        print(f"{name:<28} {r['avg_pressure']:>9.1f} {r_str:>10} {g_str:>10} "
              f"{stii:>8.2f} {r['reswick']['final_damage_fraction']:>10.2f} {min_perf:>10.2f}")

    print("-" * 120)
    print("\nSTII - STRAIN-TIME INJURY INDEX (Linder-Ganz & Gefen 2007):")
    print("  Sigmoid strain tolerance: tissue tolerates 65% strain for 1h, drops to 40% by 4h")
    print("  STII >= 1.0 indicates injury threshold exceeded")
    print("\nTRADITIONAL MODELS:")
    print("  - Reswick-Rogers (1976): Surface damage pressure-time relationship")
    print("  - Gefen DTI (2008): Deep tissue injury at bone-muscle interface")
    print("\nPERFUSION RECOVERY MODEL:")
    print("  - Blood flow doesn't instantly return when pressure removed")
    print("  - Recovery rate depends on ischemia duration:")
    print(f"      Short (<{SHORT_ISCHEMIA_THRESHOLD:.0f} min): Fast recovery ({PERFUSION_RECOVERY_RATE_FAST*100:.0f}%/min)")
    print(f"      Medium ({SHORT_ISCHEMIA_THRESHOLD:.0f}-{PROLONGED_ISCHEMIA_THRESHOLD:.0f} min): Normal recovery ({PERFUSION_RECOVERY_RATE_BASE*100:.0f}%/min)")
    print(f"      Prolonged (>{PROLONGED_ISCHEMIA_THRESHOLD:.0f} min): Slow recovery ({PERFUSION_RECOVERY_RATE_SLOW*100:.0f}%/min)")
    print(f"  - Tissue {RECOVERY_VULNERABILITY_FACTOR:.0%} more vulnerable during incomplete recovery")


if __name__ == "__main__":
    create_time_to_damage_analysis()
