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
        'Standard Foam': {'type': 'foam'},
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

        if config['type'] != 'foam':
            pattern = config['pattern']
            mattress = MultiDynamicAirMattress(cell_size_cm=5, movement_pattern=pattern)
            realistic_state = RealisticMattressState(mattress.rows, mattress.cols)

            scale_y = mattress.rows / body_pressure.shape[0]
            scale_x = mattress.cols / body_pressure.shape[1]
            body_resampled = zoom(body_pressure, (scale_y, scale_x), order=1)

        pressure_history = []

        for time_sec in time_points_sec:
            if config['type'] == 'foam':
                # Foam mattress provides ~85% pressure redistribution
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
    """Create multi-panel visualization of time-to-damage predictions."""

    config_names = list(all_results.keys())
    n_configs = len(config_names)

    # Extract metrics for bar charts
    avg_pressures = [all_results[name]['avg_pressure'] for name in config_names]

    # Time to damage (use total_time if no damage occurred)
    reswick_ttd = []
    gefen_ttd = []
    for name in config_names:
        r_ttd = all_results[name]['reswick']['time_to_damage_hours']
        g_ttd = all_results[name]['gefen']['time_to_damage_hours']
        reswick_ttd.append(r_ttd if r_ttd is not None else total_time_hours)
        gefen_ttd.append(g_ttd if g_ttd is not None else total_time_hours)

    # Damage fraction at 8 hours
    reswick_damage = [all_results[name]['reswick']['final_damage_fraction'] for name in config_names]
    gefen_damage = [all_results[name]['gefen']['final_damage_fraction'] for name in config_names]

    # Colors based on evolved optimal highlighting
    colors = ['#2ecc71' if name == 'Evolved Optimal' else
              '#e74c3c' if name == 'Standard Foam' else
              '#3498db' for name in config_names]

    # Extract STII damage for bar chart
    stii_ttd = []
    for name in config_names:
        s_ttd = all_results[name]['reswick'].get('time_to_stii_damage_hours')
        stii_ttd.append(s_ttd if s_ttd is not None else total_time_hours)

    # Calculate percent changes for table
    baseline_pressure = avg_pressures[0]
    baseline_surface = reswick_ttd[0]
    baseline_dti = gefen_ttd[0]
    baseline_stii = all_results[config_names[0]]['reswick'].get('cumulative_stii', 0)
    baseline_damage = reswick_damage[0]

    pct_pressure = [((p - baseline_pressure) / baseline_pressure * 100) if baseline_pressure > 0 else 0 for p in avg_pressures]
    pct_dti = [((d - baseline_dti) / baseline_dti * 100) if baseline_dti > 0 else 0 for d in gefen_ttd]
    stii_values = [all_results[name]['reswick'].get('cumulative_stii', 0) for name in config_names]
    pct_stii = [((s - baseline_stii) / baseline_stii * 100) if baseline_stii > 0 else 0 for s in stii_values]
    pct_damage = [((d - baseline_damage) / baseline_damage * 100) if baseline_damage > 0 else 0 for d in reswick_damage]
    dti_efficiency = [(d / baseline_dti * 100) if baseline_dti > 0 else 100 for d in gefen_ttd]

    # Create figure with 4 rows: bar charts, damage curves, STII/perfusion, data table
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            '<b>Time to Surface Damage</b><br><sup>Reswick-Rogers Model</sup>',
            '<b>Time to Deep Tissue Injury</b><br><sup>Gefen DTI Model</sup>',
            '<b>Damage Accumulation (Surface)</b><br><sup>Reswick-Rogers</sup>',
            '<b>Damage Accumulation (DTI)</b><br><sup>Gefen Model</sup>',
            '<b>STII - Strain-Time Injury Index</b><br><sup>Linder-Ganz & Gefen (2007)</sup>',
            '<b>Tissue Perfusion Level</b><br><sup>1.0 = perfused, 0.0 = ischemic</sup>',
            '<b>Data Table: Raw Values</b>',
            '<b>Data Table: Percent Change from Foam</b>',
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "table"}, {"type": "table"}]],
        row_heights=[0.22, 0.22, 0.22, 0.34]
    )

    # Panel 1: Time to surface damage (bar chart)
    fig.add_trace(
        go.Bar(
            x=config_names,
            y=reswick_ttd,
            marker_color=colors,
            text=[f'{t:.1f}h' if t < total_time_hours else '>8h' for t in reswick_ttd],
            textposition='outside',
            hovertemplate='%{x}<br>Time to damage: %{y:.1f} hours<extra></extra>'
        ),
        row=1, col=1
    )

    # Panel 2: Time to deep tissue injury (bar chart)
    fig.add_trace(
        go.Bar(
            x=config_names,
            y=gefen_ttd,
            marker_color=colors,
            text=[f'{t:.1f}h' if t < total_time_hours else '>8h' for t in gefen_ttd],
            textposition='outside',
            hovertemplate='%{x}<br>Time to DTI: %{y:.1f} hours<extra></extra>'
        ),
        row=1, col=2
    )

    # Panel 3: Damage accumulation curves (Reswick-Rogers)
    time_hours = all_results[config_names[0]]['time_hours']
    for i, name in enumerate(config_names):
        damage_history = all_results[name]['reswick']['damage_history']
        fig.add_trace(
            go.Scatter(
                x=time_hours,
                y=damage_history,
                mode='lines',
                name=name,
                line=dict(
                    color='#2ecc71' if name == 'Evolved Optimal' else
                          '#e74c3c' if name == 'Standard Foam' else
                          '#3498db',
                    width=3 if name in ['Evolved Optimal', 'Standard Foam'] else 1,
                ),
                opacity=1.0 if name in ['Evolved Optimal', 'Standard Foam'] else 0.4,
                showlegend=True,
                legendgroup='reswick',
                hovertemplate=f'{name}<br>Time: %{{x:.1f}}h<br>Damage: %{{y:.2f}}<extra></extra>'
            ),
            row=2, col=1
        )

    # Add damage threshold line
    fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                  annotation_text="Damage Threshold", row=2, col=1)

    # Panel 4: Damage accumulation curves (Gefen)
    for i, name in enumerate(config_names):
        damage_history = all_results[name]['gefen']['damage_history']
        fig.add_trace(
            go.Scatter(
                x=time_hours,
                y=damage_history,
                mode='lines',
                name=name,
                line=dict(
                    color='#2ecc71' if name == 'Evolved Optimal' else
                          '#e74c3c' if name == 'Standard Foam' else
                          '#3498db',
                    width=3 if name in ['Evolved Optimal', 'Standard Foam'] else 1,
                ),
                opacity=1.0 if name in ['Evolved Optimal', 'Standard Foam'] else 0.4,
                showlegend=False,
                legendgroup='gefen',
                hovertemplate=f'{name}<br>Time: %{{x:.1f}}h<br>Damage: %{{y:.2f}}<extra></extra>'
            ),
            row=2, col=2
        )

    # Add damage threshold line
    fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                  annotation_text="Damage Threshold", row=2, col=2)

    # Panel 5: STII - Strain-Time Injury Index curves
    for i, name in enumerate(config_names):
        stii_history = all_results[name]['reswick'].get('stii_history', [0.0] * len(time_hours))
        fig.add_trace(
            go.Scatter(
                x=time_hours,
                y=stii_history,
                mode='lines',
                name=name,
                line=dict(
                    color='#2ecc71' if name == 'Evolved Optimal' else
                          '#e74c3c' if name == 'Standard Foam' else
                          '#3498db',
                    width=3 if name in ['Evolved Optimal', 'Standard Foam'] else 1,
                ),
                opacity=1.0 if name in ['Evolved Optimal', 'Standard Foam'] else 0.4,
                showlegend=False,
                hovertemplate=f'{name}<br>Time: %{{x:.1f}}h<br>STII: %{{y:.2f}}<extra></extra>'
            ),
            row=3, col=1
        )

    # Add STII threshold line
    fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                  annotation_text="Injury Threshold", row=3, col=1)

    # Panel 6: Perfusion level over time
    for i, name in enumerate(config_names):
        perfusion_history = all_results[name]['reswick'].get('perfusion_history', [1.0] * len(time_hours))
        fig.add_trace(
            go.Scatter(
                x=time_hours,
                y=perfusion_history,
                mode='lines',
                name=name,
                line=dict(
                    color='#2ecc71' if name == 'Evolved Optimal' else
                          '#e74c3c' if name == 'Standard Foam' else
                          '#3498db',
                    width=3 if name in ['Evolved Optimal', 'Standard Foam'] else 1,
                ),
                opacity=1.0 if name in ['Evolved Optimal', 'Standard Foam'] else 0.4,
                showlegend=False,
                hovertemplate=f'{name}<br>Time: %{{x:.1f}}h<br>Perfusion: %{{y:.2f}}<extra></extra>'
            ),
            row=3, col=2
        )

    # Add perfusion threshold line
    fig.add_hline(y=INCOMPLETE_RECOVERY_THRESHOLD, line_dash="dash", line_color="orange",
                  annotation_text="Recovery Threshold", row=3, col=2)

    # Panel 7: Data Table - Raw Values
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Configuration</b>', '<b>Pressure<br>(mmHg)</b>', '<b>Surface<br>(hours)</b>',
                        '<b>DTI<br>(hours)</b>', '<b>STII</b>', '<b>Damage</b>'],
                fill_color='#2c3e50',
                font=dict(color='white', size=11),
                align='left',
                height=28
            ),
            cells=dict(
                values=[
                    config_names,
                    [f'{p:.1f}' for p in avg_pressures],
                    [f'{s:.1f}' for s in reswick_ttd],
                    [f'{d:.1f}' for d in gefen_ttd],
                    [f'{s:.2f}' for s in stii_values],
                    [f'{d:.2f}' for d in reswick_damage],
                ],
                fill_color=[['#2ecc71' if n == 'Evolved Optimal' else '#e74c3c' if n == 'Standard Foam' else 'white' for n in config_names]] + [['white'] * len(config_names)] * 5,
                font=dict(size=10),
                align='left',
                height=22
            )
        ),
        row=4, col=1
    )

    # Panel 8: Data Table - Percent Change
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Configuration</b>', '<b>Pressure<br>Change (%)</b>', '<b>DTI<br>Change (%)</b>',
                        '<b>STII<br>Change (%)</b>', '<b>Damage<br>Change (%)</b>', '<b>DTI<br>Efficiency (%)</b>'],
                fill_color='#2c3e50',
                font=dict(color='white', size=11),
                align='left',
                height=28
            ),
            cells=dict(
                values=[
                    config_names,
                    ['Baseline' if i == 0 else f'{pct_pressure[i]:+.1f}' for i in range(len(config_names))],
                    ['Baseline' if i == 0 else f'{pct_dti[i]:+.1f}' for i in range(len(config_names))],
                    ['Baseline' if i == 0 else f'{pct_stii[i]:+.1f}' for i in range(len(config_names))],
                    ['Baseline' if i == 0 else f'{pct_damage[i]:+.1f}' for i in range(len(config_names))],
                    [f'{e:.0f}' for e in dti_efficiency],
                ],
                fill_color=[['#2ecc71' if n == 'Evolved Optimal' else '#e74c3c' if n == 'Standard Foam' else 'white' for n in config_names]] + [['white'] * len(config_names)] * 5,
                font=dict(size=10),
                align='left',
                height=22
            )
        ),
        row=4, col=2
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Tissue Damage Analysis: STII (Strain-Time Injury Index)</b><br>'
                 '<sup>Sacrum region | 8-hour simulation | Linder-Ganz & Gefen (2007), Reswick-Rogers</sup>',
            x=0.5,
            font=dict(size=20)
        ),
        height=1600,
        width=1500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5
        ),
    )

    # Update axes for chart rows (tables don't need axis updates)
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_yaxes(title_text="Time (hours)", range=[0, total_time_hours * 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Time (hours)", range=[0, total_time_hours * 1.1], row=1, col=2)
    # Row 2: Damage accumulation
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative Damage", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Damage", row=2, col=2)
    # Row 3: STII and Perfusion
    fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=3, col=2)
    fig.update_yaxes(title_text="Cumulative STII", row=3, col=1)
    fig.update_yaxes(title_text="Perfusion Level", range=[0, 1.1], row=3, col=2)

    fig.write_html('time_to_damage.html', include_plotlyjs=True, full_html=True)
    print(f"\nSaved: time_to_damage.html")

    # Print summary table
    print("\n" + "=" * 120)
    print("TIME-TO-DAMAGE PREDICTION SUMMARY (8 HOURS) - STII FORMAL INDEX")
    print("=" * 120)
    print(f"{'Configuration':<28} {'Avg Press':>9} {'Surface':>10} {'DTI':>10} {'STII':>8} {'Damage':>10} {'Min Perf':>10}")
    print(f"{'':28} {'(mmHg)':>9} {'(hours)':>10} {'(hours)':>10} {'':>8} {'Fraction':>10} {'':>10}")
    print("-" * 120)

    foam_reswick = all_results['Standard Foam']['reswick']['time_to_damage_hours']
    foam_gefen = all_results['Standard Foam']['gefen']['time_to_damage_hours']

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
