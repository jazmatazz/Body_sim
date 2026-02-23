#!/usr/bin/env python3
"""
Evolutionary Mattress Pattern Optimizer
========================================
Uses genetic algorithm to evolve optimal pressure redistribution patterns.
Each "genome" defines cycling behavior per body region.
Fitness is measured by minimizing damage, at-risk cells, and shear.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from scipy.ndimage import zoom
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from multidynamic_mattress_optimization import (
    SMPLBodyPressureModel, MultiDynamicAirMattress,
    CAPILLARY_CLOSING_PRESSURE
)
from all_mattress_configs import (
    RealisticMattressState, calculate_cumulative_damage,
    QUADRIPLEGIC_SHEAR_FACTOR as AMC_SHEAR_FACTOR
)

# PTI calculation constants (matching all_mattress_configs.py exactly)
PRESSURE_THRESHOLD = 32  # mmHg - capillary closing pressure
SHEAR_THRESHOLD = 10
SHEAR_SYNERGY = 1.5
QUADRIPLEGIC_SHEAR_FACTOR = 1.3

# Pressure-Time Integral (PTI) based damage model
# Unit: mmHg·hours above capillary closing threshold
# Reference: Reswick & Rogers 1976, Linder-Ganz et al. 2006
# Critical PTI threshold for HAPI (mmHg·hours) for very high risk patient (Braden ≤9)
CRITICAL_PTI = 20.0

BRADEN_MULTIPLIERS = {
    'very_high': 2.0,  # Score 6-9
    'high': 1.5,       # Score 10-12
    'moderate': 1.2,   # Score 13-14
    'at_risk': 1.0,    # Score 15-18
}

# Physical constraints for realistic air mattress behavior
TRANSITION_TIME = 45.0  # Seconds for full inflation/deflation (0→1 or 1→0)
CELL_HEIGHT_MAX = 12.0  # cm when fully inflated
CELL_HEIGHT_MIN = 4.0   # cm when deflated


# =============================================================================
# GENOME DEFINITION
# =============================================================================

@dataclass
class RegionGenome:
    """Genetic parameters for a single body region."""
    cycle_speed: float      # How fast to cycle (0.5 to 5.0x base rate)
    wave_type: str          # 'sine', 'square', 'triangle', 'sawtooth'
    phase_offset: float     # Phase offset (0 to 1)
    min_inflation: float    # Minimum cell inflation (0 to 0.5)
    max_inflation: float    # Maximum cell inflation (0.5 to 1.0)

    def mutate(self, mutation_rate: float = 0.2):
        """Apply random mutations to this region's parameters."""
        if random.random() < mutation_rate:
            self.cycle_speed = np.clip(self.cycle_speed + random.gauss(0, 0.5), 0.5, 5.0)
        if random.random() < mutation_rate:
            self.wave_type = random.choice(['sine', 'square', 'triangle', 'sawtooth'])
        if random.random() < mutation_rate:
            self.phase_offset = (self.phase_offset + random.gauss(0, 0.2)) % 1.0
        if random.random() < mutation_rate:
            self.min_inflation = np.clip(self.min_inflation + random.gauss(0, 0.1), 0.0, 0.4)
        if random.random() < mutation_rate:
            self.max_inflation = np.clip(self.max_inflation + random.gauss(0, 0.1), 0.6, 1.0)

    def copy(self) -> 'RegionGenome':
        return RegionGenome(
            self.cycle_speed, self.wave_type, self.phase_offset,
            self.min_inflation, self.max_inflation
        )


@dataclass
class PatternGenome:
    """Complete genome defining a mattress pattern."""
    regions: Dict[str, RegionGenome] = field(default_factory=dict)
    fitness: float = 0.0
    generation: int = 0

    # Body region definitions (row_start, row_end, col_start, col_end)
    REGION_BOUNDS = {
        'head': (0.0, 0.08, 0.3, 0.7),
        'scapulae': (0.10, 0.25, 0.2, 0.8),
        'thoracic': (0.25, 0.40, 0.25, 0.75),
        'sacrum': (0.40, 0.55, 0.35, 0.65),
        'thighs': (0.55, 0.75, 0.2, 0.8),
        'calves': (0.75, 0.92, 0.25, 0.75),
        'left_heel': (0.92, 1.0, 0.30, 0.42),
        'right_heel': (0.92, 1.0, 0.58, 0.70),
        'default': (0.0, 1.0, 0.0, 1.0),  # Fallback for uncovered areas
    }

    # Clinical importance weights (higher = more important to protect)
    REGION_WEIGHTS = {
        'sacrum': 3.6,      # 36% of pressure ulcers
        'left_heel': 1.5,
        'right_heel': 1.5,
        'scapulae': 0.3,
        'head': 0.2,
        'thoracic': 0.3,
        'thighs': 0.2,
        'calves': 0.3,
        'default': 0.1,
    }

    @classmethod
    def random(cls) -> 'PatternGenome':
        """Create a random genome."""
        genome = cls()
        for region in cls.REGION_BOUNDS.keys():
            genome.regions[region] = RegionGenome(
                cycle_speed=random.uniform(0.5, 4.0),
                wave_type=random.choice(['sine', 'square', 'triangle', 'sawtooth']),
                phase_offset=random.random(),
                min_inflation=random.uniform(0.0, 0.3),
                max_inflation=random.uniform(0.7, 1.0),
            )
        return genome

    def get_cell_state(self, row: int, col: int, rows: int, cols: int, phase: float) -> float:
        """Calculate cell inflation state based on genome."""
        row_pos = row / rows
        col_pos = col / cols

        # Find which region this cell belongs to
        active_region = 'default'
        for region, (r0, r1, c0, c1) in self.REGION_BOUNDS.items():
            if region == 'default':
                continue
            if r0 <= row_pos < r1 and c0 <= col_pos <= c1:
                active_region = region
                break

        genome = self.regions[active_region]
        local_phase = (phase * genome.cycle_speed + genome.phase_offset) % 1.0

        # Apply wave type
        if genome.wave_type == 'sine':
            wave = 0.5 + 0.5 * np.cos(2 * np.pi * local_phase)
        elif genome.wave_type == 'square':
            wave = 1.0 if local_phase < 0.5 else 0.0
        elif genome.wave_type == 'triangle':
            wave = 1.0 - 2 * abs(local_phase - 0.5)
        elif genome.wave_type == 'sawtooth':
            wave = 1.0 - local_phase
        else:
            wave = 0.5

        # Scale to min/max inflation range
        return genome.min_inflation + wave * (genome.max_inflation - genome.min_inflation)

    def mutate(self, mutation_rate: float = 0.2):
        """Apply mutations to all regions."""
        for region in self.regions.values():
            region.mutate(mutation_rate)

    def crossover(self, other: 'PatternGenome') -> 'PatternGenome':
        """Create offspring by combining with another genome."""
        child = PatternGenome()
        for region in self.REGION_BOUNDS.keys():
            # 50% chance to inherit from each parent
            if random.random() < 0.5:
                child.regions[region] = self.regions[region].copy()
            else:
                child.regions[region] = other.regions[region].copy()
        return child

    def copy(self) -> 'PatternGenome':
        """Create a deep copy."""
        new_genome = PatternGenome()
        new_genome.generation = self.generation
        new_genome.fitness = self.fitness
        for region, params in self.regions.items():
            new_genome.regions[region] = params.copy()
        return new_genome


# =============================================================================
# FITNESS EVALUATION
# =============================================================================

class GenomeMattress:
    """Mattress adapter that uses a genome for cell states with realistic physics."""

    def __init__(self, genome: PatternGenome, cell_size_cm: int = 5,
                 transition_time: float = TRANSITION_TIME):
        self.genome = genome
        self.cell_size_cm = cell_size_cm
        self.rows = 40
        self.cols = 18
        self.phase = 0.0
        self.cycle_time = 300  # 5-minute base cycle
        self.transition_time = transition_time  # Seconds for full 0→1 transition

        # Max rate of change per second (0→1 takes transition_time seconds)
        self.max_rate = 1.0 / transition_time

        # Cell states - actual vs target
        self.inflation = np.ones((self.rows, self.cols))  # Actual current state
        self.target = np.ones((self.rows, self.cols))     # Target state from pattern
        self.relief_time = np.zeros((self.rows, self.cols))

        self.last_time = 0.0

    def update(self, time_seconds: float):
        """Update cell states with realistic transition physics."""
        self.phase = (time_seconds / self.cycle_time) % 1.0
        dt = time_seconds - self.last_time
        self.last_time = time_seconds

        # Maximum change possible in this time step
        max_change = self.max_rate * dt

        for row in range(self.rows):
            for col in range(self.cols):
                # Get target state from genome pattern
                target = self.genome.get_cell_state(row, col, self.rows, self.cols, self.phase)
                self.target[row, col] = target

                # Current actual state
                current = self.inflation[row, col]

                # Move toward target at limited rate (realistic pump speed)
                diff = target - current
                if abs(diff) <= max_change:
                    # Can reach target this step
                    new_state = target
                else:
                    # Move toward target at max rate
                    new_state = current + np.sign(diff) * max_change

                self.inflation[row, col] = new_state

                # Track relief time (when actually deflated, not just targeting)
                if new_state < 0.3:
                    self.relief_time[row, col] += 1

    def get_effective_interface_pressure(self, body_pressure: np.ndarray) -> np.ndarray:
        """Calculate pressure with mattress redistribution."""
        # Resize body pressure to match mattress grid if needed
        if body_pressure.shape != (self.rows, self.cols):
            scale_y = self.rows / body_pressure.shape[0]
            scale_x = self.cols / body_pressure.shape[1]
            body_pressure = zoom(body_pressure, (scale_y, scale_x), order=1)

        # Deflated cells redistribute pressure to neighbors
        effective = body_pressure.copy()

        for row in range(self.rows):
            for col in range(self.cols):
                inflation = self.inflation[row, col]

                if inflation < 0.5:
                    # This cell provides relief (matching all_mattress_configs.py)
                    relief_factor = 1 - inflation
                    pressure_relieved = effective[row, col] * relief_factor * 0.7
                    effective[row, col] -= pressure_relieved

        return np.clip(effective, 0, None)


class GenomePattern:
    """Adapter to use a PatternGenome with MultiDynamicAirMattress."""

    def __init__(self, genome: PatternGenome):
        self.genome = genome
        self.name = "Genome Pattern"
        self.pattern_type = "genome"

    def get_cell_state(self, row: int, col: int, rows: int, cols: int, phase: float,
                       smooth: bool = False) -> float:
        return self.genome.get_cell_state(row, col, rows, cols, phase)


def evaluate_fitness(genome: PatternGenome, body_pressure: np.ndarray,
                     shear_map: np.ndarray, braden_score: int = 8,
                     simulation_minutes: int = 120) -> Tuple[float, dict]:
    """
    Evaluate genome fitness using EXACT same code path as all_mattress_configs.py.
    Returns (fitness_score, metrics_dict).
    Higher fitness = better pattern.
    """
    # Create mattress with genome pattern (same as all_mattress_configs.py)
    pattern = GenomePattern(genome)
    mattress = MultiDynamicAirMattress(cell_size_cm=5, movement_pattern=pattern)
    realistic_state = RealisticMattressState(mattress.rows, mattress.cols)

    # Simulation parameters (matching all_mattress_configs.py exactly)
    total_time = simulation_minutes * 60
    n_frames = 48
    time_points = np.linspace(0, total_time, n_frames, endpoint=False)
    time_step_min = (time_points[1] - time_points[0]) / 60

    # Resample body maps to mattress grid (same as all_mattress_configs.py)
    scale_y = mattress.rows / body_pressure.shape[0]
    scale_x = mattress.cols / body_pressure.shape[1]
    body_resampled = zoom(body_pressure, (scale_y, scale_x), order=1)
    shear_resampled = zoom(shear_map, (scale_y, scale_x), order=1) * AMC_SHEAR_FACTOR

    pressure_history = []
    shear_history = []

    for time_sec in time_points:
        # Get target state from pattern
        mattress.update(time_sec)
        target_state = mattress.cell_state.copy()

        # Apply realistic transition physics (same as all_mattress_configs.py)
        actual_state = realistic_state.update(target_state, time_sec)

        # Calculate pressure with ACTUAL cell states (same as all_mattress_configs.py)
        effective_pressure = body_resampled.copy()
        for row in range(mattress.rows):
            for col in range(mattress.cols):
                inflation = actual_state[row, col]
                if inflation < 0.5:
                    relief_factor = 1 - inflation
                    pressure_relieved = effective_pressure[row, col] * relief_factor * 0.7
                    effective_pressure[row, col] -= pressure_relieved

        # Shear is reduced when cells deflate (same as all_mattress_configs.py)
        effective_shear = shear_resampled * actual_state

        # Resample back to display size (same as all_mattress_configs.py)
        scale_y_back = body_pressure.shape[0] / effective_pressure.shape[0]
        scale_x_back = body_pressure.shape[1] / effective_pressure.shape[1]
        effective_pressure = zoom(effective_pressure, (scale_y_back, scale_x_back), order=1)
        effective_shear = zoom(effective_shear, (scale_y_back, scale_x_back), order=1)

        pressure_history.append(effective_pressure)
        shear_history.append(effective_shear)

    # Use the EXACT same damage calculation as all_mattress_configs.py
    final_damage, _ = calculate_cumulative_damage(pressure_history, shear_history, time_step_min, braden_score)

    # Calculate fitness metrics (using PTI - mmHg·hours)
    max_pti = final_damage.max()
    at_risk_cells = (final_damage >= CRITICAL_PTI).sum()
    avg_peak_pressure = np.mean([p.max() for p in pressure_history])
    avg_cells_over = np.mean([(p > PRESSURE_THRESHOLD).sum() for p in pressure_history])
    avg_shear = np.mean([s.mean() for s in shear_history])

    # Relief coverage: cells that stayed below PTI threshold
    relief_coverage = (final_damage < CRITICAL_PTI).sum() / final_damage.size

    # Weighted fitness (lower PTI = higher fitness)
    # We want to MINIMIZE these, so fitness = 1 / (weighted sum)
    fitness = 1000 / (
        max_pti * 2.0 +              # Heavily penalize max PTI
        at_risk_cells * 10.0 +       # Penalize at-risk cells
        avg_peak_pressure * 0.5 +    # Moderate penalty for peak pressure
        avg_shear * 1.0 +            # Shear penalty
        (1 - relief_coverage) * 50   # Bonus for good relief coverage
    )

    metrics = {
        'max_pti': max_pti,  # mmHg·hours
        'at_risk_cells': at_risk_cells,
        'avg_peak_pressure': avg_peak_pressure,
        'avg_cells_over_threshold': avg_cells_over,
        'avg_shear': avg_shear,
        'relief_coverage': relief_coverage,
    }

    return fitness, metrics


# =============================================================================
# GENETIC ALGORITHM
# =============================================================================

class GeneticOptimizer:
    """Genetic algorithm for evolving optimal mattress patterns."""

    def __init__(self, population_size: int = 50, elite_count: int = 5,
                 mutation_rate: float = 0.2, crossover_rate: float = 0.7):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.population: List[PatternGenome] = []
        self.best_genome: PatternGenome = None
        self.best_fitness: float = 0.0
        self.generation: int = 0
        self.history: List[dict] = []

        # Body model (matching all_mattress_configs.py: 40x18 grid)
        self.model = SMPLBodyPressureModel(75, 30)
        self.body_pressure, self.shear_map = self.model.calculate_pressure_map(40, 18)

    def initialize_population(self):
        """Create initial random population."""
        self.population = [PatternGenome.random() for _ in range(self.population_size)]
        print(f"Initialized population of {self.population_size} random genomes")

    def evaluate_population(self, verbose: bool = True):
        """Evaluate fitness of all genomes."""
        for i, genome in enumerate(self.population):
            fitness, metrics = evaluate_fitness(
                genome, self.body_pressure, self.shear_map,
                braden_score=8, simulation_minutes=120
            )
            genome.fitness = fitness
            genome.generation = self.generation

            if verbose and (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{self.population_size} genomes...")

        # Sort by fitness (descending)
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Update best
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_genome = self.population[0].copy()

    def select_parents(self) -> Tuple[PatternGenome, PatternGenome]:
        """Tournament selection for parents."""
        def tournament(k: int = 3) -> PatternGenome:
            candidates = random.sample(self.population, k)
            return max(candidates, key=lambda g: g.fitness)

        return tournament(), tournament()

    def evolve_generation(self):
        """Create next generation through selection, crossover, mutation."""
        new_population = []

        # Elitism: keep best genomes
        for i in range(self.elite_count):
            new_population.append(self.population[i].copy())

        # Fill rest with offspring
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()

            if random.random() < self.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = parent1.copy()

            child.mutate(self.mutation_rate)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def run(self, generations: int = 30, target_fitness: float = None,
            convergence_threshold: int = 5, verbose: bool = True) -> PatternGenome:
        """
        Run genetic algorithm for specified generations or until convergence.

        Args:
            generations: Maximum number of generations
            target_fitness: Stop early if this fitness is reached
            convergence_threshold: Stop if no improvement for this many generations
            verbose: Print progress

        Returns:
            Best genome found
        """
        print("\n" + "=" * 70)
        print("EVOLUTIONARY MATTRESS PATTERN OPTIMIZER")
        print("=" * 70)
        print(f"Population: {self.population_size} | Generations: {generations}")
        print(f"Elite: {self.elite_count} | Mutation: {self.mutation_rate:.0%} | Crossover: {self.crossover_rate:.0%}")
        print("=" * 70)

        self.initialize_population()

        no_improvement_count = 0
        last_best = 0.0

        for gen in range(generations):
            if verbose:
                print(f"\nGeneration {gen + 1}/{generations}")

            self.evaluate_population(verbose=verbose)

            # Get stats
            best = self.population[0]
            _, metrics = evaluate_fitness(best, self.body_pressure, self.shear_map)

            avg_fitness = np.mean([g.fitness for g in self.population])

            self.history.append({
                'generation': gen + 1,
                'best_fitness': best.fitness,
                'avg_fitness': avg_fitness,
                'max_pti': metrics['max_pti'],
                'at_risk_cells': metrics['at_risk_cells'],
                'avg_shear': metrics['avg_shear'],
            })

            if verbose:
                print(f"  Best Fitness: {best.fitness:.4f}")
                print(f"  Avg Fitness:  {avg_fitness:.4f}")
                print(f"  Peak PTI:     {metrics['max_pti']:.1f} mmHg·h")
                print(f"  At Risk:      {metrics['at_risk_cells']}")
                print(f"  Avg Shear:    {metrics['avg_shear']:.1f}")

            # Check for convergence
            if best.fitness <= last_best:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
                last_best = best.fitness

            if no_improvement_count >= convergence_threshold:
                print(f"\nConverged after {gen + 1} generations (no improvement for {convergence_threshold} generations)")
                break

            if target_fitness and best.fitness >= target_fitness:
                print(f"\nTarget fitness {target_fitness} reached!")
                break

            # Evolve next generation
            self.evolve_generation()

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)

        return self.best_genome

    def create_visualization(self, save_path: str = 'evolved_mattress.html'):
        """Create visualization of optimization results."""
        if not self.history:
            print("No optimization history to visualize")
            return

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '<b>Fitness Over Generations</b>',
                '<b>PTI (mmHg·h) Over Generations</b>',
                '<b>Best Pattern - Region Parameters</b>',
                '<b>Pressure Distribution</b>'
            ],
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'table'}, {'type': 'heatmap'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Panel 1: Fitness over generations
        gens = [h['generation'] for h in self.history]
        best_fitness = [h['best_fitness'] for h in self.history]
        avg_fitness = [h['avg_fitness'] for h in self.history]

        fig.add_trace(go.Scatter(
            x=gens, y=best_fitness,
            mode='lines+markers',
            name='Best Fitness',
            line=dict(color='green', width=2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=gens, y=avg_fitness,
            mode='lines',
            name='Avg Fitness',
            line=dict(color='blue', dash='dash')
        ), row=1, col=1)

        # Panel 2: PTI metrics
        max_pti = [h['max_pti'] for h in self.history]
        at_risk = [h['at_risk_cells'] for h in self.history]

        fig.add_trace(go.Scatter(
            x=gens, y=max_pti,
            mode='lines+markers',
            name='Peak PTI (mmHg·h)',
            line=dict(color='red', width=2)
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=gens, y=at_risk,
            mode='lines+markers',
            name='At Risk Cells',
            line=dict(color='orange', width=2),
            yaxis='y2'
        ), row=1, col=2)

        # Panel 3: Best pattern parameters
        if self.best_genome:
            table_data = []
            for region, params in self.best_genome.regions.items():
                table_data.append([
                    region.replace('_', ' ').title(),
                    f"{params.cycle_speed:.2f}x",
                    params.wave_type,
                    f"{params.phase_offset:.2f}",
                    f"{params.min_inflation:.2f}",
                    f"{params.max_inflation:.2f}",
                ])

            fig.add_trace(go.Table(
                header=dict(
                    values=['<b>Region</b>', '<b>Speed</b>', '<b>Wave</b>',
                            '<b>Phase</b>', '<b>Min</b>', '<b>Max</b>'],
                    fill_color='lightsteelblue',
                    align='left',
                    font=dict(size=11)
                ),
                cells=dict(
                    values=list(zip(*table_data)) if table_data else [[]]*6,
                    fill_color=[['white', 'whitesmoke']*len(table_data)],
                    align='left',
                    font=dict(size=10)
                )
            ), row=2, col=1)

        # Panel 4: Final pressure distribution
        if self.best_genome:
            mattress = GenomeMattress(self.best_genome)
            mattress.update(150)  # Sample at 2.5 minutes
            effective = mattress.get_effective_interface_pressure(self.body_pressure)

            colorscale = [
                [0, 'rgb(0, 100, 0)'],
                [0.32, 'rgb(144, 238, 144)'],
                [0.40, 'rgb(255, 255, 0)'],
                [0.60, 'rgb(255, 165, 0)'],
                [0.80, 'rgb(255, 0, 0)'],
                [1.0, 'rgb(100, 0, 0)']
            ]

            fig.add_trace(go.Heatmap(
                z=effective,
                colorscale=colorscale,
                zmin=0,
                zmax=100,
                colorbar=dict(title='mmHg', len=0.4, y=0.25),
                hovertemplate='Pressure: %{z:.1f} mmHg<extra></extra>'
            ), row=2, col=2)

        # Layout
        final_metrics = self.history[-1] if self.history else {}

        fig.update_layout(
            title=dict(
                text=f'<b>Evolved Mattress Pattern</b><br>'
                     f'<sup>Generation {len(self.history)} | '
                     f'Peak PTI: {final_metrics.get("max_pti", 0):.1f} mmHg·h | '
                     f'At Risk: {final_metrics.get("at_risk_cells", 0)}</sup>',
                x=0.5,
                font=dict(size=18)
            ),
            height=900,
            width=1200,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )

        fig.update_xaxes(title_text='Generation', row=1, col=1)
        fig.update_yaxes(title_text='Fitness', row=1, col=1)
        fig.update_xaxes(title_text='Generation', row=1, col=2)
        fig.update_yaxes(title_text='PTI (mmHg·h)', row=1, col=2)

        fig.write_html(save_path, include_plotlyjs=True, full_html=True)
        print(f"\nSaved visualization: {save_path}")


def export_best_genome(genome: PatternGenome, filename: str = 'evolved_pattern.py'):
    """Export the best genome as a Python class for use in other scripts."""

    code = '''#!/usr/bin/env python3
"""
Auto-Generated Evolved Mattress Pattern
========================================
Generated by evolve_optimal_mattress.py
Fitness: {fitness:.4f}
Generation: {generation}
"""

import numpy as np


class EvolvedOptimalPattern:
    """Evolved optimal pressure redistribution pattern."""

    name = "Evolved Optimal"
    pattern_type = "evolved"

    # Region parameters (evolved)
    REGIONS = {{
'''.format(fitness=genome.fitness, generation=genome.generation)

    for region, params in genome.regions.items():
        code += f'''        '{region}': {{
            'bounds': {PatternGenome.REGION_BOUNDS[region]},
            'cycle_speed': {params.cycle_speed:.3f},
            'wave_type': '{params.wave_type}',
            'phase_offset': {params.phase_offset:.3f},
            'min_inflation': {params.min_inflation:.3f},
            'max_inflation': {params.max_inflation:.3f},
        }},
'''

    code += '''    }

    def get_cell_state(self, row: int, col: int, rows: int, cols: int, phase: float,
                        smooth: bool = False) -> float:
        """Calculate cell inflation based on evolved parameters.

        Args:
            row, col: Cell position
            rows, cols: Grid dimensions
            phase: Current cycle phase (0-1)
            smooth: If True, use smooth transitions for visualization
        """
        row_pos = row / rows
        col_pos = col / cols

        # Find active region
        active_region = 'default'
        for region, params in self.REGIONS.items():
            if region == 'default':
                continue
            r0, r1, c0, c1 = params['bounds']
            if r0 <= row_pos < r1 and c0 <= col_pos <= c1:
                active_region = region
                break

        params = self.REGIONS[active_region]
        local_phase = (phase * params['cycle_speed'] + params['phase_offset']) % 1.0

        # Apply wave type
        wave_type = params['wave_type']
        if wave_type == 'sine':
            wave = 0.5 + 0.5 * np.cos(2 * np.pi * local_phase)
        elif wave_type == 'square':
            if smooth:
                # Smooth square wave using sigmoid-like transition
                k = 20
                wave = 1.0 / (1.0 + np.exp(k * (local_phase - 0.25))) + \\
                       1.0 / (1.0 + np.exp(-k * (local_phase - 0.75))) - 1.0
                wave = np.clip(wave, 0, 1)
            else:
                wave = 1.0 if local_phase < 0.5 else 0.0
        elif wave_type == 'triangle':
            wave = 1.0 - 2 * abs(local_phase - 0.5)
        elif wave_type == 'sawtooth':
            wave = 1.0 - local_phase
        else:
            wave = 0.5

        # Scale to inflation range
        return params['min_inflation'] + wave * (params['max_inflation'] - params['min_inflation'])
'''

    with open(filename, 'w') as f:
        f.write(code)

    print(f"Exported evolved pattern to: {filename}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evolve optimal mattress pattern')
    parser.add_argument('--population', type=int, default=30, help='Population size')
    parser.add_argument('--generations', type=int, default=20, help='Max generations')
    parser.add_argument('--mutation', type=float, default=0.25, help='Mutation rate')
    parser.add_argument('--elite', type=int, default=3, help='Elite count')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    args = parser.parse_args()

    # Run optimization
    optimizer = GeneticOptimizer(
        population_size=args.population,
        elite_count=args.elite,
        mutation_rate=args.mutation,
    )

    best = optimizer.run(
        generations=args.generations,
        convergence_threshold=5,
        verbose=not args.quiet
    )

    # Print best pattern
    print("\n" + "=" * 70)
    print("BEST EVOLVED PATTERN")
    print("=" * 70)

    for region, params in best.regions.items():
        print(f"\n{region.upper()}:")
        print(f"  Cycle Speed:  {params.cycle_speed:.2f}x")
        print(f"  Wave Type:    {params.wave_type}")
        print(f"  Phase Offset: {params.phase_offset:.2f}")
        print(f"  Inflation:    {params.min_inflation:.2f} - {params.max_inflation:.2f}")

    # Create visualization
    optimizer.create_visualization('evolved_mattress.html')

    # Export as Python code
    export_best_genome(best, 'evolved_pattern.py')

    print("\n" + "=" * 70)
    print("Files created:")
    print("  - evolved_mattress.html (visualization)")
    print("  - evolved_pattern.py (usable pattern class)")
    print("=" * 70)
