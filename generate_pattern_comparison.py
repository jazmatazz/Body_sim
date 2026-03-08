#!/usr/bin/env python3
"""Generate pattern comparison visualization for all movement patterns."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from multidynamic_mattress_optimization import (
    AlternatingCheckerboard, WaveHorizontal, WaveVertical,
    ZoneBased, CircularWave, DiagonalWave, RowGroups,
    MultiFrequencyZone, SequentialRows
)


# Patterns with best phase to show their behavior clearly
PATTERNS = [
    ('Alternating Checkerboard', AlternatingCheckerboard(), 0.0),
    ('Horizontal Wave', WaveHorizontal(wave_width=3), 0.25),
    ('Vertical Wave', WaveVertical(wave_width=2), 0.25),
    ('Zone-Based Adaptive', ZoneBased(), 0.2),
    ('Circular Wave', CircularWave(), 0.3),
    ('Diagonal Wave', DiagonalWave(), 0.25),
    ('Row Groups (1)', RowGroups(group_size=1), 0.0),
    ('Row Groups (2)', RowGroups(group_size=2), 0.0),
    ('Row Groups (3)', RowGroups(group_size=3), 0.0),
    ('Multi-Frequency', MultiFrequencyZone(), 0.0),
    ('Sequential Rows', SequentialRows(), 0.3),
]


def create_pattern_comparison():
    """Create simple grid visualization of all patterns."""
    n_patterns = len(PATTERNS)
    cols = 4
    rows = (n_patterns + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[name for name, _, _ in PATTERNS],
        vertical_spacing=0.12,
        horizontal_spacing=0.05,
    )

    # Grid size
    grid_rows, grid_cols = 40, 18  # Higher resolution

    for idx, (name, pattern, phase) in enumerate(PATTERNS):
        pattern_grid = np.zeros((grid_rows, grid_cols))

        for i in range(grid_rows):
            for j in range(grid_cols):
                val = pattern.get_cell_state(i, j, grid_rows, grid_cols, phase)
                # Zone-Based uses gradient to show different zones
                if name == 'Zone-Based Adaptive':
                    pattern_grid[i, j] = val
                else:
                    pattern_grid[i, j] = 1.0 if val > 0.5 else 0.0

        row = idx // cols + 1
        col = idx % cols + 1

        fig.add_trace(go.Heatmap(
            z=pattern_grid,
            colorscale=[[0, '#e74c3c'], [1, '#27ae60']],
            showscale=False,
            zmin=0,
            zmax=1,
        ), row=row, col=col)

    fig.update_layout(
        title=dict(
            text='<b>APM Movement Patterns</b><br>'
                 '<sup>Green = Inflated | Red = Deflated</sup>',
            x=0.5,
            font=dict(size=20),
        ),
        height=280 * rows,
        width=900,
    )

    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    fig.write_html('pattern_comparison.html', include_plotlyjs=True, full_html=True)
    print(f"Saved: pattern_comparison.html ({n_patterns} patterns)")


if __name__ == "__main__":
    create_pattern_comparison()
