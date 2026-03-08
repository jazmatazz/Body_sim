#!/usr/bin/env python3
"""Generate pattern comparison visualization for all movement patterns."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import all pattern classes
from multidynamic_mattress_optimization import (
    AlternatingCheckerboard, WaveHorizontal, WaveVertical,
    ZoneBased, CircularWave, DiagonalWave, RowGroups,
    MultiFrequencyZone, SequentialRows
)
from evolved_pattern import OptimalPattern


# All patterns to visualize
PATTERNS = {
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
    'optimal': OptimalPattern(),
}


def create_pattern_comparison():
    """Create visualization comparing all movement patterns."""
    patterns = list(PATTERNS.keys())
    n_patterns = len(patterns)
    cols = 4
    rows = (n_patterns + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[PATTERNS[p].name for p in patterns],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )
    
    # Visualize each pattern at 25% phase
    test_rows, test_cols = 40, 18  # Grid representing mattress
    
    for idx, pattern_name in enumerate(patterns):
        pattern = PATTERNS[pattern_name]
        pattern_grid = np.zeros((test_rows, test_cols))
        
        phase = 0.25
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
            zmin=0,
            zmax=1,
            colorbar=dict(
                title='Inflation',
                x=1.02,
                tickvals=[0, 0.5, 1],
                ticktext=['Deflated', 'Mid', 'Inflated']
            ) if idx == 0 else None,
        ), row=row, col=col)
    
    fig.update_layout(
        title=dict(
            text='<b>Movement Pattern Comparison</b><br>'
                 '<sup>Green = Inflated (support), Red = Deflated (relief) | Showing 25% through cycle</sup>',
            x=0.5,
            font=dict(size=18),
        ),
        height=250 * rows,
        width=1400,
    )
    
    # Hide axis labels for cleaner look
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    fig.write_html('pattern_comparison.html', include_plotlyjs=True, full_html=True)
    print(f"Saved: pattern_comparison.html")
    print(f"Total patterns: {n_patterns}")
    
    return fig


if __name__ == "__main__":
    create_pattern_comparison()
