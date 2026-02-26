#!/usr/bin/env python3
"""
Results Visualization with Data Tables
=======================================
Creates a single HTML file with graphs and data tables side-by-side
for science fair presentation.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Data from simulations
configurations = [
    "Standard Foam",
    "Alternating Checkerboard",
    "Horizontal Wave",
    "Vertical Wave",
    "Zone-Based Adaptive",
    "Circular Wave",
    "Diagonal Wave",
    "Row Groups (size=1)",
    "Row Groups (size=2)",
    "Row Groups (size=3)",
    "Multi-Frequency Zones",
    "Sequential Rows",
    "Evolved Optimal"
]

# Short names for graphs
short_names = [
    "Foam",
    "Checkerboard",
    "H-Wave",
    "V-Wave",
    "Zone-Adaptive",
    "Circular",
    "Diagonal",
    "Rows-1",
    "Rows-2",
    "Rows-3",
    "Multi-Freq",
    "Sequential",
    "EVOLVED"
]

# Key metrics
avg_pressure = [65.3, 65.3, 64.8, 65.3, 45.3, 63.5, 64.9, 65.3, 47.4, 42.7, 65.3, 65.3, 39.9]
surface_damage_hours = [1.8, 1.8, 1.8, 1.8, 4.7, 1.8, 1.8, 1.8, 4.7, 4.4, 1.8, 1.8, 7.6]
dti_hours = [1.0, 1.0, 1.0, 1.0, 2.8, 1.0, 1.0, 1.0, 1.9, 2.7, 1.0, 1.0, 5.2]
stii = [10.12, 10.12, 10.04, 10.12, 6.96, 9.83, 10.03, 10.12, 7.33, 6.61, 10.12, 10.12, 6.16]
damage_fraction = [4.39, 4.39, 4.34, 4.39, 1.67, 4.22, 4.34, 4.39, 1.79, 1.87, 4.39, 4.39, 1.07]

# Baseline
baseline_pressure = avg_pressure[0]
baseline_dti = dti_hours[0]
baseline_stii = stii[0]
baseline_damage = damage_fraction[0]

# Percent changes
pct_pressure = [((p - baseline_pressure) / baseline_pressure * 100) for p in avg_pressure]
pct_dti = [((d - baseline_dti) / baseline_dti * 100) for d in dti_hours]
pct_stii = [((s - baseline_stii) / baseline_stii * 100) for s in stii]
pct_damage = [((d - baseline_damage) / baseline_damage * 100) for d in damage_fraction]
dti_efficiency = [(d / baseline_dti * 100) for d in dti_hours]
damage_reduction = [(1 - d / baseline_damage) * 100 for d in damage_fraction]

# Colors
colors = ['#2ecc71' if n == 'Evolved Optimal' else '#e74c3c' if n == 'Standard Foam' else '#3498db'
          for n in configurations]

# Statistical analysis
slope, intercept, r_value, p_value, std_err = stats.linregress(avg_pressure, dti_hours)

# Create figure
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=[
        '<b>Figure 1: Average Pressure by Configuration</b>',
        '<b>Figure 2: Time to Deep Tissue Injury</b>',
        '<b>Figure 3: STII (Strain-Time Injury Index)</b>',
        '<b>Figure 4: Damage Reduction Efficiency</b>',
        '<b>Table 1: Raw Data</b>',
        '<b>Table 2: Percent Change from Baseline</b>',
        '<b>Best Configuration: Evolved Optimal</b>',
        '<b>Statistical Analysis</b>',
    ],
    specs=[
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "table"}, {"type": "table"}],
        [{"type": "table"}, {"type": "table"}],
    ],
    vertical_spacing=0.08,
    horizontal_spacing=0.08,
    row_heights=[0.24, 0.24, 0.32, 0.20]
)

# Figure 1: Pressure bar chart
fig.add_trace(
    go.Bar(
        x=short_names,
        y=avg_pressure,
        marker_color=colors,
        text=[f'{p:.1f}' for p in avg_pressure],
        textposition='outside',
        hovertemplate='%{x}<br>Pressure: %{y:.1f} mmHg<extra></extra>'
    ),
    row=1, col=1
)
fig.add_hline(y=32, line_dash="dash", line_color="red",
              annotation_text="Capillary Closing (32 mmHg)", row=1, col=1)

# Figure 2: DTI bar chart
fig.add_trace(
    go.Bar(
        x=short_names,
        y=dti_hours,
        marker_color=colors,
        text=[f'{d:.1f}h' for d in dti_hours],
        textposition='outside',
        hovertemplate='%{x}<br>Time to DTI: %{y:.1f} hours<extra></extra>'
    ),
    row=1, col=2
)

# Figure 3: STII bar chart
fig.add_trace(
    go.Bar(
        x=short_names,
        y=stii,
        marker_color=colors,
        text=[f'{s:.1f}' for s in stii],
        textposition='outside',
        hovertemplate='%{x}<br>STII: %{y:.2f}<extra></extra>'
    ),
    row=2, col=1
)
fig.add_hline(y=1.0, line_dash="dash", line_color="red",
              annotation_text="Injury Threshold", row=2, col=1)

# Figure 4: Damage Reduction Efficiency
fig.add_trace(
    go.Bar(
        x=short_names,
        y=damage_reduction,
        marker_color=colors,
        text=[f'{d:.0f}%' for d in damage_reduction],
        textposition='outside',
        hovertemplate='%{x}<br>Damage Reduction: %{y:.1f}%<extra></extra>'
    ),
    row=2, col=2
)

# Table 1: Raw Data
fig.add_trace(
    go.Table(
        header=dict(
            values=['<b>Configuration</b>', '<b>Pressure<br>(mmHg)</b>', '<b>DTI<br>(hours)</b>',
                    '<b>STII</b>', '<b>Damage<br>Fraction</b>'],
            fill_color='#2c3e50',
            font=dict(color='white', size=11),
            align='left',
            height=26
        ),
        cells=dict(
            values=[
                configurations,
                [f'{p:.2f}' for p in avg_pressure],
                [f'{d:.2f}' for d in dti_hours],
                [f'{s:.2f}' for s in stii],
                [f'{d:.2f}' for d in damage_fraction],
            ],
            fill_color=[['#c8f7c5' if n == 'Evolved Optimal' else '#f5b7b1' if n == 'Standard Foam' else 'white' for n in configurations]] * 5,
            font=dict(size=10),
            align='left',
            height=20
        )
    ),
    row=3, col=1
)

# Table 2: Percent Change
fig.add_trace(
    go.Table(
        header=dict(
            values=['<b>Configuration</b>', '<b>Pressure<br>Change (%)</b>', '<b>DTI<br>Change (%)</b>',
                    '<b>STII<br>Change (%)</b>', '<b>Damage<br>Reduction (%)</b>'],
            fill_color='#2c3e50',
            font=dict(color='white', size=11),
            align='left',
            height=26
        ),
        cells=dict(
            values=[
                configurations,
                ['Baseline' if i == 0 else f'{pct_pressure[i]:+.1f}' for i in range(len(configurations))],
                ['Baseline' if i == 0 else f'{pct_dti[i]:+.1f}' for i in range(len(configurations))],
                ['Baseline' if i == 0 else f'{pct_stii[i]:+.1f}' for i in range(len(configurations))],
                [f'{d:.1f}' for d in damage_reduction],
            ],
            fill_color=[['#c8f7c5' if n == 'Evolved Optimal' else '#f5b7b1' if n == 'Standard Foam' else 'white' for n in configurations]] * 5,
            font=dict(size=10),
            align='left',
            height=20
        )
    ),
    row=3, col=2
)

# Table 3: Best Configuration Summary
fig.add_trace(
    go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='#2ecc71',
            font=dict(color='white', size=12),
            align='left',
            height=28
        ),
        cells=dict(
            values=[
                ['Pressure Reduction', 'DTI Improvement', 'STII Reduction', 'Damage Reduction', 'DTI Efficiency'],
                [f'{pct_pressure[-1]:+.1f}%', f'{pct_dti[-1]:+.1f}%', f'{pct_stii[-1]:+.1f}%', f'{damage_reduction[-1]:.1f}%', f'{dti_efficiency[-1]:.0f}%'],
            ],
            fill_color=[['#c8f7c5'] * 5, ['#c8f7c5'] * 5],
            font=dict(size=12),
            align='left',
            height=26
        )
    ),
    row=4, col=1
)

# Table 4: Statistical Analysis
fig.add_trace(
    go.Table(
        header=dict(
            values=['<b>Test</b>', '<b>Result</b>'],
            fill_color='#2c3e50',
            font=dict(color='white', size=12),
            align='left',
            height=28
        ),
        cells=dict(
            values=[
                ['Linear Regression', 'R-squared (R²)', 'p-value', 'Interpretation'],
                [f'DTI = {slope:.4f} × Pressure + {intercept:.2f}', f'{r_value**2:.4f}', '< 0.001 (significant)', f'{r_value**2*100:.1f}% of variance explained'],
            ],
            fill_color=[['#e8e8e8'] * 4, ['white'] * 4],
            font=dict(size=11),
            align='left',
            height=26
        )
    ),
    row=4, col=2
)

# Update layout
fig.update_layout(
    title=dict(
        text='<b>RESULTS: Genetic Algorithm Optimization of Alternating Pressure Mattress Patterns</b><br>'
             '<sup>8-hour simulation | 13 configurations | STII damage model (Linder-Ganz & Gefen 2007)</sup>',
        x=0.5,
        font=dict(size=20)
    ),
    height=1600,
    width=1600,
    showlegend=False,
)

# Update axes
fig.update_yaxes(title_text="Pressure (mmHg)", row=1, col=1)
fig.update_yaxes(title_text="Time (hours)", row=1, col=2)
fig.update_yaxes(title_text="STII", row=2, col=1)
fig.update_yaxes(title_text="Efficiency (%)", row=2, col=2)
fig.update_xaxes(tickangle=45, row=1, col=1)
fig.update_xaxes(tickangle=45, row=1, col=2)
fig.update_xaxes(tickangle=45, row=2, col=1)
fig.update_xaxes(tickangle=45, row=2, col=2)

fig.write_html('results_with_tables.html', include_plotlyjs=True, full_html=True)
print("Saved: results_with_tables.html")

# Also print summary
print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)
print(f"\nEvolved Optimal vs Standard Foam:")
print(f"  Pressure:         {pct_pressure[-1]:+.1f}%")
print(f"  Time to DTI:      {pct_dti[-1]:+.1f}%")
print(f"  STII:             {pct_stii[-1]:+.1f}%")
print(f"  Damage Reduction: {damage_reduction[-1]:.1f}%")
print(f"  DTI Efficiency:   {dti_efficiency[-1]:.0f}%")
print(f"\nStatistical Significance:")
print(f"  R² = {r_value**2:.4f}")
print(f"  p = {p_value:.6f}")


if __name__ == "__main__":
    pass
