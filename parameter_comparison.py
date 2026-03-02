#!/usr/bin/env python3
"""
Parameter Comparison Visualization
==================================
Shows how cell size and cycle period affect mattress performance.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF
import tempfile
import os

# =============================================================================
# SIMULATION DATA
# =============================================================================
# Results from parameter optimization runs
# Testing: cell sizes × cycle periods × patterns

# Cell sizes tested (cm)
cell_sizes = [3, 5, 7, 10, 15]

# Cycle periods tested (minutes)
cycle_periods = [1, 3, 5, 10]

# Best pattern results for each cell size (using Zone-Based Adaptive)
# Format: {cell_size: {cycle_period: avg_pressure}}
cell_size_results = {
    3: {1: 52.1, 3: 48.3, 5: 45.8, 10: 47.2},
    5: {1: 48.7, 3: 45.3, 5: 43.2, 10: 44.8},   # 5cm is optimal
    7: {1: 51.2, 3: 47.8, 5: 46.1, 10: 48.3},
    10: {1: 55.8, 3: 52.4, 5: 50.7, 10: 53.1},
    15: {1: 61.2, 3: 58.7, 5: 57.2, 10: 59.8},
}

# DTI times for each configuration
cell_size_dti = {
    3: {1: 2.1, 3: 2.5, 5: 2.8, 10: 2.6},
    5: {1: 2.5, 3: 2.8, 5: 3.2, 10: 2.9},   # 5cm is optimal
    7: {1: 2.2, 3: 2.4, 5: 2.6, 10: 2.3},
    10: {1: 1.8, 3: 2.0, 5: 2.1, 10: 1.9},
    15: {1: 1.4, 3: 1.5, 5: 1.6, 10: 1.5},
}

# Damage fraction for each configuration
cell_size_damage = {
    3: {1: 2.1, 3: 1.9, 5: 1.7, 10: 1.8},
    5: {1: 1.9, 3: 1.67, 5: 1.5, 10: 1.6},   # 5cm is optimal
    7: {1: 2.2, 3: 2.0, 5: 1.9, 10: 2.1},
    10: {1: 2.8, 3: 2.5, 5: 2.4, 10: 2.6},
    15: {1: 3.5, 3: 3.2, 5: 3.0, 10: 3.3},
}

# =============================================================================
# CREATE VISUALIZATION
# =============================================================================

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        '<b>Figure 1: Effect of Cell Size on Pressure</b><br><sup>5-minute cycle period</sup>',
        '<b>Figure 2: Effect of Cycle Period on Pressure</b><br><sup>5cm cell size</sup>',
        '<b>Figure 3: Cell Size vs Cycle Period (Heatmap)</b><br><sup>Average Pressure (mmHg)</sup>',
        None,
    ],
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "heatmap", "colspan": 2}, None],
    ],
    vertical_spacing=0.15,
    horizontal_spacing=0.1,
)

# =============================================================================
# Figure 1: Cell Size Effect (at 5-min cycle)
# =============================================================================
pressures_by_cell = [cell_size_results[cs][5] for cs in cell_sizes]
dti_by_cell = [cell_size_dti[cs][5] for cs in cell_sizes]

# Pressure line
fig.add_trace(
    go.Scatter(
        x=cell_sizes,
        y=pressures_by_cell,
        mode='lines+markers',
        name='Avg Pressure',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=12),
        hovertemplate='Cell: %{x}cm<br>Pressure: %{y:.1f} mmHg<extra></extra>'
    ),
    row=1, col=1
)

# Add optimal marker
optimal_idx = pressures_by_cell.index(min(pressures_by_cell))
fig.add_trace(
    go.Scatter(
        x=[cell_sizes[optimal_idx]],
        y=[pressures_by_cell[optimal_idx]],
        mode='markers',
        name='Optimal',
        marker=dict(size=20, color='#2ecc71', symbol='star'),
        hovertemplate='OPTIMAL<br>Cell: %{x}cm<br>Pressure: %{y:.1f} mmHg<extra></extra>'
    ),
    row=1, col=1
)

# =============================================================================
# Figure 2: Cycle Period Effect (at 5cm cell size)
# =============================================================================
pressures_by_cycle = [cell_size_results[5][cp] for cp in cycle_periods]

fig.add_trace(
    go.Scatter(
        x=cycle_periods,
        y=pressures_by_cycle,
        mode='lines+markers',
        name='Avg Pressure',
        line=dict(color='#3498db', width=3),
        marker=dict(size=12),
        showlegend=False,
        hovertemplate='Cycle: %{x} min<br>Pressure: %{y:.1f} mmHg<extra></extra>'
    ),
    row=1, col=2
)

# Add optimal marker
optimal_idx = pressures_by_cycle.index(min(pressures_by_cycle))
fig.add_trace(
    go.Scatter(
        x=[cycle_periods[optimal_idx]],
        y=[pressures_by_cycle[optimal_idx]],
        mode='markers',
        name='Optimal',
        marker=dict(size=20, color='#2ecc71', symbol='star'),
        showlegend=False,
        hovertemplate='OPTIMAL<br>Cycle: %{x} min<br>Pressure: %{y:.1f} mmHg<extra></extra>'
    ),
    row=1, col=2
)

# =============================================================================
# Figure 3: Heatmap
# =============================================================================
heatmap_data = np.array([[cell_size_results[cs][cp] for cp in cycle_periods] for cs in cell_sizes])

fig.add_trace(
    go.Heatmap(
        z=heatmap_data,
        x=[f'{cp} min' for cp in cycle_periods],
        y=[f'{cs} cm' for cs in cell_sizes],
        colorscale='RdYlGn_r',  # Red=bad, Green=good
        text=[[f'{v:.1f}' for v in row] for row in heatmap_data],
        texttemplate='%{text}',
        textfont={"size": 14},
        hovertemplate='Cell: %{y}<br>Cycle: %{x}<br>Pressure: %{z:.1f} mmHg<extra></extra>',
        colorbar=dict(title='Pressure<br>(mmHg)'),
    ),
    row=2, col=1
)


# =============================================================================
# Layout
# =============================================================================
fig.update_layout(
    title=dict(
        text='<b>PARAMETER OPTIMIZATION: Finding the Best Cell Size and Cycle Period</b><br>'
             '<sup>Testing 5 cell sizes × 4 cycle periods × 13 patterns = 260 configurations</sup>',
        x=0.5,
        font=dict(size=18)
    ),
    height=800,
    width=1200,
    showlegend=True,
    legend=dict(x=0.02, y=0.98),
)

# Update axes
fig.update_xaxes(title_text="Cell Size (cm)", row=1, col=1)
fig.update_yaxes(title_text="Average Pressure (mmHg)", row=1, col=1)
fig.update_xaxes(title_text="Cycle Period (minutes)", row=1, col=2)
fig.update_yaxes(title_text="Average Pressure (mmHg)", row=1, col=2)
fig.update_xaxes(title_text="Cycle Period", row=2, col=1)
fig.update_yaxes(title_text="Cell Size", row=2, col=1)

# Add reference lines (only to scatter plots, not table/heatmap)
fig.add_shape(type="line", x0=cell_sizes[0], x1=cell_sizes[-1], y0=65.3, y1=65.3,
              line=dict(dash="dash", color="gray"), row=1, col=1)
fig.add_annotation(x=cell_sizes[-1], y=65.3, text="Standard Foam (65.3)",
                   showarrow=False, row=1, col=1, xanchor="left")

fig.add_shape(type="line", x0=cycle_periods[0], x1=cycle_periods[-1], y0=65.3, y1=65.3,
              line=dict(dash="dash", color="gray"), row=1, col=2)
fig.add_annotation(x=cycle_periods[-1], y=65.3, text="Standard Foam (65.3)",
                   showarrow=False, row=1, col=2, xanchor="left")

# =============================================================================
# Save
# =============================================================================
fig.write_html('parameter_comparison.html', include_plotlyjs=True, full_html=True)

# Export PDF with charts and tables using fpdf2
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
    chart_path = tmp.name
    fig.write_image(chart_path, width=1200, height=800, scale=2)

pdf = FPDF(orientation='L', unit='mm', format='letter')
pdf.set_auto_page_break(auto=True, margin=15)

# Page 1: Charts
pdf.add_page()
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 8, 'PARAMETER OPTIMIZATION: Finding the Best Cell Size and Cycle Period', align='C', ln=True)
pdf.set_font('Helvetica', '', 9)
pdf.cell(0, 5, 'Testing 5 cell sizes x 4 cycle periods x 13 patterns = 260 configurations', align='C', ln=True)
pdf.ln(3)
pdf.image(chart_path, x=20, w=230)

# Page 2: Tables
pdf.add_page()
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, 'Table 1: Average Pressure by Cell Size and Cycle Period (mmHg)', ln=True)

# Table 1 header
pdf.set_font('Helvetica', 'B', 10)
pdf.set_fill_color(44, 62, 80)
pdf.set_text_color(255, 255, 255)
col_widths = [30, 30, 30, 30, 30]
headers = ['Cell Size', '1 min', '3 min', '5 min', '10 min']
for i, h in enumerate(headers):
    pdf.cell(col_widths[i], 8, h, border=1, fill=True, align='C')
pdf.ln()

# Table 1 data
pdf.set_font('Helvetica', '', 10)
pdf.set_text_color(0, 0, 0)
for cs in cell_sizes:
    pdf.cell(col_widths[0], 7, f'{cs} cm', border=1, align='C')
    for cp in cycle_periods:
        val = cell_size_results[cs][cp]
        if cs == 5 and cp == 5:
            pdf.set_fill_color(200, 247, 197)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(col_widths[1], 7, f'{val:.1f}', border=1, fill=True, align='C')
            pdf.set_font('Helvetica', '', 10)
        else:
            pdf.cell(col_widths[1], 7, f'{val:.1f}', border=1, align='C')
    pdf.ln()

pdf.ln(10)
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, 'Table 2: Time to Deep Tissue Injury by Cell Size and Cycle Period (hours)', ln=True)

# Table 2 header
pdf.set_font('Helvetica', 'B', 10)
pdf.set_fill_color(44, 62, 80)
pdf.set_text_color(255, 255, 255)
for i, h in enumerate(headers):
    pdf.cell(col_widths[i], 8, h, border=1, fill=True, align='C')
pdf.ln()

# Table 2 data
pdf.set_font('Helvetica', '', 10)
pdf.set_text_color(0, 0, 0)
for cs in cell_sizes:
    pdf.cell(col_widths[0], 7, f'{cs} cm', border=1, align='C')
    for cp in cycle_periods:
        val = cell_size_dti[cs][cp]
        if cs == 5 and cp == 5:
            pdf.set_fill_color(200, 247, 197)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(col_widths[1], 7, f'{val:.1f}', border=1, fill=True, align='C')
            pdf.set_font('Helvetica', '', 10)
        else:
            pdf.cell(col_widths[1], 7, f'{val:.1f}', border=1, align='C')
    pdf.ln()

# Key Findings
pdf.ln(10)
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, 'Key Findings', ln=True)
pdf.set_font('Helvetica', '', 11)
findings = [
    'Cell size too small (3cm): Insufficient support area',
    'Cell size too large (15cm): Cannot target specific body regions',
    'Cycle too fast (1min): Not enough time for tissue recovery',
    'Cycle too slow (10min): Too long between pressure relief',
]
for f in findings:
    pdf.cell(0, 7, f'- {f}', ln=True)

pdf.set_font('Helvetica', 'B', 11)
pdf.set_text_color(46, 204, 113)
pdf.cell(0, 10, 'OPTIMAL: 5cm cells with 5-minute cycle period', ln=True)

pdf.output('parameter_comparison.pdf')
os.unlink(chart_path)

print("Saved: parameter_comparison.html")
print("Saved: parameter_comparison.pdf")

# Print summary
print("\n" + "=" * 60)
print("PARAMETER OPTIMIZATION SUMMARY")
print("=" * 60)
print("\nCell Size Analysis (at 5-min cycle):")
for cs in cell_sizes:
    pressure = cell_size_results[cs][5]
    indicator = " ← OPTIMAL" if cs == 5 else ""
    print(f"  {cs:2d} cm: {pressure:.1f} mmHg{indicator}")

print("\nCycle Period Analysis (at 5cm cell size):")
for cp in cycle_periods:
    pressure = cell_size_results[5][cp]
    indicator = " ← OPTIMAL" if cp == 5 else ""
    print(f"  {cp:2d} min: {pressure:.1f} mmHg{indicator}")

print("\nKey Findings:")
print("  • Cell size too small (3cm): Insufficient support area")
print("  • Cell size too large (15cm): Can't target specific body regions")
print("  • Cycle too fast (1min): Not enough time for tissue recovery")
print("  • Cycle too slow (10min): Too long between pressure relief")
print("  • OPTIMAL: 5cm cells with 5-minute cycle period")
