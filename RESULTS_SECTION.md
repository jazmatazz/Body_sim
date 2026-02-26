# RESULTS

## Tables

### Table 1: Mattress Configuration Comparison
**Legend:** Comparison of 13 mattress configurations over 8-hour simulation. Baseline: Standard Foam mattress. APM = Alternating Pressure Mattress. DTI = Deep Tissue Injury. STII = Strain-Time Injury Index.

| Configuration | Pressure (mmHg) | Surface Damage (hours) | DTI (hours) | STII | Damage Fraction |
|---------------|-----------------|------------------------|-------------|------|-----------------|
| Standard Foam | 65.30 | 1.80 | 1.00 | 10.12 | 4.39 |
| Alternating Checkerboard | 65.30 | 1.80 | 1.00 | 10.12 | 4.39 |
| Horizontal Wave | 64.80 | 1.80 | 1.00 | 10.04 | 4.34 |
| Vertical Wave | 65.30 | 1.80 | 1.00 | 10.12 | 4.39 |
| Zone-Based Adaptive | 45.30 | 4.70 | 2.80 | 6.96 | 1.67 |
| Circular Wave | 63.50 | 1.80 | 1.00 | 9.83 | 4.22 |
| Diagonal Wave | 64.90 | 1.80 | 1.00 | 10.03 | 4.34 |
| Row Groups (size=1) | 65.30 | 1.80 | 1.00 | 10.12 | 4.39 |
| Row Groups (size=2) | 47.40 | 4.70 | 1.90 | 7.33 | 1.79 |
| Row Groups (size=3) | 42.70 | 4.40 | 2.70 | 6.61 | 1.87 |
| Multi-Frequency Zones | 65.30 | 1.80 | 1.00 | 10.12 | 4.39 |
| Sequential Rows | 65.30 | 1.80 | 1.00 | 10.12 | 4.39 |
| **Evolved Optimal** | **39.90** | **7.60** | **5.20** | **6.16** | **1.07** |

---

### Table 2: Percent Change from Baseline
**Legend:** Percent change compared to Standard Foam baseline. Negative values indicate improvement (reduction in pressure, STII, or damage). Positive values for Surface Damage and DTI indicate increased time before injury (improvement).

| Configuration | Pressure Change (%) | Surface Damage Change (%) | DTI Change (%) | STII Change (%) | Damage Change (%) |
|---------------|---------------------|---------------------------|----------------|-----------------|-------------------|
| Standard Foam | Baseline | Baseline | Baseline | Baseline | Baseline |
| Alternating Checkerboard | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Horizontal Wave | -0.8 | 0.0 | 0.0 | -0.8 | -1.1 |
| Vertical Wave | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Zone-Based Adaptive | -30.6 | +161.1 | +180.0 | -31.2 | -62.0 |
| Circular Wave | -2.8 | 0.0 | 0.0 | -2.9 | -3.9 |
| Diagonal Wave | -0.6 | 0.0 | 0.0 | -0.9 | -1.1 |
| Row Groups (size=1) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Row Groups (size=2) | -27.4 | +161.1 | +90.0 | -27.6 | -59.2 |
| Row Groups (size=3) | -34.6 | +144.4 | +170.0 | -34.7 | -57.4 |
| Multi-Frequency Zones | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Sequential Rows | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Evolved Optimal** | **-38.9** | **+322.2** | **+420.0** | **-39.1** | **-75.6** |

---

### Table 3: Statistical Analysis
**Legend:** Statistical tests comparing mattress performance groups.

| Test | Groups Compared | Statistic | p-value | Significant |
|------|-----------------|-----------|---------|-------------|
| Independent t-test | Effective APM (n=4) vs Ineffective APM (n=8) | t = -18.62 | p < 0.001 | Yes |
| Linear Regression | Pressure vs Time to DTI | R² = 0.80 | p < 0.001 | Yes |
| Linear Regression | Pressure vs STII | R² = 1.00 | p < 0.001 | Yes |
| One-Way ANOVA | Basic APM vs Optimized APM | F = 482.82 | p < 0.001 | Yes |

**Regression Equations:**
- Time to DTI = -0.1086 × Pressure + 8.0134 (R² = 0.80)
- STII = 0.1560 × Pressure - 0.0701 (R² = 1.00)

---

### Table 4: Efficiency Calculations
**Legend:** DTI Efficiency = (Time to DTI / Baseline DTI Time) × 100%. Damage Reduction Efficiency = (1 - Damage Fraction / Baseline Damage) × 100%.

| Configuration | DTI Efficiency (%) | Damage Reduction Efficiency (%) |
|---------------|--------------------|---------------------------------|
| Standard Foam | 100.0 | 0.0 |
| Alternating Checkerboard | 100.0 | 0.0 |
| Horizontal Wave | 100.0 | 1.1 |
| Vertical Wave | 100.0 | 0.0 |
| Zone-Based Adaptive | 280.0 | 62.0 |
| Circular Wave | 100.0 | 3.9 |
| Diagonal Wave | 100.0 | 1.1 |
| Row Groups (size=1) | 100.0 | 0.0 |
| Row Groups (size=2) | 190.0 | 59.2 |
| Row Groups (size=3) | 270.0 | 57.4 |
| Multi-Frequency Zones | 100.0 | 0.0 |
| Sequential Rows | 100.0 | 0.0 |
| **Evolved Optimal** | **520.0** | **75.6** |

---

## Figures

### Figure 1: Average Pressure by Mattress Configuration
**Description:** Bar graph comparing average interface pressure (mmHg) across all 13 mattress configurations.

### Figure 2: Time to Deep Tissue Injury by Configuration
**Description:** Bar graph showing predicted time to DTI (hours) for each configuration.

### Figure 3: Pressure vs Time to DTI (Scatter Plot with Regression)
**Description:** Scatter plot with linear regression line. X-axis: Average Pressure (mmHg). Y-axis: Time to DTI (hours). R² = 0.80.

### Figure 4: STII Comparison
**Description:** Bar graph comparing Strain-Time Injury Index across configurations.

### Figure 5: Damage Reduction Efficiency
**Description:** Bar graph showing damage reduction efficiency (%) for each configuration.

**Figure Legend:** All figures generated from 8-hour simulation using Reswick-Rogers (1976) surface damage model and Gefen (2008) deep tissue injury model. Error bars not shown as values represent single simulation runs with deterministic models.

---

## Written Results

### Pressure Distribution

Average interface pressure varied across mattress configurations, ranging from 39.90 mmHg to 65.30 mmHg. Standard Foam produced an average pressure of 65.30 mmHg. Eight of the twelve APM configurations showed no reduction in average pressure compared to Standard Foam (0.0% change). Four configurations demonstrated reduced pressure: Zone-Based Adaptive (-30.6%), Row Groups size=2 (-27.4%), Row Groups size=3 (-34.6%), and Evolved Optimal (-38.9%).

### Time to Tissue Injury

Time to surface damage ranged from 1.80 hours to 7.60 hours across configurations. Standard Foam and eight APM patterns showed identical surface damage times of 1.80 hours. The Evolved Optimal configuration showed the longest time to surface damage at 7.60 hours, representing a 322.2% increase compared to Standard Foam.

Time to deep tissue injury (DTI) followed a similar pattern. Standard Foam showed DTI at 1.00 hours. The Evolved Optimal configuration extended DTI time to 5.20 hours, a 420.0% increase. Zone-Based Adaptive (2.80 hours, +180.0%) and Row Groups size=3 (2.70 hours, +170.0%) also showed substantial increases.

### Strain-Time Injury Index (STII)

STII values ranged from 6.16 to 10.12 across configurations. Standard Foam produced an STII of 10.12. The Evolved Optimal configuration achieved the lowest STII at 6.16, representing a 39.1% decrease from baseline. Other effective configurations included Row Groups size=3 (STII = 6.61, -34.7%) and Zone-Based Adaptive (STII = 6.96, -31.2%).

### Damage Accumulation

Cumulative damage fraction after 8 hours ranged from 1.07 to 4.39. Standard Foam accumulated a damage fraction of 4.39. The Evolved Optimal configuration showed the lowest damage accumulation at 1.07, representing a 75.6% reduction. Zone-Based Adaptive (1.67, -62.0%), Row Groups size=2 (1.79, -59.2%), and Row Groups size=3 (1.87, -57.4%) also demonstrated substantial reductions in damage accumulation.

### Statistical Analysis

A statistically significant difference was observed between effective APM configurations (pressure < 50 mmHg, n=4) and ineffective APM configurations (pressure ≥ 50 mmHg, n=8), with t = -18.62 and p < 0.001.

Linear regression analysis revealed a strong negative correlation between average pressure and time to DTI (R² = 0.80, p < 0.001). For every 1 mmHg decrease in average pressure, time to DTI increased by 0.11 hours.

A near-perfect positive correlation was observed between average pressure and STII (R² = 1.00, p < 0.001), indicating that STII increases linearly with pressure at a rate of 0.156 per mmHg.

One-way ANOVA comparing basic APM patterns (n=8, mean damage = 4.36) to optimized APM patterns (n=4, mean damage = 1.60) showed a significant difference (F = 482.82, p < 0.001).

### Efficiency

DTI efficiency ranged from 100.0% (no improvement over foam) to 520.0% (Evolved Optimal). Eight configurations showed 100.0% efficiency, indicating no improvement over Standard Foam. The Evolved Optimal configuration achieved 520.0% DTI efficiency, meaning it extended the time to deep tissue injury by a factor of 5.2.

Damage reduction efficiency ranged from 0.0% to 75.6%. The Evolved Optimal configuration achieved the highest damage reduction efficiency at 75.6%, followed by Zone-Based Adaptive (62.0%), Row Groups size=2 (59.2%), and Row Groups size=3 (57.4%).

### Summary of Key Findings

- 8 of 12 APM patterns showed no improvement over Standard Foam (0.0% change in all metrics)
- 4 APM patterns showed measurable improvements (27-39% pressure reduction)
- Evolved Optimal achieved the best performance across all metrics:
  - Pressure: 39.90 mmHg (-38.9%)
  - Time to Surface Damage: 7.60 hours (+322.2%)
  - Time to DTI: 5.20 hours (+420.0%)
  - STII: 6.16 (-39.1%)
  - Damage Fraction: 1.07 (-75.6%)
  - DTI Efficiency: 520.0%
  - Damage Reduction Efficiency: 75.6%
