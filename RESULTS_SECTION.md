# RESULTS

## Tables

### Table 1: Mattress Configuration Comparison
**Legend:** Comparison of 13 mattress configurations over 8-hour simulation. Baseline: Manual Repositioning (2h). APM = Alternating Pressure Mattress. DTI = Deep Tissue Injury. STII = Strain-Time Injury Index.

| Configuration | Pressure (mmHg) | Surface Damage (hours) | DTI (hours) | STII | Damage Fraction |
|---------------|-----------------|------------------------|-------------|------|-----------------|
| Manual Repositioning (2h) | 52.26 | 1.4 | 0.8 | 8.00 | 3.71 |
| Alternating Checkerboard | 69.58 | 1.4 | 0.8 | 10.77 | 5.81 |
| Horizontal Wave | 69.05 | 1.4 | 0.8 | 10.69 | 5.74 |
| Vertical Wave | 69.58 | 1.4 | 0.8 | 10.77 | 5.81 |
| Zone-Based Adaptive | 48.29 | 3.4 | 1.8 | 7.42 | 2.51 |
| Circular Wave | 67.67 | 1.4 | 0.8 | 10.47 | 5.57 |
| Diagonal Wave | 69.07 | 1.4 | 0.8 | 10.69 | 5.74 |
| Row Groups (size=1) | 69.58 | 1.4 | 0.8 | 10.77 | 5.81 |
| Row Groups (size=2) | 50.52 | 2.8 | 1.5 | 7.80 | 2.55 |
| Row Groups (size=3) | 45.47 | 3.3 | 2.4 | 7.04 | 2.59 |
| Multi-Frequency Zones | 69.58 | 1.4 | 0.8 | 10.77 | 5.81 |
| Sequential Rows | 69.58 | 1.4 | 0.8 | 10.77 | 5.81 |
| **Optimal** | **44.08** | **4.3** | **2.7** | **6.81** | **1.83** |

---

### Table 2: Percent Change from Baseline
**Legend:** Percent change compared to Manual Repositioning (2h) baseline. Negative values indicate improvement (reduction in pressure, STII, or damage). Positive values for Surface Damage and DTI indicate increased time before injury (improvement).

| Configuration | Pressure Change (%) | Surface Damage Change (%) | DTI Change (%) | STII Change (%) | Damage Change (%) |
|---------------|---------------------|---------------------------|----------------|-----------------|-------------------|
| Manual Repositioning (2h) | Baseline | Baseline | Baseline | Baseline | Baseline |
| Alternating Checkerboard | +33.2 | 0.0 | 0.0 | +34.6 | +56.6 |
| Horizontal Wave | +32.1 | 0.0 | 0.0 | +33.6 | +54.7 |
| Vertical Wave | +33.2 | 0.0 | 0.0 | +34.6 | +56.6 |
| Zone-Based Adaptive | -7.6 | +142.9 | +125.0 | -7.3 | -32.3 |
| Circular Wave | +29.5 | 0.0 | 0.0 | +30.9 | +50.1 |
| Diagonal Wave | +32.2 | 0.0 | 0.0 | +33.6 | +54.7 |
| Row Groups (size=1) | +33.2 | 0.0 | 0.0 | +34.6 | +56.6 |
| Row Groups (size=2) | -3.3 | +100.0 | +87.5 | -2.5 | -31.3 |
| Row Groups (size=3) | -13.0 | +135.7 | +200.0 | -12.0 | -30.2 |
| Multi-Frequency Zones | +33.2 | 0.0 | 0.0 | +34.6 | +56.6 |
| Sequential Rows | +33.2 | 0.0 | 0.0 | +34.6 | +56.6 |
| **Optimal** | **-15.7** | **+207.1** | **+237.5** | **-14.9** | **-50.7** |

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
| Manual Repositioning (2h) | 100.0 | 0.0 |
| Alternating Checkerboard | 100.0 | -56.6 |
| Horizontal Wave | 100.0 | -54.7 |
| Vertical Wave | 100.0 | -56.6 |
| Zone-Based Adaptive | 225.0 | 32.3 |
| Circular Wave | 100.0 | -50.1 |
| Diagonal Wave | 100.0 | -54.7 |
| Row Groups (size=1) | 100.0 | -56.6 |
| Row Groups (size=2) | 187.5 | 31.3 |
| Row Groups (size=3) | 300.0 | 30.2 |
| Multi-Frequency Zones | 100.0 | -56.6 |
| Sequential Rows | 100.0 | -56.6 |
| **Optimal** | **337.5** | **50.7** |

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

Average interface pressure varied across mattress configurations, ranging from 44.08 mmHg to 69.58 mmHg. Manual Repositioning (2h) produced an average pressure of 52.26 mmHg. Eight of the twelve APM configurations showed higher pressure than Manual Repositioning. Four configurations demonstrated reduced pressure: Zone-Based Adaptive (-7.6%), Row Groups size=2 (-3.3%), Row Groups size=3 (-13.0%), and Optimal (-15.7%).

### Time to Tissue Injury

Time to surface damage ranged from 1.4 hours to 4.3 hours across configurations. Manual Repositioning and eight APM patterns showed identical surface damage times of 1.4 hours. The Optimal configuration showed the longest time to surface damage at 4.3 hours, representing a 207.1% increase compared to Manual Repositioning.

Time to deep tissue injury (DTI) followed a similar pattern. Manual Repositioning showed DTI at 0.8 hours. The Optimal configuration extended DTI time to 2.7 hours, a 237.5% increase. Row Groups size=3 (2.4 hours, +200.0%) and Zone-Based Adaptive (1.8 hours, +125.0%) also showed substantial increases.

### Strain-Time Injury Index (STII)

STII values ranged from 6.81 to 10.77 across configurations. Manual Repositioning produced an STII of 8.00. The Optimal configuration achieved the lowest STII at 6.81, representing a 14.9% decrease from baseline. Other effective configurations included Row Groups size=3 (STII = 7.04, -12.0%) and Zone-Based Adaptive (STII = 7.42, -7.3%).

### Damage Accumulation

Cumulative damage fraction after 8 hours ranged from 1.83 to 5.81. Manual Repositioning accumulated a damage fraction of 3.71. The Optimal configuration showed the lowest damage accumulation at 1.83, representing a 50.7% reduction. Zone-Based Adaptive (2.51, -32.3%), Row Groups size=2 (2.55, -31.3%), and Row Groups size=3 (2.59, -30.2%) also demonstrated substantial reductions in damage accumulation.

### Statistical Analysis

A statistically significant difference was observed between effective APM configurations (pressure < 50 mmHg, n=4) and ineffective APM configurations (pressure ≥ 50 mmHg, n=8), with t = -18.62 and p < 0.001.

Linear regression analysis revealed a strong negative correlation between average pressure and time to DTI (R² = 0.80, p < 0.001). For every 1 mmHg decrease in average pressure, time to DTI increased by 0.11 hours.

A near-perfect positive correlation was observed between average pressure and STII (R² = 1.00, p < 0.001), indicating that STII increases linearly with pressure at a rate of 0.156 per mmHg.

One-way ANOVA comparing basic APM patterns (n=8, mean damage = 5.73) to optimized APM patterns (n=4, mean damage = 2.37) showed a significant difference (F = 482.82, p < 0.001).

### Efficiency

DTI efficiency ranged from 100.0% (no improvement over Manual Repositioning) to 337.5% (Optimal). Eight configurations showed 100.0% efficiency, indicating no improvement over Manual Repositioning. The Optimal configuration achieved 337.5% DTI efficiency, meaning it extended the time to deep tissue injury by a factor of 3.4.

Damage reduction efficiency ranged from -56.6% (worse than baseline) to 50.7%. The Optimal configuration achieved the highest damage reduction efficiency at 50.7%, followed by Zone-Based Adaptive (32.3%), Row Groups size=2 (31.3%), and Row Groups size=3 (30.2%).

### Summary of Key Findings

- 8 of 12 APM patterns showed worse performance than Manual Repositioning (higher pressure, more damage)
- 4 APM patterns showed measurable improvements (3-16% pressure reduction)
- Optimal achieved the best performance across all metrics:
  - Pressure: 44.08 mmHg (-15.7%)
  - Time to Surface Damage: 4.3 hours (+207.1%)
  - Time to DTI: 2.7 hours (+237.5%)
  - STII: 6.81 (-14.9%)
  - Damage Fraction: 1.83 (-50.7%)
  - DTI Efficiency: 337.5%
  - Damage Reduction Efficiency: 50.7%
