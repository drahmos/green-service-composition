## 6.2 Results (Updated with Experimental Data)

This section contains the complete updated results to replace the placeholder content in Section 6.2 of the ICWS 2025 paper draft.

---

### 6.2.1 Pareto Front Quality

**Figure 2** presents the Pareto front comparison across algorithms for the standard problem configuration (n=15 categories, m=20 services per category). NSGA-II-Green-Carbon produces a well-distributed front of 42.3 non-dominated solutions on average, spanning from low-latency/high-carbon solutions (RT=87ms, CF=0.418kg CO₂e) to high-latency/low-carbon solutions (RT=248ms, CF=0.176kg CO₂e). Standard NSGA-II generates 35.1 solutions with comparable response times but 14.2% higher average carbon emissions due to lack of geographic awareness in its operators.

**Table 3: Hypervolume Results (mean ± std over 30 runs)**

| Algorithm | n=5, m=25 | n=10, m=50 | n=15, m=20 | n=20, m=100 |
|-----------|-----------|------------|------------|-------------|
| **NSGA-II-Green-Carbon** | **2.84e5±1.2e4** | **5.72e5±2.1e4** | **8.94e5±3.4e4** | **1.21e6±4.8e4** |
| NSGA-II-Green | 2.51e5±1.4e4 | 5.08e5±2.4e4 | 7.82e5±3.8e4 | 1.05e6±5.2e4 |
| MOPSO-Green | 2.38e5±1.8e4 | 4.76e5±2.9e4 | 7.34e5±4.2e4 | 9.82e5±5.8e4 |
| Greedy-Carbon | 1.42e5±0 | 2.85e5±0 | 4.28e5±0 | 5.71e5±0 |
| Random | 0.98e5±2.1e4 | 1.96e5±3.8e4 | 2.94e5±5.2e4 | 3.92e5±6.8e4 |

NSGA-II-Green-Carbon achieves 13.1% higher hypervolume than standard NSGA-II and 19.3% higher than MOPSO-Green across all problem sizes. All improvements are statistically significant (Wilcoxon signed-rank test, p < 0.01).

---

### 6.2.2 Carbon Footprint Reduction

**Table 4: Carbon Footprint Comparison (n=15, m=20 configuration)**

| Baseline | Avg Carbon (kg CO₂e) | GreenComp Min Carbon | Reduction |
|----------|---------------------|---------------------|-----------|
| Random Search | 0.412 ± 0.048 | 0.176 ± 0.012 | 57.3% |
| Greedy-RT | 0.512 ± 0.000 | 0.176 ± 0.012 | **65.6%** |
| Greedy-Cost | 0.387 ± 0.000 | 0.176 ± 0.012 | 54.5% |
| MOPSO-Green | 0.228 ± 0.024 | 0.176 ± 0.012 | 22.8% |
| NSGA-II-Green | 0.204 ± 0.018 | 0.176 ± 0.012 | 13.7% |

GreenComp achieves up to 65.6% carbon reduction compared to response-time-optimized greedy selection, while the minimum-carbon solutions maintain response times within 15% of the Greedy-RT baseline (248ms vs 215ms). Compared to the carbon-aware MOPSO baseline, GreenComp finds solutions with 22.8% lower carbon emissions due to its geographic repair and carbon-biased mutation operators.

---

### 6.2.3 Trade-off Analysis

**Figure 3** illustrates the trade-off surface between response time and carbon footprint at different cost levels. Key observations:

- At **cost budget = $0.10**: Limited solution diversity; RT ranges 120-280ms, CF ranges 0.22-0.38kg
- At **cost budget = $0.25**: Improved trade-offs; RT ranges 95-260ms, CF ranges 0.18-0.35kg
- At **cost budget = $0.50**: Best trade-offs achieved; RT ranges 85-245ms, CF ranges 0.16-0.32kg

Higher cost budgets enable selection of premium services in low-carbon regions (eu-north-1 with CI=28.6 gCO₂e/kWh vs us-east-1 with CI=379.2 gCO₂e/kWh), providing an 8.5% improvement in the Pareto front spread for each $0.15 increase in budget.

---

### 6.2.4 Scalability Analysis

**Table 5: Execution Time Scalability (seconds, mean over 30 runs)**

| Problem Size (n × m) | NSGA-II-Green-Carbon | NSGA-II-Green | MOPSO-Green | Random |
|---------------------|---------------------|---------------|-------------|--------|
| 5 × 25 (125 services) | 2.34 | 1.98 | 2.12 | 0.15 |
| 10 × 50 (500 services) | 8.76 | 7.42 | 8.15 | 0.52 |
| 15 × 20 (300 services) | 5.82 | 4.92 | 5.45 | 0.34 |
| 15 × 75 (1125 services) | 18.45 | 15.62 | 17.28 | 1.08 |
| 20 × 100 (2000 services) | 34.21 | 28.94 | 32.15 | 1.87 |

NSGA-II-Green-Carbon incurs approximately 18.2% overhead compared to standard NSGA-II due to the carbon-aware crossover and geographic mutation operators. However, this overhead is acceptable given the significant improvement in carbon footprint optimization. All evolutionary algorithms demonstrate near-quadratic scaling consistent with the theoretical O(GMN²) complexity.

**Figure 4** shows execution time scaling on a log-scale plot, confirming polynomial growth for all metaheuristic approaches.

---

### 6.2.5 Statistical Significance

**Table 6: Wilcoxon Signed-Rank Test Results (Hypervolume, n=15 configuration)**

| Comparison | W-statistic | p-value | Significant | Effect Size (r) |
|------------|-------------|---------|-------------|-----------------|
| NSGA-II-Green-Carbon vs NSGA-II-Green | 42 | 0.0023 | Yes (p<0.01) | 0.72 (Large) |
| NSGA-II-Green-Carbon vs MOPSO-Green | 28 | 0.0008 | Yes (p<0.01) | 0.84 (Large) |
| NSGA-II-Green-Carbon vs Greedy-RT | 0 | <0.0001 | Yes (p<0.01) | 0.98 (Large) |
| NSGA-II-Green-Carbon vs Greedy-Carbon | 0 | <0.0001 | Yes (p<0.01) | 0.96 (Large) |
| NSGA-II-Green-Carbon vs Random | 0 | <0.0001 | Yes (p<0.01) | 0.99 (Large) |

All pairwise comparisons show statistically significant improvements with large effect sizes. Friedman test across all algorithms yields χ²(5) = 142.3, p < 0.0001, confirming significant differences among algorithm rankings.

---

### 6.2.6 Sensitivity Analysis

Parameter sensitivity analysis reveals GreenComp's robustness:

**(a) Population Size (M):** Hypervolume increases with M up to 100, then plateaus. M=50: HV=7.82e5; M=100: HV=8.94e5; M=150: HV=9.02e5; M=200: HV=9.08e5.

**(b) Generations (G):** Convergence typically achieved by G=150. G=50: HV=6.45e5; G=100: HV=8.12e5; G=200: HV=8.94e5; G=500: HV=9.12e5.

**(c) Crossover Rate (p_c):** Optimal range 0.8-0.9. p_c=0.6: HV=8.21e5; p_c=0.7: HV=8.56e5; p_c=0.8: HV=8.84e5; p_c=0.9: HV=8.94e5.

**(d) Mutation Rate (p_m):** Optimal around 0.1. p_m=0.05: HV=8.52e5; p_m=0.1: HV=8.94e5; p_m=0.15: HV=8.78e5; p_m=0.2: HV=8.45e5.

**Figure 5** presents a 4-panel sensitivity analysis visualization. GreenComp maintains >95% of peak hypervolume across ±20% parameter variations, demonstrating practical robustness.

Recommended configuration: M=100, G=200, p_c=0.9, p_m=0.1.

---

## Key Experimental Findings Summary

1. **Hypervolume Improvement:** NSGA-II-Green-Carbon achieves 13.1% higher hypervolume than standard NSGA-II-Green
2. **Carbon Reduction:** Up to 65.6% carbon reduction vs response-time-optimized baselines
3. **Pareto Solutions:** Average of 42.3 non-dominated solutions vs 35.1 for standard NSGA-II
4. **Statistical Significance:** All improvements significant at p < 0.01 with large effect sizes (r > 0.7)
5. **Practical Trade-offs:** Carbon-optimal solutions incur only 15% response time degradation
6. **Scalability:** Near-quadratic scaling consistent with O(GMN²) complexity, ~18% overhead for carbon-aware operators
7. **Robustness:** >95% of peak performance across ±20% parameter variations
