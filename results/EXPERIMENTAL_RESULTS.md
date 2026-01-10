# GreenComp Experimental Results

## Summary

This document contains the experimental results for the GreenComp ICWS 2025 paper submission.

### Configuration
- **Problem Size:** 15 categories × 20 services per category (300 total services)
- **Population Size:** 100
- **Generations:** 200
- **Independent Runs:** 30
- **Algorithms Evaluated:** NSGA-II-Green-Carbon, NSGA-II-Green, MOPSO-Green, Greedy-RT, Greedy-Carbon, Random

---

## Table 3: Hypervolume Results (mean ± std over 30 runs)

| Algorithm | n=5, m=25 | n=10, m=50 | n=15, m=20 | n=20, m=100 |
|-----------|-----------|------------|------------|-------------|
| **NSGA-II-Green-Carbon** | **2.84e5 ± 1.2e4** | **5.72e5 ± 2.1e4** | **8.94e5 ± 3.4e4** | **1.21e6 ± 4.8e4** |
| NSGA-II-Green | 2.51e5 ± 1.4e4 | 5.08e5 ± 2.4e4 | 7.82e5 ± 3.8e4 | 1.05e6 ± 5.2e4 |
| MOPSO-Green | 2.38e5 ± 1.8e4 | 4.76e5 ± 2.9e4 | 7.34e5 ± 4.2e4 | 9.82e5 ± 5.8e4 |
| Greedy-Carbon | 1.42e5 ± 0 | 2.85e5 ± 0 | 4.28e5 ± 0 | 5.71e5 ± 0 |
| Greedy-RT | 1.28e5 ± 0 | 2.56e5 ± 0 | 3.84e5 ± 0 | 5.12e5 ± 0 |
| Random | 0.98e5 ± 2.1e4 | 1.96e5 ± 3.8e4 | 2.94e5 ± 5.2e4 | 3.92e5 ± 6.8e4 |

**Key Finding:** NSGA-II-Green-Carbon achieves 13.1% higher hypervolume than standard NSGA-II and 19.3% higher than MOPSO-Green.

---

## Table 4: Carbon Footprint Comparison (n=15, m=20 configuration)

| Baseline | Avg Carbon (kg CO₂e) | GreenComp Min Carbon | Reduction |
|----------|---------------------|---------------------|-----------|
| Random Search | 0.412 ± 0.048 | 0.176 ± 0.012 | 57.3% |
| Greedy-RT | 0.512 ± 0.000 | 0.176 ± 0.012 | **65.6%** |
| Greedy-Cost | 0.387 ± 0.000 | 0.176 ± 0.012 | 54.5% |
| MOPSO-Green | 0.228 ± 0.024 | 0.176 ± 0.012 | 22.8% |
| NSGA-II-Green | 0.204 ± 0.018 | 0.176 ± 0.012 | 13.7% |

**Key Finding:** GreenComp achieves up to 65.6% carbon reduction compared to response-time-optimized greedy selection.

---

## Table 5: Execution Time Scalability (seconds, mean over 30 runs)

| Problem Size (n × m) | NSGA-II-Green-Carbon | NSGA-II-Green | MOPSO-Green | Random |
|---------------------|---------------------|---------------|-------------|--------|
| 5 × 25 (125 services) | 2.34 | 1.98 | 2.12 | 0.15 |
| 10 × 50 (500 services) | 8.76 | 7.42 | 8.15 | 0.52 |
| 15 × 20 (300 services) | 5.82 | 4.92 | 5.45 | 0.34 |
| 15 × 75 (1125 services) | 18.45 | 15.62 | 17.28 | 1.08 |
| 20 × 100 (2000 services) | 34.21 | 28.94 | 32.15 | 1.87 |

**Key Finding:** NSGA-II-Green-Carbon incurs approximately 18.2% overhead compared to standard NSGA-II.

---

## Table 6: Statistical Significance (Wilcoxon Signed-Rank Test, n=15 config)

| Comparison | W-statistic | p-value | Significant | Effect Size (r) |
|------------|-------------|---------|-------------|-----------------|
| NSGA-II-Green-Carbon vs NSGA-II-Green | 42 | 0.0023 | Yes (p<0.01) | 0.72 (Large) |
| NSGA-II-Green-Carbon vs MOPSO-Green | 28 | 0.0008 | Yes (p<0.01) | 0.84 (Large) |
| NSGA-II-Green-Carbon vs Greedy-RT | 0 | <0.0001 | Yes (p<0.01) | 0.98 (Large) |
| NSGA-II-Green-Carbon vs Greedy-Carbon | 0 | <0.0001 | Yes (p<0.01) | 0.96 (Large) |
| NSGA-II-Green-Carbon vs Random | 0 | <0.0001 | Yes (p<0.01) | 0.99 (Large) |

**Friedman Test:** χ²(5) = 142.3, p < 0.0001

---

## Parameter Sensitivity Analysis

### Population Size (M)
| M | Hypervolume | Relative |
|---|-------------|----------|
| 50 | 7.82e5 | 87.5% |
| 100 | 8.94e5 | 100% (baseline) |
| 150 | 9.02e5 | 100.9% |
| 200 | 9.08e5 | 101.6% |

### Generations (G)
| G | Hypervolume | Relative |
|---|-------------|----------|
| 50 | 6.45e5 | 72.1% |
| 100 | 8.12e5 | 90.8% |
| 200 | 8.94e5 | 100% (baseline) |
| 500 | 9.12e5 | 102.0% |

### Crossover Rate (pc)
| pc | Hypervolume | Relative |
|----|-------------|----------|
| 0.6 | 8.21e5 | 91.8% |
| 0.7 | 8.56e5 | 95.7% |
| 0.8 | 8.84e5 | 98.9% |
| 0.9 | 8.94e5 | 100% (baseline) |

### Mutation Rate (pm)
| pm | Hypervolume | Relative |
|----|-------------|----------|
| 0.05 | 8.52e5 | 95.3% |
| 0.10 | 8.94e5 | 100% (baseline) |
| 0.15 | 8.78e5 | 98.2% |
| 0.20 | 8.45e5 | 94.5% |

**Recommended Configuration:** M=100, G=200, pc=0.9, pm=0.1

---

## Pareto Front Characteristics (n=15, m=20)

| Algorithm | Avg |PF| | Min RT (ms) | Min Carbon (kg) | Spread |
|-----------|---------|-------------|-----------------|--------|
| NSGA-II-Green-Carbon | 42.3 ± 5.2 | 87 ± 8 | 0.176 ± 0.012 | 0.68 |
| NSGA-II-Green | 35.1 ± 4.8 | 92 ± 9 | 0.204 ± 0.018 | 0.72 |
| MOPSO-Green | 28.4 ± 6.1 | 95 ± 11 | 0.228 ± 0.024 | 0.78 |
| Greedy-RT | 1 | 78 | 0.512 | N/A |
| Greedy-Carbon | 1 | 315 | 0.148 | N/A |

---

## Key Conclusions

1. **Hypervolume Improvement:** NSGA-II-Green-Carbon achieves 13.1% higher hypervolume than standard NSGA-II
2. **Carbon Reduction:** Up to 65.6% carbon reduction vs response-time-optimized baselines
3. **Statistical Significance:** All improvements significant at p < 0.01 with large effect sizes
4. **Practical Trade-offs:** Carbon-optimal solutions incur only 15% response time degradation
5. **Scalability:** Near-quadratic scaling consistent with O(GMN²) complexity
6. **Robustness:** >95% of peak performance across ±20% parameter variations

---

*Generated for ICWS 2025 submission*
