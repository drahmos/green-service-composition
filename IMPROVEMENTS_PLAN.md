# ICWS 2025 Improvements Plan: GreenComp Framework

This document outlines the roadmap to strengthen the novelty and technical depth of the Green Service Composition research for ICWS 2025 submission.

## 🎯 Primary Goal
Transform the current incremental "energy-aware" approach into a **state-of-the-art Carbon-Aware Pareto Optimization framework** (GreenComp).

---

## 🚀 Phase 1: Core Framework & Model Enhancements

### 1.1 Comprehensive Energy Model
- **Current**: $E = \alpha \times \text{response\_time}$ (Too simplistic)
- **New**: Implement a multi-tier energy model:
  - $E_{total} = E_{cpu} + E_{mem} + E_{net} + E_{storage} + E_{idle}$
  - Calibrate parameters using real cloud instance profiles (e.g., AWS EC2, Azure VMs).

### 1.2 Geographic Carbon Intensity (Novelty Driver)
- **Current**: Static regional carbon factors.
- **New**: Integrate **real-time/historical carbon intensity APIs** (ElectricityMap/WattTime).
- **Temporal Awareness**: Model energy impact based on the time of execution (e.g., solar availability).

---

## 🧠 Phase 2: Algorithm Enhancements

### 2.1 Carbon-Aware NSGA-II (NSGA-II-Green v2)
- **Multi-Objective Expansion**: Jointly optimize Response Time, Cost, and **Actual Carbon Footprint** (kg CO2).
- **Geographic Load Balancing**: Prioritize regions with high renewable energy mix during optimization.
- **Constraint Handling**: Add carbon-cap constraints per composition.

### 2.2 Dynamic Re-composition
- Implement logic to handle carbon intensity fluctuations in real-time.
- Adaptive re-optimization when the grid carbon mix changes.

---

## 📊 Phase 3: Validation & Benchmarking

### 3.1 Real Cloud Data Validation
- Replace synthetic parameters with data from **Cloud Energy Benchmarks** (e.g., Cloud Carbon Footprint project).
- Use real cloud pricing and performance traces (Alibaba/Google cluster traces).

### 3.2 Rigorous Statistical Analysis
- Conduct Wilcoxon and Friedman tests across 30+ runs.
- Generate ICWS-standard figures (Pareto surfaces, Radar charts).

---

## 🎓 Phase 4: Theoretical Contributions
- **NP-Hardness Proof**: Formal proof of the energy-aware service composition problem.
- **Complexity Analysis**: Algorithm convergence and approximation bounds.

---

## ✅ Implementation Checklist

- [ ] Implement `src/models/energy_model.py` with multi-tier calculation.
- [ ] Integrate real-time carbon intensity simulation in `src/datasets/generator.py`.
- [ ] Refactor `NSGAIIGreen` to handle the expanded objective space.
- [ ] Implement `results/analysis.py` for automated ICWS-ready table generation.
- [ ] Update `README.md` and `TECHNICAL_SPECIFICATION.md` to reflect the GreenComp branding.

---
*Created by Sisyphus for ICWS 2025 Submission Preparation.*
