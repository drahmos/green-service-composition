"""
Full ICWS 2025 Experiments for GreenComp Paper

Runs complete 30-run experiment suite with statistical tests.
Usage: python experiments/run_full_experiments.py
Expected runtime: 1-2 hours on Google Colab
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats
from src.datasets import generate_dataset
from src.algorithms import NSGAIIGreen, NSGAIIGreenCarbonAware
from src.algorithms.baselines import QoSGreedy, CarbonGreedy, RandomSearch, MOPSOGreen
from src.evaluation import MetricsCalculator


@dataclass
class Config:
    num_runs: int = 30
    num_categories: int = 15
    services_per_category: int = 20
    population_size: int = 100
    num_generations: int = 200
    seed_start: int = 42


def get_algorithms(cfg):
    return {
        "NSGA-II-Green-Carbon": lambda s: NSGAIIGreenCarbonAware(
            population_size=cfg.population_size, num_generations=cfg.num_generations, seed=s),
        "NSGA-II-Green": lambda s: NSGAIIGreen(
            population_size=cfg.population_size, num_generations=cfg.num_generations, seed=s),
        "MOPSO-Green": lambda s: MOPSOGreen(
            swarm_size=cfg.population_size, num_iterations=cfg.num_generations, seed=s),
        "Greedy-RT": lambda s: QoSGreedy("response_time"),
        "Greedy-Carbon": lambda s: CarbonGreedy(),
        "Random": lambda s: RandomSearch(num_samples=1000, seed=s)
    }


def run_experiment(algo, problem, metrics_calc):
    start = time.time()
    solutions = algo.optimize(problem, verbose=False)
    runtime = time.time() - start
    metrics = metrics_calc.calculate_all(solutions, problem)
    metrics["runtime"] = runtime
    metrics["num_pareto"] = len(solutions)
    if solutions:
        obj = np.array([s.objectives for s in solutions])
        metrics["min_carbon"] = float(obj[:, 1].min())
        metrics["min_rt"] = float(obj[:, 0].min())
    return metrics


def analyze(results):
    summary = {}
    for algo, runs in results.items():
        hv = [r["hypervolume"] for r in runs]
        carbon = [r.get("min_carbon", 0) for r in runs]
        runtime = [r["runtime"] for r in runs]
        pareto = [r.get("num_pareto", 0) for r in runs]
        summary[algo] = {
            "hv_mean": np.mean(hv), "hv_std": np.std(hv),
            "carbon_mean": np.mean(carbon), "carbon_std": np.std(carbon),
            "runtime_mean": np.mean(runtime), "pareto_mean": np.mean(pareto)
        }
    return summary


def wilcoxon_tests(results):
    tests = {}
    proposed = "NSGA-II-Green-Carbon"
    proposed_hv = [r["hypervolume"] for r in results[proposed]]
    for baseline in ["NSGA-II-Green", "MOPSO-Green", "Random"]:
        if baseline in results:
            baseline_hv = [r["hypervolume"] for r in results[baseline]]
            try:
                stat, p = stats.wilcoxon(proposed_hv, baseline_hv)
                tests[f"{proposed} vs {baseline}"] = {
                    "p_value": float(p), "significant": p < 0.05
                }
            except Exception as e:
                tests[f"{proposed} vs {baseline}"] = {"error": str(e)}
    return tests


def print_table(summary):
    print("\n" + "=" * 85)
    print("RESULTS (mean +/- std over 30 runs)")
    print("=" * 85)
    print(f"{"Algorithm":<22} {"Hypervolume":<22} {"Min Carbon":<18} {"Time (s)":<10}")
    print("-" * 85)
    for algo, s in summary.items():
        hv = f"{s[\"hv_mean\"]:.2e} +/- {s[\"hv_std\"]:.2e}"
        carbon = f"{s[\"carbon_mean\"]:.4f} +/- {s[\"carbon_std\"]:.4f}"
        print(f"{algo:<22} {hv:<22} {carbon:<18} {s[\"runtime_mean\"]:.2f}")


def save_results(results, summary, tests, outdir):
    Path(outdir / "raw").mkdir(parents=True, exist_ok=True)
    Path(outdir / "processed").mkdir(parents=True, exist_ok=True)
    with open(outdir / "raw" / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(outdir / "processed" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(outdir / "processed" / "statistical_tests.json", "w") as f:
        json.dump(tests, f, indent=2)
    rows = [{"Algorithm": a, **s} for a, s in summary.items()]
    pd.DataFrame(rows).to_csv(outdir / "processed" / "summary.csv", index=False)
    print(f"Results saved to {outdir}/")


def main():
    print("=" * 70)
    print("GreenComp ICWS 2025 - Full Experiment Suite")
    print("=" * 70)
    
    cfg = Config()
    print(f"Config: {cfg.num_runs} runs, {cfg.num_categories}x{cfg.services_per_category} problem")
    print(f"Population: {cfg.population_size}, Generations: {cfg.num_generations}\n")
    
    # Generate problem
    problem = generate_dataset(cfg.num_categories, cfg.services_per_category, seed=42)
    print(f"Generated problem with {problem.total_services} services")
    
    algorithms = get_algorithms(cfg)
    metrics_calc = MetricsCalculator()
    results = {}
    
    for algo_name, algo_factory in algorithms.items():
        print(f"\nRunning {algo_name}...")
        results[algo_name] = []
        for run in range(cfg.num_runs):
            algo = algo_factory(cfg.seed_start + run)
            metrics = run_experiment(algo, problem, metrics_calc)
            results[algo_name].append(metrics)
            if (run + 1) % 10 == 0:
                print(f"  {run+1}/{cfg.num_runs} completed")
    
    # Analyze
    summary = analyze(results)
    tests = wilcoxon_tests(results)
    
    # Print
    print_table(summary)
    print("\nStatistical Tests (Wilcoxon, alpha=0.05):")
    for comp, res in tests.items():
        if "error" not in res:
            sig = "SIGNIFICANT" if res["significant"] else "not significant"
            print(f"  {comp}: p={res[\"p_value\"]:.4f} ({sig})")
    
    # Carbon savings
    proposed = summary["NSGA-II-Green-Carbon"]["carbon_mean"]
    greedy = summary["Greedy-RT"]["carbon_mean"]
    if greedy > 0:
        print(f"\nCarbon reduction vs Greedy-RT: {(greedy-proposed)/greedy*100:.1f}%")
    
    # Save
    save_results(results, summary, tests, Path("results"))
    print("\nDONE\!")


if __name__ == "__main__":
    main()
