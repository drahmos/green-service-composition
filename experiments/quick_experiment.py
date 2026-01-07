"""
Colab-Ready Quick Experiment
Run this script directly in Google Colab for paper-ready results.
"""

import numpy as np
import time
import sys
import os

# Setup path
sys.path.insert(0, os.path.abspath('.'))

from src.datasets import generate_dataset
from src.algorithms import NSGAIIGreen
from src.algorithms.baselines import QoSGreedy, CarbonGreedy, RandomSearch, MOPSOGreen

from src.evaluation import MetricsCalculator
from src.visualization import PaperVisualizer, ResultsFormatter


def run_quick_experiment(
    num_categories: int = 15,
    services_per_category: int = 20,
    num_runs: int = 10,
    generations: int = 100
):
    """
    Quick experiment suitable for Colab execution.
    
    Args:
        num_categories: Number of service categories
        services_per_category: Services per category
        num_runs: Number of independent runs
        generations: NSGA-II generations
    """
    print("=" * 60)
    print("Green Service Composition - Quick Experiment")
    print("=" * 60)
    print(f"Problem: {num_categories} categories × {services_per_category} services")
    print(f"Runs: {num_runs}, Generations: {generations}")
    print()
    
    # Generate problem
    print("Generating problem...")
    problem = generate_dataset(
        num_categories=num_categories,
        services_per_category=services_per_category,
        seed=42
    )
    
    # Algorithms
    algorithms = {
        'NSGA-II-Green': lambda s: NSGAIIGreen(
            population_size=80, num_generations=generations, seed=s
        ),
        'MOPSO-Green': lambda s: MOPSOGreen(
            swarm_size=80, num_iterations=generations, archive_size=80, seed=s
        ),
        'QoS-Greedy': lambda s: QoSGreedy(objective='response_time'),
        'Carbon-Greedy': lambda s: CarbonGreedy(),
        'Random': lambda s: RandomSearch(num_samples=500, seed=s)

    }
    
    results = {}
    metrics_calc = MetricsCalculator()
    
    # Run experiments
    print("\nRunning experiments...")
    for algo_name, algo_factory in algorithms.items():
        print(f"  {algo_name}...", end=" ", flush=True)
        
        algo_results = []
        start = time.time()
        
        for run in range(num_runs):
            algo = algo_factory(42 + run)
            solutions = algo.optimize(problem, verbose=False)
            metrics = metrics_calc.calculate_all(solutions, problem)
            metrics['runtime'] = 0  # Will be updated
            algo_results.append(metrics)
        
        runtime = time.time() - start
        
        # Aggregate
        results[algo_name] = {
            'hypervolume_mean': np.mean([r['hypervolume'] for r in algo_results]),
            'hypervolume_std': np.std([r['hypervolume'] for r in algo_results]),
            'carbon_mean': np.mean([r['avg_carbon'] for r in algo_results]),
            'carbon_std': np.std([r['avg_carbon'] for r in algo_results]),
            'num_solutions': np.mean([r['num_solutions'] for r in algo_results]),
            'total_runtime': runtime
        }
        
        print(f"HV={results[algo_name]['hypervolume_mean']:.2e}, "
              f"Carbon={results[algo_name]['carbon_mean']:.4f}kg")

    
    # Print results table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n{:<20} {:>12} {:>12} {:>10}".format(
        "Algorithm", "Hypervolume", "Carbon (kg)", "|PF|"
    ))
    print("-" * 58)
    
    for algo_name in sorted(results.keys(),
                           key=lambda x: -results[x]['hypervolume_mean']):
        r = results[algo_name]
        print("{:<20} {:>12.2e} {:>12.4f} {:>10.1f}".format(
            algo_name, r['hypervolume_mean'], r['carbon_mean'], r['num_solutions']
        ))
    
    # Carbon savings
    qos_carbon = results['QoS-Greedy']['carbon_mean']
    nsga_carbon = results['NSGA-II-Green']['carbon_mean']
    savings = (qos_carbon - nsga_carbon) / qos_carbon * 100
    
    print("\n" + "-" * 58)
    print(f"Carbon Savings (NSGA-II-Green vs QoS-Greedy): {savings:.1f}%")

    
    # Generate visualization
    print("\nGenerating visualization...")
    viz = PaperVisualizer("results/figures")
    
    # Get Pareto fronts
    pareto_dict = {}
    for algo_name, algo_factory in algorithms.items():
        algo = algo_factory(42)
        solutions = algo.optimize(problem, verbose=False)
        pareto_dict[algo_name] = solutions
    
    viz.plot_pareto_front(
        pareto_dict,
        title="Pareto Front: Energy vs Response Time",
        filename="quick_experiment_pareto.png"
    )
    
    # Generate LaTeX table
    formatter = ResultsFormatter()
    latex_table = formatter.format_latex_table(
        results,
        columns=['hypervolume_mean', 'num_solutions', 'carbon_mean'],
        caption="Quick experiment results (mean over 10 runs).",
        label="tab:quick_results"
    )

    
    print("\n" + "=" * 60)
    print("LATEX TABLE FOR PAPER")
    print("=" * 60)
    print(latex_table)
    
    # Save to file
    with open("results/processed/quick_experiment_results.tex", 'w') as f:
        f.write(latex_table)
    
    print("\n✓ Results saved to results/processed/")
    print("✓ Figure saved to results/figures/quick_experiment_pareto.png")
    
    return results


# Colab execution
if __name__ == "__main__":
    print("Running Quick Experiment...")
    print("Expected runtime: 5-10 minutes on Colab CPU\n")
    
    results = run_quick_experiment(
        num_categories=12,
        services_per_category=15,
        num_runs=10,
        generations=100
    )
