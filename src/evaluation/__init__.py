"""
Evaluation Metrics for Multi-Objective Optimization
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats

from src.models import Composition, CompositionProblem


class MetricsCalculator:
    """Calculates evaluation metrics for composition solutions."""
    
    def __init__(self, reference_point: Optional[np.ndarray] = None):
        """
        Initialize calculator.
        
        Args:
            reference_point: Reference point for hypervolume calculation
                           [max_response, max_energy, max_cost]
        """
        self.reference_point = reference_point
    
    def calculate_all(
        self,
        solutions: List[Composition],
        problem: CompositionProblem,
        reference_point: np.ndarray = None
    ) -> Dict:
        """Calculate all evaluation metrics."""
        if not solutions:
            return {
                'hypervolume': 0.0,
                'igd': float('inf'),
                'spread': 1.0,
                'num_solutions': 0,
                'avg_response_time': 0.0,
                'avg_energy': 0.0,
                'avg_cost': 0.0,
                'min_energy': 0.0,
                'energy_range': 0.0
            }
        
        objectives = np.array([s.objectives for s in solutions])
        
        metrics = {
            'num_solutions': len(solutions),
            'avg_response_time': float(np.mean(objectives[:, 0])),
            'avg_energy': float(np.mean(objectives[:, 1])),
            'avg_cost': float(np.mean(objectives[:, 2])),
            'min_energy': float(np.min(objectives[:, 1])),
            'max_energy': float(np.max(objectives[:, 1])),
            'energy_range': float(np.max(objectives[:, 1]) - np.min(objectives[:, 1])),
            'min_response_time': float(np.min(objectives[:, 0])),
            'max_response_time': float(np.max(objectives[:, 0]))
        }
        
        # Multi-objective metrics
        metrics['hypervolume'] = self.hypervolume(solutions, reference_point)
        metrics['spread'] = self.spread(solutions)
        
        # Calculate IGD if true Pareto front available
        if hasattr(problem, 'true_pareto') and problem.true_pareto is not None:
            metrics['igd'] = self.igd(solutions, problem.true_pareto)
        else:
            metrics['igd'] = self.igd_approx(solutions)
        
        return metrics
    
    def hypervolume(
        self,
        solutions: List[Composition],
        reference_point: np.ndarray = None
    ) -> float:
        """
        Calculate hypervolume indicator.
        
        For 2-objective case (Response Time vs Energy):
        HV = sum of rectangles formed by Pareto front and reference point.
        """
        if not solutions:
            return 0.0
        
        objectives = np.array([s.objectives for s in solutions])
        
        if reference_point is None:
            reference_point = np.max(objectives, axis=0) + 0.1 * np.max(objectives, axis=0)
        
        # Sort by first objective (response time)
        sorted_idx = np.argsort(objectives[:, 0])
        sorted_obj = objectives[sorted_idx]
        
        hv = 0.0
        n = len(sorted_obj)
        
        for i in range(n):
            width = reference_point[0] - sorted_obj[i, 0]
            
            if i < n - 1:
                height = sorted_obj[i + 1, 1] - sorted_obj[i, 1]
            else:
                height = reference_point[1] - sorted_obj[i, 1]
            
            if width > 0 and height > 0:
                hv += width * height
        
        # Add last rectangle
        last_height = reference_point[1] - sorted_obj[-1, 1]
        if last_height > 0:
            hv += (reference_point[0] - sorted_obj[-1, 0]) * last_height
        
        return float(hv)
    
    def spread(self, solutions: List[Composition]) -> float:
        """Calculate spread metric (Δ) - distribution of solutions."""
        if len(solutions) < 3:
            return 1.0
        
        objectives = np.array([s.objectives for s in solutions])
        
        # Normalize objectives
        normalized = (objectives - np.min(objectives, axis=0)) / (
            np.ptp(objectives, axis=0) + 1e-10
        )
        
        # Extreme points
        d_f = np.max(np.linalg.norm(normalized - normalized[0], axis=1))
        d_l = np.max(np.linalg.norm(normalized - normalized[-1], axis=1))
        
        # Mean distance between consecutive solutions
        distances = []
        for i in range(1, len(normalized)):
            distances.append(np.linalg.norm(normalized[i] - normalized[i - 1]))
        
        d_bar = np.mean(distances)
        
        if d_bar < 1e-10:
            return 1.0
        
        sigma = np.sum(np.abs(np.array(distances) - d_bar))
        
        delta = (d_f + d_l + sigma) / (d_f + d_l + len(distances) * d_bar)
        
        return float(delta)
    
    def igd_approx(self, solutions: List[Composition]) -> float:
        """
        Approximate Inverted Generational Distance.
        Uses solutions themselves as reference approximation.
        """
        if len(solutions) < 2:
            return 0.0
        
        objectives = np.array([s.objectives for s in solutions])
        
        # For each solution, find min distance to other solutions
        distances = []
        for i in range(len(objectives)):
            min_dist = float('inf')
            for j in range(len(objectives)):
                if i != j:
                    dist = np.linalg.norm(objectives[i] - objectives[j])
                    if dist < min_dist:
                        min_dist = dist
            distances.append(min_dist)
        
        return float(np.mean(distances))
    
    def igd(self, solutions: List[Composition], 
            reference_front: List[Composition]) -> float:
        """Calculate IGD using known reference front."""
        if not reference_front:
            return 0.0
        
        sol_obj = np.array([s.objectives for s in solutions])
        ref_obj = np.array([s.objectives for s in reference_front])
        
        distances = []
        for ref_point in ref_obj:
            min_dist = np.min(np.linalg.norm(sol_obj - ref_point, axis=1))
            distances.append(min_dist)
        
        return float(np.mean(distances))
    
    def energy_savings(
        self,
        solutions: List[Composition],
        baseline_energy: float
    ) -> Dict[str, float]:
        """Calculate energy savings compared to baseline."""
        if not solutions:
            return {'savings': 0.0, 'percentage': 0.0}
        
        min_energy = min(s.energy for s in solutions)
        avg_energy = np.mean([s.energy for s in solutions])
        
        return {
            'min_energy': min_energy,
            'avg_energy': avg_energy,
            'savings': baseline_energy - min_energy,
            'percentage': (baseline_energy - min_energy) / baseline_energy * 100,
            'avg_savings_percentage': (baseline_energy - avg_energy) / baseline_energy * 100
        }


class StatisticalAnalyzer:
    """Statistical analysis for comparing algorithms."""
    
    @staticmethod
    def wilcoxon_test(
        algorithm1_results: List[float],
        algorithm2_results: List[float],
        alpha: float = 0.05
    ) -> Dict:
        """Wilcoxon signed-rank test for pairwise comparison."""
        statistic, p_value = stats.wilcoxon(
            algorithm1_results,
            algorithm2_results,
            alternative='two-sided'
        )
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'better': 'algorithm1' if np.mean(algorithm1_results) < np.mean(algorithm2_results) else 'algorithm2'
        }
    
    @staticmethod
    def friedman_test(
        results: Dict[str, List[float]],
        alpha: float = 0.05
    ) -> Dict:
        """Friedman test for multiple algorithm comparison."""
        algorithms = list(results.keys())
        n = len(results[algorithms[0]])  # Number of runs
        k = len(algorithms)  # Number of algorithms
        
        # Rank each run
        rankings = []
        for run_idx in range(n):
            run_values = {alg: results[alg][run_idx] for alg in algorithms}
            sorted_algs = sorted(run_values, key=run_values.get)
            rankings.append({alg: sorted_algs.index(alg) + 1 for alg in algorithms})
        
        # Average ranks
        avg_ranks = {
            alg: np.mean([r[alg] for r in rankings])
            for alg in algorithms
        }
        
        # Friedman statistic
        chi2 = (12 * n / (k * (k + 1))) * sum(
            (avg_ranks[alg] - (k + 1) / 2) ** 2
            for alg in algorithms
        )
        
        p_value = 1 - stats.chi2.cdf(chi2, k - 1)
        
        return {
            'statistic': float(chi2),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'avg_ranks': avg_ranks,
            'best_algorithm': min(avg_ranks, key=avg_ranks.get)
        }
    
    @staticmethod
    def compute_confidence_interval(
        data: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute confidence interval for a list of values."""
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        
        if se < 1e-10:
            return mean, mean
        
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        
        return mean - h, mean + h
