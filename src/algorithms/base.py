"""
Base class for optimization algorithms in Green Service Composition.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional
from src.models import CompositionProblem, Composition


class BaseAlgorithm(ABC):
    """Abstract base class for service composition algorithms."""
    
    def __init__(
        self,
        population_size: int = 100,
        num_generations: int = 200,
        seed: Optional[int] = None
    ):
        self.population_size = population_size
        self.num_generations = num_generations
        self.seed = seed
        
    @abstractmethod
    def optimize(
        self,
        problem: CompositionProblem,
        verbose: bool = True
    ) -> List[Composition]:
        """Run the optimization algorithm."""
        pass
        
    def _initialize_population(self, problem: CompositionProblem) -> np.ndarray:
        """Create initial random population."""
        population = []
        for _ in range(self.population_size):
            population.append(self._create_individual(problem))
        return np.array(population)
        
    @abstractmethod
    def _create_individual(self, problem: CompositionProblem) -> np.ndarray:
        """Create a single random individual."""
        pass
        
    def _extract_pareto_front(
        self,
        population: np.ndarray,
        problem: CompositionProblem
    ) -> List[Composition]:
        """Extract non-dominated solutions from population."""
        objectives = population[:, -3:]
        pareto_idx = []
        
        for i in range(len(objectives)):
            is_dominated = False
            for j in range(len(objectives)):
                if i != j:
                    if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_idx.append(i)
        
        results = []
        for i in pareto_idx:
            encoding = population[i, :problem.num_categories].astype(int)
            selected_services = [
                problem.categories[c_idx].services[s_idx]
                for c_idx, s_idx in enumerate(encoding)
            ]
            results.append(Composition(
                composition_id=f"sol_{i}",
                selected_services=selected_services,
                objectives=objectives[i]
            ))
            
        return results
