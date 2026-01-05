"""
Base Algorithm Class for Service Composition Optimization
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from src.models import CompositionProblem, Composition


class BaseAlgorithm(ABC):
    """Abstract base class for composition optimization algorithms."""
    
    def __init__(
        self,
        population_size: int = 100,
        num_generations: int = 200,
        seed: Optional[int] = None
    ):
        self.population_size = population_size
        self.num_generations = num_generations
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
    
    @abstractmethod
    def _evaluate(self, individual: np.ndarray, problem: CompositionProblem) -> np.ndarray:
        """Evaluate objectives for an individual."""
        pass
    
    @abstractmethod
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """Perform crossover operation."""
        pass
    
    @abstractmethod
    def _mutate(self, individual: np.ndarray, problem: CompositionProblem) -> np.ndarray:
        """Perform mutation operation."""
        pass
    
    @abstractmethod
    def _create_individual(self, problem: CompositionProblem) -> np.ndarray:
        """Create a new random individual."""
        pass
    
    def optimize(
        self,
        problem: CompositionProblem,
        verbose: bool = True
    ) -> List[Composition]:
        """Run the optimization algorithm."""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        population = self._initialize_population(problem)
        history = []
        
        for generation in range(self.num_generations):
            population = self._evolve(population, problem)
            
            if verbose and generation % 20 == 0:
                pareto = self._extract_pareto_front(population, problem)
                best_energy = min(c.energy for c in pareto)
                print(f"Gen {generation}: Best Energy = {best_energy:.2f}J")
        
        final_pareto = self._extract_pareto_front(population, problem)
        return final_pareto
    
    def _initialize_population(self, problem: CompositionProblem) -> np.ndarray:
        """Initialize random population."""
        return np.array([
            self._create_individual(problem)
            for _ in range(self.population_size)
        ])
    
    def _evolve(
        self,
        population: np.ndarray,
        problem: CompositionProblem
    ) -> np.ndarray:
        """Perform one generation of evolution."""
        offspring = []
        
        for _ in range(self.population_size):
            # Tournament selection
            i1, i2 = np.random.choice(len(population), 2, replace=False)
            parent1 = population[i1] if np.random.random() < 0.5 else population[i2]
            parent2 = population[i1] if parent1 is population[i2] else population[np.random.randint(len(population))]
            
            # Crossover
            if np.random.random() < 0.9:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate(child1, problem)
            child2 = self._mutate(child2, problem)
            
            offspring.extend([child1, child2])
        
        # Evaluate offspring
        evaluated = [self._evaluate(ind, problem) for ind in offspring[:self.population_size]]
        
        # Combine and select
        combined = list(population) + [np.concatenate([ind, obj]) for ind, obj in zip(offspring[:self.population_size], evaluated)]
        
        # Simple survival selection
        fitness = np.array([ind[-3:] for ind in combined])
        indices = np.argsort(fitness[:, 1])[:self.population_size]  # Sort by energy
        
        return np.array([combined[i] for i in indices])
    
    def _extract_pareto_front(
        self,
        population: np.ndarray,
        problem: CompositionProblem
    ) -> List[Composition]:
        """Extract Pareto-optimal solutions from population."""
        objectives = population[:, -3:]
        
        pareto_indices = []
        for i in range(len(objectives)):
            is_dominated = False
            for j in range(len(objectives)):
                if i != j:
                    # Check if i is dominated by j (minimize all)
                    if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_indices.append(i)
        
        pareto_solutions = population[pareto_indices]
        
        return [
            Composition(
                composition_id=f"sol_{i}",
                selected_services=[],  # Services reconstructed from encoding
                objectives=sol[-3:]
            )
            for i, sol in enumerate(pareto_solutions)
        ]
