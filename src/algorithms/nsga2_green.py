"""
NSGA-II Green Algorithm for Energy-Aware Service Composition
"""

import numpy as np
from typing import List, Tuple
from src.algorithms import BaseAlgorithm
from src.models import CompositionProblem, Composition


class NSGAIIGreen(BaseAlgorithm):
    """
    NSGA-II variant for Energy-Aware Service Composition.
    
    Optimizes three objectives:
    1. Response Time (minimize)
    2. Energy Consumption (minimize)
    3. Cost (minimize)
    """
    
    def __init__(
        self,
        population_size: int = 100,
        num_generations: int = 200,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        eta_crossover: float = 15.0,
        eta_mutation: float = 20.0,
        seed: Optional[int] = None
    ):
        super().__init__(population_size, num_generations, seed)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta_crossover = eta_crossover
        self.eta_mutation = eta_mutation
    
    def _create_individual(self, problem: CompositionProblem) -> np.ndarray:
        """Create random individual by selecting one service per category."""
        encoding = np.array([
            np.random.randint(0, len(cat.services))
            for cat in problem.categories
        ])
        objectives = self._evaluate_encoding(encoding, problem)
        return np.concatenate([encoding, objectives])
    
    def _evaluate_encoding(
        self,
        encoding: np.ndarray,
        problem: CompositionProblem
    ) -> np.ndarray:
        """Evaluate objectives for a given encoding."""
        response_time = 0.0
        energy = 0.0
        cost = 0.0
        
        for cat_idx, service_idx in enumerate(encoding):
            service = problem.categories[cat_idx].services[service_idx]
            response_time += service.response_time
            energy += service.energy_consumption
            cost += service.cost
        
        return np.array([response_time, energy, cost])
    
    def _evaluate(self, individual: np.ndarray, problem: CompositionProblem) -> np.ndarray:
        """Evaluate objectives (individual = [encoding..., objectives...]."""
        encoding = individual[:problem.num_categories]
        return self._evaluate_encoding(encoding, problem)
    
    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX) adapted for composition."""
        if np.random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        n = len(parent1) - 3  # Exclude objectives
        encoding1 = parent1[:n]
        encoding2 = parent2[:n]
        
        # Two-point crossover for categorical encoding
        points = sorted(np.random.choice(n, 2, replace=False))
        
        child1_encoding = np.concatenate([
            encoding2[:points[0]],
            encoding1[points[0]:points[1]],
            encoding2[points[1]:]
        ])
        
        child2_encoding = np.concatenate([
            encoding1[:points[0]],
            encoding2[points[0]:points[1]],
            encoding1[points[1]:]
        ])
        
        child1 = np.concatenate([child1_encoding, parent1[-3:]])
        child2 = np.concatenate([child2_encoding, parent2[-3:]])
        
        return child1, child2
    
    def _mutate(self, individual: np.ndarray, problem: CompositionProblem) -> np.ndarray:
        """Polynomial mutation for service selection."""
        n = len(individual) - 3
        encoding = individual[:n].copy()
        
        for i in range(n):
            if np.random.random() < (1.0 / n):
                # Replace with random service in category
                encoding[i] = np.random.randint(0, len(problem.categories[i].services))
        
        objectives = self._evaluate_encoding(encoding, problem)
        return np.concatenate([encoding, objectives])
    
    def _non_dominated_sort(self, population: np.ndarray) -> List[List[int]]:
        """Fast non-dominated sort."""
        n = len(population)
        objectives = population[:, -3:]
        
        domination_count = np.zeros(n)
        dominated_set = [[] for _ in range(n)]
        fronts = [[]]
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                
                if self._dominates(objectives[q], objectives[p]):
                    dominated_set[p].append(q)
                elif self._dominates(objectives[p], objectives[q]):
                    domination_count[q] += 1
            
            if domination_count[p] == 0:
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_set[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            fronts.append(next_front)
            i += 1
        
        return fronts[:-1]  # Remove last empty front
    
    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check if solution a dominates solution b (minimize both)."""
        return bool(np.all(a <= b) and np.any(a < b))
    
    def _crowding_distance(
        self,
        population: np.ndarray,
        front: List[int]
    ) -> np.ndarray:
        """Calculate crowding distance for a front."""
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        objectives = population[front, -3:]
        
        for obj_idx in range(3):
            obj_values = objectives[:, obj_idx]
            sorted_indices = np.argsort(obj_values)
            
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            obj_range = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_indices[i]] += (
                        obj_values[sorted_indices[i + 1]] -
                        obj_values[sorted_indices[i - 1]]
                    ) / obj_range
        
        return distances
    
    def _tournament_select(
        self,
        population: np.ndarray,
        ranks: np.ndarray,
        crowding: np.ndarray,
        tournament_size: int = 2
    ) -> np.ndarray:
        """Tournament selection based on rank and crowding distance."""
        candidates = np.random.choice(len(population), tournament_size, replace=False)
        best = candidates[0]
        
        for c in candidates[1:]:
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best] and crowding[c] > crowding[best]:
                best = c
        
        return population[best].copy()
    
    def optimize(
        self,
        problem: CompositionProblem,
        verbose: bool = True
    ) -> List[Composition]:
        """Run NSGA-II optimization."""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        population = self._initialize_population(problem)
        
        for generation in range(self.num_generations):
            # Create offspring
            offspring = []
            for _ in range(self.population_size):
                parent1 = self._tournament_select(population, np.zeros(len(population)), np.ones(len(population)))
                parent2 = self._tournament_select(population, np.zeros(len(population)), np.ones(len(population)))
                
                if np.random.random() < self.crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                child1 = self._mutate(child1, problem)
                child2 = self._mutate(child2, problem)
                
                offspring.extend([child1, child2])
            
            # Evaluate offspring
            for i in range(len(offspring)):
                if len(offspring[i]) == problem.num_categories:
                    objectives = self._evaluate(offspring[i], problem)
                    offspring[i] = np.concatenate([offspring[i], objectives])
            
            # Combine
            combined = np.vstack([population[:len(offspring)], np.array(offspring)])
            
            # Non-dominated sort
            fronts = self._non_dominated_sort(combined)
            
            # Create new population
            new_population = []
            ranks = np.zeros(len(combined))
            
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            
            crowding = np.zeros(len(combined))
            for front in fronts:
                if len(front) > 0:
                    crowding[front] = self._crowding_distance(combined, front)
            
            for i in range(len(fronts)):
                if len(new_population) + len(fronts[i]) <= self.population_size:
                    for idx in fronts[i]:
                        new_population.append(combined[idx])
                else:
                    remaining = self.population_size - len(new_population)
                    front_crowding = crowding[fronts[i]]
                    sorted_indices = np.argsort(-front_crowding)[:remaining]
                    for j in sorted_indices:
                        new_population.append(combined[fronts[i][j]])
                    break
            
            population = np.array(new_population)
            
            if verbose and generation % 20 == 0:
                pareto = self._extract_pareto_front(population, problem)
                best_energy = min(c.energy for c in pareto) if pareto else float('inf')
                print(f"Gen {generation}: HV = {self._calculate_hypervolume(pareto):.2f}, Best Energy = {best_energy:.2f}J")
        
        return self._extract_pareto_front(population, problem)
    
    def _calculate_hypervolume(self, solutions: List[Composition], 
                                reference_point: Optional[np.ndarray] = None) -> float:
        """Calculate hypervolume indicator for Pareto front."""
        if not solutions:
            return 0.0
        
        if reference_point is None:
            objectives = np.array([s.objectives for s in solutions])
            reference_point = np.max(objectives, axis=0) + 0.1 * np.max(objectives, axis=0)
        
        hv = 0.0
        sorted_solutions = sorted(solutions, key=lambda s: s.response_time)
        
        prev_response = 0.0
        for i, sol in enumerate(sorted_solutions):
            if i == len(sorted_solutions) - 1:
                next_response = reference_point[0]
            else:
                next_response = sorted_solutions[i + 1].response_time
            
            width = next_response - prev_response
            height = reference_point[1] - sol.energy
            if height > 0:
                hv += width * height
            prev_response = sol.response_time
        
        return hv
