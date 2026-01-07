"""
Baseline Algorithms for Service Composition
"""

import numpy as np
from typing import List, Dict, Callable
from src.models import CompositionProblem, Composition
from src.algorithms import BaseAlgorithm


class QoSGreedy:
    """Greedy algorithm optimizing single QoS objective."""
    
    def __init__(self, objective: str = 'response_time'):
        self.objective = objective
    
    def optimize(self, problem: CompositionProblem) -> List[Composition]:
        """Greedy service selection minimizing specified objective."""
        selected = []
        
        for category in problem.categories:
            best_service = min(
                category.services,
                key=lambda s: getattr(s, self.objective)
            )
            selected.append(best_service)
        
        objectives = self._compute_objectives(selected, problem)
        
        return [Composition(
            composition_id="greedy_qos",
            selected_services=selected,
            objectives=objectives
        )]
    
    def _compute_objectives(
        self,
        services: List,
        problem: CompositionProblem
    ) -> np.ndarray:
        response_time = sum(s.response_time for s in services)
        carbon = sum(s.carbon_emission for s in services)
        cost = sum(s.cost for s in services)
        return np.array([response_time, carbon, cost])


class CarbonGreedy:
    """Greedy algorithm minimizing carbon only."""
    
    def optimize(self, problem: CompositionProblem) -> List[Composition]:
        return QoSGreedy(objective='carbon_emission').optimize(problem)



class RandomSearch:
    """Random search baseline."""
    
    def __init__(self, num_samples: int = 1000, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed
    
    def optimize(self, problem: CompositionProblem) -> List[Composition]:
        np.random.seed(self.seed)
        solutions = []
        
        for _ in range(self.num_samples):
            selected = [
                cat.services[np.random.randint(len(cat.services))]
                for cat in problem.categories
            ]
            
            response_time = sum(s.response_time for s in selected)
            carbon = sum(s.carbon_emission for s in selected)
            cost = sum(s.cost for s in selected)
            
            solutions.append(Composition(
                composition_id=f"random_{_}",
                selected_services=selected,
                objectives=np.array([response_time, carbon, cost])
            ))

        
        # Return Pareto front
        return self._extract_pareto(solutions, problem)
    
    def _extract_pareto(
        self,
        solutions: List[Composition],
        problem: CompositionProblem
    ) -> List[Composition]:
        objectives = np.array([s.objectives for s in solutions])
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
        
        return [solutions[i] for i in pareto_idx]


class GeneticAlgorithmQoS:
    """GA optimizing single QoS objective (response time)."""
    
    def __init__(
        self,
        population_size: int = 100,
        num_generations: int = 200,
        mutation_rate: float = 0.1,
        seed: int = 42
    ):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.seed = seed
    
    def optimize(self, problem: CompositionProblem) -> List[Composition]:
        np.random.seed(self.seed)
        
        # Initialize population
        population = self._init_population(problem)
        
        for gen in range(self.num_generations):
            # Evaluate fitness (response time)
            fitness = self._evaluate_population(population, problem)
            
            # Selection (tournament)
            selected = self._tournament_select(population, fitness, tournament_size=3)
            
            # Crossover
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child = self._crossover(selected[i], selected[i + 1], problem)
                    offspring.append(child)
            
            # Mutation
            mutated = [self._mutate(ind, problem) for ind in offspring]
            
            # Elitism: keep best
            best_idx = np.argmin(fitness)
            population = [population[best_idx]] + mutated[:self.population_size - 1]
        
        # Return best solution
        fitness = self._evaluate_population(population, problem)
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        
        objectives = self._evaluate_encoding(best, problem)
        
        return [Composition(
            composition_id="ga_qos",
            selected_services=[],
            objectives=objectives
        )]
    
    def _init_population(self, problem: CompositionProblem) -> List[np.ndarray]:
        return [
            np.array([
                np.random.randint(0, len(cat.services))
                for cat in problem.categories
            ])
            for _ in range(self.population_size)
        ]
    
    def _evaluate_population(
        self,
        population: List[np.ndarray],
        problem: CompositionProblem
    ) -> np.ndarray:
        return np.array([
            self._evaluate_encoding(ind, problem)[0]  # Response time
            for ind in population
        ])
    
    def _evaluate_encoding(self, encoding: np.ndarray, problem: CompositionProblem) -> np.ndarray:
        response_time = 0.0
        carbon = 0.0
        cost = 0.0
        
        for cat_idx, svc_idx in enumerate(encoding):
            service = problem.categories[cat_idx].services[svc_idx]
            response_time += service.response_time
            carbon += service.carbon_emission
            cost += service.cost
        
        return np.array([response_time, carbon, cost])

    
    def _tournament_select(
        self,
        population: List[np.ndarray],
        fitness: np.ndarray,
        tournament_size: int = 3
    ) -> List[np.ndarray]:
        selected = []
        for _ in range(self.population_size):
            candidates = np.random.choice(len(population), tournament_size, replace=False)
            winner = candidates[np.argmin(fitness[candidates])]
            selected.append(population[winner].copy())
        return selected
    
    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        problem: CompositionProblem
    ) -> np.ndarray:
        n = len(parent1)
        points = sorted(np.random.choice(n, 2, replace=False))
        child = np.concatenate([
            parent2[:points[0]],
            parent1[points[0]:points[1]],
            parent2[points[1]:]
        ])
        return child
    
    def _mutate(self, individual: np.ndarray, problem: CompositionProblem) -> np.ndarray:
        n = len(individual)
        for i in range(n):
            if np.random.random() < self.mutation_rate / n:
                individual[i] = np.random.randint(0, len(problem.categories[i].services))
        return individual.copy()


class MOPSOGreen:
    """Multi-Objective Particle Swarm Optimization for Green Composition."""
    
    def __init__(
        self,
        swarm_size: int = 100,
        num_iterations: int = 200,
        archive_size: int = 100,
        seed: int = 42
    ):
        self.swarm_size = swarm_size
        self.num_iterations = num_iterations
        self.archive_size = archive_size
        self.seed = seed
    
    def optimize(self, problem: CompositionProblem) -> List[Composition]:
        np.random.seed(self.seed)
        
        # Initialize particles
        particles = self._init_swarm(problem)
        velocities = [np.zeros(len(problem.categories)) for _ in range(self.swarm_size)]
        personal_best = [p.copy() for p in particles]
        personal_best_fitness = [self._evaluate(p, problem) for p in particles]
        
        # Initialize archive
        archive = particles[:]
        
        for iteration in range(self.num_iterations):
            # Update velocities
            w = 0.4  # Inertia weight
            c1 = 0.5  # Cognitive coefficient
            c2 = 0.5  # Social coefficient
            
            for i in range(self.swarm_size):
                r1, r2 = np.random.random(2)
                
                velocities[i] = (
                    w * velocities[i] +
                    c1 * r1 * (personal_best[i] - particles[i]) +
                    c2 * r2 * (self._select_leader(archive, particles[i]) - particles[i])
                )
                
                # Limit velocity
                velocities[i] = np.clip(velocities[i], -5, 5)
                
                # Update position
                particles[i] = np.round(particles[i] + velocities[i]).astype(int)
                
                # Bound to valid range
                for j in range(len(particles[i])):
                    particles[i][j] = np.clip(particles[i][j], 0, len(problem.categories[j].services) - 1)
                
                # Evaluate
                fitness = self._evaluate(particles[i], problem)
                
                # Update personal best (Pareto dominance)
                if self._dominates(fitness, personal_best_fitness[i]):
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                elif not self._dominates(personal_best_fitness[i], fitness) and np.random.random() < 0.5:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                
                # Update archive
                self._archive_update(archive, particles[i], fitness, problem)
        
        # Extract solutions from archive
        solutions = []
        for encoding in archive:
            objectives = self._evaluate(encoding, problem)
            solutions.append(Composition(
                composition_id="mopso",
                selected_services=[],
                objectives=objectives
            ))
        
        return self._extract_pareto(solutions, problem)
    
    def _init_swarm(self, problem: CompositionProblem) -> List[np.ndarray]:
        return [
            np.array([
                np.random.randint(0, len(cat.services))
                for cat in problem.categories
            ])
            for _ in range(self.swarm_size)
        ]
    
    def _evaluate(self, encoding: np.ndarray, problem: CompositionProblem) -> np.ndarray:
        response_time = 0.0
        carbon = 0.0
        cost = 0.0
        
        for cat_idx, svc_idx in enumerate(encoding):
            service = problem.categories[cat_idx].services[svc_idx]
            response_time += service.response_time
            carbon += service.carbon_emission
            cost += service.cost
        
        return np.array([response_time, carbon, cost])

    
    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all(a <= b) and np.any(a < b))
    
    def _select_leader(self, archive: List[np.ndarray], particle: np.ndarray) -> np.ndarray:
        if not archive:
            return particle
        return archive[np.random.randint(len(archive))]
    
    def _archive_update(
        self,
        archive: List[np.ndarray],
        particle: np.ndarray,
        fitness: np.ndarray,
        problem: CompositionProblem
    ):
        # Check if particle is dominated by any archive member
        for arch in archive:
            arch_fitness = self._evaluate(arch, problem)
            if self._dominates(arch_fitness, fitness):
                return  # Particle is dominated
        
        # Remove dominated archive members
        new_archive = []
        for arch in archive:
            arch_fitness = self._evaluate(arch, problem)
            if not self._dominates(fitness, arch_fitness):
                new_archive.append(arch)
        
        new_archive.append(particle)
        
        # Archive size management
        if len(new_archive) > self.archive_size:
            # Random removal
            new_archive = new_archive[:self.archive_size]
        
        archive[:] = new_archive
    
    def _extract_pareto(
        self,
        solutions: List[Composition],
        problem: CompositionProblem
    ) -> List[Composition]:
        objectives = np.array([s.objectives for s in solutions])
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
        
        return [solutions[i] for i in pareto_idx]
