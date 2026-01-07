"""
Composition and Problem Data Models.
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from .service import Service, ServiceCategory


@dataclass
class CompositionProblem:
    """Represents a web service composition problem."""
    problem_id: str
    categories: List[ServiceCategory]
    max_response_time: float = float('inf')
    min_availability: float = 0.0
    max_cost: float = float('inf')
    
    @property
    def num_categories(self) -> int:
        return len(self.categories)
    
    @property
    def total_services(self) -> int:
        return sum(len(cat.services) for cat in self.categories)


@dataclass
class Composition:
    """Represents a candidate service composition."""
    composition_id: str
    selected_services: List[Service]
    objectives: np.ndarray
    
    @property
    def response_time(self) -> float:
        return self.objectives[0]
    
    @property
    def energy(self) -> float:
        return self.objectives[1]
    
    @property
    def cost(self) -> float:
        return self.objectives[2]
    
    @property
    def carbon_emission(self) -> float:
        return sum(s.carbon_emission for s in self.selected_services)
    
    def is_feasible(self, problem: CompositionProblem) -> bool:
        """Check if composition satisfies all constraints."""
        if self.response_time > problem.max_response_time:
            return False
        # Avoid division by zero and check availability
        if not self.selected_services:
            return False
        avail = np.prod([s.availability for s in self.selected_services])
        if avail < problem.min_availability:
            return False
        if self.cost > problem.max_cost:
            return False
        return True
