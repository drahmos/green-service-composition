"""
Service Data Models for Energy-Aware Composition
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Service:
    """Represents a web service with QoS and energy attributes."""
    service_id: str
    category_id: int
    response_time: float  # ms
    availability: float  # [0, 1]
    throughput: float  # requests/sec
    energy_consumption: float  # Joules
    cost: float  # $ per request
    carbon_factor: float  # kg CO2/kWh
    region: str
    
    @property
    def carbon_emission(self) -> float:
        """Calculate carbon emission for this service."""
        return self.energy_consumption * self.carbon_factor


@dataclass
class ServiceCategory:
    """Represents a category of functionally equivalent services."""
    category_id: int
    name: str
    services: List[Service]


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
        avail = np.prod([s.availability for s in self.selected_services])
        if avail < problem.min_availability:
            return False
        if self.cost > problem.max_cost:
            return False
        return True
