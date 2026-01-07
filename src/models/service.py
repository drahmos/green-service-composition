"""
Service and Category Data Models.
"""

from dataclasses import dataclass
from typing import List


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
    cpu_utilization: float = 0.1  # [0, 1]
    memory_usage: float = 128.0   # MB
    data_transfer: float = 1.0    # MB
    peak_power: float = 200.0     # Watts
    idle_power: float = 100.0     # Watts
    
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
