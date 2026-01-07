"""
Comprehensive Energy Consumption Model for Web Services.
"""

from src.models import Service


class EnergyModel:
    """
    Detailed energy consumption model based on CPU, Memory, Network, and Storage.
    """
    
    def __init__(self, pue: float = 1.5):
        """
        Initialize model with Data Center PUE (Power Usage Effectiveness).
        
        Args:
            pue: Ratio of total facility energy to IT equipment energy.
        """
        self.pue = pue
        
    def calculate_service_energy(self, service: Service) -> float:
        """
        Calculate total energy consumption for a single service invocation.
        
        Formula:
        E = (P_cpu + P_mem + P_net + P_storage) * T * PUE
        
        Where:
        P_cpu = P_idle + (P_peak - P_idle) * U_cpu
        P_mem = Usage_MB * 0.0005 (Estimated 0.5mW per MB)
        P_net = Transfer_MB * 0.1 (Estimated 100mJ per MB)
        """
        # Time in seconds
        t_sec = service.response_time / 1000.0
        
        # CPU Power
        p_cpu = service.idle_power + (service.peak_power - service.idle_power) * service.cpu_utilization
        
        # Memory Power (simplified model)
        p_mem = service.memory_usage * 0.0005
        
        # Network Energy (per invocation, simplified)
        # Note: Network energy is often modeled per MB transferred
        e_net = service.data_transfer * 0.1
        
        # Total Energy in Joules
        e_it = (p_cpu + p_mem) * t_sec + e_net
        
        # Apply PUE
        e_total = e_it * self.pue
        
        return e_total

    def calculate_composition_energy(self, services: list[Service]) -> float:
        """Calculate total energy for a list of services in a composition."""
        return sum(self.calculate_service_energy(s) for s in services)
