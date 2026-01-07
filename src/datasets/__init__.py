"""
Synthetic Dataset Generator for Service Composition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


from src.models import Service, ServiceCategory, CompositionProblem, EnergyModel


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    num_categories: int = 10
    services_per_category: int = 20
    seed: int = 42
    response_time_range: Tuple[float, float] = (50.0, 1000.0)
    availability_range: Tuple[float, float] = (0.9, 0.999)
    throughput_range: Tuple[float, float] = (10.0, 1000.0)
    energy_correlation: float = 0.7  # Correlation between response time and energy
    carbon_factors: Dict[str, float] = field(default_factory=dict)


class DatasetGenerator:
    """Generates synthetic service composition datasets."""
    
    DEFAULT_CARBON_FACTORS = {
        'us-east': 0.42,
        'us-west': 0.35,
        'eu-west': 0.28,
        'eu-north': 0.18,
        'asia-east': 0.55,
        'asia-south': 0.68,
        'sa-east': 0.08,
        'au-east': 0.45
    }
    
    REGIONS = list(DEFAULT_CARBON_FACTORS.keys())
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        if self.config.carbon_factors is None:
            self.config.carbon_factors = self.DEFAULT_CARBON_FACTORS
        self.energy_model = EnergyModel()
    
    def generate(self, problem_id: str = "default") -> CompositionProblem:
        """Generate a complete composition problem."""
        np.random.seed(self.config.seed)
        
        categories = []
        
        for cat_id in range(self.config.num_categories):
            services = self._generate_category_services(cat_id)
            category = ServiceCategory(
                category_id=cat_id,
                name=f"Category_{cat_id}",
                services=services
            )
            categories.append(category)
        
        # Estimate max response time from generated data
        max_response = float(sum(
            np.mean([s.response_time for s in cat.services]) * 1.5
            for cat in categories
        ))
        
        return CompositionProblem(
            problem_id=problem_id,
            categories=categories,
            max_response_time=max_response,
            min_availability=0.8,
            max_cost=np.inf
        )
    
    def _generate_category_services(self, category_id: int) -> List[Service]:
        """Generate services for a single category."""
        services = []
        
        # Ensure carbon factors are available
        factors = self.config.carbon_factors or self.DEFAULT_CARBON_FACTORS
        base_response = np.random.uniform(
            self.config.response_time_range[0],
            self.config.response_time_range[1]
        )
        
        base_availability = np.random.uniform(
            self.config.availability_range[0],
            self.config.availability_range[1]
        )
        
        for svc_id in range(self.config.services_per_category):
            # Response time with variation
            response_time = base_response * np.random.uniform(0.5, 1.5)
            
            # Availability with slight variation
            availability = min(base_availability * np.random.uniform(0.98, 1.0), 0.999)
            
            # Throughput (inverse to response time)
            throughput = 1000.0 / response_time * np.random.uniform(0.8, 1.2)
            
            # IMPROVEMENT: Comprehensive Energy Parameters
            cpu_util = np.random.uniform(0.05, 0.4)
            mem_usage = np.random.uniform(64, 512)
            data_trans = np.random.uniform(0.1, 5.0)
            
            # Real-world inspired power parameters (Watts)
            peak_p = np.random.uniform(180, 250)
            idle_p = np.random.uniform(80, 120)
            
            # Random region
            region = np.random.choice(self.REGIONS)
            carbon_factor = factors[region]
            
            # Temporary service to calculate energy
            temp_service = Service(
                service_id=f"s_{category_id}_{svc_id}",
                category_id=category_id,
                response_time=response_time,
                availability=availability,
                throughput=throughput,
                energy_consumption=0.0,
                cost=0.0,
                carbon_factor=carbon_factor,
                region=region,
                cpu_utilization=cpu_util,
                memory_usage=mem_usage,
                data_transfer=data_trans,
                peak_power=peak_p,
                idle_power=idle_p
            )
            
            # Calculate energy using the comprehensive model
            energy = self.energy_model.calculate_service_energy(temp_service)
            
            # Cost (correlated with quality and energy)
            cost = (1 - availability) * 10 + (response_time / 1000.0) * 5 + (energy / 100.0)
            
            service = Service(
                service_id=f"s_{category_id}_{svc_id}",
                category_id=category_id,
                response_time=response_time,
                availability=availability,
                throughput=throughput,
                energy_consumption=energy,
                cost=cost,
                carbon_factor=carbon_factor,
                region=region,
                cpu_utilization=cpu_util,
                memory_usage=mem_usage,
                data_transfer=data_trans,
                peak_power=peak_p,
                idle_power=idle_p
            )

            
            # Calculate energy using the comprehensive model
            energy = self.energy_model.calculate_service_energy(temp_service)
            
            # Cost (correlated with quality and energy)
            cost = (1 - availability) * 10 + (response_time / 1000.0) * 5 + (energy / 100.0)
            
            service = Service(
                service_id=f"s_{category_id}_{svc_id}",
                category_id=category_id,
                response_time=response_time,
                availability=availability,
                throughput=throughput,
                energy_consumption=energy,
                cost=cost,
                carbon_factor=carbon_factor,
                region=region,
                cpu_utilization=cpu_util,
                memory_usage=mem_usage,
                data_transfer=data_trans,
                peak_power=peak_p,
                idle_power=idle_p
            )
            services.append(service)
        
        return services

    
    def generate_parameter_study(
        self,
        sizes: Optional[List[int]] = None
    ) -> Dict[int, CompositionProblem]:

        """Generate problems of varying sizes for scalability study."""
        if sizes is None:
            sizes = [5, 10, 15, 20, 25]
        
        problems = {}
        for size in sizes:
            config = DatasetConfig(
                num_categories=size,
                services_per_category=20,
                seed=42
            )
            generator = DatasetGenerator(config)
            problems[size] = generator.generate(f"scale_{size}")
        
        return problems
    
    def export_to_dataframe(self, problem: CompositionProblem) -> pd.DataFrame:
        """Export problem to pandas DataFrame."""
        rows = []
        
        for category in problem.categories:
            for service in category.services:
                rows.append({
                    'service_id': service.service_id,
                    'category_id': service.category_id,
                    'response_time': service.response_time,
                    'availability': service.availability,
                    'throughput': service.throughput,
                    'energy': service.energy_consumption,
                    'cost': service.cost,
                    'carbon_factor': service.carbon_factor,
                    'region': service.region,
                    'carbon_emission': service.carbon_emission
                })
        
        return pd.DataFrame(rows)


# Convenience function
def generate_dataset(
    num_categories: int = 10,
    services_per_category: int = 20,
    seed: int = 42
) -> CompositionProblem:
    """Quick dataset generation function."""
    config = DatasetConfig(
        num_categories=num_categories,
        services_per_category=services_per_category,
        seed=seed
    )
    generator = DatasetGenerator(config)
    return generator.generate()
