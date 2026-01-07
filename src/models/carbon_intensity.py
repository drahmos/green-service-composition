"""
Carbon Intensity Service for geographic and temporal awareness.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Optional


class CarbonIntensityService:
    """
    Simulates or fetches real-time carbon intensity (g CO2/kWh) for different regions.
    Supports temporal variations (day/night cycles for solar).
    """
    
    # Base intensity by region (avg values)
    BASE_INTENSITY = {
        'us-east': 420.0,
        'us-west': 350.0,
        'eu-west': 280.0,
        'eu-north': 180.0,
        'asia-east': 550.0,
        'asia-south': 680.0,
        'sa-east': 80.0,
        'au-east': 450.0
    }
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            
    def get_intensity(self, region: str, timestamp: Optional[datetime] = None) -> float:
        """
        Get carbon intensity for a region at a specific time.
        
        Args:
            region: Data center region.
            timestamp: Time of execution.
            
        Returns:
            Intensity in kg CO2 / kWh.
        """
        base = self.BASE_INTENSITY.get(region, 400.0)
        
        if timestamp is None:
            timestamp = datetime.now()
            
        # Simulate solar variation (lower intensity during day)
        hour = timestamp.hour
        solar_factor = 1.0
        if 8 <= hour <= 18:
            # Daytime - simulate solar availability
            # Peak solar at 13:00 reduces intensity by up to 30%
            dist_from_peak = abs(hour - 13)
            reduction = max(0, 0.3 * (1 - dist_from_peak / 5))
            solar_factor = 1.0 - reduction
            
        # Add some random noise
        noise = np.random.normal(1.0, 0.05)
        
        # Convert g to kg
        return (base * solar_factor * noise) / 1000.0

    def get_all_intensities(self, timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """Get intensities for all supported regions."""
        return {
            region: self.get_intensity(region, timestamp)
            for region in self.BASE_INTENSITY
        }
