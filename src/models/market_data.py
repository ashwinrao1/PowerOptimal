"""Market data model for optimization inputs."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional


@dataclass
class MarketData:
    """Hourly market data for optimization.
    
    Attributes:
        timestamp: DatetimeIndex with 8760 hourly timestamps
        lmp: Locational marginal prices in $/MWh
        gas_price: Natural gas prices in $/MMBtu
        solar_cf: Solar capacity factors (0-1)
        grid_carbon_intensity: Grid carbon intensity in kg CO2/MWh
    """
    timestamp: pd.DatetimeIndex
    lmp: np.ndarray
    gas_price: np.ndarray
    solar_cf: np.ndarray
    grid_carbon_intensity: np.ndarray
    
    def validate(self) -> bool:
        """Ensure data completeness and validity.
        
        Returns:
            True if all validation checks pass
            
        Raises:
            ValueError: If any validation check fails
        """
        # Check length
        if len(self.timestamp) != 8760:
            raise ValueError(f"Expected 8760 hours, got {len(self.timestamp)}")
        
        # Check array lengths match
        if not all(len(arr) == 8760 for arr in [self.lmp, self.gas_price, 
                                                  self.solar_cf, self.grid_carbon_intensity]):
            raise ValueError("All arrays must have length 8760")
        
        # Check for missing values
        if np.any(np.isnan(self.lmp)):
            raise ValueError("LMP data contains NaN values")
        if np.any(np.isnan(self.gas_price)):
            raise ValueError("Gas price data contains NaN values")
        if np.any(np.isnan(self.solar_cf)):
            raise ValueError("Solar capacity factor data contains NaN values")
        if np.any(np.isnan(self.grid_carbon_intensity)):
            raise ValueError("Grid carbon intensity data contains NaN values")
        
        # Validate LMP range
        if np.any(self.lmp < -100) or np.any(self.lmp > 5000):
            raise ValueError("LMP values must be between -100 and 5000 $/MWh")
        
        # Validate gas price range
        if np.any(self.gas_price < 0) or np.any(self.gas_price > 50):
            raise ValueError("Gas prices must be between 0 and 50 $/MMBtu")
        
        # Validate solar capacity factor range
        if np.any(self.solar_cf < 0) or np.any(self.solar_cf > 1):
            raise ValueError("Solar capacity factors must be between 0 and 1")
        
        # Validate carbon intensity is non-negative
        if np.any(self.grid_carbon_intensity < 0):
            raise ValueError("Grid carbon intensity must be non-negative")
        
        return True
    
    def get_statistics(self) -> dict:
        """Get summary statistics for the market data.
        
        Returns:
            Dictionary with statistics for each data series
        """
        return {
            'lmp': {
                'mean': float(np.mean(self.lmp)),
                'min': float(np.min(self.lmp)),
                'max': float(np.max(self.lmp)),
                'std': float(np.std(self.lmp))
            },
            'gas_price': {
                'mean': float(np.mean(self.gas_price)),
                'min': float(np.min(self.gas_price)),
                'max': float(np.max(self.gas_price)),
                'std': float(np.std(self.gas_price))
            },
            'solar_cf': {
                'mean': float(np.mean(self.solar_cf)),
                'min': float(np.min(self.solar_cf)),
                'max': float(np.max(self.solar_cf)),
                'std': float(np.std(self.solar_cf))
            },
            'grid_carbon_intensity': {
                'mean': float(np.mean(self.grid_carbon_intensity)),
                'min': float(np.min(self.grid_carbon_intensity)),
                'max': float(np.max(self.grid_carbon_intensity)),
                'std': float(np.std(self.grid_carbon_intensity))
            }
        }
