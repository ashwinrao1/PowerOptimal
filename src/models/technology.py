"""Technology cost and facility parameter models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TechnologyCosts:
    """Capital and operating costs for all technologies.
    
    Attributes:
        grid_capex_per_kw: Grid interconnection capital cost in $/kW
        gas_capex_per_kw: Gas peaker capital cost in $/kW
        battery_capex_per_kwh: Battery storage capital cost in $/kWh
        solar_capex_per_kw: Solar PV capital cost in $/kW
        gas_variable_om: Gas variable O&M cost in $/MWh
        gas_heat_rate: Gas heat rate in MMBtu/MWh
        gas_efficiency: Gas thermal efficiency (fraction)
        battery_degradation: Battery degradation cost in $/MWh
        battery_efficiency: Battery round-trip efficiency (fraction)
        battery_duration: Battery duration in hours
        solar_fixed_om: Solar fixed O&M cost in $/kW-year
        grid_demand_charge: Grid demand charge in $/kW-month
    """
    grid_capex_per_kw: float = 3000
    gas_capex_per_kw: float = 1000
    battery_capex_per_kwh: float = 350
    solar_capex_per_kw: float = 1200
    
    gas_variable_om: float = 15
    gas_heat_rate: float = 10
    gas_efficiency: float = 0.35
    
    battery_degradation: float = 5
    battery_efficiency: float = 0.85
    battery_duration: float = 4
    
    solar_fixed_om: float = 20
    
    grid_demand_charge: float = 15
    
    def validate(self) -> bool:
        """Validate that all cost parameters are non-negative.
        
        Returns:
            True if all validation checks pass
            
        Raises:
            ValueError: If any validation check fails
        """
        if self.grid_capex_per_kw < 0:
            raise ValueError("Grid CAPEX must be non-negative")
        if self.gas_capex_per_kw < 0:
            raise ValueError("Gas CAPEX must be non-negative")
        if self.battery_capex_per_kwh < 0:
            raise ValueError("Battery CAPEX must be non-negative")
        if self.solar_capex_per_kw < 0:
            raise ValueError("Solar CAPEX must be non-negative")
        
        if self.gas_variable_om < 0:
            raise ValueError("Gas variable O&M must be non-negative")
        if self.gas_heat_rate <= 0:
            raise ValueError("Gas heat rate must be positive")
        if not 0 < self.gas_efficiency <= 1:
            raise ValueError("Gas efficiency must be between 0 and 1")
        
        if self.battery_degradation < 0:
            raise ValueError("Battery degradation cost must be non-negative")
        if not 0 < self.battery_efficiency <= 1:
            raise ValueError("Battery efficiency must be between 0 and 1")
        if self.battery_duration <= 0:
            raise ValueError("Battery duration must be positive")
        
        if self.solar_fixed_om < 0:
            raise ValueError("Solar fixed O&M must be non-negative")
        
        if self.grid_demand_charge < 0:
            raise ValueError("Grid demand charge must be non-negative")
        
        return True


@dataclass
class FacilityParams:
    """Data center facility characteristics.
    
    Attributes:
        it_load_mw: IT equipment load in MW
        pue: Power Usage Effectiveness ratio
        total_load_mw: Total facility load in MW (computed from IT load and PUE)
        reliability_target: Target reliability as fraction (e.g., 0.9999 for 99.99%)
        carbon_budget: Optional carbon budget in tons CO2/year
        planning_horizon_years: Planning horizon in years
        discount_rate: Discount rate for NPV calculations (fraction)
        curtailment_penalty: Penalty for load curtailment in $/MWh
    """
    it_load_mw: float = 300
    pue: float = 1.05
    total_load_mw: float = field(init=False)
    reliability_target: float = 0.9999
    carbon_budget: Optional[float] = None
    planning_horizon_years: int = 20
    discount_rate: float = 0.07
    curtailment_penalty: float = 10000
    
    def __post_init__(self):
        """Calculate total load from IT load and PUE."""
        self.total_load_mw = self.it_load_mw * self.pue
    
    def validate(self) -> bool:
        """Validate facility parameters.
        
        Returns:
            True if all validation checks pass
            
        Raises:
            ValueError: If any validation check fails
        """
        if self.it_load_mw <= 0:
            raise ValueError("IT load must be positive")
        
        if self.pue < 1.0:
            raise ValueError("PUE must be at least 1.0")
        
        if not 0 < self.reliability_target <= 1:
            raise ValueError("Reliability target must be between 0 and 1")
        
        if self.carbon_budget is not None and self.carbon_budget < 0:
            raise ValueError("Carbon budget must be non-negative")
        
        if self.planning_horizon_years <= 0:
            raise ValueError("Planning horizon must be positive")
        
        if self.discount_rate < 0:
            raise ValueError("Discount rate must be non-negative")
        
        if self.curtailment_penalty < 0:
            raise ValueError("Curtailment penalty must be non-negative")
        
        return True
    
    def max_annual_curtailment_mwh(self) -> float:
        """Calculate maximum allowed annual curtailment in MWh.
        
        Based on reliability target and total load.
        For 99.99% reliability, allows 1 hour of downtime per year.
        
        Returns:
            Maximum annual curtailment in MWh
        """
        allowed_downtime_hours = 8760 * (1 - self.reliability_target)
        return allowed_downtime_hours * self.total_load_mw
