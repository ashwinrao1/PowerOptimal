"""
Configuration management module for data center energy optimization.

This module stores all constants, parameters, and configuration settings
used throughout the project.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TechnologyCosts:
    """Capital and operating costs for all technologies."""
    
    # Capital costs ($/kW or $/kWh)
    grid_capex_per_kw: float = 3000.0
    gas_capex_per_kw: float = 1000.0
    battery_capex_per_kwh: float = 350.0
    solar_capex_per_kw: float = 1200.0
    
    # Gas peaker parameters
    gas_variable_om: float = 15.0  # $/MWh
    gas_heat_rate: float = 10.0  # MMBtu/MWh
    gas_efficiency: float = 0.35
    gas_carbon_intensity: float = 0.4  # metric tons CO2/MWh
    
    # Battery parameters
    battery_degradation: float = 5.0  # $/MWh
    battery_efficiency: float = 0.85
    battery_duration: float = 4.0  # hours
    battery_min_soc: float = 0.1  # 10% minimum state of charge
    battery_max_soc: float = 0.9  # 90% maximum state of charge
    battery_max_charge_rate: float = 0.25  # 25% of capacity per hour
    
    # Solar parameters
    solar_fixed_om: float = 20.0  # $/kW-year
    
    # Grid parameters
    grid_demand_charge: float = 15.0  # $/kW-month


@dataclass
class FacilityParams:
    """Data center facility characteristics."""
    
    it_load_mw: float = 300.0
    pue: float = 1.05
    total_load_mw: float = field(init=False)
    reliability_target: float = 0.9999  # 99.99%
    carbon_budget: Optional[float] = None  # tons CO2/year
    planning_horizon_years: int = 20
    discount_rate: float = 0.07
    curtailment_penalty: float = 10000.0  # $/MWh
    
    def __post_init__(self):
        """Calculate total load including PUE overhead."""
        self.total_load_mw = self.it_load_mw * self.pue
    
    @property
    def max_annual_curtailment_mwh(self) -> float:
        """Maximum allowed curtailment based on reliability target."""
        hours_per_year = 8760
        allowed_downtime_hours = hours_per_year * (1 - self.reliability_target)
        return self.total_load_mw * allowed_downtime_hours


@dataclass
class SolverConfig:
    """Gurobi solver configuration parameters."""
    
    solver_name: str = "gurobi"
    mip_gap: float = 0.005  # 0.5%
    time_limit: int = 1800  # 30 minutes in seconds
    threads: int = 0  # 0 = automatic
    method: int = -1  # -1 = automatic, 0 = primal simplex, 1 = dual simplex, 2 = barrier
    crossover: int = 0  # 0 = automatic crossover after barrier
    numerical_focus: int = 0  # 0 = default, 1-3 = increasing numerical precision
    
    def to_dict(self) -> dict:
        """Convert to dictionary for Pyomo solver options."""
        return {
            "MIPGap": self.mip_gap,
            "TimeLimit": self.time_limit,
            "Threads": self.threads,
            "Method": self.method,
            "Crossover": self.crossover,
            "NumericFocus": self.numerical_focus
        }


@dataclass
class DataConfig:
    """Configuration for data collection and processing."""
    
    # ERCOT configuration
    ercot_hub: str = "HB_WEST"
    ercot_start_date: str = "2022-01-01"
    ercot_end_date: str = "2024-12-31"
    
    # NREL PVWatts configuration (West Texas)
    solar_latitude: float = 31.9973
    solar_longitude: float = -102.0779
    solar_tilt: float = 32.0  # Fixed tilt at latitude
    solar_azimuth: float = 180.0  # South-facing
    solar_system_capacity: float = 1.0  # 1 kW for capacity factor calculation
    solar_module_type: int = 0  # 0 = Standard
    solar_array_type: int = 1  # 1 = Fixed (open rack)
    solar_losses: float = 14.0  # System losses percentage
    
    # EIA configuration
    gas_hub: str = "WAHA"
    carbon_region: str = "ERCO"  # ERCOT
    
    # Data validation
    max_missing_pct: float = 1.0  # Maximum 1% missing data allowed
    lmp_min: float = -100.0  # $/MWh
    lmp_max: float = 5000.0  # $/MWh
    gas_price_min: float = 0.0  # $/MMBtu
    gas_price_max: float = 50.0  # $/MMBtu


@dataclass
class VisualizationConfig:
    """Configuration for plots and visualizations."""
    
    # Plotly theme
    template: str = "plotly_white"
    
    # Color scheme for technologies
    colors: dict = field(default_factory=lambda: {
        "grid": "#1f77b4",  # Blue
        "gas": "#ff7f0e",   # Orange
        "battery": "#2ca02c",  # Green
        "solar": "#ffd700",  # Gold
        "curtailment": "#d62728"  # Red
    })
    
    # Figure dimensions
    default_width: int = 1000
    default_height: int = 600
    
    # Font sizes
    title_font_size: int = 18
    axis_font_size: int = 14
    legend_font_size: int = 12


# Global configuration instances
TECH_COSTS = TechnologyCosts()
FACILITY_PARAMS = FacilityParams()
SOLVER_CONFIG = SolverConfig()
DATA_CONFIG = DataConfig()
VIZ_CONFIG = VisualizationConfig()


# Constants
HOURS_PER_YEAR = 8760
MONTHS_PER_YEAR = 12
MW_TO_KW = 1000.0
MWH_TO_KWH = 1000.0


def get_scenario_params(scenario_name: str) -> dict:
    """
    Get predefined scenario parameters.
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        Dictionary of scenario parameters
    """
    scenarios = {
        "baseline": {
            "name": "Baseline (Grid Only)",
            "gas_price_multiplier": 1.0,
            "lmp_multiplier": 1.0,
            "battery_cost_multiplier": 1.0,
            "reliability_target": 0.9999,
            "carbon_budget": None,
            "allow_gas": False,
            "allow_battery": False,
            "allow_solar": False
        },
        "optimal": {
            "name": "Optimal Portfolio",
            "gas_price_multiplier": 1.0,
            "lmp_multiplier": 1.0,
            "battery_cost_multiplier": 1.0,
            "reliability_target": 0.9999,
            "carbon_budget": None,
            "allow_gas": True,
            "allow_battery": True,
            "allow_solar": True
        },
        "high_reliability": {
            "name": "High Reliability (99.999%)",
            "gas_price_multiplier": 1.0,
            "lmp_multiplier": 1.0,
            "battery_cost_multiplier": 1.0,
            "reliability_target": 0.99999,
            "carbon_budget": None,
            "allow_gas": True,
            "allow_battery": True,
            "allow_solar": True
        },
        "carbon_free": {
            "name": "100% Carbon Free",
            "gas_price_multiplier": 1.0,
            "lmp_multiplier": 1.0,
            "battery_cost_multiplier": 1.0,
            "reliability_target": 0.9999,
            "carbon_budget": 0.0,
            "allow_gas": False,
            "allow_battery": True,
            "allow_solar": True
        }
    }
    
    return scenarios.get(scenario_name, scenarios["optimal"])
