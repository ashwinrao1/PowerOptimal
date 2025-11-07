"""Solution data models for optimization results."""

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import json
from typing import Optional, Dict, Any
from pathlib import Path

from .technology import TechnologyCosts


@dataclass
class CapacitySolution:
    """Optimal capacity investments.
    
    Attributes:
        grid_mw: Grid interconnection capacity in MW
        gas_mw: Gas peaker capacity in MW
        battery_mwh: Battery storage capacity in MWh
        solar_mw: Solar PV capacity in MW
    """
    grid_mw: float
    gas_mw: float
    battery_mwh: float
    solar_mw: float
    
    def total_capex(self, costs: TechnologyCosts) -> float:
        """Calculate total capital expenditure.
        
        Args:
            costs: Technology cost parameters
            
        Returns:
            Total CAPEX in dollars
        """
        return (
            self.grid_mw * costs.grid_capex_per_kw +
            self.gas_mw * costs.gas_capex_per_kw +
            self.battery_mwh * costs.battery_capex_per_kwh +
            self.solar_mw * costs.solar_capex_per_kw
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format.
        
        Returns:
            Dictionary with descriptive keys
        """
        return {
            "Grid Connection (MW)": self.grid_mw,
            "Gas Peakers (MW)": self.gas_mw,
            "Battery Storage (MWh)": self.battery_mwh,
            "Solar PV (MW)": self.solar_mw
        }


@dataclass
class DispatchSolution:
    """Hourly operational decisions.
    
    Attributes:
        hour: Hour indices (1-8760)
        grid_power: Grid power draw in MW for each hour
        gas_power: Gas generation in MW for each hour
        solar_power: Solar generation in MW for each hour
        battery_power: Battery power in MW (positive = charge, negative = discharge)
        curtailment: Load curtailment in MW for each hour
        battery_soc: Battery state of charge in MWh for each hour
    """
    hour: np.ndarray
    grid_power: np.ndarray
    gas_power: np.ndarray
    solar_power: np.ndarray
    battery_power: np.ndarray
    curtailment: np.ndarray
    battery_soc: np.ndarray
    
    def __post_init__(self):
        """Validate array lengths."""
        arrays = [
            self.hour, self.grid_power, self.gas_power, 
            self.solar_power, self.battery_power, 
            self.curtailment, self.battery_soc
        ]
        lengths = [len(arr) for arr in arrays]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All arrays must have the same length")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.
        
        Returns:
            DataFrame with hourly dispatch data
        """
        return pd.DataFrame({
            "Hour": self.hour,
            "Grid (MW)": self.grid_power,
            "Gas (MW)": self.gas_power,
            "Solar (MW)": self.solar_power,
            "Battery (MW)": self.battery_power,
            "Curtailment (MW)": self.curtailment,
            "Battery SOC (MWh)": self.battery_soc
        })


@dataclass
class SolutionMetrics:
    """Key performance indicators.
    
    Attributes:
        total_npv: Total 20-year net present value in dollars
        capex: Capital expenditure in dollars
        opex_annual: Annual operating expenditure in dollars/year
        lcoe: Levelized cost of energy in $/MWh
        reliability_pct: Reliability percentage
        total_curtailment_mwh: Total annual curtailment in MWh
        num_curtailment_hours: Number of hours with curtailment
        carbon_tons_annual: Annual carbon emissions in tons CO2
        carbon_intensity_g_per_kwh: Carbon intensity in g CO2/kWh
        carbon_reduction_pct: Carbon reduction vs grid-only baseline
        grid_dependence_pct: Percentage of energy from grid
        gas_capacity_factor: Gas utilization percentage
        battery_cycles_per_year: Battery cycles per year
        solar_capacity_factor: Solar capacity factor
        solve_time_seconds: Solver time in seconds
        optimality_gap_pct: Optimality gap percentage
    """
    total_npv: float
    capex: float
    opex_annual: float
    lcoe: float
    
    reliability_pct: float
    total_curtailment_mwh: float
    num_curtailment_hours: int
    
    carbon_tons_annual: float
    carbon_intensity_g_per_kwh: float
    carbon_reduction_pct: float
    
    grid_dependence_pct: float
    gas_capacity_factor: float
    battery_cycles_per_year: float
    solar_capacity_factor: float
    
    solve_time_seconds: float
    optimality_gap_pct: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.
        
        Returns:
            Dictionary with all metrics
        """
        return asdict(self)


@dataclass
class OptimizationSolution:
    """Complete optimization result.
    
    Attributes:
        capacity: Capacity investment decisions
        dispatch: Hourly dispatch decisions
        metrics: Solution performance metrics
        scenario_params: Scenario parameters used for this solution
    """
    capacity: CapacitySolution
    dispatch: DispatchSolution
    metrics: SolutionMetrics
    scenario_params: Dict[str, Any]
    
    def save(self, filepath: str) -> None:
        """Save solution to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        # Convert dispatch arrays to lists for JSON serialization
        dispatch_dict = {
            "hour": self.dispatch.hour.tolist(),
            "grid_power": self.dispatch.grid_power.tolist(),
            "gas_power": self.dispatch.gas_power.tolist(),
            "solar_power": self.dispatch.solar_power.tolist(),
            "battery_power": self.dispatch.battery_power.tolist(),
            "curtailment": self.dispatch.curtailment.tolist(),
            "battery_soc": self.dispatch.battery_soc.tolist()
        }
        
        data = {
            "capacity": self.capacity.to_dict(),
            "dispatch": dispatch_dict,
            "metrics": self.metrics.to_dict(),
            "scenario_params": self.scenario_params
        }
        
        # Create parent directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'OptimizationSolution':
        """Load solution from JSON file.
        
        Args:
            filepath: Path to input JSON file
            
        Returns:
            OptimizationSolution object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct capacity
        capacity_data = data["capacity"]
        capacity = CapacitySolution(
            grid_mw=capacity_data["Grid Connection (MW)"],
            gas_mw=capacity_data["Gas Peakers (MW)"],
            battery_mwh=capacity_data["Battery Storage (MWh)"],
            solar_mw=capacity_data["Solar PV (MW)"]
        )
        
        # Reconstruct dispatch
        dispatch_data = data["dispatch"]
        dispatch = DispatchSolution(
            hour=np.array(dispatch_data["hour"]),
            grid_power=np.array(dispatch_data["grid_power"]),
            gas_power=np.array(dispatch_data["gas_power"]),
            solar_power=np.array(dispatch_data["solar_power"]),
            battery_power=np.array(dispatch_data["battery_power"]),
            curtailment=np.array(dispatch_data["curtailment"]),
            battery_soc=np.array(dispatch_data["battery_soc"])
        )
        
        # Reconstruct metrics
        metrics_data = data["metrics"]
        metrics = SolutionMetrics(**metrics_data)
        
        # Get scenario params
        scenario_params = data["scenario_params"]
        
        return cls(
            capacity=capacity,
            dispatch=dispatch,
            metrics=metrics,
            scenario_params=scenario_params
        )
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Create a summary dictionary with key results.
        
        Returns:
            Dictionary with capacity, key metrics, and scenario info
        """
        return {
            "capacity": self.capacity.to_dict(),
            "key_metrics": {
                "Total NPV ($)": self.metrics.total_npv,
                "CAPEX ($)": self.metrics.capex,
                "Annual OPEX ($/year)": self.metrics.opex_annual,
                "LCOE ($/MWh)": self.metrics.lcoe,
                "Reliability (%)": self.metrics.reliability_pct,
                "Carbon Intensity (g CO2/kWh)": self.metrics.carbon_intensity_g_per_kwh,
                "Grid Dependence (%)": self.metrics.grid_dependence_pct
            },
            "scenario": self.scenario_params
        }
