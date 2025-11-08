"""
Solution extraction module for data center energy optimization.

This module extracts capacity decisions, hourly dispatch, and calculates
comprehensive metrics from solved Pyomo optimization models.
"""

import pyomo.environ as pyo
import numpy as np
from typing import Dict, Any, Optional
import logging

# Use absolute imports to avoid issues when imported from different contexts
try:
    from ..models.solution import (
        CapacitySolution,
        DispatchSolution,
        SolutionMetrics,
        OptimizationSolution
    )
    from ..models.technology import TechnologyCosts, FacilityParams
    from ..models.market_data import MarketData
except ImportError:
    from models.solution import (
        CapacitySolution,
        DispatchSolution,
        SolutionMetrics,
        OptimizationSolution
    )
    from models.technology import TechnologyCosts, FacilityParams
    from models.market_data import MarketData

logger = logging.getLogger(__name__)


def extract_solution(
    model: pyo.ConcreteModel,
    market_data: MarketData,
    tech_costs: TechnologyCosts,
    facility_params: FacilityParams,
    solve_time: float,
    optimality_gap: float = 0.0,
    scenario_params: Optional[Dict[str, Any]] = None
) -> OptimizationSolution:
    """
    Extract complete solution from solved optimization model.
    
    Extracts capacity decisions, hourly dispatch decisions, and calculates
    comprehensive metrics including costs, reliability, carbon emissions,
    and operational statistics.
    
    Args:
        model: Solved Pyomo ConcreteModel
        market_data: Market data used in optimization
        tech_costs: Technology cost parameters
        facility_params: Facility parameters
        solve_time: Solver time in seconds
        optimality_gap: Optimality gap as fraction (e.g., 0.005 for 0.5%)
        scenario_params: Optional dictionary of scenario parameters
        
    Returns:
        OptimizationSolution with capacity, dispatch, and metrics
        
    Example:
        >>> results, solve_time = solve_model(model)
        >>> solution = extract_solution(model, data, costs, params, solve_time)
        >>> print(f"Optimal grid capacity: {solution.capacity.grid_mw:.1f} MW")
    """
    logger.info("Extracting solution from solved model...")
    
    # Extract capacity decisions
    capacity = _extract_capacity(model)
    logger.info(f"Capacity: Grid={capacity.grid_mw:.1f} MW, "
                f"Gas={capacity.gas_mw:.1f} MW, "
                f"Battery={capacity.battery_mwh:.1f} MWh, "
                f"Solar={capacity.solar_mw:.1f} MW")
    
    # Extract hourly dispatch
    dispatch = _extract_dispatch(model)
    logger.info(f"Extracted dispatch for {len(dispatch.hour)} hours")
    
    # Calculate solution metrics
    metrics = _calculate_metrics(
        model=model,
        capacity=capacity,
        dispatch=dispatch,
        market_data=market_data,
        tech_costs=tech_costs,
        facility_params=facility_params,
        solve_time=solve_time,
        optimality_gap=optimality_gap
    )
    logger.info(f"Total NPV: ${metrics.total_npv:,.0f}, "
                f"LCOE: ${metrics.lcoe:.2f}/MWh, "
                f"Reliability: {metrics.reliability_pct:.4f}%")
    
    # Create scenario params dict if not provided
    if scenario_params is None:
        scenario_params = {
            "facility_load_mw": facility_params.total_load_mw,
            "reliability_target": facility_params.reliability_target,
            "carbon_budget": facility_params.carbon_budget,
            "planning_horizon_years": facility_params.planning_horizon_years,
            "discount_rate": facility_params.discount_rate
        }
    
    # Create complete solution object
    solution = OptimizationSolution(
        capacity=capacity,
        dispatch=dispatch,
        metrics=metrics,
        scenario_params=scenario_params
    )
    
    logger.info("Solution extraction complete")
    return solution


def _extract_capacity(model: pyo.ConcreteModel) -> CapacitySolution:
    """
    Extract capacity decisions from solved model.
    
    Args:
        model: Solved Pyomo ConcreteModel
        
    Returns:
        CapacitySolution with capacity investments
    """
    return CapacitySolution(
        grid_mw=pyo.value(model.C_grid),
        gas_mw=pyo.value(model.C_gas),
        battery_mwh=pyo.value(model.C_battery),
        solar_mw=pyo.value(model.C_solar)
    )


def _extract_dispatch(model: pyo.ConcreteModel) -> DispatchSolution:
    """
    Extract hourly dispatch decisions from solved model.
    
    Args:
        model: Solved Pyomo ConcreteModel
        
    Returns:
        DispatchSolution with hourly operational decisions
    """
    hours = sorted(model.hours)
    
    # Extract dispatch arrays
    hour_array = np.array(hours)
    grid_power = np.array([pyo.value(model.p_grid[h]) for h in hours])
    gas_power = np.array([pyo.value(model.p_gas[h]) for h in hours])
    solar_power = np.array([pyo.value(model.p_solar[h]) for h in hours])
    battery_power = np.array([pyo.value(model.p_battery[h]) for h in hours])
    curtailment = np.array([pyo.value(model.p_curtail[h]) for h in hours])
    battery_soc = np.array([pyo.value(model.SOC[h]) for h in hours])
    
    return DispatchSolution(
        hour=hour_array,
        grid_power=grid_power,
        gas_power=gas_power,
        solar_power=solar_power,
        battery_power=battery_power,
        curtailment=curtailment,
        battery_soc=battery_soc
    )


def _calculate_metrics(
    model: pyo.ConcreteModel,
    capacity: CapacitySolution,
    dispatch: DispatchSolution,
    market_data: MarketData,
    tech_costs: TechnologyCosts,
    facility_params: FacilityParams,
    solve_time: float,
    optimality_gap: float
) -> SolutionMetrics:
    """
    Calculate comprehensive solution metrics.
    
    Args:
        model: Solved Pyomo ConcreteModel
        capacity: Capacity solution
        dispatch: Dispatch solution
        market_data: Market data
        tech_costs: Technology costs
        facility_params: Facility parameters
        solve_time: Solver time in seconds
        optimality_gap: Optimality gap as fraction
        
    Returns:
        SolutionMetrics with all performance indicators
    """
    # Extract basic values
    total_npv = pyo.value(model.total_cost)
    capex = capacity.total_capex(tech_costs)
    
    # Calculate annual OPEX components
    opex_annual = _calculate_annual_opex(
        dispatch=dispatch,
        capacity=capacity,
        market_data=market_data,
        tech_costs=tech_costs
    )
    
    # Calculate LCOE
    total_energy_mwh = facility_params.total_load_mw * 8760
    lcoe = _calculate_lcoe(
        capex=capex,
        opex_annual=opex_annual,
        total_energy_mwh=total_energy_mwh,
        planning_horizon=facility_params.planning_horizon_years,
        discount_rate=facility_params.discount_rate
    )
    
    # Calculate reliability metrics
    reliability_pct, total_curtailment_mwh, num_curtailment_hours = _calculate_reliability_metrics(
        dispatch=dispatch,
        facility_load_mw=facility_params.total_load_mw
    )
    
    # Calculate carbon metrics
    carbon_tons_annual, carbon_intensity_g_per_kwh, carbon_reduction_pct = _calculate_carbon_metrics(
        dispatch=dispatch,
        market_data=market_data,
        facility_load_mw=facility_params.total_load_mw
    )
    
    # Calculate operational metrics
    grid_dependence_pct = _calculate_grid_dependence(dispatch, facility_params.total_load_mw)
    gas_capacity_factor = _calculate_gas_capacity_factor(dispatch, capacity)
    battery_cycles_per_year = _calculate_battery_cycles(dispatch, capacity)
    solar_capacity_factor = _calculate_solar_capacity_factor(dispatch, capacity)
    
    return SolutionMetrics(
        total_npv=total_npv,
        capex=capex,
        opex_annual=opex_annual,
        lcoe=lcoe,
        reliability_pct=reliability_pct,
        total_curtailment_mwh=total_curtailment_mwh,
        num_curtailment_hours=num_curtailment_hours,
        carbon_tons_annual=carbon_tons_annual,
        carbon_intensity_g_per_kwh=carbon_intensity_g_per_kwh,
        carbon_reduction_pct=carbon_reduction_pct,
        grid_dependence_pct=grid_dependence_pct,
        gas_capacity_factor=gas_capacity_factor,
        battery_cycles_per_year=battery_cycles_per_year,
        solar_capacity_factor=solar_capacity_factor,
        solve_time_seconds=solve_time,
        optimality_gap_pct=optimality_gap * 100
    )


def _calculate_annual_opex(
    dispatch: DispatchSolution,
    capacity: CapacitySolution,
    market_data: MarketData,
    tech_costs: TechnologyCosts
) -> float:
    """
    Calculate total annual operating expenditure.
    
    Args:
        dispatch: Dispatch solution
        capacity: Capacity solution
        market_data: Market data
        tech_costs: Technology costs
        
    Returns:
        Annual OPEX in dollars/year
    """
    # Grid electricity costs (energy charges)
    grid_energy_cost = np.sum(dispatch.grid_power * market_data.lmp)
    
    # Grid demand charges (monthly peak)
    peak_grid_draw = np.max(dispatch.grid_power)
    grid_demand_cost = peak_grid_draw * tech_costs.grid_demand_charge * 12
    
    # Gas fuel costs
    gas_fuel_cost = np.sum(
        dispatch.gas_power * market_data.gas_price * tech_costs.gas_heat_rate
    )
    
    # Gas O&M costs
    gas_om_cost = np.sum(dispatch.gas_power * tech_costs.gas_variable_om)
    
    # Battery degradation costs
    battery_degradation_cost = np.sum(
        np.abs(dispatch.battery_power) * tech_costs.battery_degradation
    )
    
    # Solar O&M costs
    solar_om_cost = capacity.solar_mw * tech_costs.solar_fixed_om
    
    # Total annual OPEX
    opex_annual = (
        grid_energy_cost +
        grid_demand_cost +
        gas_fuel_cost +
        gas_om_cost +
        battery_degradation_cost +
        solar_om_cost
    )
    
    return opex_annual


def _calculate_lcoe(
    capex: float,
    opex_annual: float,
    total_energy_mwh: float,
    planning_horizon: int,
    discount_rate: float
) -> float:
    """
    Calculate levelized cost of energy.
    
    Args:
        capex: Capital expenditure
        opex_annual: Annual operating expenditure
        total_energy_mwh: Total annual energy consumption
        planning_horizon: Planning horizon in years
        discount_rate: Discount rate
        
    Returns:
        LCOE in $/MWh
    """
    # Calculate NPV of OPEX
    npv_factor = sum(1 / (1 + discount_rate)**year for year in range(1, planning_horizon + 1))
    npv_opex = opex_annual * npv_factor
    
    # Calculate NPV of energy
    npv_energy = total_energy_mwh * npv_factor
    
    # LCOE = (CAPEX + NPV of OPEX) / NPV of Energy
    lcoe = (capex + npv_opex) / npv_energy
    
    return lcoe


def _calculate_reliability_metrics(
    dispatch: DispatchSolution,
    facility_load_mw: float
) -> tuple:
    """
    Calculate reliability metrics.
    
    Args:
        dispatch: Dispatch solution
        facility_load_mw: Facility load in MW
        
    Returns:
        Tuple of (reliability_pct, total_curtailment_mwh, num_curtailment_hours)
    """
    # Total curtailment
    total_curtailment_mwh = np.sum(dispatch.curtailment)
    
    # Number of hours with curtailment
    num_curtailment_hours = np.sum(dispatch.curtailment > 1e-6)
    
    # Reliability percentage
    # Total possible energy = 8760 hours * load
    total_possible_energy_mwh = 8760 * facility_load_mw
    served_energy_mwh = total_possible_energy_mwh - total_curtailment_mwh
    reliability_pct = (served_energy_mwh / total_possible_energy_mwh) * 100
    
    return reliability_pct, total_curtailment_mwh, int(num_curtailment_hours)


def _calculate_carbon_metrics(
    dispatch: DispatchSolution,
    market_data: MarketData,
    facility_load_mw: float
) -> tuple:
    """
    Calculate carbon emission metrics.
    
    Args:
        dispatch: Dispatch solution
        market_data: Market data
        facility_load_mw: Facility load in MW
        
    Returns:
        Tuple of (carbon_tons_annual, carbon_intensity_g_per_kwh, carbon_reduction_pct)
    """
    # Grid emissions (kg CO2)
    grid_emissions_kg = np.sum(dispatch.grid_power * market_data.grid_carbon_intensity)
    
    # Gas emissions (tons CO2) - 0.4 tons CO2/MWh
    gas_emissions_tons = np.sum(dispatch.gas_power * 0.4)
    
    # Total emissions in tons
    carbon_tons_annual = grid_emissions_kg / 1000 + gas_emissions_tons
    
    # Carbon intensity (g CO2/kWh)
    total_energy_mwh = facility_load_mw * 8760
    carbon_intensity_g_per_kwh = (carbon_tons_annual * 1e6) / (total_energy_mwh * 1000)
    
    # Calculate baseline (grid-only) emissions for comparison
    baseline_emissions_kg = np.sum(facility_load_mw * market_data.grid_carbon_intensity)
    baseline_emissions_tons = baseline_emissions_kg / 1000
    
    # Carbon reduction percentage
    if baseline_emissions_tons > 0:
        carbon_reduction_pct = ((baseline_emissions_tons - carbon_tons_annual) / 
                               baseline_emissions_tons) * 100
    else:
        carbon_reduction_pct = 0.0
    
    return carbon_tons_annual, carbon_intensity_g_per_kwh, carbon_reduction_pct


def _calculate_grid_dependence(
    dispatch: DispatchSolution,
    facility_load_mw: float
) -> float:
    """
    Calculate percentage of energy from grid.
    
    Args:
        dispatch: Dispatch solution
        facility_load_mw: Facility load in MW
        
    Returns:
        Grid dependence as percentage
    """
    total_grid_energy = np.sum(dispatch.grid_power)
    total_energy = facility_load_mw * 8760
    
    grid_dependence_pct = (total_grid_energy / total_energy) * 100
    
    return grid_dependence_pct


def _calculate_gas_capacity_factor(
    dispatch: DispatchSolution,
    capacity: CapacitySolution
) -> float:
    """
    Calculate gas peaker capacity factor.
    
    Args:
        dispatch: Dispatch solution
        capacity: Capacity solution
        
    Returns:
        Gas capacity factor as percentage
    """
    if capacity.gas_mw < 1e-6:
        return 0.0
    
    total_gas_energy = np.sum(dispatch.gas_power)
    max_possible_energy = capacity.gas_mw * 8760
    
    gas_capacity_factor = (total_gas_energy / max_possible_energy) * 100
    
    return gas_capacity_factor


def _calculate_battery_cycles(
    dispatch: DispatchSolution,
    capacity: CapacitySolution
) -> float:
    """
    Calculate battery cycles per year.
    
    A cycle is defined as charging the battery from empty to full.
    
    Args:
        dispatch: Dispatch solution
        capacity: Capacity solution
        
    Returns:
        Battery cycles per year
    """
    if capacity.battery_mwh < 1e-6:
        return 0.0
    
    # Total energy throughput (sum of absolute charge/discharge)
    total_throughput = np.sum(np.abs(dispatch.battery_power))
    
    # Cycles = total throughput / (2 * capacity)
    # Divide by 2 because each cycle involves both charge and discharge
    battery_cycles = total_throughput / (2 * capacity.battery_mwh)
    
    return battery_cycles


def _calculate_solar_capacity_factor(
    dispatch: DispatchSolution,
    capacity: CapacitySolution
) -> float:
    """
    Calculate solar capacity factor.
    
    Args:
        dispatch: Dispatch solution
        capacity: Capacity solution
        
    Returns:
        Solar capacity factor as percentage
    """
    if capacity.solar_mw < 1e-6:
        return 0.0
    
    total_solar_energy = np.sum(dispatch.solar_power)
    max_possible_energy = capacity.solar_mw * 8760
    
    solar_capacity_factor = (total_solar_energy / max_possible_energy) * 100
    
    return solar_capacity_factor


def extract_worst_reliability_events(
    dispatch: DispatchSolution,
    market_data: MarketData,
    top_n: int = 10
) -> list:
    """
    Identify worst-case reliability events (hours with highest curtailment).
    
    Args:
        dispatch: Dispatch solution
        market_data: Market data
        top_n: Number of worst events to return
        
    Returns:
        List of dictionaries with event details, sorted by curtailment (worst first)
    """
    # Find hours with curtailment
    curtailment_hours = np.where(dispatch.curtailment > 1e-6)[0]
    
    if len(curtailment_hours) == 0:
        return []
    
    # Create event records
    events = []
    for idx in curtailment_hours:
        hour = dispatch.hour[idx]
        events.append({
            'hour': int(hour),
            'curtailment_mw': float(dispatch.curtailment[idx]),
            'grid_power_mw': float(dispatch.grid_power[idx]),
            'gas_power_mw': float(dispatch.gas_power[idx]),
            'solar_power_mw': float(dispatch.solar_power[idx]),
            'battery_power_mw': float(dispatch.battery_power[idx]),
            'battery_soc_mwh': float(dispatch.battery_soc[idx]),
            'lmp': float(market_data.lmp[idx]),
            'gas_price': float(market_data.gas_price[idx]),
            'solar_cf': float(market_data.solar_cf[idx])
        })
    
    # Sort by curtailment (worst first)
    events.sort(key=lambda x: x['curtailment_mw'], reverse=True)
    
    # Return top N events
    return events[:top_n]
