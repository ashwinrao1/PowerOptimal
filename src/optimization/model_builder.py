"""
Optimization model builder for data center energy portfolio optimization.

This module constructs a Pyomo optimization model that determines the optimal
capacity investments and hourly dispatch strategy for a data center's energy portfolio.
"""

import pyomo.environ as pyo
from typing import Optional
import numpy as np

# Use absolute imports to avoid issues when imported from different contexts
try:
    from ..models.market_data import MarketData
    from ..models.technology import TechnologyCosts, FacilityParams
except ImportError:
    from models.market_data import MarketData
    from models.technology import TechnologyCosts, FacilityParams


def build_optimization_model(
    market_data: MarketData,
    tech_costs: TechnologyCosts,
    facility_params: FacilityParams,
    allow_gas: bool = True,
    allow_battery: bool = True,
    allow_solar: bool = True
) -> pyo.ConcreteModel:
    """
    Build Pyomo optimization model for data center energy portfolio.
    
    The model minimizes total 20-year cost (CAPEX + NPV of OPEX + curtailment penalty)
    subject to energy balance, capacity limits, battery dynamics, reliability, and
    optional carbon constraints.
    
    Args:
        market_data: Hourly market data (LMP, gas prices, solar CF, carbon intensity)
        tech_costs: Technology cost parameters
        facility_params: Facility parameters (load, reliability target, etc.)
        allow_gas: Whether to allow gas peaker investment
        allow_battery: Whether to allow battery storage investment
        allow_solar: Whether to allow solar PV investment
        
    Returns:
        Pyomo ConcreteModel ready for solving
    """
    # Validate inputs
    market_data.validate()
    tech_costs.validate()
    facility_params.validate()
    
    # Create model
    model = pyo.ConcreteModel(name="DataCenterEnergyOptimization")
    
    # Sets
    model.hours = pyo.Set(initialize=range(1, 8761), doc="Hours in year (1-8760)")
    
    # Parameters
    model.dc_load = pyo.Param(initialize=facility_params.total_load_mw, doc="Data center load (MW)")
    model.discount_rate = pyo.Param(initialize=facility_params.discount_rate, doc="Discount rate")
    model.planning_horizon = pyo.Param(initialize=facility_params.planning_horizon_years, doc="Planning horizon (years)")
    
    # Market data parameters (indexed by hour)
    lmp_dict = {h: market_data.lmp[h-1] for h in model.hours}
    gas_price_dict = {h: market_data.gas_price[h-1] for h in model.hours}
    solar_cf_dict = {h: market_data.solar_cf[h-1] for h in model.hours}
    carbon_intensity_dict = {h: market_data.grid_carbon_intensity[h-1] for h in model.hours}
    
    model.lmp = pyo.Param(model.hours, initialize=lmp_dict, doc="Locational marginal price ($/MWh)")
    model.gas_price = pyo.Param(model.hours, initialize=gas_price_dict, doc="Natural gas price ($/MMBtu)")
    model.solar_cf = pyo.Param(model.hours, initialize=solar_cf_dict, doc="Solar capacity factor")
    model.grid_carbon = pyo.Param(model.hours, initialize=carbon_intensity_dict, doc="Grid carbon intensity (kg CO2/MWh)")
    
    # Technology cost parameters
    model.grid_capex = pyo.Param(initialize=tech_costs.grid_capex_per_kw, doc="Grid CAPEX ($/kW)")
    model.gas_capex = pyo.Param(initialize=tech_costs.gas_capex_per_kw, doc="Gas CAPEX ($/kW)")
    model.battery_capex = pyo.Param(initialize=tech_costs.battery_capex_per_kwh, doc="Battery CAPEX ($/kWh)")
    model.solar_capex = pyo.Param(initialize=tech_costs.solar_capex_per_kw, doc="Solar CAPEX ($/kW)")
    
    model.gas_vom = pyo.Param(initialize=tech_costs.gas_variable_om, doc="Gas variable O&M ($/MWh)")
    model.gas_heat_rate = pyo.Param(initialize=tech_costs.gas_heat_rate, doc="Gas heat rate (MMBtu/MWh)")
    model.battery_degradation = pyo.Param(initialize=tech_costs.battery_degradation, doc="Battery degradation ($/MWh)")
    model.battery_efficiency = pyo.Param(initialize=tech_costs.battery_efficiency, doc="Battery efficiency")
    model.solar_om = pyo.Param(initialize=tech_costs.solar_fixed_om, doc="Solar O&M ($/kW-year)")
    model.demand_charge = pyo.Param(initialize=tech_costs.grid_demand_charge, doc="Demand charge ($/kW-month)")
    
    model.curtailment_penalty = pyo.Param(initialize=facility_params.curtailment_penalty, doc="Curtailment penalty ($/MWh)")
    model.max_curtailment = pyo.Param(initialize=facility_params.max_annual_curtailment_mwh(), doc="Max annual curtailment (MWh)")
    
    # Decision Variables - Capacity
    model.C_grid = pyo.Var(domain=pyo.NonNegativeReals, doc="Grid interconnection capacity (MW)")
    model.C_gas = pyo.Var(domain=pyo.NonNegativeReals, doc="Gas peaker capacity (MW)")
    model.C_battery = pyo.Var(domain=pyo.NonNegativeReals, doc="Battery storage capacity (MWh)")
    model.C_solar = pyo.Var(domain=pyo.NonNegativeReals, doc="Solar PV capacity (MW)")
    
    # Decision Variables - Hourly Dispatch
    model.p_grid = pyo.Var(model.hours, domain=pyo.NonNegativeReals, doc="Grid power draw (MW)")
    model.p_gas = pyo.Var(model.hours, domain=pyo.NonNegativeReals, doc="Gas generation (MW)")
    model.p_solar = pyo.Var(model.hours, domain=pyo.NonNegativeReals, doc="Solar generation (MW)")
    model.p_battery = pyo.Var(model.hours, domain=pyo.Reals, doc="Battery power (MW, +charge/-discharge)")
    model.p_curtail = pyo.Var(model.hours, domain=pyo.NonNegativeReals, doc="Load curtailment (MW)")
    model.SOC = pyo.Var(model.hours, domain=pyo.NonNegativeReals, doc="Battery state of charge (MWh)")
    
    # Auxiliary variables for absolute value of battery power (for degradation cost)
    model.p_battery_abs = pyo.Var(model.hours, domain=pyo.NonNegativeReals, doc="Absolute battery power (MW)")
    
    # Auxiliary variable for demand charge calculation
    model.peak_grid_draw = pyo.Var(domain=pyo.NonNegativeReals, doc="Peak monthly grid draw (MW)")
    
    # Constraints to disable technologies if not allowed
    if not allow_gas:
        model.C_gas.fix(0)
    if not allow_battery:
        model.C_battery.fix(0)
    if not allow_solar:
        model.C_solar.fix(0)
    
    # Optional carbon budget constraint
    if facility_params.carbon_budget is not None:
        model.carbon_budget = pyo.Param(initialize=facility_params.carbon_budget, doc="Carbon budget (tons CO2/year)")
    
    # Objective Function: Minimize total 20-year cost
    def objective_rule(m):
        # CAPEX
        capex = (
            m.C_grid * m.grid_capex +
            m.C_gas * m.gas_capex +
            m.C_battery * m.battery_capex +
            m.C_solar * m.solar_capex
        )
        
        # Annual OPEX components
        # Grid electricity costs (energy charges)
        grid_energy_cost = sum(m.p_grid[h] * m.lmp[h] for h in m.hours)
        
        # Grid demand charges (monthly peak)
        grid_demand_cost = m.peak_grid_draw * m.demand_charge * 12
        
        # Gas fuel costs
        gas_fuel_cost = sum(m.p_gas[h] * m.gas_price[h] * m.gas_heat_rate for h in m.hours)
        
        # Gas O&M costs
        gas_om_cost = sum(m.p_gas[h] * m.gas_vom for h in m.hours)
        
        # Battery degradation costs (using absolute value variable)
        battery_degradation_cost = sum(m.p_battery_abs[h] * m.battery_degradation for h in m.hours)
        
        # Solar O&M costs
        solar_om_cost = m.C_solar * m.solar_om
        
        # Total annual OPEX
        opex_annual = (
            grid_energy_cost +
            grid_demand_cost +
            gas_fuel_cost +
            gas_om_cost +
            battery_degradation_cost +
            solar_om_cost
        )
        
        # NPV of OPEX over planning horizon
        npv_factor = sum(1 / (1 + m.discount_rate)**year for year in range(1, int(pyo.value(m.planning_horizon)) + 1))
        npv_opex = opex_annual * npv_factor
        
        # Curtailment penalty (NPV)
        curtailment_cost_annual = sum(m.p_curtail[h] * m.curtailment_penalty for h in m.hours)
        npv_curtailment = curtailment_cost_annual * npv_factor
        
        # Total cost
        return capex + npv_opex + npv_curtailment
    
    model.total_cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize, doc="Minimize total 20-year cost")
    
    # Constraint 1: Energy Balance (for all hours)
    def energy_balance_rule(m, h):
        return m.p_grid[h] + m.p_gas[h] + m.p_solar[h] - m.p_battery[h] + m.p_curtail[h] == m.dc_load
    
    model.energy_balance = pyo.Constraint(model.hours, rule=energy_balance_rule, doc="Energy balance")
    
    # Constraint 2: Grid capacity limit
    def grid_capacity_rule(m, h):
        return m.p_grid[h] <= m.C_grid
    
    model.grid_capacity = pyo.Constraint(model.hours, rule=grid_capacity_rule, doc="Grid capacity limit")
    
    # Constraint 3: Gas capacity limit
    def gas_capacity_rule(m, h):
        return m.p_gas[h] <= m.C_gas
    
    model.gas_capacity = pyo.Constraint(model.hours, rule=gas_capacity_rule, doc="Gas capacity limit")
    
    # Constraint 4: Battery power limits (4-hour battery: ±25% per hour)
    def battery_charge_limit_rule(m, h):
        return m.p_battery[h] <= 0.25 * m.C_battery
    
    def battery_discharge_limit_rule(m, h):
        return m.p_battery[h] >= -0.25 * m.C_battery
    
    model.battery_charge_limit = pyo.Constraint(model.hours, rule=battery_charge_limit_rule, doc="Battery charge limit")
    model.battery_discharge_limit = pyo.Constraint(model.hours, rule=battery_discharge_limit_rule, doc="Battery discharge limit")
    
    # Constraint 5: Solar generation
    def solar_generation_rule(m, h):
        return m.p_solar[h] == m.C_solar * m.solar_cf[h]
    
    model.solar_generation = pyo.Constraint(model.hours, rule=solar_generation_rule, doc="Solar generation")
    
    # Constraint 6: Battery dynamics (SOC evolution)
    def battery_dynamics_rule(m, h):
        if h == 1:
            # For first hour, use last hour for periodicity (will be enforced separately)
            return pyo.Constraint.Skip
        else:
            # Positive p_battery = charging (increases SOC)
            # Negative p_battery = discharging (decreases SOC)
            # Apply efficiency: charging has efficiency loss, discharging has efficiency loss
            return m.SOC[h] == m.SOC[h-1] + m.p_battery[h] * m.battery_efficiency
    
    model.battery_dynamics = pyo.Constraint(model.hours, rule=battery_dynamics_rule, doc="Battery dynamics")
    
    # Constraint 7: Battery SOC limits (10% to 90% for battery health)
    def battery_soc_min_rule(m, h):
        return m.SOC[h] >= 0.1 * m.C_battery
    
    def battery_soc_max_rule(m, h):
        return m.SOC[h] <= 0.9 * m.C_battery
    
    model.battery_soc_min = pyo.Constraint(model.hours, rule=battery_soc_min_rule, doc="Battery SOC minimum")
    model.battery_soc_max = pyo.Constraint(model.hours, rule=battery_soc_max_rule, doc="Battery SOC maximum")
    
    # Constraint 8: Battery periodicity (SOC at end equals SOC at start)
    def battery_periodicity_rule(m):
        # SOC[1] is determined by SOC[8760] + p_battery[1]
        return m.SOC[1] == m.SOC[8760] + m.p_battery[1] * m.battery_efficiency
    
    model.battery_periodicity = pyo.Constraint(rule=battery_periodicity_rule, doc="Battery periodicity")
    
    # Constraint 9: Gas ramping constraint (50% of capacity per hour)
    def gas_ramp_up_rule(m, h):
        if h == 1:
            return pyo.Constraint.Skip
        return m.p_gas[h] - m.p_gas[h-1] <= 0.5 * m.C_gas
    
    def gas_ramp_down_rule(m, h):
        if h == 1:
            return pyo.Constraint.Skip
        return m.p_gas[h-1] - m.p_gas[h] <= 0.5 * m.C_gas
    
    model.gas_ramp_up = pyo.Constraint(model.hours, rule=gas_ramp_up_rule, doc="Gas ramp up limit")
    model.gas_ramp_down = pyo.Constraint(model.hours, rule=gas_ramp_down_rule, doc="Gas ramp down limit")
    
    # Constraint 10: Absolute value constraints for battery power (for degradation cost)
    def battery_abs_positive_rule(m, h):
        return m.p_battery_abs[h] >= m.p_battery[h]
    
    def battery_abs_negative_rule(m, h):
        return m.p_battery_abs[h] >= -m.p_battery[h]
    
    model.battery_abs_positive = pyo.Constraint(model.hours, rule=battery_abs_positive_rule, doc="Battery abs value (positive)")
    model.battery_abs_negative = pyo.Constraint(model.hours, rule=battery_abs_negative_rule, doc="Battery abs value (negative)")
    
    # Constraint 11: Reliability constraint (total annual curtailment limit)
    def reliability_rule(m):
        return sum(m.p_curtail[h] for h in m.hours) <= m.max_curtailment
    
    model.reliability = pyo.Constraint(rule=reliability_rule, doc="Reliability constraint")
    
    # Constraint 12: Peak grid draw for demand charge
    def peak_grid_draw_rule(m, h):
        return m.peak_grid_draw >= m.p_grid[h]
    
    model.peak_grid_draw_constraint = pyo.Constraint(model.hours, rule=peak_grid_draw_rule, doc="Peak grid draw")
    
    # Constraint 13: Optional carbon budget constraint
    if facility_params.carbon_budget is not None:
        def carbon_budget_rule(m):
            # Grid emissions + gas emissions (in tons CO2)
            grid_emissions = sum(m.p_grid[h] * m.grid_carbon[h] / 1000 for h in m.hours)  # Convert kg to tons
            gas_emissions = sum(m.p_gas[h] * 0.4 for h in m.hours)  # 0.4 tons CO2/MWh
            return grid_emissions + gas_emissions <= m.carbon_budget
        
        model.carbon_constraint = pyo.Constraint(rule=carbon_budget_rule, doc="Carbon budget constraint")
    
    return model
