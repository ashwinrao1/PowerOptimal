"""
Solution validation module for data center energy optimization.

This module verifies that optimization solutions satisfy all physical and
operational constraints with appropriate numerical tolerances.
"""

import pyomo.environ as pyo
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

# Use absolute imports to avoid issues when imported from different contexts
try:
    from ..models.solution import OptimizationSolution
    from ..models.technology import TechnologyCosts, FacilityParams
except ImportError:
    from models.solution import OptimizationSolution
    from models.technology import TechnologyCosts, FacilityParams

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when solution validation fails."""
    pass


def validate_solution(
    solution: OptimizationSolution,
    facility_params: FacilityParams,
    tech_costs: TechnologyCosts,
    tolerance: float = 1e-4
) -> Tuple[bool, List[str]]:
    """
    Validate that solution satisfies all constraints.
    
    Performs comprehensive validation of:
    - Energy balance at every hour
    - Capacity limits
    - Battery dynamics and SOC limits
    - Gas ramping constraints
    - Reliability constraint
    - Physical feasibility
    
    Args:
        solution: Optimization solution to validate
        facility_params: Facility parameters
        tech_costs: Technology cost parameters
        tolerance: Numerical tolerance for constraint violations
        
    Returns:
        Tuple of (is_valid, list_of_violations)
        - is_valid: True if all constraints satisfied, False otherwise
        - list_of_violations: List of violation descriptions
        
    Example:
        >>> is_valid, violations = validate_solution(solution, params, costs)
        >>> if not is_valid:
        ...     for violation in violations:
        ...         print(f"Violation: {violation}")
    """
    logger.info("Validating solution...")
    violations = []
    
    # Validate energy balance
    energy_violations = _validate_energy_balance(
        solution=solution,
        facility_load_mw=facility_params.total_load_mw,
        tolerance=tolerance
    )
    violations.extend(energy_violations)
    
    # Validate capacity limits
    capacity_violations = _validate_capacity_limits(
        solution=solution,
        tolerance=tolerance
    )
    violations.extend(capacity_violations)
    
    # Validate battery constraints
    battery_violations = _validate_battery_constraints(
        solution=solution,
        tech_costs=tech_costs,
        tolerance=tolerance
    )
    violations.extend(battery_violations)
    
    # Validate gas ramping
    gas_violations = _validate_gas_ramping(
        solution=solution,
        tolerance=tolerance
    )
    violations.extend(gas_violations)
    
    # Validate reliability constraint
    reliability_violations = _validate_reliability(
        solution=solution,
        facility_params=facility_params,
        tolerance=tolerance
    )
    violations.extend(reliability_violations)
    
    # Validate non-negativity
    nonnegativity_violations = _validate_nonnegativity(
        solution=solution,
        tolerance=tolerance
    )
    violations.extend(nonnegativity_violations)
    
    is_valid = len(violations) == 0
    
    if is_valid:
        logger.info("Solution validation passed - all constraints satisfied")
    else:
        logger.warning(f"Solution validation failed - {len(violations)} violations found")
        for violation in violations[:5]:  # Log first 5 violations
            logger.warning(f"  {violation}")
        if len(violations) > 5:
            logger.warning(f"  ... and {len(violations) - 5} more violations")
    
    return is_valid, violations


def _validate_energy_balance(
    solution: OptimizationSolution,
    facility_load_mw: float,
    tolerance: float
) -> List[str]:
    """
    Validate energy balance at every hour.
    
    Constraint: grid + gas + solar - battery + curtailment = load
    
    Args:
        solution: Optimization solution
        facility_load_mw: Facility load in MW
        tolerance: Numerical tolerance
        
    Returns:
        List of violation descriptions
    """
    violations = []
    dispatch = solution.dispatch
    
    for i in range(len(dispatch.hour)):
        hour = dispatch.hour[i]
        
        # Calculate supply
        supply = (
            dispatch.grid_power[i] +
            dispatch.gas_power[i] +
            dispatch.solar_power[i] -
            dispatch.battery_power[i] +
            dispatch.curtailment[i]
        )
        
        # Check balance
        imbalance = abs(supply - facility_load_mw)
        
        if imbalance > tolerance:
            violations.append(
                f"Energy balance violation at hour {hour}: "
                f"supply={supply:.4f} MW, demand={facility_load_mw:.4f} MW, "
                f"imbalance={imbalance:.4f} MW"
            )
    
    return violations


def _validate_capacity_limits(
    solution: OptimizationSolution,
    tolerance: float
) -> List[str]:
    """
    Validate that dispatch does not exceed installed capacity.
    
    Args:
        solution: Optimization solution
        tolerance: Numerical tolerance
        
    Returns:
        List of violation descriptions
    """
    violations = []
    capacity = solution.capacity
    dispatch = solution.dispatch
    
    for i in range(len(dispatch.hour)):
        hour = dispatch.hour[i]
        
        # Grid capacity
        if dispatch.grid_power[i] > capacity.grid_mw + tolerance:
            violations.append(
                f"Grid capacity violation at hour {hour}: "
                f"power={dispatch.grid_power[i]:.4f} MW > "
                f"capacity={capacity.grid_mw:.4f} MW"
            )
        
        # Gas capacity
        if dispatch.gas_power[i] > capacity.gas_mw + tolerance:
            violations.append(
                f"Gas capacity violation at hour {hour}: "
                f"power={dispatch.gas_power[i]:.4f} MW > "
                f"capacity={capacity.gas_mw:.4f} MW"
            )
        
        # Solar capacity (with capacity factor)
        # Solar power should equal capacity * CF, but allow small tolerance
        # This is more of a consistency check than a constraint violation
        
        # Battery power limits (±25% of capacity per hour for 4-hour battery)
        max_battery_power = 0.25 * capacity.battery_mwh
        if dispatch.battery_power[i] > max_battery_power + tolerance:
            violations.append(
                f"Battery charge limit violation at hour {hour}: "
                f"power={dispatch.battery_power[i]:.4f} MW > "
                f"limit={max_battery_power:.4f} MW"
            )
        
        if dispatch.battery_power[i] < -max_battery_power - tolerance:
            violations.append(
                f"Battery discharge limit violation at hour {hour}: "
                f"power={dispatch.battery_power[i]:.4f} MW < "
                f"limit={-max_battery_power:.4f} MW"
            )
    
    return violations


def _validate_battery_constraints(
    solution: OptimizationSolution,
    tech_costs: TechnologyCosts,
    tolerance: float
) -> List[str]:
    """
    Validate battery dynamics and SOC limits.
    
    Args:
        solution: Optimization solution
        tech_costs: Technology cost parameters
        tolerance: Numerical tolerance
        
    Returns:
        List of violation descriptions
    """
    violations = []
    capacity = solution.capacity
    dispatch = solution.dispatch
    
    if capacity.battery_mwh < 1e-6:
        # No battery, skip validation
        return violations
    
    # Validate SOC limits (10% to 90%)
    min_soc = 0.1 * capacity.battery_mwh
    max_soc = 0.9 * capacity.battery_mwh
    
    for i in range(len(dispatch.hour)):
        hour = dispatch.hour[i]
        soc = dispatch.battery_soc[i]
        
        if soc < min_soc - tolerance:
            violations.append(
                f"Battery SOC minimum violation at hour {hour}: "
                f"SOC={soc:.4f} MWh < min={min_soc:.4f} MWh"
            )
        
        if soc > max_soc + tolerance:
            violations.append(
                f"Battery SOC maximum violation at hour {hour}: "
                f"SOC={soc:.4f} MWh > max={max_soc:.4f} MWh"
            )
    
    # Validate battery dynamics (SOC evolution)
    efficiency = tech_costs.battery_efficiency
    
    for i in range(1, len(dispatch.hour)):
        hour = dispatch.hour[i]
        prev_soc = dispatch.battery_soc[i-1]
        curr_soc = dispatch.battery_soc[i]
        battery_power = dispatch.battery_power[i]
        
        # Expected SOC based on previous SOC and battery power
        expected_soc = prev_soc + battery_power * efficiency
        
        soc_error = abs(curr_soc - expected_soc)
        
        if soc_error > tolerance:
            violations.append(
                f"Battery dynamics violation at hour {hour}: "
                f"SOC={curr_soc:.4f} MWh, expected={expected_soc:.4f} MWh, "
                f"error={soc_error:.4f} MWh"
            )
    
    # Validate battery periodicity (SOC at end should match SOC at start)
    first_soc = dispatch.battery_soc[0]
    last_soc = dispatch.battery_soc[-1]
    first_power = dispatch.battery_power[0]
    
    expected_first_soc = last_soc + first_power * efficiency
    periodicity_error = abs(first_soc - expected_first_soc)
    
    if periodicity_error > tolerance:
        violations.append(
            f"Battery periodicity violation: "
            f"SOC[1]={first_soc:.4f} MWh, "
            f"expected from SOC[8760]={expected_first_soc:.4f} MWh, "
            f"error={periodicity_error:.4f} MWh"
        )
    
    return violations


def _validate_gas_ramping(
    solution: OptimizationSolution,
    tolerance: float
) -> List[str]:
    """
    Validate gas ramping constraints (50% of capacity per hour).
    
    Args:
        solution: Optimization solution
        tolerance: Numerical tolerance
        
    Returns:
        List of violation descriptions
    """
    violations = []
    capacity = solution.capacity
    dispatch = solution.dispatch
    
    if capacity.gas_mw < 1e-6:
        # No gas capacity, skip validation
        return violations
    
    max_ramp = 0.5 * capacity.gas_mw
    
    for i in range(1, len(dispatch.hour)):
        hour = dispatch.hour[i]
        prev_gas = dispatch.gas_power[i-1]
        curr_gas = dispatch.gas_power[i]
        
        ramp = curr_gas - prev_gas
        
        if ramp > max_ramp + tolerance:
            violations.append(
                f"Gas ramp up violation at hour {hour}: "
                f"ramp={ramp:.4f} MW > limit={max_ramp:.4f} MW"
            )
        
        if ramp < -max_ramp - tolerance:
            violations.append(
                f"Gas ramp down violation at hour {hour}: "
                f"ramp={ramp:.4f} MW < limit={-max_ramp:.4f} MW"
            )
    
    return violations


def _validate_reliability(
    solution: OptimizationSolution,
    facility_params: FacilityParams,
    tolerance: float
) -> List[str]:
    """
    Validate reliability constraint (total annual curtailment limit).
    
    Args:
        solution: Optimization solution
        facility_params: Facility parameters
        tolerance: Numerical tolerance
        
    Returns:
        List of violation descriptions
    """
    violations = []
    
    max_curtailment = facility_params.max_annual_curtailment_mwh()
    total_curtailment = solution.metrics.total_curtailment_mwh
    
    if total_curtailment > max_curtailment + tolerance:
        violations.append(
            f"Reliability constraint violation: "
            f"total curtailment={total_curtailment:.4f} MWh > "
            f"limit={max_curtailment:.4f} MWh "
            f"(reliability target={facility_params.reliability_target*100:.4f}%)"
        )
    
    return violations


def _validate_nonnegativity(
    solution: OptimizationSolution,
    tolerance: float
) -> List[str]:
    """
    Validate that all variables are non-negative (where required).
    
    Args:
        solution: Optimization solution
        tolerance: Numerical tolerance (allow small negative values due to numerics)
        
    Returns:
        List of violation descriptions
    """
    violations = []
    capacity = solution.capacity
    dispatch = solution.dispatch
    
    # Capacity variables must be non-negative
    if capacity.grid_mw < -tolerance:
        violations.append(f"Negative grid capacity: {capacity.grid_mw:.4f} MW")
    
    if capacity.gas_mw < -tolerance:
        violations.append(f"Negative gas capacity: {capacity.gas_mw:.4f} MW")
    
    if capacity.battery_mwh < -tolerance:
        violations.append(f"Negative battery capacity: {capacity.battery_mwh:.4f} MWh")
    
    if capacity.solar_mw < -tolerance:
        violations.append(f"Negative solar capacity: {capacity.solar_mw:.4f} MW")
    
    # Dispatch variables (except battery power which can be negative)
    for i in range(len(dispatch.hour)):
        hour = dispatch.hour[i]
        
        if dispatch.grid_power[i] < -tolerance:
            violations.append(
                f"Negative grid power at hour {hour}: {dispatch.grid_power[i]:.4f} MW"
            )
        
        if dispatch.gas_power[i] < -tolerance:
            violations.append(
                f"Negative gas power at hour {hour}: {dispatch.gas_power[i]:.4f} MW"
            )
        
        if dispatch.solar_power[i] < -tolerance:
            violations.append(
                f"Negative solar power at hour {hour}: {dispatch.solar_power[i]:.4f} MW"
            )
        
        if dispatch.curtailment[i] < -tolerance:
            violations.append(
                f"Negative curtailment at hour {hour}: {dispatch.curtailment[i]:.4f} MW"
            )
        
        if dispatch.battery_soc[i] < -tolerance:
            violations.append(
                f"Negative battery SOC at hour {hour}: {dispatch.battery_soc[i]:.4f} MWh"
            )
    
    return violations


def validate_model_constraints(
    model: pyo.ConcreteModel,
    tolerance: float = 1e-6
) -> Tuple[bool, List[str]]:
    """
    Validate that all Pyomo model constraints are satisfied.
    
    This is a lower-level validation that checks the Pyomo model directly,
    useful for debugging model formulation issues.
    
    Args:
        model: Solved Pyomo ConcreteModel
        tolerance: Numerical tolerance for constraint violations
        
    Returns:
        Tuple of (is_valid, list_of_violations)
        
    Example:
        >>> results, _ = solve_model(model)
        >>> is_valid, violations = validate_model_constraints(model)
    """
    violations = []
    
    # Check all constraints
    for constraint in model.component_objects(pyo.Constraint, active=True):
        for index in constraint:
            try:
                constraint_obj = constraint[index] if index is not None else constraint
                
                # Get constraint body value
                body_value = pyo.value(constraint_obj.body)
                
                # Check lower bound
                if constraint_obj.lower is not None:
                    lower_value = pyo.value(constraint_obj.lower)
                    if body_value < lower_value - tolerance:
                        violation_msg = (
                            f"{constraint.name}[{index}]: "
                            f"body={body_value:.6f} < lower={lower_value:.6f} "
                            f"(violation={lower_value - body_value:.6f})"
                        )
                        violations.append(violation_msg)
                
                # Check upper bound
                if constraint_obj.upper is not None:
                    upper_value = pyo.value(constraint_obj.upper)
                    if body_value > upper_value + tolerance:
                        violation_msg = (
                            f"{constraint.name}[{index}]: "
                            f"body={body_value:.6f} > upper={upper_value:.6f} "
                            f"(violation={body_value - upper_value:.6f})"
                        )
                        violations.append(violation_msg)
                        
            except Exception as e:
                violations.append(
                    f"{constraint.name}[{index}]: Error checking constraint - {e}"
                )
    
    is_valid = len(violations) == 0
    
    if is_valid:
        logger.info("Model constraint validation passed")
    else:
        logger.warning(f"Model constraint validation failed - {len(violations)} violations")
    
    return is_valid, violations


def generate_validation_report(
    solution: OptimizationSolution,
    facility_params: FacilityParams,
    tech_costs: TechnologyCosts,
    tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Generate comprehensive validation report.
    
    Args:
        solution: Optimization solution
        facility_params: Facility parameters
        tech_costs: Technology cost parameters
        tolerance: Numerical tolerance
        
    Returns:
        Dictionary with validation results and statistics
    """
    is_valid, violations = validate_solution(
        solution=solution,
        facility_params=facility_params,
        tech_costs=tech_costs,
        tolerance=tolerance
    )
    
    # Calculate validation statistics
    dispatch = solution.dispatch
    
    # Energy balance statistics
    supply = (
        dispatch.grid_power +
        dispatch.gas_power +
        dispatch.solar_power -
        dispatch.battery_power +
        dispatch.curtailment
    )
    demand = np.full(len(dispatch.hour), facility_params.total_load_mw)
    energy_balance_errors = np.abs(supply - demand)
    
    report = {
        'is_valid': is_valid,
        'num_violations': len(violations),
        'violations': violations,
        'tolerance': tolerance,
        'statistics': {
            'energy_balance': {
                'max_error': float(np.max(energy_balance_errors)),
                'mean_error': float(np.mean(energy_balance_errors)),
                'num_hours_with_error': int(np.sum(energy_balance_errors > tolerance))
            },
            'curtailment': {
                'total_mwh': solution.metrics.total_curtailment_mwh,
                'max_allowed_mwh': facility_params.max_annual_curtailment_mwh(),
                'num_hours': solution.metrics.num_curtailment_hours,
                'max_hourly_mw': float(np.max(dispatch.curtailment))
            },
            'battery': {
                'min_soc_mwh': float(np.min(dispatch.battery_soc)) if solution.capacity.battery_mwh > 0 else 0,
                'max_soc_mwh': float(np.max(dispatch.battery_soc)) if solution.capacity.battery_mwh > 0 else 0,
                'capacity_mwh': solution.capacity.battery_mwh
            }
        }
    }
    
    return report
