"""
Solver interface for data center energy optimization model.

This module provides functions to solve Pyomo optimization models using Gurobi
with appropriate configuration and comprehensive error handling.
"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition, SolverResults
import time
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

__all__ = [
    'solve_model',
    'SolverError',
    'InfeasibleError',
    'UnboundedError',
    'NumericalError',
    'TimeoutError'
]

# Set Gurobi license file location
import os
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
_license_path = _project_root / "config" / "gurobi.lic"
if _license_path.exists():
    os.environ['GRB_LICENSE_FILE'] = str(_license_path)
    logger.debug(f"Using Gurobi license file: {_license_path}")

# Try to load Gurobi WLS configuration (alternative to .lic file)
try:
    from src.optimization.gurobi_config import load_gurobi_wls_config
    load_gurobi_wls_config()
except Exception as e:
    logger.debug(f"Could not load Gurobi WLS config: {e}")


class SolverError(Exception):
    """Base exception for solver-related errors."""
    pass


class InfeasibleError(SolverError):
    """Raised when the optimization problem is infeasible."""
    pass


class UnboundedError(SolverError):
    """Raised when the optimization problem is unbounded."""
    pass


class NumericalError(SolverError):
    """Raised when the solver encounters numerical difficulties."""
    pass


class TimeoutError(SolverError):
    """Raised when the solver exceeds the time limit."""
    pass


def solve_model(
    model: pyo.ConcreteModel,
    solver_options: Optional[Dict] = None,
    time_limit: int = 1800,
    mip_gap: float = 0.005,
    verbose: bool = True,
    solver_name: str = 'gurobi'
) -> Tuple[SolverResults, float]:
    """
    Solve optimization model using Gurobi solver.
    
    Configures Gurobi with appropriate settings for the data center energy
    optimization problem and handles various solver error conditions.
    
    Args:
        model: Pyomo ConcreteModel to solve
        solver_options: Optional dictionary of additional solver options
        time_limit: Maximum solve time in seconds (default: 1800 = 30 minutes)
        mip_gap: MIP optimality gap tolerance (default: 0.005 = 0.5%)
        verbose: Whether to print solver output (default: True)
        solver_name: Name of solver to use (default: 'gurobi', alternatives: 'glpk', 'cbc')
        
    Returns:
        Tuple of (solver_results, solve_time_seconds)
        
    Raises:
        InfeasibleError: If the problem has no feasible solution
        UnboundedError: If the objective can decrease indefinitely
        NumericalError: If the solver encounters numerical difficulties
        TimeoutError: If the solver exceeds the time limit without finding optimal solution
        SolverError: For other solver-related errors
        
    Example:
        >>> model = build_optimization_model(data, costs, params)
        >>> results, solve_time = solve_model(model)
        >>> print(f"Solved in {solve_time:.2f} seconds")
        >>> print(f"Optimal cost: ${pyo.value(model.total_cost):,.0f}")
    """
    # Create solver instance
    try:
        solver = pyo.SolverFactory(solver_name)
    except Exception as e:
        raise SolverError(f"Failed to create {solver_name} solver: {e}")
    
    # Check if solver is available
    if not solver.available():
        raise SolverError(
            f"{solver_name} solver is not available. "
            f"Please ensure {solver_name} is installed and licensed."
        )
    
    # Configure solver options based on solver type
    if solver_name == 'gurobi':
        options = {
            'MIPGap': mip_gap,
            'TimeLimit': time_limit,
            'Threads': 0,  # 0 = automatic thread selection
            'Method': 2,   # 2 = barrier method for LP problems
            'Crossover': 0,  # 0 = automatic crossover decision
        }
    elif solver_name == 'glpk':
        options = {
            'tmlim': time_limit,  # Time limit in seconds
            'mipgap': mip_gap,    # MIP gap tolerance
        }
    elif solver_name == 'cbc':
        options = {
            'seconds': time_limit,
            'ratioGap': mip_gap,
        }
    else:
        # Generic options
        options = {}
    
    # Merge with user-provided options (user options take precedence)
    if solver_options:
        options.update(solver_options)
    
    # Set solver options
    for key, value in options.items():
        solver.options[key] = value
    
    # Configure output
    if verbose:
        logger.info("Starting optimization solve...")
        logger.info(f"Solver configuration: MIPGap={mip_gap}, TimeLimit={time_limit}s")
    
    # Solve the model
    start_time = time.time()
    
    try:
        results = solver.solve(model, tee=verbose)
    except Exception as e:
        raise SolverError(f"Solver execution failed: {e}")
    
    solve_time = time.time() - start_time
    
    # Check solver status and termination condition
    solver_status = results.solver.status
    termination_condition = results.solver.termination_condition
    
    if verbose:
        logger.info(f"Solve completed in {solve_time:.2f} seconds")
        logger.info(f"Solver status: {solver_status}")
        logger.info(f"Termination condition: {termination_condition}")
    
    # Handle different termination conditions
    if termination_condition == TerminationCondition.infeasible:
        error_msg = (
            "Optimization problem is infeasible. "
            "No solution exists that satisfies all constraints. "
            "Possible causes:\n"
            "  - Reliability target is too strict for available capacity\n"
            "  - Carbon budget is too restrictive\n"
            "  - Conflicting constraints between energy balance and capacity limits\n"
            "Consider relaxing reliability target or carbon constraints."
        )
        raise InfeasibleError(error_msg)
    
    elif termination_condition == TerminationCondition.unbounded:
        error_msg = (
            "Optimization problem is unbounded. "
            "The objective can decrease indefinitely. "
            "Possible causes:\n"
            "  - Missing capacity upper bounds\n"
            "  - Negative or zero cost parameters\n"
            "  - Model formulation error\n"
            "Please check model constraints and cost parameters."
        )
        raise UnboundedError(error_msg)
    
    elif termination_condition == TerminationCondition.infeasibleOrUnbounded:
        error_msg = (
            "Optimization problem is either infeasible or unbounded. "
            "Gurobi could not determine which. "
            "Try solving with different solver settings or check model formulation."
        )
        raise SolverError(error_msg)
    
    elif termination_condition == TerminationCondition.maxTimeLimit:
        # Check if we have a feasible solution
        if solver_status == SolverStatus.ok and hasattr(results.problem, 'upper_bound'):
            # We have a feasible solution but didn't reach optimality
            gap = _calculate_optimality_gap(results)
            warning_msg = (
                f"Solver reached time limit of {time_limit} seconds. "
                f"Returning best solution found with optimality gap of {gap:.2%}. "
                f"Solution may not be optimal but is feasible."
            )
            logger.warning(warning_msg)
            # Return the best solution found
            return results, solve_time
        else:
            # No feasible solution found within time limit
            error_msg = (
                f"Solver exceeded time limit of {time_limit} seconds "
                "without finding a feasible solution. "
                "Consider:\n"
                "  - Increasing time limit\n"
                "  - Relaxing constraints\n"
                "  - Simplifying the problem (e.g., fewer hours)"
            )
            raise TimeoutError(error_msg)
    
    elif hasattr(TerminationCondition, 'numericalDifficulties') and termination_condition == TerminationCondition.numericalDifficulties:
        error_msg = (
            "Solver encountered numerical difficulties. "
            "Possible causes:\n"
            "  - Poor scaling of variables or constraints\n"
            "  - Very large or very small coefficient values\n"
            "  - Ill-conditioned constraint matrix\n"
            "Suggestions:\n"
            "  - Scale variables (e.g., use GW instead of MW)\n"
            "  - Tighten feasibility tolerances\n"
            "  - Try alternative solver methods"
        )
        raise NumericalError(error_msg)
    
    elif termination_condition == TerminationCondition.other:
        error_msg = (
            f"Solver terminated with unexpected condition: {termination_condition}. "
            f"Solver status: {solver_status}. "
            "Check solver log for details."
        )
        raise SolverError(error_msg)
    
    elif termination_condition != TerminationCondition.optimal:
        # Handle any other non-optimal termination
        error_msg = (
            f"Solver did not find optimal solution. "
            f"Termination condition: {termination_condition}, "
            f"Solver status: {solver_status}"
        )
        raise SolverError(error_msg)
    
    # Check solver status for optimal solution
    if solver_status != SolverStatus.ok:
        error_msg = (
            f"Solver status is not OK: {solver_status}. "
            f"Termination condition: {termination_condition}"
        )
        raise SolverError(error_msg)
    
    # Solution is optimal
    if verbose:
        objective_value = pyo.value(model.total_cost)
        logger.info(f"Optimal solution found!")
        logger.info(f"Objective value: ${objective_value:,.0f}")
        
        # Report optimality gap if available
        gap = _calculate_optimality_gap(results)
        if gap is not None:
            logger.info(f"Optimality gap: {gap:.4%}")
    
    return results, solve_time


def _calculate_optimality_gap(results: SolverResults) -> Optional[float]:
    """
    Calculate the optimality gap from solver results.
    
    Args:
        results: Pyomo solver results
        
    Returns:
        Optimality gap as a fraction (e.g., 0.005 for 0.5%), or None if not available
    """
    try:
        if hasattr(results.problem, 'upper_bound') and hasattr(results.problem, 'lower_bound'):
            upper = results.problem.upper_bound
            lower = results.problem.lower_bound
            
            if upper is not None and lower is not None and abs(lower) > 1e-10:
                gap = abs(upper - lower) / abs(lower)
                return gap
    except Exception:
        pass
    
    return None


def get_solver_info() -> Dict[str, Any]:
    """
    Get information about the Gurobi solver installation.
    
    Returns:
        Dictionary with solver availability and version information
        
    Example:
        >>> info = get_solver_info()
        >>> print(f"Gurobi available: {info['available']}")
        >>> print(f"Version: {info['version']}")
    """
    try:
        solver = pyo.SolverFactory('gurobi')
        available = solver.available()
        
        info = {
            'available': available,
            'solver_name': 'gurobi',
        }
        
        if available:
            try:
                # Try to get version information
                version = solver.version()
                info['version'] = version
            except Exception:
                info['version'] = 'unknown'
        
        return info
        
    except Exception as e:
        return {
            'available': False,
            'solver_name': 'gurobi',
            'error': str(e)
        }


def validate_solution(
    model: pyo.ConcreteModel,
    tolerance: float = 1e-6
) -> Tuple[bool, list]:
    """
    Validate that the solution satisfies all constraints.
    
    This function checks that all constraints in the solved model are satisfied
    within the specified numerical tolerance.
    
    Args:
        model: Solved Pyomo ConcreteModel
        tolerance: Numerical tolerance for constraint violations (default: 1e-6)
        
    Returns:
        Tuple of (is_valid, list_of_violations)
        - is_valid: True if all constraints satisfied, False otherwise
        - list_of_violations: List of constraint names that are violated
        
    Example:
        >>> results, _ = solve_model(model)
        >>> is_valid, violations = validate_solution(model)
        >>> if not is_valid:
        ...     print(f"Constraint violations: {violations}")
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
                            f"body={body_value:.6f} < lower={lower_value:.6f}"
                        )
                        violations.append(violation_msg)
                
                # Check upper bound
                if constraint_obj.upper is not None:
                    upper_value = pyo.value(constraint_obj.upper)
                    if body_value > upper_value + tolerance:
                        violation_msg = (
                            f"{constraint.name}[{index}]: "
                            f"body={body_value:.6f} > upper={upper_value:.6f}"
                        )
                        violations.append(violation_msg)
                        
            except Exception as e:
                violations.append(f"{constraint.name}[{index}]: Error checking constraint - {e}")
    
    is_valid = len(violations) == 0
    return is_valid, violations
