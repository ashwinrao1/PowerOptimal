"""
Batch solver for parallel scenario execution.

This module enables solving multiple optimization scenarios concurrently
using Python multiprocessing to leverage multiple CPU cores.
"""

import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Optional, Callable
import logging
import time
import traceback
from pathlib import Path
import pandas as pd

from ..optimization.model_builder import build_optimization_model
from ..optimization.solver import solve_model, SolverError
from ..optimization.solution_extractor import extract_solution
from ..models.market_data import MarketData
from ..models.technology import TechnologyCosts, FacilityParams
from ..models.solution import OptimizationSolution

logger = logging.getLogger(__name__)


class BatchSolverError(Exception):
    """Base exception for batch solver errors."""
    pass


def solve_scenarios(
    scenarios: List[Dict[str, Any]],
    market_data: MarketData,
    base_tech_costs: TechnologyCosts,
    base_facility_params: FacilityParams,
    n_workers: Optional[int] = None,
    save_solutions: bool = True,
    output_dir: str = "results/scenarios",
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Solve multiple scenarios in parallel using multiprocessing.
    
    This function distributes scenario solving across multiple CPU cores
    for faster execution. Each scenario is solved independently, and results
    are collected into a list of solution dictionaries.
    
    Args:
        scenarios: List of scenario parameter dictionaries from scenario_generator
        market_data: Base market data (will be modified per scenario)
        base_tech_costs: Base technology costs (will be modified per scenario)
        base_facility_params: Base facility parameters (will be modified per scenario)
        n_workers: Number of worker processes (default: CPU count - 1)
        save_solutions: Whether to save individual solution JSON files
        output_dir: Directory for saving solution files
        verbose: Whether to print progress information
        
    Returns:
        List of solution dictionaries with results from all scenarios
        
    Raises:
        BatchSolverError: If all scenarios fail to solve
        
    Example:
        >>> from src.analysis.scenario_generator import generate_scenarios
        >>> scenarios = generate_scenarios(gas_price_variations=[0.5, 1.0, 1.5])
        >>> results = solve_scenarios(scenarios, market_data, costs, params)
        >>> print(f"Solved {len(results)} scenarios")
    """
    if not scenarios:
        raise ValueError("No scenarios provided")
    
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    n_workers = min(n_workers, len(scenarios))
    
    if verbose:
        logger.info(f"Starting batch solve for {len(scenarios)} scenarios")
        logger.info(f"Using {n_workers} worker processes")
    
    # Create output directory if saving solutions
    if save_solutions:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare worker arguments
    worker_args = [
        (
            i,
            scenario,
            market_data,
            base_tech_costs,
            base_facility_params,
            save_solutions,
            output_dir,
            verbose
        )
        for i, scenario in enumerate(scenarios)
    ]
    
    # Solve scenarios in parallel
    start_time = time.time()
    
    if n_workers == 1:
        # Single-threaded execution for debugging
        results = [_solve_single_scenario(args) for args in worker_args]
    else:
        # Multi-threaded execution
        with Pool(processes=n_workers) as pool:
            results = pool.map(_solve_single_scenario, worker_args)
    
    total_time = time.time() - start_time
    
    # Separate successful and failed results
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'failed']
    
    if verbose:
        logger.info(f"Batch solve completed in {total_time:.2f} seconds")
        logger.info(f"Successful: {len(successful_results)}/{len(scenarios)}")
        logger.info(f"Failed: {len(failed_results)}/{len(scenarios)}")
        
        if failed_results:
            logger.warning("Failed scenarios:")
            for result in failed_results:
                logger.warning(f"  - {result['scenario_name']}: {result['error']}")
    
    # Check if all scenarios failed
    if not successful_results:
        raise BatchSolverError(
            f"All {len(scenarios)} scenarios failed to solve. "
            "Check solver configuration and scenario parameters."
        )
    
    return results


def _solve_single_scenario(args: tuple) -> Dict[str, Any]:
    """
    Solve a single scenario (worker function for multiprocessing).
    
    This function is designed to be called by multiprocessing.Pool.map().
    It handles all errors gracefully and returns a result dictionary
    with either the solution or error information.
    
    Args:
        args: Tuple of (scenario_index, scenario_params, market_data, 
                       tech_costs, facility_params, save_solutions, 
                       output_dir, verbose)
        
    Returns:
        Dictionary with scenario results or error information
    """
    (
        scenario_idx,
        scenario_params,
        market_data,
        base_tech_costs,
        base_facility_params,
        save_solutions,
        output_dir,
        verbose
    ) = args
    
    scenario_name = scenario_params.get('scenario_name', f'scenario_{scenario_idx}')
    
    try:
        # Apply scenario modifications to market data
        modified_market_data = _apply_market_data_modifications(
            market_data, scenario_params
        )
        
        # Apply scenario modifications to technology costs
        modified_tech_costs = _apply_tech_cost_modifications(
            base_tech_costs, scenario_params
        )
        
        # Apply scenario modifications to facility parameters
        modified_facility_params = _apply_facility_param_modifications(
            base_facility_params, scenario_params
        )
        
        # Build optimization model
        model = build_optimization_model(
            market_data=modified_market_data,
            tech_costs=modified_tech_costs,
            facility_params=modified_facility_params
        )
        
        # Solve model
        results, solve_time = solve_model(
            model=model,
            verbose=False  # Suppress individual solver output in batch mode
        )
        
        # Extract solution
        solution = extract_solution(
            model=model,
            market_data=modified_market_data,
            tech_costs=modified_tech_costs,
            facility_params=modified_facility_params,
            solve_time=solve_time,
            scenario_params=scenario_params
        )
        
        # Save solution if requested
        if save_solutions:
            output_path = Path(output_dir) / f"{scenario_name}.json"
            solution.save(str(output_path))
        
        # Create result dictionary
        result = {
            'status': 'success',
            'scenario_index': scenario_idx,
            'scenario_name': scenario_name,
            'scenario_params': scenario_params,
            'capacity': solution.capacity.to_dict(),
            'metrics': solution.metrics.to_dict(),
            'solve_time': solve_time
        }
        
        if verbose:
            logger.info(f"Solved scenario {scenario_idx + 1}: {scenario_name} "
                       f"(NPV: ${solution.metrics.total_npv:,.0f}, "
                       f"Time: {solve_time:.1f}s)")
        
        return result
        
    except SolverError as e:
        # Solver-specific errors
        error_msg = f"Solver error: {str(e)}"
        logger.error(f"Scenario {scenario_idx + 1} ({scenario_name}) failed: {error_msg}")
        
        return {
            'status': 'failed',
            'scenario_index': scenario_idx,
            'scenario_name': scenario_name,
            'scenario_params': scenario_params,
            'error': error_msg,
            'error_type': type(e).__name__
        }
        
    except Exception as e:
        # Unexpected errors
        error_msg = f"Unexpected error: {str(e)}"
        error_trace = traceback.format_exc()
        logger.error(f"Scenario {scenario_idx + 1} ({scenario_name}) failed: {error_msg}")
        logger.debug(f"Traceback:\n{error_trace}")
        
        return {
            'status': 'failed',
            'scenario_index': scenario_idx,
            'scenario_name': scenario_name,
            'scenario_params': scenario_params,
            'error': error_msg,
            'error_type': type(e).__name__,
            'traceback': error_trace
        }


def _apply_market_data_modifications(
    market_data: MarketData,
    scenario_params: Dict[str, Any]
) -> MarketData:
    """
    Apply scenario-specific modifications to market data.
    
    Args:
        market_data: Base market data
        scenario_params: Scenario parameters
        
    Returns:
        Modified MarketData object
    """
    import copy
    modified_data = copy.deepcopy(market_data)
    
    # Apply LMP multiplier
    if 'lmp_multiplier' in scenario_params:
        modified_data.lmp = modified_data.lmp * scenario_params['lmp_multiplier']
    
    # Apply gas price multiplier
    if 'gas_price_multiplier' in scenario_params:
        modified_data.gas_price = modified_data.gas_price * scenario_params['gas_price_multiplier']
    
    return modified_data


def _apply_tech_cost_modifications(
    tech_costs: TechnologyCosts,
    scenario_params: Dict[str, Any]
) -> TechnologyCosts:
    """
    Apply scenario-specific modifications to technology costs.
    
    Args:
        tech_costs: Base technology costs
        scenario_params: Scenario parameters
        
    Returns:
        Modified TechnologyCosts object
    """
    import copy
    modified_costs = copy.deepcopy(tech_costs)
    
    # Apply battery cost modification
    if 'battery_cost_per_kwh' in scenario_params:
        modified_costs.battery_capex_per_kwh = scenario_params['battery_cost_per_kwh']
    
    return modified_costs


def _apply_facility_param_modifications(
    facility_params: FacilityParams,
    scenario_params: Dict[str, Any]
) -> FacilityParams:
    """
    Apply scenario-specific modifications to facility parameters.
    
    Args:
        facility_params: Base facility parameters
        scenario_params: Scenario parameters
        
    Returns:
        Modified FacilityParams object
    """
    import copy
    modified_params = copy.deepcopy(facility_params)
    
    # Apply reliability target modification
    if 'reliability_target' in scenario_params:
        modified_params.reliability_target = scenario_params['reliability_target']
    
    # Apply carbon budget modification
    if 'carbon_reduction_pct' in scenario_params:
        carbon_pct = scenario_params['carbon_reduction_pct']
        if carbon_pct is not None:
            # Calculate carbon budget based on reduction percentage
            # This requires baseline emissions, which we'll estimate
            # Baseline = grid-only emissions
            # For now, set to None if not specified, will be handled in model builder
            modified_params.carbon_budget = carbon_pct
        else:
            modified_params.carbon_budget = None
    
    return modified_params


def save_scenario_results_csv(
    results: List[Dict[str, Any]],
    output_path: str = "results/scenario_results.csv"
) -> None:
    """
    Save scenario results to CSV file for analysis.
    
    Creates a CSV file with one row per scenario containing key metrics
    and parameters for easy comparison and analysis.
    
    Args:
        results: List of result dictionaries from solve_scenarios()
        output_path: Path to output CSV file
        
    Example:
        >>> results = solve_scenarios(scenarios, data, costs, params)
        >>> save_scenario_results_csv(results)
    """
    # Filter successful results
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        logger.warning("No successful results to save")
        return
    
    # Extract data for CSV
    rows = []
    for result in successful_results:
        row = {
            'scenario_index': result['scenario_index'],
            'scenario_name': result['scenario_name'],
            'status': result['status'],
            'solve_time': result['solve_time'],
        }
        
        # Add scenario parameters
        for key, value in result['scenario_params'].items():
            row[f'param_{key}'] = value
        
        # Add capacity results
        for key, value in result['capacity'].items():
            row[f'capacity_{key}'] = value
        
        # Add metrics
        for key, value in result['metrics'].items():
            row[f'metric_{key}'] = value
        
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    
    # Create parent directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(rows)} scenario results to {output_path}")


def get_failed_scenarios(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract failed scenarios from results.
    
    Args:
        results: List of result dictionaries from solve_scenarios()
        
    Returns:
        List of failed scenario results
    """
    return [r for r in results if r['status'] == 'failed']


def get_successful_scenarios(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract successful scenarios from results.
    
    Args:
        results: List of result dictionaries from solve_scenarios()
        
    Returns:
        List of successful scenario results
    """
    return [r for r in results if r['status'] == 'success']


def retry_failed_scenarios(
    failed_results: List[Dict[str, Any]],
    market_data: MarketData,
    base_tech_costs: TechnologyCosts,
    base_facility_params: FacilityParams,
    n_workers: Optional[int] = None,
    save_solutions: bool = True,
    output_dir: str = "results/scenarios",
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Retry solving failed scenarios.
    
    Useful for scenarios that failed due to transient issues or
    when solver parameters need adjustment.
    
    Args:
        failed_results: List of failed result dictionaries
        market_data: Base market data
        base_tech_costs: Base technology costs
        base_facility_params: Base facility parameters
        n_workers: Number of worker processes
        save_solutions: Whether to save solution files
        output_dir: Directory for saving solutions
        verbose: Whether to print progress
        
    Returns:
        List of retry results
    """
    # Extract scenario parameters from failed results
    scenarios = [r['scenario_params'] for r in failed_results]
    
    logger.info(f"Retrying {len(scenarios)} failed scenarios")
    
    # Solve scenarios
    return solve_scenarios(
        scenarios=scenarios,
        market_data=market_data,
        base_tech_costs=base_tech_costs,
        base_facility_params=base_facility_params,
        n_workers=n_workers,
        save_solutions=save_solutions,
        output_dir=output_dir,
        verbose=verbose
    )


def get_batch_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get summary statistics for batch solve results.
    
    Args:
        results: List of result dictionaries from solve_scenarios()
        
    Returns:
        Dictionary with summary statistics
    """
    successful = get_successful_scenarios(results)
    failed = get_failed_scenarios(results)
    
    summary = {
        'total_scenarios': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': len(successful) / len(results) if results else 0.0
    }
    
    if successful:
        solve_times = [r['solve_time'] for r in successful]
        npvs = [r['metrics']['total_npv'] for r in successful]
        lcoes = [r['metrics']['lcoe'] for r in successful]
        reliabilities = [r['metrics']['reliability_pct'] for r in successful]
        
        summary['solve_time'] = {
            'min': min(solve_times),
            'max': max(solve_times),
            'mean': sum(solve_times) / len(solve_times),
            'total': sum(solve_times)
        }
        
        summary['npv'] = {
            'min': min(npvs),
            'max': max(npvs),
            'mean': sum(npvs) / len(npvs)
        }
        
        summary['lcoe'] = {
            'min': min(lcoes),
            'max': max(lcoes),
            'mean': sum(lcoes) / len(lcoes)
        }
        
        summary['reliability'] = {
            'min': min(reliabilities),
            'max': max(reliabilities),
            'mean': sum(reliabilities) / len(reliabilities)
        }
    
    if failed:
        error_types = {}
        for result in failed:
            error_type = result.get('error_type', 'Unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        summary['error_types'] = error_types
    
    return summary
