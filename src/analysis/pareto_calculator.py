"""
Pareto frontier calculator for multi-objective optimization analysis.

This module identifies non-dominated solutions (Pareto-optimal) from a set
of optimization results, enabling trade-off analysis between competing objectives
such as cost vs. reliability, cost vs. carbon emissions, and grid dependence vs. reliability.
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ParetoCalculatorError(Exception):
    """Base exception for Pareto calculator errors."""
    pass


def calculate_pareto_frontier(
    solutions: List[Dict[str, Any]],
    objective1: str,
    objective2: str,
    minimize_obj1: bool = True,
    minimize_obj2: bool = True
) -> pd.DataFrame:
    """
    Identify Pareto-optimal solutions for two objectives.
    
    A solution is Pareto-optimal (non-dominated) if no other solution is better
    in both objectives. This function finds all such solutions from a set of
    optimization results.
    
    Args:
        solutions: List of solution dictionaries from batch_solver
        objective1: Name of first objective (e.g., 'total_npv', 'reliability_pct')
        objective2: Name of second objective (e.g., 'carbon_tons_annual')
        minimize_obj1: If True, minimize objective1; if False, maximize
        minimize_obj2: If True, minimize objective2; if False, maximize
        
    Returns:
        DataFrame with Pareto-optimal solutions, sorted by objective1
        
    Raises:
        ParetoCalculatorError: If objectives not found or no valid solutions
        
    Example:
        >>> results = solve_scenarios(scenarios, data, costs, params)
        >>> pareto_df = calculate_pareto_frontier(
        ...     results,
        ...     objective1='total_npv',
        ...     objective2='carbon_tons_annual',
        ...     minimize_obj1=True,
        ...     minimize_obj2=True
        ... )
        >>> print(f"Found {len(pareto_df)} Pareto-optimal solutions")
    """
    if not solutions:
        raise ParetoCalculatorError("No solutions provided")
    
    # Filter successful solutions
    successful_solutions = [s for s in solutions if s.get('status') == 'success']
    
    if not successful_solutions:
        raise ParetoCalculatorError("No successful solutions to analyze")
    
    # Extract objective values
    try:
        objective_data = []
        for solution in successful_solutions:
            metrics = solution.get('metrics', {})
            
            # Try to get objective from metrics
            obj1_value = metrics.get(objective1)
            obj2_value = metrics.get(objective2)
            
            if obj1_value is None or obj2_value is None:
                logger.warning(
                    f"Solution {solution.get('scenario_name')} missing objectives: "
                    f"{objective1}={obj1_value}, {objective2}={obj2_value}"
                )
                continue
            
            objective_data.append({
                'solution': solution,
                'obj1': float(obj1_value),
                'obj2': float(obj2_value)
            })
        
        if not objective_data:
            raise ParetoCalculatorError(
                f"No solutions have both objectives: {objective1}, {objective2}"
            )
    
    except (KeyError, TypeError) as e:
        raise ParetoCalculatorError(
            f"Error extracting objectives {objective1}, {objective2}: {str(e)}"
        )
    
    # Convert to numpy arrays for efficient computation
    obj1_values = np.array([d['obj1'] for d in objective_data])
    obj2_values = np.array([d['obj2'] for d in objective_data])
    
    # Adjust for maximization objectives (convert to minimization)
    if not minimize_obj1:
        obj1_values = -obj1_values
    if not minimize_obj2:
        obj2_values = -obj2_values
    
    # Find Pareto-optimal solutions
    pareto_indices = _find_pareto_optimal_indices(obj1_values, obj2_values)
    
    # Build result DataFrame
    pareto_solutions = []
    for idx in pareto_indices:
        solution = objective_data[idx]['solution']
        
        row = {
            'scenario_index': solution.get('scenario_index'),
            'scenario_name': solution.get('scenario_name'),
            'is_pareto_optimal': True,
            objective1: objective_data[idx]['obj1'],
            objective2: objective_data[idx]['obj2']
        }
        
        # Add all metrics
        row.update(solution.get('metrics', {}))
        
        # Add capacity information
        capacity = solution.get('capacity', {})
        for key, value in capacity.items():
            row[f'capacity_{key}'] = value
        
        # Add scenario parameters
        params = solution.get('scenario_params', {})
        for key, value in params.items():
            row[f'param_{key}'] = value
        
        pareto_solutions.append(row)
    
    # Create DataFrame and sort by first objective
    pareto_df = pd.DataFrame(pareto_solutions)
    pareto_df = pareto_df.sort_values(by=objective1)
    
    logger.info(
        f"Found {len(pareto_df)} Pareto-optimal solutions out of "
        f"{len(successful_solutions)} total solutions"
    )
    
    return pareto_df


def _find_pareto_optimal_indices(obj1: np.ndarray, obj2: np.ndarray) -> List[int]:
    """
    Find indices of Pareto-optimal solutions using efficient algorithm.
    
    A solution i is Pareto-optimal if there is no solution j such that:
    - obj1[j] <= obj1[i] AND obj2[j] <= obj2[i]
    - AND at least one inequality is strict
    
    Args:
        obj1: Array of first objective values (minimization)
        obj2: Array of second objective values (minimization)
        
    Returns:
        List of indices of Pareto-optimal solutions
    """
    n = len(obj1)
    is_pareto = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not is_pareto[i]:
            continue
        
        # Check if any other solution dominates solution i
        # Solution j dominates i if:
        # obj1[j] <= obj1[i] AND obj2[j] <= obj2[i] AND (obj1[j] < obj1[i] OR obj2[j] < obj2[i])
        dominates = (
            (obj1 <= obj1[i]) & 
            (obj2 <= obj2[i]) & 
            ((obj1 < obj1[i]) | (obj2 < obj2[i]))
        )
        
        # If any solution dominates i, mark i as not Pareto-optimal
        if np.any(dominates):
            is_pareto[i] = False
    
    return np.where(is_pareto)[0].tolist()


def calculate_cost_reliability_frontier(
    solutions: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Calculate Pareto frontier for cost vs. reliability trade-off.
    
    Identifies solutions that minimize cost while maximizing reliability.
    
    Args:
        solutions: List of solution dictionaries from batch_solver
        
    Returns:
        DataFrame with Pareto-optimal solutions for cost-reliability trade-off
    """
    return calculate_pareto_frontier(
        solutions=solutions,
        objective1='total_npv',
        objective2='reliability_pct',
        minimize_obj1=True,
        minimize_obj2=False  # Maximize reliability
    )


def calculate_cost_carbon_frontier(
    solutions: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Calculate Pareto frontier for cost vs. carbon emissions trade-off.
    
    Identifies solutions that minimize both cost and carbon emissions.
    
    Args:
        solutions: List of solution dictionaries from batch_solver
        
    Returns:
        DataFrame with Pareto-optimal solutions for cost-carbon trade-off
    """
    return calculate_pareto_frontier(
        solutions=solutions,
        objective1='total_npv',
        objective2='carbon_tons_annual',
        minimize_obj1=True,
        minimize_obj2=True
    )


def calculate_grid_reliability_frontier(
    solutions: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Calculate Pareto frontier for grid dependence vs. reliability trade-off.
    
    Identifies solutions that minimize grid dependence while maximizing reliability.
    
    Args:
        solutions: List of solution dictionaries from batch_solver
        
    Returns:
        DataFrame with Pareto-optimal solutions for grid-reliability trade-off
    """
    return calculate_pareto_frontier(
        solutions=solutions,
        objective1='grid_dependence_pct',
        objective2='reliability_pct',
        minimize_obj1=True,
        minimize_obj2=False  # Maximize reliability
    )


def calculate_all_pareto_frontiers(
    solutions: List[Dict[str, Any]]
) -> Dict[str, pd.DataFrame]:
    """
    Calculate all standard Pareto frontiers.
    
    Computes Pareto frontiers for the three main objective pairs:
    - Cost vs. Reliability
    - Cost vs. Carbon
    - Grid Dependence vs. Reliability
    
    Args:
        solutions: List of solution dictionaries from batch_solver
        
    Returns:
        Dictionary mapping frontier names to DataFrames
    """
    frontiers = {}
    
    try:
        frontiers['cost_reliability'] = calculate_cost_reliability_frontier(solutions)
        logger.info("Calculated cost-reliability Pareto frontier")
    except ParetoCalculatorError as e:
        logger.warning(f"Could not calculate cost-reliability frontier: {e}")
    
    try:
        frontiers['cost_carbon'] = calculate_cost_carbon_frontier(solutions)
        logger.info("Calculated cost-carbon Pareto frontier")
    except ParetoCalculatorError as e:
        logger.warning(f"Could not calculate cost-carbon frontier: {e}")
    
    try:
        frontiers['grid_reliability'] = calculate_grid_reliability_frontier(solutions)
        logger.info("Calculated grid-reliability Pareto frontier")
    except ParetoCalculatorError as e:
        logger.warning(f"Could not calculate grid-reliability frontier: {e}")
    
    return frontiers


def save_pareto_frontiers(
    frontiers: Dict[str, pd.DataFrame],
    output_path: str = "results/pareto_frontiers.json"
) -> None:
    """
    Save Pareto frontier data to JSON file.
    
    Args:
        frontiers: Dictionary of frontier name to DataFrame
        output_path: Path to output JSON file
        
    Example:
        >>> frontiers = calculate_all_pareto_frontiers(results)
        >>> save_pareto_frontiers(frontiers)
    """
    # Convert DataFrames to dictionaries for JSON serialization
    output_data = {}
    for name, df in frontiers.items():
        output_data[name] = {
            'num_solutions': len(df),
            'solutions': df.to_dict(orient='records')
        }
    
    # Create parent directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved Pareto frontiers to {output_path}")
    
    # Log summary
    for name, df in frontiers.items():
        logger.info(f"  {name}: {len(df)} Pareto-optimal solutions")


def load_pareto_frontiers(
    input_path: str = "results/pareto_frontiers.json"
) -> Dict[str, pd.DataFrame]:
    """
    Load Pareto frontier data from JSON file.
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        Dictionary of frontier name to DataFrame
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    frontiers = {}
    for name, frontier_data in data.items():
        frontiers[name] = pd.DataFrame(frontier_data['solutions'])
    
    logger.info(f"Loaded Pareto frontiers from {input_path}")
    return frontiers


def get_pareto_summary(
    pareto_df: pd.DataFrame,
    objective1: str,
    objective2: str
) -> Dict[str, Any]:
    """
    Get summary statistics for a Pareto frontier.
    
    Args:
        pareto_df: DataFrame with Pareto-optimal solutions
        objective1: Name of first objective
        objective2: Name of second objective
        
    Returns:
        Dictionary with summary statistics
    """
    if pareto_df.empty:
        return {
            'num_solutions': 0,
            'objective1': objective1,
            'objective2': objective2
        }
    
    summary = {
        'num_solutions': len(pareto_df),
        'objective1': {
            'name': objective1,
            'min': float(pareto_df[objective1].min()),
            'max': float(pareto_df[objective1].max()),
            'range': float(pareto_df[objective1].max() - pareto_df[objective1].min())
        },
        'objective2': {
            'name': objective2,
            'min': float(pareto_df[objective2].min()),
            'max': float(pareto_df[objective2].max()),
            'range': float(pareto_df[objective2].max() - pareto_df[objective2].min())
        }
    }
    
    # Add extreme points
    obj1_min_idx = pareto_df[objective1].idxmin()
    obj1_max_idx = pareto_df[objective1].idxmax()
    obj2_min_idx = pareto_df[objective2].idxmin()
    obj2_max_idx = pareto_df[objective2].idxmax()
    
    summary['extreme_points'] = {
        'min_objective1': {
            'scenario_name': pareto_df.loc[obj1_min_idx, 'scenario_name'],
            objective1: float(pareto_df.loc[obj1_min_idx, objective1]),
            objective2: float(pareto_df.loc[obj1_min_idx, objective2])
        },
        'max_objective1': {
            'scenario_name': pareto_df.loc[obj1_max_idx, 'scenario_name'],
            objective1: float(pareto_df.loc[obj1_max_idx, objective1]),
            objective2: float(pareto_df.loc[obj1_max_idx, objective2])
        },
        'min_objective2': {
            'scenario_name': pareto_df.loc[obj2_min_idx, 'scenario_name'],
            objective1: float(pareto_df.loc[obj2_min_idx, objective1]),
            objective2: float(pareto_df.loc[obj2_min_idx, objective2])
        },
        'max_objective2': {
            'scenario_name': pareto_df.loc[obj2_max_idx, 'scenario_name'],
            objective1: float(pareto_df.loc[obj2_max_idx, objective1]),
            objective2: float(pareto_df.loc[obj2_max_idx, objective2])
        }
    }
    
    return summary


def identify_knee_point(
    pareto_df: pd.DataFrame,
    objective1: str,
    objective2: str
) -> Dict[str, Any]:
    """
    Identify the knee point on the Pareto frontier.
    
    The knee point represents the best trade-off between objectives,
    where small improvements in one objective require large sacrifices
    in the other. Uses the maximum distance from the line connecting
    extreme points.
    
    Args:
        pareto_df: DataFrame with Pareto-optimal solutions
        objective1: Name of first objective
        objective2: Name of second objective
        
    Returns:
        Dictionary with knee point information
    """
    if len(pareto_df) < 3:
        logger.warning("Need at least 3 points to identify knee point")
        return {}
    
    # Normalize objectives to [0, 1] range
    obj1_values = pareto_df[objective1].values
    obj2_values = pareto_df[objective2].values
    
    obj1_norm = (obj1_values - obj1_values.min()) / (obj1_values.max() - obj1_values.min() + 1e-10)
    obj2_norm = (obj2_values - obj2_values.min()) / (obj2_values.max() - obj2_values.min() + 1e-10)
    
    # Calculate distance from line connecting extreme points
    # Line from (0, 0) to (1, 1) in normalized space
    distances = np.abs(obj1_norm - obj2_norm) / np.sqrt(2)
    
    # Find point with maximum distance
    knee_idx = np.argmax(distances)
    
    knee_point = {
        'scenario_name': pareto_df.iloc[knee_idx]['scenario_name'],
        'scenario_index': int(pareto_df.iloc[knee_idx]['scenario_index']),
        objective1: float(pareto_df.iloc[knee_idx][objective1]),
        objective2: float(pareto_df.iloc[knee_idx][objective2]),
        'distance_from_line': float(distances[knee_idx])
    }
    
    logger.info(f"Identified knee point: {knee_point['scenario_name']}")
    
    return knee_point


def compare_to_baseline(
    pareto_df: pd.DataFrame,
    baseline_solution: Dict[str, Any],
    objective1: str,
    objective2: str
) -> pd.DataFrame:
    """
    Compare Pareto frontier solutions to a baseline solution.
    
    Calculates improvement percentages for each Pareto-optimal solution
    relative to the baseline.
    
    Args:
        pareto_df: DataFrame with Pareto-optimal solutions
        baseline_solution: Baseline solution dictionary
        objective1: Name of first objective
        objective2: Name of second objective
        
    Returns:
        DataFrame with added comparison columns
    """
    df = pareto_df.copy()
    
    # Get baseline objective values
    baseline_metrics = baseline_solution.get('metrics', {})
    baseline_obj1 = baseline_metrics.get(objective1)
    baseline_obj2 = baseline_metrics.get(objective2)
    
    if baseline_obj1 is None or baseline_obj2 is None:
        logger.warning("Baseline solution missing objectives, skipping comparison")
        return df
    
    # Calculate improvement percentages
    df[f'{objective1}_improvement_pct'] = (
        (baseline_obj1 - df[objective1]) / baseline_obj1 * 100
    )
    df[f'{objective2}_improvement_pct'] = (
        (baseline_obj2 - df[objective2]) / baseline_obj2 * 100
    )
    
    return df


def filter_pareto_by_constraint(
    pareto_df: pd.DataFrame,
    constraint_column: str,
    constraint_value: float,
    constraint_type: str = 'max'
) -> pd.DataFrame:
    """
    Filter Pareto frontier by an additional constraint.
    
    Useful for finding solutions that meet specific requirements
    (e.g., maximum cost, minimum reliability).
    
    Args:
        pareto_df: DataFrame with Pareto-optimal solutions
        constraint_column: Column name for constraint
        constraint_value: Constraint threshold value
        constraint_type: 'max' for upper bound, 'min' for lower bound
        
    Returns:
        Filtered DataFrame
    """
    if constraint_type == 'max':
        filtered_df = pareto_df[pareto_df[constraint_column] <= constraint_value]
    elif constraint_type == 'min':
        filtered_df = pareto_df[pareto_df[constraint_column] >= constraint_value]
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")
    
    logger.info(
        f"Filtered Pareto frontier: {len(filtered_df)}/{len(pareto_df)} solutions "
        f"meet constraint {constraint_column} {constraint_type} {constraint_value}"
    )
    
    return filtered_df
