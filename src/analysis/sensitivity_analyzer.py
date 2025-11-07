"""
Sensitivity analyzer for parameter impact assessment.

This module analyzes how changes in input parameters affect optimization results,
calculating elasticities, identifying breakeven points, and ranking parameters
by their impact on total NPV and other key metrics.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SensitivityAnalyzerError(Exception):
    """Base exception for sensitivity analyzer errors."""
    pass


def analyze_sensitivity(
    base_solution: Dict[str, Any],
    varied_solutions: List[Dict[str, Any]],
    parameter_name: str,
    parameter_values: Optional[List[float]] = None,
    metric: str = 'total_npv'
) -> Dict[str, Any]:
    """
    Analyze sensitivity of optimization results to a parameter.
    
    Calculates how changes in a parameter affect the objective and other metrics,
    including elasticity (percentage change in metric per percentage change in parameter),
    breakeven points where optimal decisions change, and impact ranking.
    
    Args:
        base_solution: Base case solution dictionary from batch_solver
        varied_solutions: List of solutions with varied parameter values
        parameter_name: Name of the parameter being varied (e.g., 'gas_price_multiplier')
        parameter_values: List of parameter values corresponding to varied_solutions
                         If None, extracted from scenario_params
        metric: Metric to analyze sensitivity for (default: 'total_npv')
        
    Returns:
        Dictionary with sensitivity analysis results including:
        - elasticity: % change in metric per % change in parameter
        - breakeven_points: Parameter values where decisions change
        - impact_ranking: Relative importance score
        - regression_coefficients: Linear fit parameters
        
    Example:
        >>> base = solve_optimization(base_params)
        >>> varied = [solve_optimization(p) for p in gas_price_scenarios]
        >>> sensitivity = analyze_sensitivity(
        ...     base,
        ...     varied,
        ...     'gas_price_multiplier',
        ...     metric='total_npv'
        ... )
        >>> print(f"Elasticity: {sensitivity['elasticity']:.2f}")
    """
    if not varied_solutions:
        raise SensitivityAnalyzerError("No varied solutions provided")
    
    # Filter successful solutions
    successful_solutions = [s for s in varied_solutions if s.get('status') == 'success']
    
    if not successful_solutions:
        raise SensitivityAnalyzerError("No successful varied solutions to analyze")
    
    # Extract parameter values if not provided
    if parameter_values is None:
        parameter_values = []
        for solution in successful_solutions:
            param_value = solution.get('scenario_params', {}).get(parameter_name)
            if param_value is None:
                raise SensitivityAnalyzerError(
                    f"Parameter {parameter_name} not found in solution scenario_params"
                )
            parameter_values.append(param_value)
    
    # Extract metric values
    metric_values = []
    for solution in successful_solutions:
        metric_value = solution.get('metrics', {}).get(metric)
        if metric_value is None:
            raise SensitivityAnalyzerError(
                f"Metric {metric} not found in solution metrics"
            )
        metric_values.append(metric_value)
    
    # Get base values
    base_param_value = base_solution.get('scenario_params', {}).get(parameter_name)
    base_metric_value = base_solution.get('metrics', {}).get(metric)
    
    if base_param_value is None or base_metric_value is None:
        raise SensitivityAnalyzerError(
            f"Base solution missing parameter {parameter_name} or metric {metric}"
        )
    
    # Calculate elasticity
    elasticity = _calculate_elasticity(
        parameter_values,
        metric_values,
        base_param_value,
        base_metric_value
    )
    
    # Identify breakeven points
    breakeven_points = _identify_breakeven_points(
        successful_solutions,
        parameter_values,
        parameter_name
    )
    
    # Calculate regression coefficients
    regression = _calculate_regression(parameter_values, metric_values)
    
    # Calculate impact score
    impact_score = _calculate_impact_score(
        parameter_values,
        metric_values,
        base_param_value,
        base_metric_value
    )
    
    # Calculate percentage changes
    pct_changes = _calculate_percentage_changes(
        parameter_values,
        metric_values,
        base_param_value,
        base_metric_value
    )
    
    return {
        'parameter_name': parameter_name,
        'metric': metric,
        'base_parameter_value': base_param_value,
        'base_metric_value': base_metric_value,
        'elasticity': elasticity,
        'impact_score': impact_score,
        'breakeven_points': breakeven_points,
        'regression': regression,
        'parameter_values': parameter_values,
        'metric_values': metric_values,
        'percentage_changes': pct_changes,
        'num_scenarios': len(successful_solutions)
    }


def _calculate_elasticity(
    parameter_values: List[float],
    metric_values: List[float],
    base_param: float,
    base_metric: float
) -> float:
    """
    Calculate elasticity: % change in metric per % change in parameter.
    
    Uses linear regression on percentage changes to estimate elasticity.
    
    Args:
        parameter_values: List of parameter values
        metric_values: List of corresponding metric values
        base_param: Base parameter value
        base_metric: Base metric value
        
    Returns:
        Elasticity coefficient
    """
    if base_param == 0 or base_metric == 0:
        logger.warning("Base parameter or metric is zero, cannot calculate elasticity")
        return 0.0
    
    # Calculate percentage changes
    param_pct_changes = [(p - base_param) / base_param * 100 for p in parameter_values]
    metric_pct_changes = [(m - base_metric) / base_metric * 100 for m in metric_values]
    
    # Filter out zero parameter changes
    valid_indices = [i for i, p in enumerate(param_pct_changes) if abs(p) > 1e-6]
    
    if len(valid_indices) < 2:
        logger.warning("Not enough variation in parameter to calculate elasticity")
        return 0.0
    
    param_pct_changes = [param_pct_changes[i] for i in valid_indices]
    metric_pct_changes = [metric_pct_changes[i] for i in valid_indices]
    
    # Calculate elasticity as slope of metric % change vs parameter % change
    param_array = np.array(param_pct_changes)
    metric_array = np.array(metric_pct_changes)
    
    # Linear regression through origin (elasticity definition)
    elasticity = np.sum(param_array * metric_array) / np.sum(param_array ** 2)
    
    return float(elasticity)


def _identify_breakeven_points(
    solutions: List[Dict[str, Any]],
    parameter_values: List[float],
    parameter_name: str
) -> List[Dict[str, Any]]:
    """
    Identify breakeven points where optimal capacity decisions change.
    
    A breakeven point occurs when the optimal capacity mix changes significantly
    as the parameter varies.
    
    Args:
        solutions: List of solution dictionaries
        parameter_values: List of parameter values
        parameter_name: Name of parameter being varied
        
    Returns:
        List of breakeven point dictionaries
    """
    if len(solutions) < 2:
        return []
    
    breakeven_points = []
    
    # Sort solutions by parameter value
    sorted_indices = np.argsort(parameter_values)
    sorted_solutions = [solutions[i] for i in sorted_indices]
    sorted_params = [parameter_values[i] for i in sorted_indices]
    
    # Check for significant changes in capacity decisions
    for i in range(1, len(sorted_solutions)):
        prev_capacity = sorted_solutions[i-1].get('capacity', {})
        curr_capacity = sorted_solutions[i].get('capacity', {})
        
        # Calculate relative changes in each capacity type
        changes = {}
        for key in prev_capacity.keys():
            prev_val = prev_capacity.get(key, 0)
            curr_val = curr_capacity.get(key, 0)
            
            if prev_val > 0:
                rel_change = abs((curr_val - prev_val) / prev_val)
            elif curr_val > 0:
                rel_change = 1.0  # New capacity added
            else:
                rel_change = 0.0
            
            changes[key] = rel_change
        
        # If any capacity changes by more than 20%, mark as breakeven point
        max_change = max(changes.values()) if changes else 0
        
        if max_change > 0.2:
            breakeven_points.append({
                'parameter_value': (sorted_params[i-1] + sorted_params[i]) / 2,
                'parameter_range': [sorted_params[i-1], sorted_params[i]],
                'capacity_changes': changes,
                'max_change_pct': max_change * 100,
                'description': _describe_capacity_change(prev_capacity, curr_capacity)
            })
    
    return breakeven_points


def _describe_capacity_change(
    prev_capacity: Dict[str, float],
    curr_capacity: Dict[str, float]
) -> str:
    """
    Generate human-readable description of capacity change.
    
    Args:
        prev_capacity: Previous capacity dictionary
        curr_capacity: Current capacity dictionary
        
    Returns:
        Description string
    """
    descriptions = []
    
    for key, curr_val in curr_capacity.items():
        prev_val = prev_capacity.get(key, 0)
        
        if prev_val == 0 and curr_val > 0:
            descriptions.append(f"{key} added")
        elif prev_val > 0 and curr_val == 0:
            descriptions.append(f"{key} removed")
        elif abs(curr_val - prev_val) / prev_val > 0.2:
            change_pct = (curr_val - prev_val) / prev_val * 100
            direction = "increased" if change_pct > 0 else "decreased"
            descriptions.append(f"{key} {direction} by {abs(change_pct):.1f}%")
    
    return "; ".join(descriptions) if descriptions else "No significant changes"


def _calculate_regression(
    parameter_values: List[float],
    metric_values: List[float]
) -> Dict[str, float]:
    """
    Calculate linear regression coefficients.
    
    Fits: metric = slope * parameter + intercept
    
    Args:
        parameter_values: List of parameter values
        metric_values: List of metric values
        
    Returns:
        Dictionary with slope, intercept, and R-squared
    """
    param_array = np.array(parameter_values)
    metric_array = np.array(metric_values)
    
    # Calculate linear regression
    A = np.vstack([param_array, np.ones(len(param_array))]).T
    slope, intercept = np.linalg.lstsq(A, metric_array, rcond=None)[0]
    
    # Calculate R-squared
    metric_pred = slope * param_array + intercept
    ss_res = np.sum((metric_array - metric_pred) ** 2)
    ss_tot = np.sum((metric_array - np.mean(metric_array)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_squared)
    }


def _calculate_impact_score(
    parameter_values: List[float],
    metric_values: List[float],
    base_param: float,
    base_metric: float
) -> float:
    """
    Calculate impact score for ranking parameters.
    
    Impact score is the range of metric values normalized by base metric value.
    Higher score means parameter has larger impact.
    
    Args:
        parameter_values: List of parameter values
        metric_values: List of metric values
        base_param: Base parameter value
        base_metric: Base metric value
        
    Returns:
        Impact score (0-100+)
    """
    if base_metric == 0:
        return 0.0
    
    metric_range = max(metric_values) - min(metric_values)
    impact_score = (metric_range / abs(base_metric)) * 100
    
    return float(impact_score)


def _calculate_percentage_changes(
    parameter_values: List[float],
    metric_values: List[float],
    base_param: float,
    base_metric: float
) -> Dict[str, List[float]]:
    """
    Calculate percentage changes from base case.
    
    Args:
        parameter_values: List of parameter values
        metric_values: List of metric values
        base_param: Base parameter value
        base_metric: Base metric value
        
    Returns:
        Dictionary with parameter and metric percentage changes
    """
    param_pct = [(p - base_param) / base_param * 100 if base_param != 0 else 0 
                 for p in parameter_values]
    metric_pct = [(m - base_metric) / base_metric * 100 if base_metric != 0 else 0 
                  for m in metric_values]
    
    return {
        'parameter_pct_change': param_pct,
        'metric_pct_change': metric_pct
    }


def analyze_multiple_parameters(
    base_solution: Dict[str, Any],
    scenario_results: List[Dict[str, Any]],
    parameters: List[str],
    metric: str = 'total_npv'
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze sensitivity for multiple parameters.
    
    Groups scenario results by parameter and performs sensitivity analysis
    for each parameter independently.
    
    Args:
        base_solution: Base case solution
        scenario_results: List of all scenario results
        parameters: List of parameter names to analyze
        metric: Metric to analyze
        
    Returns:
        Dictionary mapping parameter names to sensitivity analysis results
        
    Example:
        >>> results = solve_scenarios(all_scenarios, data, costs, params)
        >>> sensitivity = analyze_multiple_parameters(
        ...     base_solution,
        ...     results,
        ...     ['gas_price_multiplier', 'lmp_multiplier', 'battery_cost_per_kwh']
        ... )
    """
    sensitivity_results = {}
    
    for parameter_name in parameters:
        # Filter scenarios that vary this parameter
        varied_solutions = _filter_solutions_by_parameter(
            scenario_results,
            parameter_name,
            base_solution
        )
        
        if not varied_solutions:
            logger.warning(f"No solutions found varying parameter: {parameter_name}")
            continue
        
        try:
            sensitivity = analyze_sensitivity(
                base_solution=base_solution,
                varied_solutions=varied_solutions,
                parameter_name=parameter_name,
                metric=metric
            )
            sensitivity_results[parameter_name] = sensitivity
            
        except SensitivityAnalyzerError as e:
            logger.error(f"Failed to analyze {parameter_name}: {e}")
            continue
    
    return sensitivity_results


def _filter_solutions_by_parameter(
    solutions: List[Dict[str, Any]],
    parameter_name: str,
    base_solution: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Filter solutions that vary only the specified parameter.
    
    Returns solutions where the specified parameter differs from base
    but other parameters remain constant.
    
    Args:
        solutions: List of all solutions
        parameter_name: Parameter to filter by
        base_solution: Base solution for comparison
        
    Returns:
        Filtered list of solutions
    """
    base_params = base_solution.get('scenario_params', {})
    base_param_value = base_params.get(parameter_name)
    
    filtered = []
    
    for solution in solutions:
        if solution.get('status') != 'success':
            continue
        
        params = solution.get('scenario_params', {})
        param_value = params.get(parameter_name)
        
        # Check if this parameter is varied
        if param_value is None or param_value == base_param_value:
            continue
        
        # Check if other parameters are constant (optional - for pure sensitivity)
        # For now, include all solutions with this parameter varied
        filtered.append(solution)
    
    return filtered


def rank_parameters_by_impact(
    sensitivity_results: Dict[str, Dict[str, Any]],
    metric: str = 'impact_score'
) -> pd.DataFrame:
    """
    Rank parameters by their impact on optimization results.
    
    Creates a ranked table showing which parameters have the largest
    effect on the objective function.
    
    Args:
        sensitivity_results: Dictionary of sensitivity analysis results
        metric: Metric to rank by ('impact_score', 'elasticity', etc.)
        
    Returns:
        DataFrame with ranked parameters
        
    Example:
        >>> sensitivity = analyze_multiple_parameters(base, results, params)
        >>> ranking = rank_parameters_by_impact(sensitivity)
        >>> print(ranking)
    """
    if not sensitivity_results:
        return pd.DataFrame()
    
    rows = []
    for param_name, analysis in sensitivity_results.items():
        rows.append({
            'parameter': param_name,
            'impact_score': analysis.get('impact_score', 0),
            'elasticity': analysis.get('elasticity', 0),
            'r_squared': analysis.get('regression', {}).get('r_squared', 0),
            'num_breakeven_points': len(analysis.get('breakeven_points', [])),
            'num_scenarios': analysis.get('num_scenarios', 0)
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by specified metric
    if metric in df.columns:
        df = df.sort_values(by=metric, ascending=False, key=abs)
    
    # Add rank column
    df.insert(0, 'rank', range(1, len(df) + 1))
    
    return df


def generate_sensitivity_metrics(
    sensitivity_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate summary metrics for all sensitivity analyses.
    
    Args:
        sensitivity_results: Dictionary of sensitivity analysis results
        
    Returns:
        Dictionary with summary metrics
    """
    if not sensitivity_results:
        return {}
    
    # Calculate aggregate statistics
    all_elasticities = [r['elasticity'] for r in sensitivity_results.values()]
    all_impact_scores = [r['impact_score'] for r in sensitivity_results.values()]
    all_r_squared = [r['regression']['r_squared'] for r in sensitivity_results.values()]
    
    return {
        'num_parameters_analyzed': len(sensitivity_results),
        'elasticity': {
            'mean': float(np.mean(all_elasticities)),
            'std': float(np.std(all_elasticities)),
            'min': float(np.min(all_elasticities)),
            'max': float(np.max(all_elasticities))
        },
        'impact_score': {
            'mean': float(np.mean(all_impact_scores)),
            'std': float(np.std(all_impact_scores)),
            'min': float(np.min(all_impact_scores)),
            'max': float(np.max(all_impact_scores))
        },
        'r_squared': {
            'mean': float(np.mean(all_r_squared)),
            'std': float(np.std(all_r_squared)),
            'min': float(np.min(all_r_squared)),
            'max': float(np.max(all_r_squared))
        }
    }


def save_sensitivity_results(
    sensitivity_results: Dict[str, Dict[str, Any]],
    output_path: str = "results/sensitivity_analysis.json"
) -> None:
    """
    Save sensitivity analysis results to JSON file.
    
    Args:
        sensitivity_results: Dictionary of sensitivity analysis results
        output_path: Path to output JSON file
        
    Example:
        >>> sensitivity = analyze_multiple_parameters(base, results, params)
        >>> save_sensitivity_results(sensitivity)
    """
    # Create parent directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Add summary metrics
    output_data = {
        'summary': generate_sensitivity_metrics(sensitivity_results),
        'parameters': sensitivity_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved sensitivity analysis results to {output_path}")


def load_sensitivity_results(
    input_path: str = "results/sensitivity_analysis.json"
) -> Dict[str, Dict[str, Any]]:
    """
    Load sensitivity analysis results from JSON file.
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        Dictionary of sensitivity analysis results
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded sensitivity analysis results from {input_path}")
    return data.get('parameters', {})


def create_sensitivity_dataframe(
    sensitivity_result: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create DataFrame from single parameter sensitivity analysis.
    
    Useful for plotting and further analysis.
    
    Args:
        sensitivity_result: Sensitivity analysis result for one parameter
        
    Returns:
        DataFrame with parameter values, metric values, and percentage changes
    """
    param_name = sensitivity_result['parameter_name']
    metric_name = sensitivity_result['metric']
    
    df = pd.DataFrame({
        param_name: sensitivity_result['parameter_values'],
        metric_name: sensitivity_result['metric_values'],
        f'{param_name}_pct_change': sensitivity_result['percentage_changes']['parameter_pct_change'],
        f'{metric_name}_pct_change': sensitivity_result['percentage_changes']['metric_pct_change']
    })
    
    return df


def identify_critical_parameters(
    sensitivity_results: Dict[str, Dict[str, Any]],
    impact_threshold: float = 10.0
) -> List[str]:
    """
    Identify critical parameters with high impact on results.
    
    Args:
        sensitivity_results: Dictionary of sensitivity analysis results
        impact_threshold: Minimum impact score to be considered critical
        
    Returns:
        List of critical parameter names
    """
    critical_params = []
    
    for param_name, analysis in sensitivity_results.items():
        impact_score = analysis.get('impact_score', 0)
        if abs(impact_score) >= impact_threshold:
            critical_params.append(param_name)
    
    logger.info(f"Identified {len(critical_params)} critical parameters")
    return critical_params


def compare_parameter_impacts(
    sensitivity_results: Dict[str, Dict[str, Any]],
    parameter1: str,
    parameter2: str
) -> Dict[str, Any]:
    """
    Compare the impact of two parameters.
    
    Args:
        sensitivity_results: Dictionary of sensitivity analysis results
        parameter1: First parameter name
        parameter2: Second parameter name
        
    Returns:
        Dictionary with comparison metrics
    """
    if parameter1 not in sensitivity_results or parameter2 not in sensitivity_results:
        raise SensitivityAnalyzerError(
            f"One or both parameters not found in sensitivity results"
        )
    
    result1 = sensitivity_results[parameter1]
    result2 = sensitivity_results[parameter2]
    
    return {
        'parameter1': parameter1,
        'parameter2': parameter2,
        'impact_score_ratio': result1['impact_score'] / result2['impact_score'] if result2['impact_score'] != 0 else float('inf'),
        'elasticity_ratio': result1['elasticity'] / result2['elasticity'] if result2['elasticity'] != 0 else float('inf'),
        'parameter1_impact': result1['impact_score'],
        'parameter2_impact': result2['impact_score'],
        'parameter1_elasticity': result1['elasticity'],
        'parameter2_elasticity': result2['elasticity'],
        'more_impactful': parameter1 if abs(result1['impact_score']) > abs(result2['impact_score']) else parameter2
    }
