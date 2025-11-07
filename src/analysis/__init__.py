"""Analysis module for scenario generation and sensitivity analysis."""

from .scenario_generator import (
    generate_scenarios,
    generate_gas_price_scenarios,
    generate_lmp_scenarios,
    generate_battery_cost_scenarios,
    generate_reliability_scenarios,
    generate_carbon_scenarios,
    generate_full_sensitivity_scenarios,
    generate_pareto_scenarios,
    filter_scenarios,
    get_scenario_summary
)

from .batch_solver import (
    solve_scenarios,
    save_scenario_results_csv,
    get_failed_scenarios,
    get_successful_scenarios,
    retry_failed_scenarios,
    get_batch_summary,
    BatchSolverError
)

from .pareto_calculator import (
    calculate_pareto_frontier,
    calculate_cost_reliability_frontier,
    calculate_cost_carbon_frontier,
    calculate_grid_reliability_frontier,
    calculate_all_pareto_frontiers,
    save_pareto_frontiers,
    load_pareto_frontiers,
    get_pareto_summary,
    identify_knee_point,
    compare_to_baseline,
    filter_pareto_by_constraint,
    ParetoCalculatorError
)

from .sensitivity_analyzer import (
    analyze_sensitivity,
    analyze_multiple_parameters,
    rank_parameters_by_impact,
    generate_sensitivity_metrics,
    save_sensitivity_results,
    load_sensitivity_results,
    create_sensitivity_dataframe,
    identify_critical_parameters,
    compare_parameter_impacts,
    SensitivityAnalyzerError
)

__all__ = [
    'generate_scenarios',
    'generate_gas_price_scenarios',
    'generate_lmp_scenarios',
    'generate_battery_cost_scenarios',
    'generate_reliability_scenarios',
    'generate_carbon_scenarios',
    'generate_full_sensitivity_scenarios',
    'generate_pareto_scenarios',
    'filter_scenarios',
    'get_scenario_summary',
    'solve_scenarios',
    'save_scenario_results_csv',
    'get_failed_scenarios',
    'get_successful_scenarios',
    'retry_failed_scenarios',
    'get_batch_summary',
    'BatchSolverError',
    'calculate_pareto_frontier',
    'calculate_cost_reliability_frontier',
    'calculate_cost_carbon_frontier',
    'calculate_grid_reliability_frontier',
    'calculate_all_pareto_frontiers',
    'save_pareto_frontiers',
    'load_pareto_frontiers',
    'get_pareto_summary',
    'identify_knee_point',
    'compare_to_baseline',
    'filter_pareto_by_constraint',
    'ParetoCalculatorError',
    'analyze_sensitivity',
    'analyze_multiple_parameters',
    'rank_parameters_by_impact',
    'generate_sensitivity_metrics',
    'save_sensitivity_results',
    'load_sensitivity_results',
    'create_sensitivity_dataframe',
    'identify_critical_parameters',
    'compare_parameter_impacts',
    'SensitivityAnalyzerError'
]
