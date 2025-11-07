# Analysis Module

This module provides tools for scenario generation, batch solving, and sensitivity analysis.

## Components

### Scenario Generator (`scenario_generator.py`)

Generates parameter combinations for systematic exploration of the solution space.

**Key Functions:**
- `generate_scenarios()`: Generate all combinations of parameter variations
- `generate_gas_price_scenarios()`: Vary only gas prices
- `generate_lmp_scenarios()`: Vary only grid LMP prices
- `generate_battery_cost_scenarios()`: Vary only battery costs
- `generate_reliability_scenarios()`: Vary only reliability targets
- `generate_carbon_scenarios()`: Vary only carbon constraints
- `generate_full_sensitivity_scenarios()`: Generate comprehensive scenario set
- `generate_pareto_scenarios()`: Generate scenarios for Pareto frontier analysis

**Example:**
```python
from src.analysis import generate_scenarios

scenarios = generate_scenarios(
    gas_price_variations=[0.5, 1.0, 1.5],
    reliability_variations=[0.999, 0.9999, 0.99999]
)
# Returns 9 scenarios (3 gas prices × 3 reliability levels)
```

### Batch Solver (`batch_solver.py`)

Solves multiple optimization scenarios in parallel using multiprocessing.

**Key Functions:**
- `solve_scenarios()`: Solve multiple scenarios concurrently
- `save_scenario_results_csv()`: Save results to CSV for analysis
- `get_successful_scenarios()`: Filter successful results
- `get_failed_scenarios()`: Filter failed results
- `retry_failed_scenarios()`: Retry scenarios that failed
- `get_batch_summary()`: Get summary statistics

**Example:**
```python
from src.analysis import generate_scenarios, solve_scenarios, save_scenario_results_csv
from src.models.market_data import MarketData
from src.models.technology import TechnologyCosts, FacilityParams

# Load data
market_data = MarketData.load("data/processed/market_data.csv")
tech_costs = TechnologyCosts()
facility_params = FacilityParams(it_load_mw=300)

# Generate scenarios
scenarios = generate_scenarios(
    gas_price_variations=[0.5, 1.0, 1.5],
    lmp_variations=[0.7, 1.0, 1.3]
)

# Solve in parallel
results = solve_scenarios(
    scenarios=scenarios,
    market_data=market_data,
    base_tech_costs=tech_costs,
    base_facility_params=facility_params,
    n_workers=4,  # Use 4 CPU cores
    save_solutions=True,
    output_dir="results/scenarios"
)

# Save summary to CSV
save_scenario_results_csv(results, "results/scenario_results.csv")
```

### Pareto Calculator (`pareto_calculator.py`)

Identifies Pareto-optimal solutions for multi-objective trade-off analysis.

**Key Functions:**
- `calculate_pareto_frontier()`: Find non-dominated solutions for any two objectives
- `calculate_cost_reliability_frontier()`: Cost vs. reliability trade-off
- `calculate_cost_carbon_frontier()`: Cost vs. carbon emissions trade-off
- `calculate_grid_reliability_frontier()`: Grid dependence vs. reliability trade-off
- `calculate_all_pareto_frontiers()`: Calculate all standard frontiers
- `save_pareto_frontiers()`: Save frontier data to JSON
- `load_pareto_frontiers()`: Load frontier data from JSON
- `get_pareto_summary()`: Get summary statistics for a frontier
- `identify_knee_point()`: Find best trade-off point on frontier
- `compare_to_baseline()`: Compare frontier to baseline solution
- `filter_pareto_by_constraint()`: Filter frontier by additional constraints

**Example:**
```python
from src.analysis import (
    generate_pareto_scenarios,
    solve_scenarios,
    calculate_all_pareto_frontiers,
    save_pareto_frontiers
)

# Generate scenarios for Pareto analysis
scenarios = generate_pareto_scenarios(objective_pair='cost_reliability')

# Solve scenarios
results = solve_scenarios(
    scenarios=scenarios,
    market_data=market_data,
    base_tech_costs=tech_costs,
    base_facility_params=facility_params
)

# Calculate all Pareto frontiers
frontiers = calculate_all_pareto_frontiers(results)

# Save to file
save_pareto_frontiers(frontiers, "results/pareto_frontiers.json")

# Analyze cost-reliability frontier
cost_rel_frontier = frontiers['cost_reliability']
print(f"Found {len(cost_rel_frontier)} Pareto-optimal solutions")

# Identify knee point (best trade-off)
from src.analysis import identify_knee_point
knee = identify_knee_point(cost_rel_frontier, 'total_npv', 'reliability_pct')
print(f"Best trade-off: {knee['scenario_name']}")
```

**Pareto Optimality:**

A solution is Pareto-optimal if no other solution is better in all objectives. For example, in cost-reliability analysis:
- Solution A: $2.0B NPV, 99.9% reliability
- Solution B: $2.4B NPV, 99.99% reliability
- Solution C: $2.2B NPV, 99.95% reliability

Solutions A and B are Pareto-optimal (A is cheaper but less reliable, B is more expensive but more reliable). Solution C is dominated by interpolating between A and B.

**Supported Objective Pairs:**

1. **Cost vs. Reliability**: Trade-off between total NPV and uptime percentage
2. **Cost vs. Carbon**: Trade-off between total NPV and annual emissions
3. **Grid Dependence vs. Reliability**: Trade-off between grid reliance and uptime

**Knee Point:**

The knee point represents the best compromise between objectives, where small improvements in one objective require large sacrifices in the other. It's identified as the point with maximum distance from the line connecting extreme points.

## Performance Considerations

### Parallel Execution

The batch solver uses Python's `multiprocessing` module to solve scenarios concurrently:

- **Default workers**: CPU count - 1 (leaves one core free)
- **Memory usage**: Each worker loads a copy of the data
- **Recommended**: 4-8 workers for typical laptops

### Solve Time Estimates

For a full-year optimization (8760 hours):
- Single scenario: 5-30 minutes
- 10 scenarios with 4 workers: 15-75 minutes
- 100 scenarios with 8 workers: 2-6 hours

For faster testing, use weekly data (168 hours):
- Single scenario: 10-60 seconds
- 10 scenarios with 4 workers: 30-150 seconds

### Error Handling

The batch solver handles errors gracefully:
- Solver failures (infeasible, unbounded, timeout)
- Numerical issues
- Unexpected exceptions

Failed scenarios are logged and can be retried with adjusted parameters.

## Scenario Parameter Format

Scenarios are dictionaries with the following structure:

```python
{
    'scenario_name': 'gas150_rel99999',
    'gas_price_multiplier': 1.5,
    'lmp_multiplier': 1.0,
    'battery_cost_per_kwh': 350.0,
    'reliability_target': 0.99999,
    'carbon_reduction_pct': None
}
```

## Output Format

### Result Dictionary

Each scenario produces a result dictionary:

```python
{
    'status': 'success',  # or 'failed'
    'scenario_index': 0,
    'scenario_name': 'base',
    'scenario_params': {...},
    'capacity': {
        'Grid Connection (MW)': 200.0,
        'Gas Peakers (MW)': 150.0,
        'Battery Storage (MWh)': 400.0,
        'Solar PV (MW)': 100.0
    },
    'metrics': {
        'total_npv': 2450000000.0,
        'capex': 850000000.0,
        'opex_annual': 95000000.0,
        'lcoe': 42.5,
        'reliability_pct': 99.99,
        'carbon_tons_annual': 125000.0,
        ...
    },
    'solve_time': 245.3
}
```

### CSV Output

The CSV file contains one row per scenario with columns:
- `scenario_index`, `scenario_name`, `status`, `solve_time`
- `param_*`: All scenario parameters
- `capacity_*`: All capacity decisions
- `metric_*`: All solution metrics

This format enables easy analysis in Excel, Python, or R.

## Common Use Cases

### 1. Pareto Frontier Analysis

```python
# Generate scenarios exploring reliability-cost trade-off
scenarios = generate_pareto_scenarios(objective_pair='cost_reliability')
results = solve_scenarios(scenarios, data, costs, params)

# Calculate Pareto frontier
frontier = calculate_cost_reliability_frontier(results)

# Find knee point (best trade-off)
knee = identify_knee_point(frontier, 'total_npv', 'reliability_pct')
print(f"Recommended solution: {knee['scenario_name']}")
print(f"  Cost: ${knee['total_npv']:,.0f}")
print(f"  Reliability: {knee['reliability_pct']:.3f}%")
```

### 2. Gas Price Sensitivity

```python
scenarios = generate_gas_price_scenarios(
    variations=[0.5, 0.75, 1.0, 1.25, 1.5]
)
results = solve_scenarios(scenarios, data, costs, params)
```

### 2. Reliability-Cost Trade-off

```python
scenarios = generate_reliability_scenarios(
    variations=[0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999]
)
results = solve_scenarios(scenarios, data, costs, params)
```

### 3. Carbon Constraint Analysis

```python
scenarios = generate_carbon_scenarios(
    variations=[None, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
)
results = solve_scenarios(scenarios, data, costs, params)
```

### 4. Full Sensitivity Analysis

```python
scenarios = generate_full_sensitivity_scenarios()
# Generates 324 scenarios (3×3×3×3×4)
results = solve_scenarios(scenarios, data, costs, params, n_workers=8)
```

## Troubleshooting

### All Scenarios Fail

Check:
1. Gurobi license is valid
2. Data is loaded correctly
3. Base scenario parameters are feasible
4. Sufficient memory available

### Some Scenarios Fail

Common causes:
- Infeasible: Reliability target too strict or carbon budget too low
- Timeout: Increase time limit in solver options
- Numerical: Check for extreme parameter values

### Slow Performance

Improvements:
- Use more workers (up to CPU count)
- Reduce problem size (use weekly instead of annual data)
- Pre-solve common scenarios and cache results
- Use faster solver settings (larger MIP gap)

### Sensitivity Analyzer (`sensitivity_analyzer.py`)

Analyzes how changes in input parameters affect optimization results, calculating elasticities, identifying breakeven points, and ranking parameters by impact.

**Key Functions:**
- `analyze_sensitivity()`: Analyze sensitivity to a single parameter
- `analyze_multiple_parameters()`: Analyze sensitivity for multiple parameters
- `rank_parameters_by_impact()`: Rank parameters by their impact on results
- `generate_sensitivity_metrics()`: Generate summary metrics for all analyses
- `save_sensitivity_results()`: Save sensitivity analysis to JSON
- `load_sensitivity_results()`: Load sensitivity analysis from JSON
- `create_sensitivity_dataframe()`: Create DataFrame for plotting
- `identify_critical_parameters()`: Identify high-impact parameters
- `compare_parameter_impacts()`: Compare impact of two parameters

**Example:**
```python
from src.analysis import (
    generate_scenarios,
    solve_scenarios,
    analyze_sensitivity,
    analyze_multiple_parameters,
    rank_parameters_by_impact,
    save_sensitivity_results
)

# Generate base solution
base_params = {'gas_price_multiplier': 1.0, 'lmp_multiplier': 1.0}
base_solution = solve_optimization(base_params, data, costs, facility_params)

# Generate scenarios varying gas prices
gas_scenarios = generate_gas_price_scenarios(
    variations=[0.5, 0.75, 1.0, 1.25, 1.5]
)
gas_results = solve_scenarios(gas_scenarios, data, costs, facility_params)

# Analyze gas price sensitivity
gas_sensitivity = analyze_sensitivity(
    base_solution=base_solution,
    varied_solutions=gas_results,
    parameter_name='gas_price_multiplier',
    metric='total_npv'
)

print(f"Elasticity: {gas_sensitivity['elasticity']:.3f}")
print(f"  (% change in NPV per % change in gas price)")
print(f"Impact Score: {gas_sensitivity['impact_score']:.2f}%")
print(f"Breakeven Points: {len(gas_sensitivity['breakeven_points'])}")

# Analyze multiple parameters
all_scenarios = (
    generate_gas_price_scenarios() +
    generate_lmp_scenarios() +
    generate_battery_cost_scenarios()
)
all_results = solve_scenarios(all_scenarios, data, costs, facility_params)

sensitivity_results = analyze_multiple_parameters(
    base_solution=base_solution,
    scenario_results=all_results,
    parameters=['gas_price_multiplier', 'lmp_multiplier', 'battery_cost_per_kwh'],
    metric='total_npv'
)

# Rank parameters by impact
ranking = rank_parameters_by_impact(sensitivity_results)
print("\nParameter Ranking:")
print(ranking)

# Save results
save_sensitivity_results(sensitivity_results, "results/sensitivity_analysis.json")
```

**Key Metrics:**

1. **Elasticity**: Percentage change in metric per percentage change in parameter
   - Formula: (Δmetric/metric) / (Δparameter/parameter)
   - Example: Elasticity of 0.5 means 1% increase in parameter causes 0.5% increase in NPV
   - Calculated using linear regression on percentage changes

2. **Impact Score**: Range of metric values as percentage of base metric
   - Formula: (max_metric - min_metric) / base_metric × 100
   - Example: Impact score of 20% means parameter variations cause 20% swing in NPV
   - Higher score = more impactful parameter

3. **Breakeven Points**: Parameter values where optimal capacity decisions change significantly
   - Identified when capacity mix changes by >20%
   - Indicates threshold where different technologies become optimal
   - Example: At gas price multiplier 1.3, optimal solution switches from gas-heavy to battery-heavy

4. **Regression Coefficients**: Linear fit of metric vs. parameter
   - Slope: Change in metric per unit change in parameter
   - R-squared: Quality of linear fit (0-1, higher is better)
   - Used to predict metric values for untested parameter values

**Example Output:**

```
Parameter: gas_price_multiplier
Elasticity: 0.160
  (% change in NPV per % change in gas price)
Impact Score: 16.00%
  (Range of NPV as % of base NPV)
Regression:
  Slope: 400,000,000
  R-squared: 0.9950
Breakeven Points: 2
  1. At gas price multiplier ~1.25
     Gas Peakers (MW) decreased by 33.3%; Battery Storage (MWh) increased by 50.0%
  2. At gas price multiplier ~1.75
     Gas Peakers (MW) removed; Solar PV (MW) increased by 100.0%
```

**Use Cases:**

1. **Identify Critical Parameters**: Which inputs have the largest impact on results?
2. **Risk Assessment**: How sensitive is the optimal solution to market uncertainties?
3. **Procurement Strategy**: At what price points do technology choices change?
4. **Scenario Planning**: Which parameters should be monitored most closely?

## Testing

Run the test scripts to verify functionality:

```bash
# Test batch solver
python examples/test_batch_solver.py

# Test Pareto calculator
python examples/test_pareto_calculator.py

# Test sensitivity analyzer
python examples/test_sensitivity_analyzer.py
```

These run small scenarios with synthetic data to verify:
- Parallel execution works
- Results are collected correctly
- CSV/JSON export functions properly
- Error handling works
- Calculations are accurate
