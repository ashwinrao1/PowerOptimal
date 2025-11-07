# Optimization Module

This module contains the core optimization model, solver interface, solution extraction, and validation components for the data center energy optimization system.

## Components

### 1. Model Builder (`model_builder.py`)

Constructs the Pyomo optimization model with:
- Decision variables for capacity investments and hourly dispatch
- Objective function minimizing 20-year total cost (CAPEX + NPV of OPEX)
- Comprehensive constraints for energy balance, capacity limits, battery dynamics, gas ramping, and reliability

**Key Function:**
```python
build_optimization_model(
    market_data: MarketData,
    tech_costs: TechnologyCosts,
    facility_params: FacilityParams,
    allow_gas: bool = True,
    allow_battery: bool = True,
    allow_solar: bool = True
) -> pyo.ConcreteModel
```

### 2. Solver Interface (`solver.py`)

Provides interface to Gurobi solver with:
- Automatic solver configuration
- Comprehensive error handling (infeasibility, unbounded, numerical issues, timeout)
- Solution validation at the Pyomo constraint level

**Key Functions:**
```python
solve_model(
    model: pyo.ConcreteModel,
    solver_options: Optional[Dict] = None,
    time_limit: int = 1800,
    mip_gap: float = 0.005,
    verbose: bool = True
) -> Tuple[SolverResults, float]

get_solver_info() -> Dict[str, Any]

validate_solution(
    model: pyo.ConcreteModel,
    tolerance: float = 1e-6
) -> Tuple[bool, List[str]]
```

### 3. Solution Extractor (`solution_extractor.py`)

Extracts complete solution from solved models:
- Capacity decisions (grid, gas, battery, solar)
- Hourly dispatch for all 8760 hours
- Comprehensive metrics:
  - Cost metrics: NPV, CAPEX, OPEX, LCOE
  - Reliability metrics: uptime percentage, curtailment statistics
  - Carbon metrics: annual emissions, intensity, reduction vs baseline
  - Operational metrics: grid dependence, capacity factors, battery cycles

**Key Functions:**
```python
extract_solution(
    model: pyo.ConcreteModel,
    market_data: MarketData,
    tech_costs: TechnologyCosts,
    facility_params: FacilityParams,
    solve_time: float,
    optimality_gap: float = 0.0,
    scenario_params: Optional[Dict[str, Any]] = None
) -> OptimizationSolution

extract_worst_reliability_events(
    dispatch: DispatchSolution,
    market_data: MarketData,
    top_n: int = 10
) -> list
```

### 4. Solution Validator (`validator.py`)

Validates solutions satisfy all physical and operational constraints:
- Energy balance at every hour
- Capacity limits for all technologies
- Battery dynamics and SOC limits
- Gas ramping constraints
- Reliability constraint
- Non-negativity constraints

**Key Functions:**
```python
validate_solution(
    solution: OptimizationSolution,
    facility_params: FacilityParams,
    tech_costs: TechnologyCosts,
    tolerance: float = 1e-4
) -> Tuple[bool, List[str]]

validate_model_constraints(
    model: pyo.ConcreteModel,
    tolerance: float = 1e-6
) -> Tuple[bool, List[str]]

generate_validation_report(
    solution: OptimizationSolution,
    facility_params: FacilityParams,
    tech_costs: TechnologyCosts,
    tolerance: float = 1e-4
) -> Dict[str, Any]
```

## Usage Example

```python
from src.models.market_data import MarketData
from src.models.technology import TechnologyCosts, FacilityParams
from src.optimization import (
    build_optimization_model,
    solve_model,
    extract_solution,
    validate_solution
)

# Load market data
market_data = MarketData(...)

# Create parameters
tech_costs = TechnologyCosts()
facility_params = FacilityParams(
    it_load_mw=300,
    reliability_target=0.9999
)

# Build and solve model
model = build_optimization_model(market_data, tech_costs, facility_params)
results, solve_time = solve_model(model)

# Extract solution
solution = extract_solution(
    model=model,
    market_data=market_data,
    tech_costs=tech_costs,
    facility_params=facility_params,
    solve_time=solve_time
)

# Validate solution
is_valid, violations = validate_solution(
    solution=solution,
    facility_params=facility_params,
    tech_costs=tech_costs
)

if is_valid:
    print("Solution is valid!")
    print(f"Optimal LCOE: ${solution.metrics.lcoe:.2f}/MWh")
    print(f"Reliability: {solution.metrics.reliability_pct:.4f}%")
else:
    print(f"Solution has {len(violations)} violations")

# Save solution
solution.save("results/solutions/optimal_portfolio.json")
```

## Error Handling

The module provides comprehensive error handling:

- `InfeasibleError`: Problem has no feasible solution
- `UnboundedError`: Objective can decrease indefinitely
- `NumericalError`: Solver numerical difficulties
- `TimeoutError`: Solver exceeded time limit
- `ValidationError`: Solution validation failed

## Testing

Test scripts are available in `examples/`:
- `test_solution_extraction.py`: Full end-to-end test (requires Gurobi)
- `test_solution_extraction_mock.py`: Validation test with mock data (no solver required)

Run tests:
```bash
python examples/test_solution_extraction_mock.py
```

## Requirements

- Python 3.10+
- Pyomo 6.7+
- Gurobi 11.0+ (with valid license)
- NumPy 1.24+
- Pandas 2.0+

## Performance

- Full year optimization (8760 hours): 5-30 minutes depending on hardware
- Problem size: ~35,000 variables, ~70,000 constraints
- Memory usage: <8 GB for full year problem
