# Data Center Energy Optimization

A mathematical optimization system that determines the optimal energy portfolio and hourly dispatch strategy for large-scale AI training data centers. The system addresses the critical trilemma of cost minimization, reliability maximization, and carbon emission reduction while considering behind-the-meter power generation options.

## Business Problem

Large-scale AI training facilities face a critical energy challenge:

- **High Costs**: Electricity represents 20-30% of total operating costs, with annual bills exceeding $95M for a 300MW facility
- **Reliability Requirements**: AI training workloads cannot tolerate interruptions; a single hour of downtime wastes millions in GPU compute
- **Carbon Commitments**: Corporate climate goals and regulatory pressure demand low-carbon operations
- **Price Volatility**: Wholesale electricity markets exhibit extreme price swings, creating both risk and opportunity

Traditional approaches (grid-only, diesel backup, renewable PPAs) fail to optimize across all three dimensions simultaneously.

## Solution Approach

This project uses mathematical optimization to determine the economically optimal combination of energy assets:

1. **Grid Connection**: Purchase electricity from wholesale markets
2. **Natural Gas Peakers**: On-site generation for backup and peak shaving
3. **Battery Storage**: Energy arbitrage and reliability backup
4. **Solar PV**: On-site renewable generation

The optimization model solves a large-scale linear program with ~35,000 decision variables and ~70,000 constraints to minimize 20-year total cost while meeting reliability targets (99.99%+ uptime) and optional carbon constraints.

### Key Results

For a 300MW AI data center in West Texas, the optimal portfolio achieves:

- **57.9% cost reduction** over 20 years ($588M NPV savings)
- **55.6% carbon reduction** compared to grid-only baseline
- **0.7-year payback period** on $55.7M behind-the-meter investment
- **100% reliability** maintained

See the [Case Study](docs/case_study.md) for detailed analysis.

## Features

- **Automated Data Pipeline**: Collect real-world market data from ERCOT, NREL, and EIA APIs
- **Mathematical Optimization**: Pyomo modeling framework with Gurobi solver
- **Scenario Analysis**: Systematic exploration of parameter sensitivity and trade-offs
- **Pareto Frontier Analysis**: Visualize cost vs. reliability vs. carbon trade-offs
- **Interactive Dashboard**: Streamlit web interface for exploring scenarios
- **Comprehensive Visualizations**: Capacity mix, dispatch heatmaps, cost breakdowns, reliability analysis

## Project Structure

```
datacenter-energy-optimization/
├── data/                          # Data files and configurations
│   ├── processed/                 # Cleaned market data (ERCOT LMP, solar CF, gas prices, carbon intensity)
│   ├── raw/                       # Raw data from APIs (not committed to git)
│   └── tech_costs.json            # Technology cost parameters from NREL ATB 2024
│
├── src/                           # Core source code
│   ├── data_pipeline/             # Data collection and validation
│   │   ├── ercot_collector.py     # ERCOT LMP data collection
│   │   ├── solar_collector.py     # NREL PVWatts solar profile generation
│   │   ├── gas_collector.py       # EIA natural gas price collection
│   │   ├── carbon_collector.py    # Grid carbon intensity data
│   │   └── validator.py           # Data quality validation
│   │
│   ├── models/                    # Data models and structures
│   │   ├── market_data.py         # MarketData dataclass
│   │   ├── technology.py          # TechnologyCosts and FacilityParams
│   │   └── solution.py            # Solution data structures
│   │
│   ├── optimization/              # Optimization model and solver
│   │   ├── model_builder.py       # Pyomo model construction
│   │   ├── solver.py              # Gurobi solver interface
│   │   ├── solution_extractor.py  # Extract results from solved model
│   │   └── validator.py           # Constraint validation
│   │
│   ├── analysis/                  # Scenario and sensitivity analysis
│   │   ├── scenario_generator.py  # Generate parameter variations
│   │   ├── batch_solver.py        # Parallel scenario solving
│   │   ├── pareto_calculator.py   # Pareto frontier identification
│   │   └── sensitivity_analyzer.py # Parameter sensitivity analysis
│   │
│   ├── visualization/             # Plotting and visualization
│   │   ├── capacity_viz.py        # Capacity mix charts
│   │   ├── dispatch_viz.py        # Hourly dispatch heatmaps
│   │   ├── cost_viz.py            # Cost breakdown visualizations
│   │   ├── pareto_viz.py          # Pareto frontier plots
│   │   ├── reliability_viz.py     # Reliability analysis charts
│   │   └── sensitivity_viz.py     # Sensitivity tornado charts
│   │
│   └── config.py                  # Configuration management
│
├── dashboard/                     # Interactive Streamlit dashboard
│   ├── app.py                     # Main dashboard entry point
│   ├── pages/                     # Dashboard pages
│   │   ├── setup.py               # Optimization setup page
│   │   ├── portfolio.py           # Optimal portfolio results
│   │   ├── dispatch.py            # Hourly dispatch visualization
│   │   ├── scenarios.py           # Scenario comparison
│   │   └── case_study.py          # 300MW West Texas case study
│   ├── utils.py                   # Dashboard utilities
│   └── cache_config.py            # Caching configuration
│
├── scripts/                       # Utility scripts
│   ├── run_baseline.py            # Run grid-only baseline optimization
│   ├── run_optimal_portfolio.py   # Run optimal portfolio optimization
│   ├── run_case_study.py          # Generate case study results
│   ├── run_dashboard.sh           # Launch dashboard (Unix/Mac)
│   └── run_dashboard.bat          # Launch dashboard (Windows)
│
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests for individual modules
│   └── integration/               # End-to-end integration tests
│
├── docs/                          # Documentation
│   └── case_study.md              # Detailed 300MW West Texas case study
│
├── results/                       # Optimization results (not committed to git)
│   ├── solutions/                 # Individual solution JSON files
│   ├── scenarios/                 # Scenario analysis results
│   └── figures/                   # Generated visualizations
│
├── .kiro/specs/                   # Project specifications
│   └── datacenter-energy-optimization/
│       ├── requirements.md        # Detailed requirements document
│       ├── design.md              # System design and architecture
│       └── tasks.md               # Implementation task list
│
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore patterns
└── README.md                      # This file
```

## Setup Instructions

### Prerequisites

**Required:**
- Python 3.10 or higher
- Gurobi Optimizer 11.0+ (commercial solver with free academic license)

**Optional:**
- Git for version control
- 8GB+ RAM recommended for full-year optimizations
- Multi-core CPU for parallel scenario analysis

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/your-org/datacenter-energy-optimization.git
cd datacenter-energy-optimization
```

#### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- **Optimization**: Pyomo 6.7+, Gurobi 11.0+
- **Data Processing**: Pandas 2.0+, NumPy 1.24+
- **Visualization**: Plotly 5.18+, Matplotlib 3.8+
- **Dashboard**: Streamlit 1.28+
- **Testing**: pytest 7.4+, pytest-cov 4.1+

#### 4. Set Up Gurobi License

Gurobi is a commercial solver but offers free licenses for academic use.

**Option A: Web License Service (WLS) - Recommended**

If you have WLS credentials, use the interactive setup script:

```bash
python setup_gurobi.py
```

Or manually create `config/gurobi_wls.json`:

```json
{
  "WLSACCESSID": "your-wls-access-id",
  "WLSSECRET": "your-wls-secret",
  "LICENSEID": 12345
}
```

Get your WLS credentials at: https://license.gurobi.com/manager/licenses

**Option B: Academic License (Free)**

1. Register at [gurobi.com/academia](https://www.gurobi.com/academia/)
2. Download your license file
3. Follow Gurobi's activation instructions:
   ```bash
   grbgetkey YOUR-LICENSE-KEY
   ```

**Option C: Evaluation License (Free 30-day trial)**

1. Visit [gurobi.com/downloads](https://www.gurobi.com/downloads/)
2. Request
python -c "import gurobipy; print(gurobipy.gurobi.version())"
```

#### 5. Verify Setup

Run a quick test to ensure everything is installed correctly:

```bash
python -c "import pyomo.environ as pyo; import pandas as pd; import plotly; import streamlit; print('Setup successful!')"
```

### Quick Start

#### Option 1: Run Pre-Computed Case Study

The fastest way to explore the results is to use the pre-computed case study data:

```bash
# Launch the dashboard
streamlit run dashboard/app.py

# Or use the convenience script:
./scripts/run_dashboard.sh      # On Unix/Mac
scripts\run_dashboard.bat       # On Windows
```

The dashboard will open in your browser at `http://localhost:8501`. Navigate to the "Case Study" page to see the 300MW West Texas analysis.

#### Option 2: Run Baseline Optimization

Run the grid-only baseline scenario:

```bash
python scripts/run_baseline.py
```

This will:
- Load processed market data from `data/processed/`
- Build and solve the optimization model
- Save results to `results/solutions/baseline_grid_only.json`
- Generate summary statistics

Expected runtime: 2-5 minutes

#### Option 3: Run Optimal Portfolio Optimization

Run the full optimization with all technologies:

```bash
python scripts/run_optimal_portfolio.py
```

This will:
- Load processed market data
- Build the complete optimization model (~35,000 variables)
- Solve using Gurobi (may take 10-30 minutes)
- Save results to `results/solutions/optimal_portfolio.json`
- Generate visualizations in `results/figures/`

Expected runtime: 10-30 minutes depending on hardware

#### Option 4: Run Complete Case Study

Generate all case study results and visualizations:

```bash
python scripts/run_case_study.py
```

This will:
- Run both baseline and optimal scenarios
- Calculate comparative metrics
- Generate all visualizations
- Update the case study report

Expected runtime: 15-40 minutes

## Interactive Dashboard

The Streamlit dashboard provides a web-based interface for exploring optimization scenarios without writing code.

### Launching the Dashboard

```bash
streamlit run dashboard/app.py
```

Or use the convenience scripts:
```bash
./scripts/run_dashboard.sh      # Unix/Mac
scripts\run_dashboard.bat       # Windows
```

Access at: `http://localhost:8501`

### Dashboard Pages

#### 1. Optimization Setup
Configure and run custom optimization scenarios:

- **Facility Parameters**:
  - Facility size: 100-500 MW
  - Reliability target: 99.9% to 99.999% uptime
  - Carbon reduction goal: 0-100%
  
- **Technology Selection**:
  - Enable/disable grid connection, gas peakers, battery storage, solar PV
  - Customize cost parameters
  
- **Market Data**:
  - Select location (West Texas, Memphis, etc.)
  - Choose year scenario (2022, 2023, 2024)

- **Run Optimization**:
  - Click "Run Optimization" button
  - Monitor progress with status indicator
  - View solve time and optimality gap

#### 2. Optimal Portfolio
View and analyze optimization results:

- **Capacity Mix Visualization**:
  - Bar chart showing MW/MWh capacity for each technology
  - Pie chart showing percentage breakdown
  - Comparison to baseline scenario

- **Key Metrics Dashboard**:
  - Total NPV (20-year cost)
  - Levelized Cost of Energy (LCOE)
  - Reliability percentage
  - Carbon intensity (g CO2/kWh)
  - Grid independence percentage

- **Cost Breakdown**:
  - Waterfall chart showing CAPEX and OPEX components
  - Technology-specific cost contributions
  - Savings vs. baseline

- **Export Results**:
  - Download solution as JSON
  - Export metrics to CSV
  - Save visualizations as HTML

#### 3. Hourly Dispatch
Explore 8760-hour operational decisions:

- **Dispatch Heatmap**:
  - Interactive heatmap showing power contribution from each source
  - Color-coded by MW output
  - Hover tooltips with LMP, gas price, solar CF

- **Time Range Selection**:
  - Zoom into specific weeks or days
  - Compare summer vs. winter patterns
  - Identify peak demand periods

- **Operational Statistics**:
  - Gas utilization hours and capacity factor
  - Battery cycles per year
  - Solar capacity factor
  - Peak grid draw and timing

- **Battery State of Charge**:
  - Time series plot of battery SOC
  - Charging/discharging patterns
  - Correlation with electricity prices

#### 4. Scenario Comparison
Multi-scenario analysis and trade-off exploration:

- **Scenario Generator**:
  - Configure parameter variations (gas prices, battery costs, etc.)
  - Run batch optimizations in parallel
  - Compare results across scenarios

- **Pareto Frontier Plots**:
  - Cost vs. Reliability trade-off
  - Cost vs. Carbon trade-off
  - Grid Dependence vs. Reliability trade-off
  - Identify non-dominated solutions

- **Sensitivity Analysis**:
  - Tornado chart showing parameter impacts
  - Elasticity calculations
  - Breakeven point identification

- **Scenario Comparison Table**:
  - Side-by-side metrics for all scenarios
  - Sortable and filterable
  - Export to CSV

#### 5. Case Study
Pre-computed 300MW West Texas analysis:

- **Executive Summary**:
  - Key findings and recommendations
  - Financial metrics (NPV savings, payback period)
  - Carbon reduction achievements

- **Baseline vs. Optimal Comparison**:
  - Side-by-side capacity mix
  - Cost breakdown comparison
  - Reliability and carbon metrics

- **Strategic Insights**:
  - Why battery storage is the key enabler
  - Solar oversizing rationale
  - Comparison to alternative strategies (nuclear PPA, renewable PPA, diesel backup)

- **Implementation Roadmap**:
  - Phased deployment plan
  - Risk mitigation strategies
  - Timeline and milestones

- **Download Report**:
  - Export complete case study as PDF
  - Include all visualizations
  - Share with stakeholders

### Dashboard Performance

The dashboard uses caching to improve performance:
- Market data is cached on first load
- Optimization results are persisted in session state
- Visualizations are rendered on-demand

For large optimizations (full year, all technologies), expect:
- Initial solve: 10-30 minutes
- Subsequent page navigation: <2 seconds
- Visualization rendering: <1 second

## Usage Examples

### Example 1: Basic Optimization

Run a simple optimization for a 300MW facility:

```python
import pandas as pd
from src.optimization.model_builder import build_optimization_model
from src.optimization.solver import solve_model
from src.optimization.solution_extractor import extract_solution
from src.models.market_data import MarketData
from src.models.technology import TechnologyCosts, FacilityParams

# Load market data
lmp_data = pd.read_csv("data/processed/ercot_lmp_hourly_2022_2024.csv")
solar_cf = pd.read_csv("data/processed/solar_cf_west_texas.csv")
gas_prices = pd.read_csv("data/processed/gas_prices_hourly.csv")
carbon_intensity = pd.read_csv("data/processed/grid_carbon_intensity.csv")

# Create market data object
market_data = MarketData(
    timestamp=pd.to_datetime(lmp_data['timestamp']),
    lmp=lmp_data['lmp'].values,
    gas_price=gas_prices['price_mmbtu'].values,
    solar_cf=solar_cf['capacity_factor'].values,
    grid_carbon_intensity=carbon_intensity['carbon_intensity_kg_per_mwh'].values
)

# Configure facility parameters
facility = FacilityParams(
    it_load_mw=285.7,
    pue=1.05,
    reliability_target=0.9999,
    carbon_budget=None  # No carbon constraint
)

# Build optimization model
model = build_optimization_model(
    market_data=market_data,
    facility_params=facility,
    tech_costs=TechnologyCosts()
)

# Solve model
results, solve_time = solve_model(model)
print(f"Solved in {solve_time:.2f} seconds")

# Extract solution
solution = extract_solution(model)
print(f"Total NPV: ${solution.metrics.total_npv:,.0f}")
print(f"LCOE: ${solution.metrics.lcoe:.2f}/MWh")
print(f"Reliability: {solution.metrics.reliability_pct:.4f}%")
print(f"Carbon Intensity: {solution.metrics.carbon_intensity_g_per_kwh:.1f} g CO2/kWh")

# Save solution
solution.save("results/solutions/my_solution.json")
```

### Example 2: Scenario Analysis

Run multiple scenarios with parameter variations:

```python
from src.analysis.scenario_generator import generate_scenarios
from src.analysis.batch_solver import solve_scenarios

# Define base parameters
base_params = {
    "facility_size_mw": 300,
    "reliability_target": 0.9999,
    "carbon_budget": None
}

# Define parameter variations
variations = {
    "gas_price_multiplier": [0.5, 1.0, 1.5],
    "battery_cost_per_kwh": [200, 350, 500],
    "reliability_target": [0.999, 0.9999, 0.99999]
}

# Generate scenario parameter sets
scenarios = generate_scenarios(base_params, variations)
print(f"Generated {len(scenarios)} scenarios")

# Solve scenarios in parallel
solutions = solve_scenarios(scenarios, n_workers=4)

# Analyze results
for i, solution in enumerate(solutions):
    print(f"Scenario {i+1}: NPV=${solution.metrics.total_npv:,.0f}, "
          f"Carbon={solution.metrics.carbon_intensity_g_per_kwh:.1f} g/kWh")
```

### Example 3: Pareto Frontier Analysis

Identify trade-offs between competing objectives:

```python
from src.analysis.pareto_calculator import calculate_pareto_frontier
from src.visualization.pareto_viz import plot_pareto_frontier

# Calculate Pareto frontier for cost vs. carbon
pareto_df = calculate_pareto_frontier(
    solutions=solutions,
    objective1="total_npv",
    objective2="carbon_tons_annual"
)

print(f"Found {len(pareto_df)} Pareto-optimal solutions")

# Visualize Pareto frontier
fig = plot_pareto_frontier(
    solutions=solutions,
    x_metric="carbon_tons_annual",
    y_metric="total_npv",
    title="Cost vs. Carbon Trade-off"
)
fig.write_html("results/figures/pareto_cost_carbon.html")
```

### Example 4: Sensitivity Analysis

Quantify parameter impacts on optimal solution:

```python
from src.analysis.sensitivity_analyzer import analyze_sensitivity
from src.visualization.sensitivity_viz import plot_sensitivity_tornado

# Analyze sensitivity to gas prices
sensitivity_results = analyze_sensitivity(
    base_solution=baseline_solution,
    varied_solutions=gas_price_scenarios,
    parameter_name="gas_price"
)

print(f"Elasticity: {sensitivity_results['elasticity']:.2f}")
print(f"Breakeven point: ${sensitivity_results['breakeven_point']:.2f}/MMBtu")

# Create tornado chart
fig = plot_sensitivity_tornado(sensitivity_results)
fig.write_html("results/figures/sensitivity_tornado.html")
```

### Example 5: Custom Visualization

Create custom visualizations of results:

```python
from src.visualization.capacity_viz import plot_capacity_mix
from src.visualization.dispatch_viz import plot_dispatch_heatmap
from src.visualization.cost_viz import plot_cost_breakdown

# Capacity mix bar chart
fig1 = plot_capacity_mix(solution, format="bar")
fig1.write_html("results/figures/capacity_mix.html")

# Hourly dispatch heatmap (first week)
fig2 = plot_dispatch_heatmap(
    dispatch_df=solution.dispatch.to_dataframe(),
    time_range=(1, 168)  # First week
)
fig2.write_html("results/figures/dispatch_week1.html")

# Cost breakdown waterfall chart
fig3 = plot_cost_breakdown(solution)
fig3.write_html("results/figures/cost_breakdown.html")
```

### Example 6: Load and Compare Solutions

Load saved solutions and compare:

```python
from src.models.solution import OptimizationSolution

# Load solutions
baseline = OptimizationSolution.load("results/solutions/baseline_grid_only.json")
optimal = OptimizationSolution.load("results/solutions/optimal_portfolio.json")

# Compare metrics
print("Baseline vs. Optimal Comparison:")
print(f"NPV: ${baseline.metrics.total_npv:,.0f} vs ${optimal.metrics.total_npv:,.0f}")
print(f"Savings: ${baseline.metrics.total_npv - optimal.metrics.total_npv:,.0f} "
      f"({100*(1 - optimal.metrics.total_npv/baseline.metrics.total_npv):.1f}%)")
print(f"Carbon: {baseline.metrics.carbon_tons_annual:,.0f} vs "
      f"{optimal.metrics.carbon_tons_annual:,.0f} tons CO2/year")
print(f"Reduction: {100*(1 - optimal.metrics.carbon_tons_annual/baseline.metrics.carbon_tons_annual):.1f}%")
```

## Documentation

### Project Documentation

- **[Case Study](docs/case_study.md)**: Detailed analysis of 300MW West Texas facility
  - Executive summary with key findings
  - Baseline vs. optimal portfolio comparison
  - Financial analysis and ROI calculations
  - Strategic insights and recommendations
  - Implementation roadmap

- **[Requirements Document](.kiro/specs/datacenter-energy-optimization/requirements.md)**: Complete system requirements
  - User stories and acceptance criteria
  - EARS-compliant requirement specifications
  - Glossary of technical terms

- **[Design Document](.kiro/specs/datacenter-energy-optimization/design.md)**: System architecture and design
  - Component architecture
  - Mathematical formulation
  - Data models and interfaces
  - Error handling strategy
  - Testing approach

- **[Implementation Tasks](.kiro/specs/datacenter-energy-optimization/tasks.md)**: Development task list
  - Completed implementation tasks
  - Task dependencies and requirements
  - Progress tracking

### Module Documentation

- **[Optimization Module](src/optimization/README.md)**: Detailed documentation of optimization model
  - Mathematical formulation
  - Decision variables and constraints
  - Solver configuration
  - Solution extraction

- **[Analysis Module](src/analysis/README.md)**: Scenario and sensitivity analysis
  - Scenario generation
  - Batch solving
  - Pareto frontier calculation
  - Sensitivity analysis methods

- **[Visualization Module](src/visualization/README.md)**: Plotting and visualization
  - Available chart types
  - Customization options
  - Export formats

- **[Dashboard Documentation](dashboard/README.md)**: Dashboard usage guide
  - Page descriptions
  - Input parameter explanations
  - Performance optimization tips

### API Reference

For detailed API documentation, see the docstrings in each module. Key classes and functions:

- `build_optimization_model()`: Construct Pyomo optimization model
- `solve_model()`: Solve model using Gurobi
- `extract_solution()`: Extract results from solved model
- `generate_scenarios()`: Create parameter variation scenarios
- `calculate_pareto_frontier()`: Identify non-dominated solutions
- `plot_*()`: Visualization functions for various chart types

### External Resources

- **ERCOT Market Data**: [ercot.com/gridinfo](http://www.ercot.com/gridinfo/load/load_hist)
- **NREL PVWatts**: [pvwatts.nrel.gov](https://pvwatts.nrel.gov/)
- **EIA Natural Gas Prices**: [eia.gov/naturalgas](https://www.eia.gov/naturalgas/)
- **NREL Annual Technology Baseline**: [atb.nrel.gov](https://atb.nrel.gov/)
- **Gurobi Documentation**: [gurobi.com/documentation](https://www.gurobi.com/documentation/)
- **Pyomo Documentation**: [pyomo.readthedocs.io](https://pyomo.readthedocs.io/)

## Case Study Highlights

The project includes a comprehensive case study for a **300MW AI training data center in West Texas**:

### Baseline Scenario (Grid-Only)
- **Total NPV**: $1,017M over 20 years
- **Annual Cost**: $95.9M/year
- **LCOE**: $34.80/MWh
- **Carbon Intensity**: 340.5 g CO2/kWh
- **Reliability**: 100%

### Optimal Portfolio
- **Capacity Mix**:
  - Grid Connection: 3,860.5 MW
  - Battery Storage: 92,800.8 MWh
  - Solar PV: 9,664.9 MW
  - Gas Peakers: 0 MW

- **Financial Performance**:
  - **Total NPV**: $429M over 20 years
  - **NPV Savings**: $588M (57.9% reduction)
  - **Annual Cost**: $35.2M/year
  - **LCOE**: $14.67/MWh
  - **Payback Period**: 0.7 years

- **Environmental Performance**:
  - **Carbon Intensity**: 151.2 g CO2/kWh
  - **Emissions Reduction**: 55.6%
  - **Grid Dependence**: 44.1%

- **Operational Metrics**:
  - **Reliability**: 100%
  - **Battery Cycles**: 15.4/year
  - **Solar Capacity Factor**: 1.8%

### Key Insights

1. **Battery storage is the key enabler** for energy arbitrage in volatile markets
2. **Solar oversizing** (3.2x facility load) maximizes economic value
3. **Gas peakers are not cost-effective** compared to battery storage
4. **Grid connection should be oversized** to enable arbitrage opportunities
5. **Location matters**: West Texas offers unique advantages (negative prices, high solar resource)
6. **Carbon and cost objectives align**: Economic optimization naturally favors renewables
7. **Payback period is extremely short**: 0.7 years for $55.7M investment

See the [complete case study](docs/case_study.md) for detailed analysis.

## Testing

The project includes comprehensive unit and integration tests.

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run with coverage report:
```bash
pytest --cov=src --cov-report=html tests/
```

Run specific test modules:
```bash
pytest tests/unit/test_optimization_model.py
pytest tests/integration/test_end_to_end.py
```

Run with verbose output:
```bash
pytest -v tests/
```

### Test Coverage

The test suite covers:
- **Data Pipeline**: API collectors, validators, data models
- **Optimization Model**: Model construction, constraint generation, solution extraction
- **Analysis**: Scenario generation, Pareto frontier calculation, sensitivity analysis
- **Visualization**: Chart generation and formatting
- **Integration**: End-to-end optimization pipeline

Target coverage: >90% for core modules

### Test Data

Tests use:
- **Synthetic data**: Controlled test cases with known outcomes
- **Historical subsets**: 1-week samples for fast integration tests
- **Mock API responses**: Simulated API data for unit tests

## Performance Considerations

### Optimization Solve Time

Expected solve times on a modern laptop (8-core CPU, 16GB RAM):

| Problem Size | Variables | Constraints | Solve Time |
|--------------|-----------|-------------|------------|
| 1 day (24 hours) | ~100 | ~200 | <10 seconds |
| 1 week (168 hours) | ~700 | ~1,400 | 30-60 seconds |
| 1 month (720 hours) | ~3,000 | ~6,000 | 2-5 minutes |
| 1 year (8760 hours) | ~35,000 | ~70,000 | 10-30 minutes |

### Memory Usage

- **Data loading**: ~500MB for full year of hourly data
- **Model construction**: ~1GB for Pyomo model objects
- **Solver**: ~2-4GB during optimization
- **Total peak**: ~6-8GB for full-year optimization

### Optimization Tips

1. **Use MIP gap tolerance**: Set to 0.5% for faster solving
2. **Parallel scenario solving**: Use multiprocessing for batch runs
3. **Cache data**: Load market data once and reuse
4. **Reduce problem size**: Use weekly or monthly aggregation for initial exploration
5. **Warm start**: Use previous solution as starting point for similar scenarios

## Troubleshooting

### Common Issues

**Issue: Gurobi license not found**
```
GurobiError: No Gurobi license found
```
**Solution**: Activate your Gurobi license using `grbgetkey YOUR-LICENSE-KEY`

**Issue: Optimization is infeasible**
```
WARNING: Loading a SolverResults object with a warning status into model
```
**Solution**: 
- Check reliability constraint (may be too strict)
- Verify capacity bounds are reasonable
- Review data for anomalies (negative prices, missing values)

**Issue: Solve time is too long**
```
Optimization taking >1 hour
```
**Solution**:
- Increase MIP gap tolerance to 1-2%
- Reduce problem size (use weekly instead of annual)
- Check for numerical issues (scale variables)
- Ensure Gurobi is using all CPU cores

**Issue: Memory error during solving**
```
MemoryError: Unable to allocate array
```
**Solution**:
- Close other applications
- Reduce problem size
- Use 64-bit Python
- Consider cloud computing resources

**Issue: Dashboard is slow**
```
Streamlit app is unresponsive
```
**Solution**:
- Clear Streamlit cache: `streamlit cache clear`
- Reduce visualization time ranges
- Pre-compute scenarios offline
- Use caching decorators (`@st.cache_data`)

### Getting Help

If you encounter issues:

1. Check the [documentation](#documentation) for detailed guides
2. Review [test examples](tests/) for usage patterns
3. Examine [example scripts](examples/) for working code
4. Check Gurobi and Pyomo documentation for solver-specific issues

## Contributing

Contributions are welcome! Areas for improvement:

- **Additional data sources**: Support for other ISOs (CAISO, PJM, etc.)
- **New technologies**: Add hydrogen storage, fuel cells, wind turbines
- **Advanced features**: Multi-year planning, demand response, grid services
- **Performance**: Faster solving algorithms, better parallelization
- **Visualization**: Additional chart types, interactive dashboards
- **Documentation**: Tutorials, video guides, case studies

Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation
6. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this project in academic research, please cite:

```bibtex
@software{datacenter_energy_optimization,
  title = {Data Center Energy Optimization: Mathematical Optimization for AI Training Facilities},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-org/datacenter-energy-optimization}
}
```

## Acknowledgments

- **ERCOT** for providing open access to market data
- **NREL** for PVWatts API and Annual Technology Baseline
- **EIA** for natural gas price data and grid carbon intensity
- **Gurobi** for academic licenses
- **Pyomo** development team for the optimization framework

## Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: your.email@example.com
- **GitHub Issues**: [github.com/your-org/datacenter-energy-optimization/issues](https://github.com/your-org/datacenter-energy-optimization/issues)
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

**Built with**: Python, Pyomo, Gurobi, Streamlit, Plotly

**Last Updated**: 2024
