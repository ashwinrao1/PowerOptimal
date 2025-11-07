# Visualization Module

This module provides interactive visualization functions for data center energy optimization results using Plotly.

## Capacity Visualizations

The `capacity_viz.py` module provides functions for visualizing optimal capacity mix decisions.

### Functions

#### `plot_capacity_mix(solution, format="bar", title=None, show_values=True)`

Create interactive capacity mix visualization in multiple formats.

**Parameters:**
- `solution`: CapacitySolution, OptimizationSolution, or dict with capacity data
- `format`: Visualization format - "bar", "pie", or "waterfall"
- `title`: Custom title for the plot (optional)
- `show_values`: Whether to show values on the plot (default: True)

**Returns:**
- Plotly Figure object

**Supported Formats:**

1. **Bar Chart** (`format="bar"`): Grouped bar chart showing capacity for each technology
   - Grid Connection (MW)
   - Gas Peakers (MW)
   - Solar PV (MW)
   - Battery Storage (MWh)

2. **Pie Chart** (`format="pie"`): Percentage breakdown of capacity
   - Battery storage is converted to MW equivalent (assuming 4-hour duration)
   - Shows relative contribution of each technology

3. **Waterfall Chart** (`format="waterfall"`): Cumulative capacity buildup
   - Shows how total capacity is built up from individual technologies
   - Useful for understanding capacity composition

**Example Usage:**

```python
from src.models.solution import OptimizationSolution
from src.visualization import plot_capacity_mix

# Load solution
solution = OptimizationSolution.load("results/solutions/optimal_portfolio.json")

# Create bar chart
fig_bar = plot_capacity_mix(solution, format="bar")
fig_bar.show()

# Create pie chart
fig_pie = plot_capacity_mix(solution, format="pie", title="Capacity Distribution")
fig_pie.write_html("capacity_pie.html")

# Create waterfall chart
fig_waterfall = plot_capacity_mix(solution, format="waterfall", show_values=False)
fig_waterfall.show()
```

#### `plot_capacity_comparison(solutions, title=None, show_values=True)`

Create grouped bar chart comparing capacity across multiple scenarios.

**Parameters:**
- `solutions`: Dictionary mapping scenario names to solution objects
- `title`: Custom title for the plot (optional)
- `show_values`: Whether to show values on bars (default: True)

**Returns:**
- Plotly Figure object

**Example Usage:**

```python
from src.models.solution import OptimizationSolution
from src.visualization import plot_capacity_comparison

# Load multiple solutions
baseline = OptimizationSolution.load("results/solutions/baseline_grid_only.json")
optimal = OptimizationSolution.load("results/solutions/optimal_portfolio.json")

# Create comparison chart
solutions = {
    "Baseline (Grid Only)": baseline,
    "Optimal Portfolio": optimal
}

fig = plot_capacity_comparison(solutions)
fig.show()
```

### Features

- **Interactive**: All visualizations are interactive with hover tooltips
- **Customizable**: Support for custom titles and value display options
- **Flexible Input**: Accepts CapacitySolution, OptimizationSolution, or dict
- **Color-Coded**: Consistent color scheme across all visualizations
  - Grid Connection: Blue (#1f77b4)
  - Gas Peakers: Orange (#ff7f0e)
  - Solar PV: Green (#2ca02c)
  - Battery Storage: Red (#d62728)

### Demo Scripts

Two example scripts are provided in the `examples/` directory:

1. **`test_capacity_viz.py`**: Basic functionality test
   - Tests all visualization formats
   - Validates with real solution data
   - Prints summary information

2. **`capacity_viz_demo.py`**: Comprehensive demonstration
   - Creates all visualization types
   - Saves HTML files to `results/figures/`
   - Includes comparison charts

Run the demo:
```bash
python examples/capacity_viz_demo.py
```

This will create interactive HTML files in `results/figures/` that can be opened in any web browser.

## Dispatch Visualizations

The `dispatch_viz.py` module provides functions for visualizing hourly dispatch decisions and operational patterns.

### Functions

#### `plot_dispatch_heatmap(solution, market_data=None, time_range=None, title=None, height=600)`

Create interactive 2D heatmap of hourly dispatch decisions showing power contribution from each source across all hours.

**Parameters:**
- `solution`: DispatchSolution, OptimizationSolution, or dict with dispatch data
- `market_data`: Optional MarketData object, DataFrame, or dict with LMP, gas prices, and solar CF for enhanced hover tooltips
- `time_range`: Optional tuple (start_hour, end_hour) for zooming into specific period. Hours are 1-indexed (1-8760)
- `title`: Custom title for the plot (optional)
- `height`: Figure height in pixels (default: 600)

**Returns:**
- Plotly Figure object with interactive heatmap

**Heatmap Layout:**
- **X-axis**: Hours of the year (1-8760)
- **Y-axis**: Power sources (Grid, Gas, Solar, Battery Discharge, Battery Charge)
- **Color**: MW contribution from each source (Viridis colorscale)
- **Hover tooltips**: Show hour, power value, and optional market data (LMP, gas price, solar CF)

**Features:**
- Interactive range slider for zooming across time periods
- Separate battery charge/discharge for clarity
- Highlights curtailment events in hover text
- Shows battery state of charge for battery rows

**Example Usage:**

```python
from src.models.solution import OptimizationSolution
from src.visualization import plot_dispatch_heatmap

# Load solution
solution = OptimizationSolution.load("results/solutions/optimal_portfolio.json")

# Full year heatmap
fig1 = plot_dispatch_heatmap(solution)
fig1.show()

# Focus on first week
fig2 = plot_dispatch_heatmap(solution, time_range=(1, 168))
fig2.write_html("dispatch_week1.html")

# With market data for enhanced tooltips
import pandas as pd
market_data = {
    "lmp": pd.read_csv("data/processed/ercot_lmp_hourly_2022_2024.csv").iloc[:8760, 1].values,
    "gas_price": pd.read_csv("data/processed/gas_prices_hourly.csv").iloc[:8760, 1].values,
    "solar_cf": pd.read_csv("data/processed/solar_cf_west_texas.csv").iloc[:8760, 1].values
}
fig3 = plot_dispatch_heatmap(solution, market_data=market_data, time_range=(1, 168))
fig3.show()
```

#### `plot_dispatch_stacked_area(solution, time_range=None, title=None, height=500)`

Create stacked area chart of hourly dispatch decisions showing power contribution as stacked areas over time.

**Parameters:**
- `solution`: DispatchSolution, OptimizationSolution, or dict with dispatch data
- `time_range`: Optional tuple (start_hour, end_hour) for zooming into specific period
- `title`: Custom title for the plot (optional)
- `height`: Figure height in pixels (default: 500)

**Returns:**
- Plotly Figure object with stacked area chart

**Features:**
- Shows overall generation mix and how it changes over time
- Stacked areas make it easy to see total supply
- Unified hover mode shows all sources at once
- Only shows battery discharge (not charge) for clarity

**Example Usage:**

```python
from src.models.solution import OptimizationSolution
from src.visualization import plot_dispatch_stacked_area

# Load solution
solution = OptimizationSolution.load("results/solutions/optimal_portfolio.json")

# First month stacked area
fig = plot_dispatch_stacked_area(solution, time_range=(1, 744))
fig.show()
```

### Color Scheme

Consistent colors across all dispatch visualizations:
- **Grid**: Blue (#1f77b4)
- **Gas**: Orange (#ff7f0e)
- **Solar**: Green (#2ca02c)
- **Battery**: Red (#d62728)

### Demo Scripts

Two example scripts are provided in the `examples/` directory:

1. **`test_dispatch_viz.py`**: Comprehensive test suite
   - Tests all visualization functions
   - Tests with different time ranges
   - Tests with and without market data
   - Validates error handling
   - Saves multiple HTML outputs

2. **`dispatch_viz_demo.py`**: Interactive demonstration
   - Shows key visualization capabilities
   - Opens visualizations in browser
   - Demonstrates time range selection

Run the test:
```bash
python examples/test_dispatch_viz.py
```

Run the demo:
```bash
python examples/dispatch_viz_demo.py
```

### Use Cases

**Full Year Analysis:**
- Identify seasonal patterns in dispatch
- See when different technologies are utilized
- Spot periods of high grid dependence

**Weekly/Monthly Focus:**
- Analyze daily dispatch cycles
- Understand battery charge/discharge patterns
- Examine response to weather patterns

**With Market Data:**
- Correlate dispatch decisions with electricity prices
- See how solar generation tracks capacity factors
- Understand gas peaker economics

## Cost Breakdown Visualizations

The `cost_viz.py` module provides functions for visualizing cost breakdowns by technology and cost type (CAPEX vs OPEX).

### Functions

#### `plot_cost_breakdown(solution, tech_costs=None, title=None, show_values=True, format="waterfall")`

Create cost breakdown visualization showing CAPEX and OPEX components by technology.

**Parameters:**
- `solution`: OptimizationSolution or dict with solution data
- `tech_costs`: TechnologyCosts object for calculating component costs. If None, uses default costs.
- `title`: Custom title for the plot (optional)
- `show_values`: Whether to show values on the chart (default: True)
- `format`: Visualization format - "waterfall" or "stacked_bar"

**Returns:**
- Plotly Figure object

**Supported Formats:**

1. **Waterfall Chart** (`format="waterfall"`): Shows cost buildup step by step
   - Individual CAPEX components (Grid, Gas, Battery, Solar)
   - Total CAPEX subtotal
   - Individual Annual OPEX components (Grid, Gas, Battery, Solar)
   - Total Annual OPEX subtotal
   - Displays Total 20-Year NPV in annotation

2. **Stacked Bar Chart** (`format="stacked_bar"`): Shows CAPEX vs OPEX by technology
   - Each technology has a stacked bar showing CAPEX and Annual OPEX
   - Annotation shows totals (CAPEX, Annual OPEX, 20-Year NPV)
   - Easy comparison of cost structure across technologies

**Cost Components:**

The function breaks down costs into:
- **Grid CAPEX**: Interconnection capacity × $3000/kW
- **Gas CAPEX**: Peaker capacity × $1000/kW
- **Battery CAPEX**: Storage capacity × $350/kWh
- **Solar CAPEX**: PV capacity × $1200/kW
- **Grid OPEX**: Energy purchases, demand charges (annual)
- **Gas OPEX**: Fuel costs, O&M (annual)
- **Battery OPEX**: Degradation costs (annual)
- **Solar OPEX**: Fixed O&M (annual)

**Example Usage:**

```python
from src.models.solution import OptimizationSolution
from src.visualization import plot_cost_breakdown

# Load solution
solution = OptimizationSolution.load("results/solutions/optimal_portfolio.json")

# Create waterfall chart
fig_waterfall = plot_cost_breakdown(solution, format="waterfall")
fig_waterfall.show()

# Create stacked bar chart
fig_stacked = plot_cost_breakdown(
    solution, 
    format="stacked_bar",
    title="Cost Breakdown by Technology"
)
fig_stacked.write_html("cost_breakdown.html")

# Use custom technology costs
from src.models.technology import TechnologyCosts
custom_costs = TechnologyCosts(battery_capex_per_kwh=300)
fig = plot_cost_breakdown(solution, tech_costs=custom_costs)
fig.show()
```

#### `plot_cost_comparison(solutions, tech_costs=None, title=None, metric="total_npv")`

Create bar chart comparing costs across multiple scenarios.

**Parameters:**
- `solutions`: Dictionary mapping scenario names to solution objects
- `tech_costs`: TechnologyCosts object for calculating component costs
- `title`: Custom title for the plot (optional)
- `metric`: Cost metric to compare - "total_npv", "capex", or "opex_annual"

**Returns:**
- Plotly Figure object

**Supported Metrics:**
- `total_npv`: Total 20-year net present value
- `capex`: Capital expenditure (upfront investment)
- `opex_annual`: Annual operating expenditure

**Example Usage:**

```python
from src.models.solution import OptimizationSolution
from src.visualization import plot_cost_comparison

# Load multiple solutions
baseline = OptimizationSolution.load("results/solutions/baseline_grid_only.json")
optimal = OptimizationSolution.load("results/solutions/optimal_portfolio.json")

solutions = {
    "Baseline (Grid Only)": baseline,
    "Optimal Portfolio": optimal
}

# Compare total NPV
fig_npv = plot_cost_comparison(solutions, metric="total_npv")
fig_npv.show()

# Compare CAPEX
fig_capex = plot_cost_comparison(solutions, metric="capex")
fig_capex.show()

# Compare annual OPEX
fig_opex = plot_cost_comparison(solutions, metric="opex_annual")
fig_opex.show()
```

### Features

- **Interactive**: All visualizations are interactive with hover tooltips showing exact dollar amounts
- **Formatted Values**: Displays costs in millions (e.g., "$55.7M") for readability
- **Comprehensive**: Shows both upfront (CAPEX) and ongoing (OPEX) costs
- **Flexible Input**: Accepts OptimizationSolution or dict format
- **Customizable**: Support for custom technology costs and titles

### Demo Scripts

Two example scripts are provided in the `examples/` directory:

1. **`test_cost_viz.py`**: Comprehensive test suite
   - Tests waterfall and stacked bar formats
   - Tests cost comparison across scenarios
   - Tests with dict format (JSON compatibility)
   - Validates with real solution data

2. **`cost_viz_demo.py`**: Simple demonstration
   - Shows basic usage patterns
   - Creates both visualization formats
   - Prints cost summary

Run the test:
```bash
python examples/test_cost_viz.py
```

Run the demo:
```bash
python examples/cost_viz_demo.py
```

This will create interactive HTML files in `results/figures/` that can be opened in any web browser.

### Use Cases

**Investment Analysis:**
- Compare upfront CAPEX across different portfolio options
- Understand which technologies require the most capital
- Evaluate total 20-year NPV for different scenarios

**Operating Cost Analysis:**
- Identify which technologies have highest ongoing costs
- Compare annual OPEX across scenarios
- Understand cost structure (CAPEX-heavy vs OPEX-heavy)

**Technology Trade-offs:**
- See cost breakdown by technology
- Compare grid-only vs diversified portfolios
- Evaluate impact of adding solar, battery, or gas

## Pareto Frontier Visualizations

The `pareto_viz.py` module provides functions for visualizing Pareto frontiers and multi-objective trade-off analysis.

### Functions

#### `plot_pareto_frontier(solutions, objective1, objective2, title=None, baseline_solution=None, optimal_solution=None, show_all_solutions=True, height=600)`

Create scatter plot showing Pareto frontier with two objectives, highlighting non-dominated solutions.

**Parameters:**
- `solutions`: One of:
  - DataFrame with Pareto frontier data (from pareto_calculator)
  - List of solution dictionaries from batch_solver
  - Dict mapping scenario names to OptimizationSolution objects
- `objective1`: Name of first objective (x-axis)
  - Common values: 'total_npv', 'grid_dependence_pct', 'carbon_tons_annual'
- `objective2`: Name of second objective (y-axis)
  - Common values: 'reliability_pct', 'carbon_tons_annual', 'carbon_intensity_g_per_kwh'
- `title`: Custom title for the plot (optional)
- `baseline_solution`: Optional baseline solution to annotate (e.g., grid-only)
- `optimal_solution`: Optional optimal solution to annotate
- `show_all_solutions`: If True, show all solutions; if False, only show Pareto-optimal (default: True)
- `height`: Figure height in pixels (default: 600)

**Returns:**
- Plotly Figure object with interactive scatter plot

**Features:**
- **Pareto-Optimal Highlighting**: Diamond markers with connecting line for Pareto frontier
- **Non-Dominated Solutions**: Gray circles for dominated solutions (if show_all_solutions=True)
- **Key Solution Annotations**: Special markers and labels for baseline and optimal solutions
- **Extreme Point Labels**: Automatic annotation of min/max points on Pareto frontier
- **Smart Axis Formatting**: Automatic formatting based on objective type (currency, percentage, etc.)

**Common Objective Pairs:**
1. **Cost vs Reliability**: `('total_npv', 'reliability_pct')`
   - Shows trade-off between cost and uptime
   - Higher reliability requires more investment
   
2. **Cost vs Carbon**: `('total_npv', 'carbon_tons_annual')`
   - Shows trade-off between cost and emissions
   - Lower carbon typically costs more
   
3. **Grid Dependence vs Reliability**: `('grid_dependence_pct', 'reliability_pct')`
   - Shows trade-off between grid independence and reliability
   - Less grid dependence may reduce reliability

**Example Usage:**

```python
from src.analysis.pareto_calculator import calculate_pareto_frontier
from src.visualization import plot_pareto_frontier

# Calculate Pareto frontier from batch results
pareto_df = calculate_pareto_frontier(
    batch_results,
    objective1='total_npv',
    objective2='reliability_pct',
    minimize_obj1=True,
    minimize_obj2=False  # Maximize reliability
)

# Basic plot
fig1 = plot_pareto_frontier(
    pareto_df,
    objective1='total_npv',
    objective2='reliability_pct'
)
fig1.show()

# With baseline and optimal annotations
baseline = OptimizationSolution.load("results/solutions/baseline_grid_only.json")
optimal = OptimizationSolution.load("results/solutions/optimal_portfolio.json")

fig2 = plot_pareto_frontier(
    pareto_df,
    objective1='total_npv',
    objective2='carbon_tons_annual',
    baseline_solution=baseline,
    optimal_solution=optimal,
    title='Cost vs Carbon: Baseline and Optimal Solutions'
)
fig2.write_html("pareto_cost_carbon.html")

# Show only Pareto-optimal solutions
fig3 = plot_pareto_frontier(
    pareto_df,
    objective1='grid_dependence_pct',
    objective2='reliability_pct',
    show_all_solutions=False
)
fig3.show()
```

#### `plot_multiple_pareto_frontiers(frontiers, objective_pairs=None, title=None, height=500)`

Create subplots showing multiple Pareto frontiers for easy comparison of different objective trade-offs.

**Parameters:**
- `frontiers`: Dictionary mapping frontier names to DataFrames
  - Expected keys: 'cost_reliability', 'cost_carbon', 'grid_reliability'
- `objective_pairs`: Optional list of (obj1, obj2) tuples for each frontier. If None, uses standard pairs based on frontier names.
- `title`: Custom title for the overall figure (optional)
- `height`: Figure height in pixels per subplot row (default: 500)

**Returns:**
- Plotly Figure object with subplots

**Standard Frontier Names:**
- `cost_reliability`: Cost vs Reliability trade-off
- `cost_carbon`: Cost vs Carbon Emissions trade-off
- `grid_reliability`: Grid Dependence vs Reliability trade-off

**Features:**
- **Multi-Panel Layout**: Automatically arranges frontiers in grid (up to 2 columns)
- **Consistent Styling**: All frontiers use same visual style
- **Interactive**: Each subplot is independently interactive
- **Automatic Labeling**: Axes and titles automatically formatted

**Example Usage:**

```python
from src.analysis.pareto_calculator import calculate_all_pareto_frontiers
from src.visualization import plot_multiple_pareto_frontiers

# Calculate all standard Pareto frontiers
frontiers = calculate_all_pareto_frontiers(batch_results)

# Create multi-panel plot
fig = plot_multiple_pareto_frontiers(
    frontiers,
    title='Comprehensive Pareto Frontier Analysis'
)
fig.show()

# Save to file
fig.write_html("results/figures/all_pareto_frontiers.html")
```

### Supported Objectives

The visualization automatically formats axes based on objective type:

**Cost Objectives** (formatted as currency):
- `total_npv`: Total 20-year net present value
- `capex`: Capital expenditure
- `opex_annual`: Annual operating expenditure
- `lcoe`: Levelized cost of energy

**Performance Objectives** (formatted as percentage):
- `reliability_pct`: Reliability percentage
- `grid_dependence_pct`: Grid dependence percentage
- `carbon_reduction_pct`: Carbon reduction percentage
- `gas_capacity_factor`: Gas utilization
- `solar_capacity_factor`: Solar capacity factor

**Emissions Objectives** (formatted as numbers):
- `carbon_tons_annual`: Annual carbon emissions (tons CO2)
- `carbon_intensity_g_per_kwh`: Carbon intensity (g CO2/kWh)

**Other Objectives**:
- `total_curtailment_mwh`: Total curtailment
- `battery_cycles_per_year`: Battery cycles

### Color Scheme

Consistent colors across Pareto visualizations:
- **Pareto-Optimal Solutions**: Blue diamonds (#1f77b4) with connecting line
- **Non-Pareto Solutions**: Light gray circles
- **Baseline Solution**: Red square (#d62728)
- **Optimal Solution**: Green star (#2ca02c)
- **Extreme Points**: Gray annotations

### Demo Scripts

Two example scripts are provided in the `examples/` directory:

1. **`test_pareto_viz.py`**: Comprehensive test suite
   - Tests with existing Pareto frontier data
   - Tests with mock data
   - Tests all visualization functions
   - Tests with and without annotations
   - Validates error handling

2. **`pareto_usage_example.py`**: Usage examples
   - Example 1: Basic Pareto frontier plot
   - Example 2: Plot with baseline and optimal annotations
   - Example 3: Multiple Pareto frontiers in subplots
   - Example 4: Create Pareto frontier from batch results
   - Example 5: Show both Pareto and non-Pareto solutions

Run the test:
```bash
python examples/test_pareto_viz.py
```

Run the examples:
```bash
python examples/pareto_usage_example.py
```

This will create interactive HTML files in `results/figures/` that can be opened in any web browser.

### Use Cases

**Trade-off Analysis:**
- Visualize cost vs reliability trade-offs
- Understand cost of reducing carbon emissions
- Evaluate grid independence vs reliability

**Scenario Comparison:**
- Compare multiple optimization scenarios
- Identify non-dominated solutions
- Find best compromise solutions

**Decision Support:**
- Annotate baseline and optimal solutions
- Highlight extreme points (min cost, max reliability, etc.)
- Show all solutions to understand solution space

**Reporting:**
- Create multi-panel reports showing all trade-offs
- Export interactive HTML for stakeholder review
- Generate publication-quality visualizations

## Reliability Analysis Visualizations

The `reliability_viz.py` module provides functions for analyzing and visualizing system reliability, curtailment events, and reserve margins.

### Functions

#### `plot_reliability_analysis(solution, title=None, height=800)`

Create comprehensive multi-panel reliability analysis visualization showing:
1. Histogram of hourly curtailment events
2. Time series of reserve margin over the year
3. Top 10 worst-case reliability events
4. Summary statistics table

**Parameters:**
- `solution`: OptimizationSolution or dict with solution data
- `title`: Custom title for the plot (optional)
- `height`: Figure height in pixels (default: 800)

**Returns:**
- Plotly Figure object with 4-panel layout (2×2 grid)

**Panel Descriptions:**

1. **Curtailment Event Histogram** (Top Left)
   - Distribution of curtailment magnitudes
   - Shows frequency of different curtailment levels
   - Displays "No curtailment events" if 100% reliability achieved

2. **Reserve Margin Over Time** (Top Right)
   - Weekly average reserve margin as percentage
   - Green markers for positive reserve, red for negative
   - Dashed line at zero reserve
   - Reserve margin = (Available Generation - Load) / Load × 100%

3. **Top 10 Worst Reliability Events** (Bottom Left)
   - Bar chart of hours with highest curtailment
   - Shows specific hour numbers for investigation
   - Displays "No significant curtailment events" if none found

4. **Reliability Statistics Table** (Bottom Right)
   - Reliability target and actual reliability
   - Total uptime and curtailment hours
   - Total curtailment energy (MWh)
   - Maximum single-hour curtailment
   - Average curtailment when it occurs
   - Number of curtailment events

**Example Usage:**

```python
from src.models.solution import OptimizationSolution
from src.visualization import plot_reliability_analysis

# Load solution
solution = OptimizationSolution.load("results/solutions/optimal_portfolio.json")

# Create comprehensive reliability analysis
fig = plot_reliability_analysis(solution)
fig.show()

# Save to file
fig.write_html("results/figures/reliability_analysis.html")

# With custom title
fig = plot_reliability_analysis(
    solution,
    title="Reliability Analysis - 300MW West Texas Facility"
)
fig.show()
```

#### `plot_curtailment_histogram(solution, title=None, bins=30, height=500)`

Create histogram showing distribution of curtailment event magnitudes.

**Parameters:**
- `solution`: OptimizationSolution or dict with solution data
- `title`: Custom title for the plot (optional)
- `bins`: Number of histogram bins (default: 30)
- `height`: Figure height in pixels (default: 500)

**Returns:**
- Plotly Figure object with curtailment histogram

**Features:**
- Shows only hours where curtailment occurred (> 0.01 MW)
- Annotation box with statistics (total events, mean, max)
- Red color scheme to highlight reliability issues
- Displays success message if no curtailment events

**Example Usage:**

```python
from src.visualization import plot_curtailment_histogram

# Basic histogram
fig = plot_curtailment_histogram(solution)
fig.show()

# With more bins for detailed distribution
fig = plot_curtailment_histogram(solution, bins=50)
fig.write_html("curtailment_detailed.html")
```

#### `plot_reserve_margin_timeseries(solution, load_mw=285, time_range=None, title=None, height=500)`

Create time series plot showing reserve margin over the year.

**Parameters:**
- `solution`: OptimizationSolution or dict with solution data
- `load_mw`: Data center load in MW (default: 285)
- `time_range`: Optional tuple (start_hour, end_hour) for zooming into specific period
- `title`: Custom title for the plot (optional)
- `height`: Figure height in pixels (default: 500)

**Returns:**
- Plotly Figure object with reserve margin time series

**Features:**
- Reserve margin = (Actual Supply - Load) / Load × 100%
- Blue line shows reserve margin over time
- Dashed gray line at zero reserve
- Red X markers highlight curtailment events
- Negative reserve indicates insufficient capacity

**Example Usage:**

```python
from src.visualization import plot_reserve_margin_timeseries

# Full year reserve margin
fig = plot_reserve_margin_timeseries(solution)
fig.show()

# Focus on summer months (hours 4000-5000)
fig = plot_reserve_margin_timeseries(
    solution,
    time_range=(4000, 5000),
    title="Reserve Margin - Summer Period"
)
fig.show()

# First week analysis
fig = plot_reserve_margin_timeseries(
    solution,
    time_range=(1, 168),
    title="Reserve Margin - First Week"
)
fig.write_html("reserve_margin_week1.html")
```

#### `plot_worst_reliability_events(solution, n_events=10, title=None, height=500)`

Identify and visualize the N worst-case reliability events (hours with highest curtailment).

**Parameters:**
- `solution`: OptimizationSolution or dict with solution data
- `n_events`: Number of worst events to show (default: 10)
- `title`: Custom title for the plot (optional)
- `height`: Figure height in pixels (default: 500)

**Returns:**
- Plotly Figure object with worst events bar chart

**Features:**
- Shows hour number and day/time for each event
- Red bars highlight severity
- Dashed orange line shows mean curtailment
- Labels include both hour of year and day/hour breakdown
- Displays success message if no significant events

**Example Usage:**

```python
from src.visualization import plot_worst_reliability_events

# Top 10 worst events
fig = plot_worst_reliability_events(solution)
fig.show()

# Top 20 worst events
fig = plot_worst_reliability_events(
    solution,
    n_events=20,
    title="Top 20 Worst Reliability Events"
)
fig.write_html("worst_events_top20.html")
```

### Key Metrics

**Reliability Percentage:**
- Calculated as: (1 - Total Curtailment MWh / (Load MW × Hours)) × 100%
- Target: 99.99% (equivalent to ~1 hour downtime per year for 285 MW load)

**Reserve Margin:**
- Indicates excess capacity beyond required load
- Positive values: System has spare capacity
- Negative values: Insufficient capacity (curtailment occurs)
- Calculated hourly: (Available Generation - Load) / Load × 100%

**Curtailment Events:**
- Any hour where load cannot be fully met
- Measured in MW of unserved load
- Critical for understanding system stress points

### Color Scheme

Consistent colors across reliability visualizations:
- **Curtailment/Issues**: Red (#d62728)
- **Good Performance**: Green (#2ca02c)
- **Reserve Margin**: Blue (#1f77b4)
- **Mean/Average**: Orange (#ff7f0e)

### Demo Scripts

Three example scripts are provided in the `examples/` directory:

1. **`test_reliability_viz.py`**: Comprehensive test suite
   - Tests all visualization functions
   - Tests with real solution data (optimal portfolio)
   - Tests with dict format (JSON compatibility)
   - Tests time range filtering
   - Saves multiple HTML outputs

2. **`test_reliability_viz_baseline.py`**: Baseline solution test
   - Tests with baseline (grid-only) solution
   - Useful for comparing reliability across scenarios

3. **`reliability_viz_demo.py`**: Interactive demonstration with mock data
   - Creates mock solution with curtailment events
   - Demonstrates all visualization types
   - Shows full capabilities with realistic data
   - Includes both full-year and zoomed views

Run the tests:
```bash
python examples/test_reliability_viz.py
python examples/test_reliability_viz_baseline.py
```

Run the demo:
```bash
python examples/reliability_viz_demo.py
```

This will create interactive HTML files in `results/figures/` that can be opened in any web browser.

### Use Cases

**Reliability Assessment:**
- Verify system meets 99.99% reliability target
- Identify frequency and magnitude of curtailment events
- Understand when system is most stressed

**Capacity Planning:**
- Analyze reserve margins to ensure adequate capacity
- Identify periods of insufficient capacity
- Determine if additional capacity investments needed

**Operational Analysis:**
- Find worst-case reliability events for investigation
- Understand seasonal patterns in reliability
- Correlate curtailment with market conditions

**Scenario Comparison:**
- Compare reliability across different portfolio options
- Evaluate impact of adding battery storage or backup generation
- Assess trade-offs between cost and reliability

**Reporting:**
- Create comprehensive reliability reports for stakeholders
- Document system performance against targets
- Identify areas for improvement

### Integration with Other Modules

Reliability visualizations work seamlessly with:
- **Dispatch Visualizations**: Cross-reference curtailment events with dispatch patterns
- **Cost Visualizations**: Understand cost of achieving high reliability
- **Pareto Visualizations**: Explore cost vs reliability trade-offs

Example integrated analysis:
```python
from src.models.solution import OptimizationSolution
from src.visualization import (
    plot_reliability_analysis,
    plot_dispatch_heatmap,
    plot_cost_breakdown
)

solution = OptimizationSolution.load("results/solutions/optimal_portfolio.json")

# Comprehensive analysis
reliability_fig = plot_reliability_analysis(solution)
dispatch_fig = plot_dispatch_heatmap(solution, time_range=(1, 168))
cost_fig = plot_cost_breakdown(solution)

# Save all
reliability_fig.write_html("reliability.html")
dispatch_fig.write_html("dispatch.html")
cost_fig.write_html("costs.html")
```

## Sensitivity Analysis Visualizations

The `sensitivity_viz.py` module provides functions for visualizing parameter sensitivity analysis results, showing how changes in input parameters affect optimization outcomes.

### Functions

#### `plot_sensitivity_tornado(sensitivity_results, metric='total_npv', title=None, top_n=None, show_values=True, height=600)`

Create tornado chart showing parameter impacts on NPV or other metrics. A tornado chart displays horizontal bars showing how each parameter affects the objective metric, with parameters sorted by magnitude of impact (largest at top).

**Parameters:**
- `sensitivity_results`: Dictionary of sensitivity analysis results from `analyze_multiple_parameters()` or similar
- `metric`: Metric being analyzed (default: 'total_npv')
- `title`: Custom title for the plot (optional)
- `top_n`: Show only top N most impactful parameters (optional)
- `show_values`: Whether to show values on the bars (default: True)
- `height`: Figure height in pixels (default: 600)

**Returns:**
- Plotly Figure object with interactive tornado chart

**Features:**
- **Sorted by Impact**: Parameters ordered by magnitude of impact (largest at top)
- **Bidirectional Bars**: 
  - Red bars (left): Impact of low parameter values
  - Green bars (right): Impact of high parameter values
- **Base Case Reference**: Vertical line at 0% showing base case
- **Interactive Tooltips**: Show parameter values, percentage changes, and metric values
- **Percentage Scale**: Shows % change in metric from base case

**Example Usage:**

```python
from src.analysis.sensitivity_analyzer import analyze_multiple_parameters
from src.visualization import plot_sensitivity_tornado

# Run sensitivity analysis
sensitivity = analyze_multiple_parameters(
    base_solution,
    scenario_results,
    parameters=['gas_price_multiplier', 'lmp_multiplier', 'battery_cost_per_kwh'],
    metric='total_npv'
)

# Create tornado chart
fig = plot_sensitivity_tornado(
    sensitivity,
    metric='total_npv',
    title="Parameter Sensitivity Analysis: Impact on Total NPV"
)
fig.show()

# Show only top 5 most impactful parameters
fig = plot_sensitivity_tornado(
    sensitivity,
    top_n=5,
    title="Top 5 Most Impactful Parameters"
)
fig.write_html("sensitivity_top5.html")

# Clean view without value labels
fig = plot_sensitivity_tornado(
    sensitivity,
    show_values=False
)
fig.show()
```

#### `plot_sensitivity_comparison(sensitivity_results, metrics=['total_npv', 'carbon_tons_annual', 'reliability_pct'], title=None, height=400)`

Create comparison chart showing parameter impacts across multiple metrics in a multi-panel layout.

**Parameters:**
- `sensitivity_results`: Dictionary of sensitivity analysis results
- `metrics`: List of metrics to compare (default: ['total_npv', 'carbon_tons_annual', 'reliability_pct'])
- `title`: Custom title for the plot (optional)
- `height`: Figure height in pixels per metric (default: 400)

**Returns:**
- Plotly Figure object with subplots

**Features:**
- **Multi-Metric View**: Compare parameter impacts across different objectives
- **Consistent Ranking**: See which parameters matter most for each metric
- **Stacked Layout**: Vertical arrangement for easy comparison
- **Impact Scores**: Shows relative importance of each parameter

**Example Usage:**

```python
from src.visualization import plot_sensitivity_comparison

# Compare impacts across multiple metrics
fig = plot_sensitivity_comparison(
    sensitivity,
    metrics=['total_npv', 'carbon_tons_annual', 'reliability_pct'],
    title="Parameter Impact Comparison: Cost, Carbon, and Reliability"
)
fig.show()

# Focus on cost and carbon
fig = plot_sensitivity_comparison(
    sensitivity,
    metrics=['total_npv', 'carbon_tons_annual']
)
fig.write_html("sensitivity_cost_carbon.html")
```

### Supported Parameters

The visualization automatically formats common parameter names:

**Economic Parameters:**
- `gas_price_multiplier`: Gas Price
- `lmp_multiplier`: Grid LMP
- `discount_rate`: Discount Rate
- `curtailment_penalty`: Curtailment Penalty

**Technology Cost Parameters:**
- `battery_cost_per_kwh`: Battery Cost
- `solar_cost_per_kw`: Solar Cost

**Operational Parameters:**
- `reliability_target`: Reliability Target
- `carbon_budget`: Carbon Budget

### Supported Metrics

All standard optimization metrics are supported:

**Cost Metrics:**
- `total_npv`: Total NPV
- `capex`: CAPEX
- `opex_annual`: Annual OPEX
- `lcoe`: LCOE

**Performance Metrics:**
- `reliability_pct`: Reliability
- `grid_dependence_pct`: Grid Dependence
- `gas_capacity_factor`: Gas Capacity Factor
- `solar_capacity_factor`: Solar Capacity Factor

**Environmental Metrics:**
- `carbon_tons_annual`: Annual Carbon Emissions
- `carbon_intensity_g_per_kwh`: Carbon Intensity
- `carbon_reduction_pct`: Carbon Reduction

### Color Scheme

Consistent colors across sensitivity visualizations:
- **Low Parameter Values**: Red (#d62728) - typically shows cost reduction or performance decrease
- **High Parameter Values**: Green (#2ca02c) - typically shows cost increase or performance improvement
- **Base Case**: Black vertical line at 0%
- **Comparison Charts**: Blue (#1f77b4) for impact scores

### Demo Scripts

Two example scripts are provided in the `examples/` directory:

1. **`test_sensitivity_viz.py`**: Comprehensive test suite with mock data
   - Tests basic tornado chart
   - Tests top N parameter filtering
   - Tests with and without value labels
   - Tests multi-metric comparison
   - Creates multiple HTML outputs

2. **`sensitivity_viz_demo.py`**: Real data demonstration
   - Loads actual sensitivity analysis results
   - Creates tornado chart from real data
   - Prints parameter impact summary
   - Shows integration with sensitivity analyzer

Run the test:
```bash
python examples/test_sensitivity_viz.py
```

Run the demo:
```bash
python examples/sensitivity_viz_demo.py
```

This will create interactive HTML files in `results/figures/` that can be opened in any web browser.

### Use Cases

**Parameter Prioritization:**
- Identify which parameters have the largest impact on outcomes
- Focus data collection efforts on high-impact parameters
- Understand which uncertainties matter most

**Risk Assessment:**
- Visualize range of possible outcomes based on parameter uncertainty
- Identify parameters that could significantly change optimal decisions
- Assess robustness of optimal solution to parameter variations

**Scenario Planning:**
- Understand how different market conditions affect results
- Evaluate impact of technology cost changes
- Assess sensitivity to policy parameters (carbon constraints, reliability targets)

**Decision Support:**
- Communicate parameter importance to stakeholders
- Support investment decisions with sensitivity analysis
- Identify parameters requiring more accurate estimation

**Reporting:**
- Create publication-quality sensitivity charts
- Document assumptions and their impacts
- Support regulatory filings and investor presentations

### Integration with Sensitivity Analyzer

Sensitivity visualizations are designed to work seamlessly with the sensitivity analyzer module:

```python
from src.analysis.sensitivity_analyzer import (
    analyze_multiple_parameters,
    rank_parameters_by_impact
)
from src.visualization import plot_sensitivity_tornado

# Run batch optimization with parameter variations
from src.analysis.batch_solver import solve_scenarios
from src.analysis.scenario_generator import generate_scenarios

# Generate scenarios
scenarios = generate_scenarios(
    base_params,
    variations={
        'gas_price_multiplier': [0.5, 1.0, 1.5],
        'lmp_multiplier': [0.7, 1.0, 1.3],
        'battery_cost_per_kwh': [200, 350, 500]
    }
)

# Solve all scenarios
results = solve_scenarios(scenarios, market_data, tech_costs, facility_params)

# Analyze sensitivity
sensitivity = analyze_multiple_parameters(
    base_solution,
    results,
    parameters=['gas_price_multiplier', 'lmp_multiplier', 'battery_cost_per_kwh']
)

# Visualize results
fig = plot_sensitivity_tornado(sensitivity)
fig.show()

# Rank parameters
ranking = rank_parameters_by_impact(sensitivity)
print(ranking)
```

### Interpretation Guide

**Reading the Tornado Chart:**

1. **Bar Length**: Longer bars indicate parameters with larger impact on the metric
2. **Bar Direction**: 
   - Left (red): Lower parameter values decrease the metric
   - Right (green): Higher parameter values increase the metric
3. **Asymmetry**: Asymmetric bars indicate non-linear relationships
4. **Ordering**: Parameters at top have largest impact, bottom have smallest

**Example Interpretation:**

If "Grid LMP" has a long bar extending right (+14%) and left (-14%):
- Grid electricity price has high impact on total NPV
- 30% increase in LMP increases NPV by 14%
- 30% decrease in LMP decreases NPV by 14%
- Relationship is approximately linear (symmetric bars)

If "Battery Cost" has a short bar:
- Battery cost has relatively small impact on total NPV
- Optimal solution is not very sensitive to battery cost changes
- Less critical to get precise battery cost estimates

### Best Practices

1. **Parameter Selection**: Include parameters with uncertainty or policy relevance
2. **Variation Range**: Use realistic ranges based on historical data or forecasts
3. **Base Case**: Ensure base case represents most likely scenario
4. **Multiple Metrics**: Analyze sensitivity for multiple objectives (cost, carbon, reliability)
5. **Documentation**: Document parameter ranges and assumptions
6. **Validation**: Cross-check with domain experts and historical data
