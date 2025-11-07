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

## Future Visualizations

Additional visualization modules will be added for:
- Cost breakdowns (CAPEX/OPEX)
- Pareto frontiers (trade-off analysis)
- Reliability analysis
- Sensitivity tornado charts
