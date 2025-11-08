# Dashboard

Interactive Streamlit dashboard for data center energy optimization.

## Structure

```
dashboard/
├── app.py              # Main application entry point
├── utils.py            # Shared utility functions
├── pages/              # Individual page modules
│   ├── setup.py        # Optimization setup page
│   ├── portfolio.py    # Optimal portfolio results
│   ├── dispatch.py     # Hourly dispatch visualization
│   ├── scenarios.py    # Scenario comparison
│   └── case_study.py   # 300MW West Texas case study
└── README.md           # This file
```

## Running the Dashboard

### Local Development

```bash
# From project root
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Configuration

The dashboard uses Streamlit's default configuration. To customize, create a `.streamlit/config.toml` file:

```toml
[server]
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## Pages

### 1. Optimization Setup
Configure facility parameters and run optimization:
- Facility size (100-500 MW)
- Reliability target (99.9-99.999%)
- Carbon reduction target (0-100%)
- Location selection
- Technology options

### 2. Optimal Portfolio
View optimization results:
- Capacity mix visualization
- Cost breakdown
- Key performance metrics
- Export functionality

### 3. Hourly Dispatch
Explore operational decisions:
- 8760-hour dispatch heatmap
- Time range selection
- Operational statistics
- Interactive tooltips

### 4. Scenario Comparison
Multi-scenario analysis:
- Pareto frontier plots
- Sensitivity analysis
- Scenario comparison table
- Tornado charts

### 5. Case Study
Detailed 300MW West Texas analysis:
- Baseline vs. optimal comparison
- Financial analysis
- Strategic recommendations
- Downloadable report

## Session State

The dashboard maintains state across page changes:

### Input Parameters
- `facility_size_mw`: Facility size in MW
- `reliability_target`: Reliability target in %
- `carbon_reduction_pct`: Carbon reduction target in %
- `location`: Geographic location
- `year_scenario`: Year for market data
- `available_technologies`: Dict of enabled technologies

### Results
- `optimization_result`: Complete optimization solution
- `optimization_status`: Current status (Not started, Running, Complete, Failed)
- `solve_time`: Optimization solve time in seconds
- `scenario_results`: Multi-scenario analysis results
- `pareto_frontiers`: Pareto frontier data

### Cached Data
- `market_data_loaded`: Flag for data loading status
- `lmp_data`: ERCOT LMP data
- `solar_cf`: Solar capacity factors
- `gas_prices`: Natural gas prices
- `grid_carbon`: Grid carbon intensity

## Development

### Adding a New Page

1. Create a new file in `pages/` directory
2. Implement a `render()` function
3. Import and call from `app.py`

Example:

```python
# pages/new_page.py
import streamlit as st

def render():
    st.title("New Page")
    st.write("Page content here")
```

```python
# app.py
elif page == "New Page":
    from pages import new_page
    new_page.render()
```

### Using Utilities

Import shared utilities:

```python
from utils import (
    format_currency,
    format_energy,
    display_metric_card,
    get_color_scheme
)
```

### Caching

Use Streamlit caching for expensive operations:

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    return pd.read_csv("data.csv")

@st.cache_resource
def get_solver():
    return pyo.SolverFactory('gurobi')
```

## Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect Streamlit Cloud to repository
3. Select `dashboard/app.py` as main file
4. Configure Python version (3.10+)
5. Deploy

### Requirements

Ensure `requirements.txt` includes:
```
streamlit>=1.28.0
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.24.0
```

## Troubleshooting

### Data Files Not Found
Run data collection scripts:
```bash
python scripts/download_all_data.py
```

### Import Errors
Ensure `src/` directory is in Python path (handled by `app.py`)

### Slow Performance
- Use caching decorators
- Pre-compute common scenarios
- Reduce data resolution for previews

### Memory Issues
- Clear session state periodically
- Limit concurrent optimizations
- Use data downsampling for large visualizations
