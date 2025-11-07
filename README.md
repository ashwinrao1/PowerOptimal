# Data Center Energy Optimization

An optimization model that determines the optimal energy portfolio and hourly dispatch strategy for large-scale AI training data centers. The system addresses the trilemma of cost minimization, reliability maximization, and carbon emission reduction.

## Overview

This project solves a large-scale linear programming problem to determine:
- Optimal capacity investments (grid connection, natural gas peakers, battery storage, solar PV)
- Hourly dispatch strategy for 8760 hours per year
- Trade-offs between cost, reliability, and carbon emissions

The optimization model minimizes 20-year total cost while meeting reliability targets (99.99%+ uptime) and optional carbon constraints.

## Features

- Automated data collection from ERCOT, NREL, and EIA APIs
- Mathematical optimization using Pyomo and Gurobi
- Scenario analysis and sensitivity studies
- Interactive Streamlit dashboard for exploration
- Case study analysis for 300MW West Texas facility

## Project Structure

```
.
├── data/                   # Data files
│   ├── raw/               # Raw data from APIs
│   └── processed/         # Cleaned and processed data
├── src/                   # Source code
│   ├── data_pipeline/     # Data collection and validation
│   ├── optimization/      # Optimization model and solver
│   ├── analysis/          # Scenario and sensitivity analysis
│   └── visualization/     # Plotting functions
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── notebooks/            # Jupyter notebooks for exploration
├── dashboard/            # Streamlit dashboard
│   └── pages/           # Dashboard pages
├── docs/                # Documentation
└── results/             # Optimization results
    ├── solutions/       # Individual solutions
    └── scenarios/       # Scenario analysis results
```

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Gurobi Optimizer (free academic license available)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd datacenter-energy-optimization
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Gurobi license:
   - Obtain a free academic license from https://www.gurobi.com/academia/
   - Follow Gurobi's instructions to activate your license

### Quick Start

1. Download data:
```bash
python scripts/download_all_data.py
```

2. Run baseline optimization:
```bash
python scripts/run_baseline.py
```

3. Launch interactive dashboard:
```bash
streamlit run dashboard/app.py
```

## Usage

### Running Optimization

```python
from src.optimization.model_builder import build_optimization_model
from src.optimization.solver import solve_model
from src.data_pipeline.loader import load_market_data

# Load data
market_data = load_market_data("data/processed/")

# Build and solve model
model = build_optimization_model(market_data, load_mw=300)
results, solve_time = solve_model(model)

# Extract solution
from src.optimization.solution_extractor import extract_solution
solution = extract_solution(model)
```

### Running Scenario Analysis

```bash
python scripts/run_scenarios.py
```

## Documentation

- [Design Document](.kiro/specs/datacenter-energy-optimization/design.md)
- [Requirements](.kiro/specs/datacenter-energy-optimization/requirements.md)
- [Implementation Tasks](.kiro/specs/datacenter-energy-optimization/tasks.md)

## Case Study

The project includes a detailed case study for a 300MW AI training data center in West Texas, comparing:
- Baseline: Grid-only power supply
- Optimal: Mixed portfolio with grid, gas, battery, and solar

Results show significant cost savings and carbon reduction while maintaining 99.99% reliability.

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## License

[Add license information]

## Contact

[Add contact information]
