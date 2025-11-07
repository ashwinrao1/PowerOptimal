# Optimization Scripts

This directory contains utility scripts for running optimization scenarios and data collection.

## Available Scripts

### run_baseline.py

Runs the baseline grid-only optimization scenario for a 300 MW data center.

**Purpose**: Establishes a cost and carbon baseline by optimizing a data center that relies entirely on grid electricity with no behind-the-meter generation or storage assets.

**Usage**:
```bash
python scripts/run_baseline.py
```

**What it does**:
1. Loads market data (LMP, gas prices, solar capacity factors, carbon intensity) for 2023
2. Loads technology costs from `data/tech_costs.json`
3. Builds an optimization model with only grid connection allowed (gas, battery, and solar disabled)
4. Solves the optimization using GLPK (open-source) or Gurobi (if available)
5. Extracts capacity decisions, hourly dispatch, and comprehensive metrics
6. Saves results to `results/solutions/baseline_grid_only.json`
7. Generates a human-readable summary at `results/solutions/baseline_grid_only_summary.txt`

**Output Files**:
- `results/solutions/baseline_grid_only.json` - Complete solution with capacity, dispatch, and metrics
- `results/solutions/baseline_grid_only_summary.txt` - Human-readable summary of key results

**Key Metrics Calculated**:
- Total 20-year NPV
- CAPEX and annual OPEX
- Levelized Cost of Energy (LCOE)
- Reliability percentage
- Annual carbon emissions
- Carbon intensity (g CO2/kWh)

**Baseline Results** (300 MW IT load, 315 MW total with PUE 1.05):
- Grid Connection: 315.0 MW
- Total NPV: ~$1.02 billion
- LCOE: ~$34.80/MWh
- Annual Emissions: ~940,000 tons CO2/year
- Carbon Intensity: ~340 g CO2/kWh
- Reliability: 100%

**Next Steps**:
After running the baseline, run the optimal portfolio optimization with `python scripts/run_optimal_portfolio.py` to compare and quantify the value of behind-the-meter assets.

---

### run_optimal_portfolio.py

Runs the optimal portfolio optimization scenario with all technologies enabled.

**Purpose**: Determines the optimal combination of grid connection, natural gas peakers, battery storage, and solar PV that minimizes total 20-year cost while meeting reliability and carbon constraints.

**Usage**:
```bash
python scripts/run_optimal_portfolio.py
```

**Prerequisites**:
- Baseline optimization must be run first (`python scripts/run_baseline.py`)
- Baseline results must exist at `results/solutions/baseline_grid_only.json`

**What it does**:
1. Loads market data for 2023 (same as baseline)
2. Loads technology costs from `data/tech_costs.json`
3. Loads baseline solution for comparison
4. Builds an optimization model with ALL technologies enabled (grid, gas, battery, solar)
5. Allows optimizer to determine optimal capacity mix
6. Solves the optimization using GLPK or Gurobi
7. Extracts capacity decisions, hourly dispatch, and comprehensive metrics
8. Calculates comparison metrics vs baseline (cost savings, carbon reduction, payback period)
9. Saves results to `results/solutions/optimal_portfolio.json`
10. Generates a human-readable summary at `results/solutions/optimal_portfolio_summary.txt`

**Output Files**:
- `results/solutions/optimal_portfolio.json` - Complete solution with capacity, dispatch, metrics, and comparison to baseline
- `results/solutions/optimal_portfolio_summary.txt` - Human-readable summary with key insights

**Key Metrics Calculated**:
- Optimal capacity mix for each technology
- Total 20-year NPV and cost savings vs baseline
- CAPEX investment required for BTM assets
- Annual OPEX savings
- Simple payback period
- Levelized Cost of Energy (LCOE)
- Reliability percentage and improvement
- Annual carbon emissions and reduction vs baseline
- Grid dependence percentage
- Battery cycles per year
- Solar and gas capacity factors

**Optimal Portfolio Results** (300 MW IT load, 315 MW total):
- Grid Connection: ~3,860 MW
- Gas Peakers: 0 MW (not economical)
- Battery Storage: ~92,800 MWh
- Solar PV: ~9,665 MW
- Total NPV: ~$429 million
- NPV Savings: ~$588 million (57.9% reduction)
- Annual OPEX Savings: ~$61 million/year (63.3% reduction)
- BTM CAPEX Investment: ~$44 million
- Simple Payback: ~0.7 years
- Carbon Reduction: ~522,000 tons CO2/year (55.6% reduction)
- Grid Dependence: 44.1% (vs 100% baseline)
- LCOE: ~$14.67/MWh (vs $34.80/MWh baseline)

**Key Insights**:
- Battery storage enables significant energy arbitrage, buying cheap grid power and storing it
- Large solar PV capacity reduces grid dependence and carbon emissions
- Gas peakers not selected (battery + solar more economical)
- Extremely fast payback period (<1 year) makes BTM investment highly attractive
- 55.6% carbon reduction achieved while saving money

**Next Steps**:
After running the optimal portfolio, you can:
1. Analyze sensitivity to key parameters (tasks 15-18)
2. Create visualizations for presentation (tasks 19-24)
3. Build interactive dashboard (tasks 25-31)
4. Run case study analysis (tasks 32-33)

## Solver Requirements

The scripts support multiple solvers:
- **GLPK** (open-source, free) - Installed via `brew install glpk` on macOS
- **Gurobi** (commercial, faster) - Requires license but free for academic use
- **CBC** (open-source, free) - Alternative to GLPK

The scripts will automatically use GLPK if Gurobi is not available. For production use or faster solving, Gurobi is recommended.

## Data Requirements

Before running any optimization scripts, ensure the following data files exist:
- `data/processed/ercot_lmp_hourly_2022_2024.csv`
- `data/processed/gas_prices_hourly.csv`
- `data/processed/solar_cf_west_texas.csv`
- `data/processed/grid_carbon_intensity.csv`
- `data/tech_costs.json`

These files should be generated by the data pipeline (tasks 2-6).
