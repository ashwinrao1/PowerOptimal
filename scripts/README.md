# Optimization Scripts

This directory contains utility scripts for running optimization scenarios and data collection.

## Available Scripts

### download_all_data.py

Automates the collection of all required data from external sources.

**Purpose**: Downloads and processes ERCOT LMP data, solar capacity factors, natural gas prices, and grid carbon intensity data for use in optimization models.

**Usage**:
```bash
# Download all data for 2022-2024
python scripts/download_all_data.py

# Download with custom date range
python scripts/download_all_data.py --start-date 2023-01-01 --end-date 2023-12-31

# Skip specific datasets
python scripts/download_all_data.py --skip-ercot --skip-solar

# Use NREL API for solar data (requires API key)
python scripts/download_all_data.py --use-nrel-api
```

**Command-line Options**:
- `--start-date`: Start date in YYYY-MM-DD format (default: 2022-01-01)
- `--end-date`: End date in YYYY-MM-DD format (default: 2024-12-31)
- `--output-dir`: Output directory for processed data (default: data/processed)
- `--use-nrel-api`: Use NREL PVWatts API for solar data (requires API key)
- `--skip-ercot`: Skip ERCOT LMP data download
- `--skip-solar`: Skip solar profile generation
- `--skip-gas`: Skip gas price download
- `--skip-carbon`: Skip carbon intensity download

**What it does**:
1. Downloads hourly ERCOT LMP data for West Texas hub
2. Generates solar capacity factor profile for West Texas location
3. Downloads and interpolates natural gas prices from Waha Hub
4. Downloads grid carbon intensity data for ERCOT region
5. Validates all data for completeness and reasonable ranges
6. Saves processed data to CSV files in data/processed/

**Output Files**:
- `data/processed/ercot_lmp_hourly_2022_2024.csv` - Hourly LMP data
- `data/processed/solar_cf_west_texas.csv` - Hourly solar capacity factors
- `data/processed/gas_prices_hourly.csv` - Hourly natural gas prices
- `data/processed/grid_carbon_intensity.csv` - Hourly grid carbon intensity

**Error Handling**:
- Retries failed API requests with exponential backoff
- Falls back to synthetic data generation if APIs are unavailable
- Validates data completeness and value ranges
- Reports detailed error messages for troubleshooting

**Next Steps**:
After downloading data, run the baseline optimization with `python scripts/run_baseline.py`.

---

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
1. Run scenario analysis with `python scripts/run_scenarios.py`
2. Create visualizations for presentation (tasks 19-24)
3. Build interactive dashboard (tasks 25-31)
4. Run case study analysis (tasks 32-33)

---

### run_scenarios.py

Runs comprehensive scenario analysis with parameter variations.

**Purpose**: Executes multiple optimization scenarios with varied parameters to analyze sensitivity and trade-offs between cost, reliability, and carbon emissions.

**Usage**:
```bash
# Run all sensitivity analyses
python scripts/run_scenarios.py

# Run specific analysis type
python scripts/run_scenarios.py --analysis-type gas
python scripts/run_scenarios.py --analysis-type lmp
python scripts/run_scenarios.py --analysis-type battery
python scripts/run_scenarios.py --analysis-type reliability

# Use different year or facility size
python scripts/run_scenarios.py --year 2024 --facility-size 500
```

**Command-line Options**:
- `--year`: Year for market data (choices: 2022, 2023, 2024; default: 2023)
- `--facility-size`: IT load in MW (default: 300)
- `--output-dir`: Output directory for scenario results (default: results/scenarios)
- `--analysis-type`: Type of sensitivity analysis (choices: all, gas, lmp, battery, reliability; default: all)

**What it does**:
1. Loads market data and technology costs
2. Runs multiple optimization scenarios with varied parameters:
   - **Gas Price Sensitivity**: 50%, 75%, 100%, 125%, 150% of baseline
   - **LMP Sensitivity**: 70%, 85%, 100%, 115%, 130% of baseline
   - **Battery Cost Sensitivity**: $200, $275, $350, $425, $500 per kWh
   - **Reliability Sensitivity**: 99.9%, 99.99%, 99.999% uptime targets
3. Solves each scenario independently
4. Collects capacity decisions and key metrics for all scenarios
5. Saves comprehensive results to timestamped JSON file

**Output Files**:
- `results/scenarios/scenario_results_YYYYMMDD_HHMMSS.json` - Complete scenario results with:
  - Scenario parameters
  - Optimal capacity mix for each scenario
  - Cost, reliability, and carbon metrics
  - Solve time and status

**Key Analyses**:

**Gas Price Sensitivity**:
- Shows how optimal portfolio changes with natural gas price variations
- Identifies breakeven points where gas peakers become economical
- Quantifies impact on total NPV and OPEX

**LMP Sensitivity**:
- Analyzes impact of grid electricity price variations
- Shows how battery and solar investments change with LMP levels
- Identifies optimal strategies for different electricity markets

**Battery Cost Sensitivity**:
- Determines battery investment levels at different cost points
- Identifies cost threshold where battery becomes economical
- Shows relationship between battery cost and grid dependence

**Reliability Sensitivity**:
- Analyzes cost of achieving different reliability levels
- Shows capacity mix changes for higher reliability targets
- Quantifies trade-off between cost and reliability

**Example Results**:
```
Scenario                                 NPV ($M)        Reliability (%)    Carbon (tons/yr)
gas_price_50pct                          412.5           99.9900            418,234
gas_price_100pct                         429.1           99.9900            417,892
gas_price_150pct                         445.8           99.9900            417,550
lmp_70pct                                315.2           99.9900            425,678
lmp_130pct                               543.0           99.9900            410,106
battery_200usd_per_kwh                   398.7           99.9900            415,234
battery_500usd_per_kwh                   459.5           99.9900            420,550
reliability_99.900pct                    401.2           99.9000            412,345
reliability_99.999pct                    457.8           99.9990            423,567
```

**Next Steps**:
After running scenario analysis:
1. Generate Pareto frontier plots to visualize trade-offs
2. Create sensitivity tornado charts
3. Analyze which parameters have the biggest impact on NPV
4. Use insights to inform investment decisions

---

### run_case_study.py

Runs comprehensive case study analysis for 300 MW West Texas data center.

**Purpose**: Executes a complete case study including baseline, optimal portfolio, financial analysis, and visualization generation.

**Usage**:
```bash
python scripts/run_case_study.py
```

**What it does**:
1. Runs baseline grid-only optimization
2. Runs optimal portfolio optimization with all technologies
3. Calculates comprehensive financial metrics
4. Generates all visualizations (capacity mix, dispatch heatmap, cost breakdown, reliability analysis)
5. Saves complete case study results

**Output Files**:
- `results/solutions/case_study_results.json` - Complete case study data
- Multiple visualization files in `results/figures/`

**Note**: This script provides a complete end-to-end analysis suitable for presentation to stakeholders.

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
