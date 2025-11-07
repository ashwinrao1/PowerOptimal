# Design Document

## Overview

This document describes the architecture and design of the data center energy optimization system. The system consists of three main components: a data pipeline for ingesting real-world market data, a mathematical optimization model implemented in Python using Pyomo and Gurobi, and an interactive dashboard for scenario analysis and visualization.

The optimization model solves a large-scale linear programming problem with approximately 35,000 decision variables (4 capacity decisions + 8760 hours × 4 operational decisions) and 50,000+ constraints to determine the optimal energy portfolio and hourly dispatch strategy that minimizes 20-year total cost while meeting reliability and optional carbon constraints.

## Architecture

### System Components

The system follows a three-layer architecture:

1. **Data Layer**: Automated data collection and preprocessing pipelines
2. **Optimization Layer**: Mathematical model formulation and solving
3. **Presentation Layer**: Interactive dashboard and visualization

### Technology Stack

- **Programming Language**: Python 3.10+
- **Optimization Framework**: Pyomo 6.7+ (algebraic modeling language)
- **Solver**: Gurobi 11.0+ (commercial solver with free academic license)
- **Data Processing**: Pandas 2.0+, NumPy 1.24+
- **Visualization**: Plotly 5.18+, Matplotlib 3.8+
- **Dashboard**: Streamlit 1.28+ (for rapid development and deployment)
- **API Clients**: requests, beautifulsoup4 (for data scraping)
- **Testing**: pytest 7.4+

### Deployment Architecture

```
User Browser
    ↓
Streamlit Cloud (Frontend + Backend)
    ↓
Optimization Engine (Pyomo + Gurobi)
    ↓
Data Storage (CSV files + JSON configs)
```


## Components and Interfaces

### 1. Data Pipeline Component

#### Purpose
Automate collection, validation, and preprocessing of real-world energy market data from multiple sources.

#### Sub-Components

**1.1 ERCOT LMP Collector**
- **Input**: ERCOT API endpoint for Settlement Point Prices
- **Output**: `data/ercot_lmp_hourly_2022_2024.csv`
- **Interface**: 
  ```python
  def fetch_ercot_lmp(start_date: str, end_date: str, hub: str) -> pd.DataFrame:
      """
      Fetch hourly LMP data from ERCOT public API
      Returns DataFrame with columns: [timestamp, lmp_dam, lmp_rtm]
      """
  ```
- **Data Format**: Timestamp (ISO 8601), DAM Price ($/MWh), RTM Price ($/MWh)
- **Error Handling**: Retry logic for API failures, forward-fill for <1% missing data

**1.2 Solar Profile Generator**
- **Input**: NREL PVWatts API with coordinates (31.9973°N, 102.0779°W)
- **Output**: `data/solar_cf_west_texas.csv`
- **Interface**:
  ```python
  def generate_solar_profile(lat: float, lon: float, tilt: float, azimuth: float) -> pd.DataFrame:
      """
      Generate 8760 hourly capacity factors using NREL PVWatts
      Returns DataFrame with columns: [hour_of_year, capacity_factor]
      """
  ```
- **Data Format**: Hour (1-8760), Capacity Factor (0-1)
- **Parameters**: Fixed tilt at latitude (32°), south-facing (180°), 20% efficient panels

**1.3 Natural Gas Price Collector**
- **Input**: EIA API for Waha Hub daily prices
- **Output**: `data/gas_prices_hourly.csv`
- **Interface**:
  ```python
  def fetch_gas_prices(start_date: str, end_date: str, hub: str) -> pd.DataFrame:
      """
      Fetch daily gas prices and interpolate to hourly
      Returns DataFrame with columns: [timestamp, price_mmbtu]
      """
  ```
- **Data Format**: Timestamp, Price ($/MMBtu)
- **Interpolation**: Apply 10% peak/off-peak differential for hourly granularity

**1.4 Grid Carbon Intensity Collector**
- **Input**: EIA Hourly Electric Grid Monitor for ERCOT West
- **Output**: `data/grid_carbon_intensity.csv`
- **Interface**:
  ```python
  def fetch_grid_carbon(start_date: str, end_date: str, region: str) -> pd.DataFrame:
      """
      Fetch hourly grid carbon intensity
      Returns DataFrame with columns: [timestamp, carbon_intensity_kg_per_mwh]
      """
  ```
- **Data Format**: Timestamp, Carbon Intensity (kg CO2/MWh)

**1.5 Technology Cost Database**
- **Input**: Manual compilation from NREL ATB 2024
- **Output**: `data/tech_costs.json`
- **Data Structure**:
  ```json
  {
    "grid_interconnection": {"capex_per_kw": 3000, "fixed_om_per_kw_year": 0},
    "gas_peaker": {"capex_per_kw": 1000, "variable_om_per_mwh": 15, "heat_rate_mmbtu_per_mwh": 10},
    "battery": {"capex_per_kwh": 350, "degradation_per_mwh": 5, "efficiency": 0.85},
    "solar": {"capex_per_kw": 1200, "fixed_om_per_kw_year": 20}
  }
  ```

**1.6 Data Validator**
- **Purpose**: Ensure data quality and completeness
- **Checks**:
  - No more than 1% missing values
  - Timestamps are continuous and complete
  - Values are within reasonable ranges (e.g., LMP > -$100/MWh, LMP < $5000/MWh)
  - Capacity factors between 0 and 1
- **Interface**:
  ```python
  def validate_data(df: pd.DataFrame, data_type: str) -> Tuple[bool, List[str]]:
      """
      Validate data quality
      Returns (is_valid, list_of_issues)
      """
  ```


### 2. Optimization Model Component

#### Purpose
Formulate and solve the mixed-integer linear program to determine optimal capacity investments and hourly dispatch.

#### Mathematical Formulation

**Decision Variables**:
- Capacity: `C_grid`, `C_gas`, `C_battery`, `C_solar` (continuous, ≥ 0)
- Hourly dispatch: `p_grid[h]`, `p_gas[h]`, `p_battery[h]`, `p_curtail[h]` for h ∈ {1..8760}
- Battery state: `SOC[h]` for h ∈ {1..8760}

**Objective Function**:
```
Minimize: CAPEX + NPV(OPEX) + NPV(Curtailment_Penalty)

CAPEX = C_grid × 3000 + C_gas × 1000 + C_battery × 350 + C_solar × 1200

OPEX_annual = Σ_h [p_grid[h] × LMP[h] + 
                   p_gas[h] × gas_price[h] × heat_rate / efficiency +
                   p_gas[h] × 15 +
                   |p_battery[h]| × 5] +
              max_h(p_grid[h]) × 15 × 12 +
              C_solar × 20

NPV(OPEX) = Σ_{year=1..20} OPEX_annual / (1 + discount_rate)^year

Curtailment_Penalty = Σ_h p_curtail[h] × 10000
```

**Constraints**:

1. Energy Balance (8760 constraints):
   ```
   p_grid[h] + p_gas[h] + p_solar[h] - p_battery[h] + p_curtail[h] = Load  ∀h
   ```

2. Capacity Limits (26,280 constraints):
   ```
   p_grid[h] ≤ C_grid  ∀h
   p_gas[h] ≤ C_gas  ∀h
   -0.25 × C_battery ≤ p_battery[h] ≤ 0.25 × C_battery  ∀h
   ```

3. Solar Generation (8760 constraints):
   ```
   p_solar[h] = C_solar × CF_solar[h]  ∀h
   ```

4. Battery Dynamics (8760 constraints):
   ```
   SOC[h] = SOC[h-1] + p_battery[h] × efficiency × Δt  ∀h
   ```

5. Battery SOC Limits (17,520 constraints):
   ```
   0.1 × C_battery ≤ SOC[h] ≤ 0.9 × C_battery  ∀h
   ```

6. Battery Periodicity (1 constraint):
   ```
   SOC[8760] = SOC[1]
   ```

7. Gas Ramping (8759 constraints):
   ```
   |p_gas[h] - p_gas[h-1]| ≤ 0.5 × C_gas  ∀h > 1
   ```

8. Reliability (1 constraint):
   ```
   Σ_h p_curtail[h] ≤ 2.85
   ```

9. Optional Carbon Constraint (1 constraint if enabled):
   ```
   Σ_h [p_grid[h] × carbon_intensity[h] + p_gas[h] × 0.4] ≤ carbon_budget
   ```

**Total Problem Size**: ~35,000 variables, ~70,000 constraints

#### Sub-Components

**2.1 Model Builder**
- **Purpose**: Construct Pyomo model from input data
- **Interface**:
  ```python
  def build_optimization_model(
      lmp_data: pd.DataFrame,
      solar_cf: pd.DataFrame,
      gas_prices: pd.DataFrame,
      tech_costs: dict,
      load_mw: float,
      reliability_target: float,
      carbon_budget: Optional[float] = None
  ) -> pyo.ConcreteModel:
      """
      Build Pyomo optimization model
      Returns: Pyomo ConcreteModel ready for solving
      """
  ```
- **Design Pattern**: Factory pattern for creating model variants

**2.2 Solver Interface**
- **Purpose**: Configure and execute Gurobi solver
- **Interface**:
  ```python
  def solve_model(
      model: pyo.ConcreteModel,
      solver_options: dict = None
  ) -> Tuple[pyo.SolverResults, float]:
      """
      Solve optimization model using Gurobi
      Returns: (solver_results, solve_time_seconds)
      """
  ```
- **Solver Configuration**:
  - MIPGap: 0.005 (0.5%)
  - TimeLimit: 1800 seconds (30 minutes)
  - Threads: Auto (use all available cores)
  - Method: Barrier with crossover for LP, Branch-and-cut for MILP

**2.3 Solution Extractor**
- **Purpose**: Extract and format optimization results
- **Interface**:
  ```python
  def extract_solution(model: pyo.ConcreteModel) -> dict:
      """
      Extract solution from solved model
      Returns: Dictionary with capacity decisions, hourly dispatch, and metrics
      """
  ```
- **Output Structure**:
  ```python
  {
      "capacity": {
          "grid_mw": float,
          "gas_mw": float,
          "battery_mwh": float,
          "solar_mw": float
      },
      "dispatch": pd.DataFrame,  # 8760 rows × 5 columns
      "metrics": {
          "total_cost_npv": float,
          "capex": float,
          "opex_annual": float,
          "lcoe": float,
          "reliability_pct": float,
          "carbon_tons_annual": float,
          "carbon_intensity_g_per_kwh": float
      }
  }
  ```

**2.4 Constraint Validator**
- **Purpose**: Verify all constraints are satisfied in solution
- **Interface**:
  ```python
  def validate_solution(model: pyo.ConcreteModel, tolerance: float = 1e-6) -> List[str]:
      """
      Check for constraint violations
      Returns: List of violated constraints (empty if all satisfied)
      """
  ```


### 3. Scenario Analysis Component

#### Purpose
Enable systematic exploration of parameter sensitivity and trade-offs.

#### Sub-Components

**3.1 Scenario Generator**
- **Purpose**: Create parameter combinations for scenario analysis
- **Interface**:
  ```python
  def generate_scenarios(base_params: dict, variations: dict) -> List[dict]:
      """
      Generate scenario parameter sets
      variations = {
          "gas_price_multiplier": [0.5, 1.0, 1.5],
          "reliability_target": [0.999, 0.9999, 0.99999],
          "carbon_reduction_pct": [0, 50, 80, 100]
      }
      Returns: List of parameter dictionaries
      """
  ```

**3.2 Batch Solver**
- **Purpose**: Run optimization for multiple scenarios in parallel
- **Interface**:
  ```python
  def solve_scenarios(
      scenarios: List[dict],
      n_workers: int = 4
  ) -> List[dict]:
      """
      Solve multiple scenarios using multiprocessing
      Returns: List of solution dictionaries
      """
  ```
- **Parallelization**: Use Python multiprocessing to solve independent scenarios concurrently

**3.3 Pareto Frontier Calculator**
- **Purpose**: Identify non-dominated solutions for trade-off analysis
- **Interface**:
  ```python
  def calculate_pareto_frontier(
      solutions: List[dict],
      objective1: str,
      objective2: str
  ) -> pd.DataFrame:
      """
      Find Pareto-optimal solutions
      objectives: "cost", "reliability", "carbon_emissions", "grid_dependence"
      Returns: DataFrame of non-dominated solutions
      """
  ```

**3.4 Sensitivity Analyzer**
- **Purpose**: Quantify impact of parameter changes on optimal solution
- **Interface**:
  ```python
  def analyze_sensitivity(
      base_solution: dict,
      varied_solutions: List[dict],
      parameter_name: str
  ) -> dict:
      """
      Calculate sensitivity metrics
      Returns: {
          "elasticity": float,  # % change in cost / % change in parameter
          "breakeven_point": float,  # parameter value where decision changes
          "impact_ranking": int  # relative importance vs other parameters
      }
      """
  ```

### 4. Visualization Component

#### Purpose
Create interactive and static visualizations for analysis and communication.

#### Sub-Components

**4.1 Capacity Mix Visualizer**
- **Charts**: Stacked bar chart, pie chart
- **Data**: Optimal capacity for each technology
- **Interface**:
  ```python
  def plot_capacity_mix(solution: dict, format: str = "bar") -> plotly.Figure:
      """
      Visualize optimal capacity portfolio
      format: "bar", "pie", "waterfall"
      """
  ```

**4.2 Dispatch Heatmap**
- **Chart**: 2D heatmap with hour on x-axis, power source on y-axis
- **Data**: 8760 hourly dispatch decisions
- **Interface**:
  ```python
  def plot_dispatch_heatmap(
      dispatch_df: pd.DataFrame,
      time_range: Optional[Tuple[int, int]] = None
  ) -> plotly.Figure:
      """
      Create interactive dispatch heatmap
      time_range: (start_hour, end_hour) for zooming
      """
  ```
- **Interactivity**: Hover tooltips showing LMP, gas price, solar CF

**4.3 Cost Breakdown Visualizer**
- **Charts**: Waterfall chart, stacked bar chart
- **Data**: CAPEX and OPEX components
- **Interface**:
  ```python
  def plot_cost_breakdown(solution: dict) -> plotly.Figure:
      """
      Visualize cost components
      Shows: Grid CAPEX, Gas CAPEX, Battery CAPEX, Solar CAPEX,
             Grid OPEX, Gas OPEX, Battery OPEX, Solar OPEX
      """
  ```

**4.4 Pareto Frontier Plotter**
- **Chart**: Scatter plot with Pareto frontier highlighted
- **Data**: Multiple scenario solutions
- **Interface**:
  ```python
  def plot_pareto_frontier(
      solutions: pd.DataFrame,
      x_metric: str,
      y_metric: str
  ) -> plotly.Figure:
      """
      Plot trade-off frontier
      Highlights non-dominated solutions
      """
  ```

**4.5 Reliability Analyzer**
- **Charts**: Histogram, time series, event identification
- **Data**: Curtailment events and reserve margins
- **Interface**:
  ```python
  def plot_reliability_analysis(solution: dict) -> plotly.Figure:
      """
      Multi-panel reliability visualization
      Panels: curtailment histogram, worst events, reserve margin over time
      """
  ```

**4.6 Sensitivity Tornado Chart**
- **Chart**: Horizontal bar chart showing parameter impacts
- **Data**: Sensitivity analysis results
- **Interface**:
  ```python
  def plot_sensitivity_tornado(sensitivity_results: dict) -> plotly.Figure:
      """
      Create tornado chart for sensitivity analysis
      Shows which parameters have biggest impact on NPV
      """
  ```


### 5. Interactive Dashboard Component

#### Purpose
Provide web-based interface for exploring scenarios and visualizations.

#### Technology Choice: Streamlit

**Rationale**:
- Rapid development (pure Python, no HTML/CSS/JS required)
- Built-in widgets for sliders, dropdowns, checkboxes
- Automatic reactivity (re-runs on input change)
- Free deployment on Streamlit Cloud
- Good performance for moderate data sizes

**Alternative Considered**: Plotly Dash
- More customizable but slower development
- Better for production applications
- Requires more boilerplate code

#### Dashboard Layout

**Page 1: Optimization Setup**
- Input widgets:
  - Slider: Facility size (100-500 MW)
  - Slider: Reliability target (99.9-99.999%)
  - Slider: Carbon reduction target (0-100%)
  - Dropdown: Location (West Texas, Memphis, etc.)
  - Dropdown: Year scenario (2022, 2023, 2024)
  - Checkboxes: Available technologies (grid, gas, battery, solar)
- Action button: "Run Optimization"
- Status indicator: Solving progress and time

**Page 2: Optimal Portfolio**
- Capacity mix visualization (bar chart)
- Cost breakdown (waterfall chart)
- Key metrics cards:
  - Total NPV
  - LCOE ($/MWh)
  - Reliability (%)
  - Carbon intensity (g CO2/kWh)
  - Grid independence (%)
- Download button: Export results to CSV/JSON

**Page 3: Hourly Dispatch**
- Dispatch heatmap (full year)
- Time range selector for zooming
- Specific day/week viewer
- Statistics panel:
  - Gas utilization hours
  - Battery cycles per year
  - Solar capacity factor
  - Peak grid draw

**Page 4: Scenario Comparison**
- Multi-scenario runner
- Pareto frontier plots:
  - Cost vs. Reliability
  - Cost vs. Carbon
  - Grid dependence vs. Reliability
- Scenario comparison table
- Sensitivity tornado chart

**Page 5: Case Study**
- Pre-computed results for 300MW West Texas facility
- Narrative explanation of findings
- Comparison to alternative strategies
- Downloadable report (PDF)

#### Interface Design

```python
# Streamlit app structure
def main():
    st.set_page_config(page_title="Data Center Energy Optimizer", layout="wide")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Navigate", [
        "Optimization Setup",
        "Optimal Portfolio",
        "Hourly Dispatch",
        "Scenario Comparison",
        "Case Study"
    ])
    
    if page == "Optimization Setup":
        render_setup_page()
    elif page == "Optimal Portfolio":
        render_portfolio_page()
    # ... etc
```

#### State Management

Use Streamlit session state to persist:
- Input parameters across page changes
- Optimization results (avoid re-solving)
- Cached data (LMP, solar CF, etc.)

```python
if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None

if st.button("Run Optimization"):
    with st.spinner("Solving optimization model..."):
        result = solve_optimization(params)
        st.session_state.optimization_result = result
```

#### Performance Optimization

**Challenge**: Optimization can take 5-30 minutes
**Solutions**:
1. Pre-compute common scenarios and cache results
2. Use Streamlit caching for data loading:
   ```python
   @st.cache_data
   def load_lmp_data():
       return pd.read_csv("data/ercot_lmp_hourly.csv")
   ```
3. Show progress bar during solving
4. Allow users to download results and upload later
5. Consider async solving with result notification


## Data Models

### Input Data Models

**1. Market Data**
```python
@dataclass
class MarketData:
    """Hourly market data for optimization"""
    timestamp: pd.DatetimeIndex  # 8760 hours
    lmp: np.ndarray  # $/MWh
    gas_price: np.ndarray  # $/MMBtu
    solar_cf: np.ndarray  # 0-1
    grid_carbon_intensity: np.ndarray  # kg CO2/MWh
    
    def validate(self) -> bool:
        """Ensure data completeness and validity"""
        assert len(self.timestamp) == 8760
        assert np.all(self.lmp >= -100) and np.all(self.lmp <= 5000)
        assert np.all(self.gas_price >= 0) and np.all(self.gas_price <= 50)
        assert np.all(self.solar_cf >= 0) and np.all(self.solar_cf <= 1)
        return True
```

**2. Technology Parameters**
```python
@dataclass
class TechnologyCosts:
    """Capital and operating costs for all technologies"""
    grid_capex_per_kw: float = 3000
    gas_capex_per_kw: float = 1000
    battery_capex_per_kwh: float = 350
    solar_capex_per_kw: float = 1200
    
    gas_variable_om: float = 15  # $/MWh
    gas_heat_rate: float = 10  # MMBtu/MWh
    gas_efficiency: float = 0.35
    
    battery_degradation: float = 5  # $/MWh
    battery_efficiency: float = 0.85
    battery_duration: float = 4  # hours
    
    solar_fixed_om: float = 20  # $/kW-year
    
    grid_demand_charge: float = 15  # $/kW-month
```

**3. Facility Parameters**
```python
@dataclass
class FacilityParams:
    """Data center facility characteristics"""
    it_load_mw: float = 300
    pue: float = 1.05
    total_load_mw: float = field(init=False)
    reliability_target: float = 0.9999  # 99.99%
    carbon_budget: Optional[float] = None  # tons CO2/year
    planning_horizon_years: int = 20
    discount_rate: float = 0.07
    curtailment_penalty: float = 10000  # $/MWh
    
    def __post_init__(self):
        self.total_load_mw = self.it_load_mw * self.pue
```

### Output Data Models

**1. Capacity Solution**
```python
@dataclass
class CapacitySolution:
    """Optimal capacity investments"""
    grid_mw: float
    gas_mw: float
    battery_mwh: float
    solar_mw: float
    
    def total_capex(self, costs: TechnologyCosts) -> float:
        return (self.grid_mw * costs.grid_capex_per_kw +
                self.gas_mw * costs.gas_capex_per_kw +
                self.battery_mwh * costs.battery_capex_per_kwh +
                self.solar_mw * costs.solar_capex_per_kw)
    
    def to_dict(self) -> dict:
        return {
            "Grid Connection (MW)": self.grid_mw,
            "Gas Peakers (MW)": self.gas_mw,
            "Battery Storage (MWh)": self.battery_mwh,
            "Solar PV (MW)": self.solar_mw
        }
```

**2. Dispatch Solution**
```python
@dataclass
class DispatchSolution:
    """Hourly operational decisions"""
    hour: np.ndarray  # 1-8760
    grid_power: np.ndarray  # MW
    gas_power: np.ndarray  # MW
    solar_power: np.ndarray  # MW
    battery_power: np.ndarray  # MW (positive = charge, negative = discharge)
    curtailment: np.ndarray  # MW
    battery_soc: np.ndarray  # MWh
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Hour": self.hour,
            "Grid (MW)": self.grid_power,
            "Gas (MW)": self.gas_power,
            "Solar (MW)": self.solar_power,
            "Battery (MW)": self.battery_power,
            "Curtailment (MW)": self.curtailment,
            "Battery SOC (MWh)": self.battery_soc
        })
```

**3. Solution Metrics**
```python
@dataclass
class SolutionMetrics:
    """Key performance indicators"""
    total_npv: float  # $
    capex: float  # $
    opex_annual: float  # $/year
    lcoe: float  # $/MWh
    
    reliability_pct: float  # %
    total_curtailment_mwh: float
    num_curtailment_hours: int
    
    carbon_tons_annual: float
    carbon_intensity_g_per_kwh: float
    carbon_reduction_pct: float  # vs grid-only baseline
    
    grid_dependence_pct: float  # % of energy from grid
    gas_capacity_factor: float  # % utilization
    battery_cycles_per_year: float
    solar_capacity_factor: float
    
    solve_time_seconds: float
    optimality_gap_pct: float
```

**4. Complete Solution**
```python
@dataclass
class OptimizationSolution:
    """Complete optimization result"""
    capacity: CapacitySolution
    dispatch: DispatchSolution
    metrics: SolutionMetrics
    scenario_params: dict
    
    def save(self, filepath: str):
        """Save solution to JSON"""
        data = {
            "capacity": self.capacity.to_dict(),
            "dispatch": self.dispatch.to_dataframe().to_dict(),
            "metrics": asdict(self.metrics),
            "scenario_params": self.scenario_params
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'OptimizationSolution':
        """Load solution from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Reconstruct objects from dict
        # ... implementation
```


## Error Handling

### Data Collection Errors

**API Failures**
- **Error**: ERCOT/NREL/EIA API returns 500 or timeout
- **Handling**: 
  - Retry with exponential backoff (3 attempts)
  - Log error details
  - Fall back to cached data if available
  - Raise informative exception if all retries fail

**Missing Data**
- **Error**: Gaps in time series data
- **Handling**:
  - If gaps < 1%: Forward-fill missing values
  - If gaps > 1%: Raise exception requiring manual intervention
  - Log all imputation operations

**Invalid Data**
- **Error**: Values outside expected ranges
- **Handling**:
  - Validate all data using data model validation methods
  - Clip extreme outliers (e.g., LMP > $5000/MWh) with warning
  - Reject data if >5% of values are invalid

### Optimization Errors

**Infeasibility**
- **Error**: No feasible solution exists
- **Cause**: Reliability constraint too strict, capacity limits too low
- **Handling**:
  - Compute Irreducible Inconsistent Subsystem (IIS) using Gurobi
  - Report conflicting constraints to user
  - Suggest relaxing reliability target or increasing capacity bounds

**Numerical Issues**
- **Error**: Solver reports numerical difficulties
- **Handling**:
  - Scale variables and constraints (e.g., use GW instead of MW)
  - Tighten feasibility tolerances
  - Try alternative solver methods (simplex vs barrier)

**Timeout**
- **Error**: Solver exceeds time limit
- **Handling**:
  - Return best solution found so far
  - Report optimality gap
  - Suggest reducing problem size (e.g., weekly instead of hourly)

**Unbounded**
- **Error**: Objective can decrease indefinitely
- **Cause**: Missing capacity constraints or cost parameters
- **Handling**:
  - Check for zero or negative costs
  - Verify all capacity variables have upper bounds
  - Report specific unbounded variables

### Dashboard Errors

**Solve Timeout in UI**
- **Error**: User waits too long for results
- **Handling**:
  - Show progress bar with estimated time remaining
  - Allow cancellation
  - Offer to email results when complete (future enhancement)

**Invalid Input Parameters**
- **Error**: User enters unrealistic values
- **Handling**:
  - Validate inputs before solving
  - Show error messages with acceptable ranges
  - Provide default/recommended values

**Memory Issues**
- **Error**: Large problem exhausts memory
- **Handling**:
  - Monitor memory usage
  - Limit concurrent scenario runs
  - Suggest reducing problem size

### File I/O Errors

**Missing Files**
- **Error**: Required data file not found
- **Handling**:
  - Check for file existence before loading
  - Provide clear error message with expected file path
  - Offer to download/generate missing data

**Corrupted Files**
- **Error**: CSV/JSON parsing fails
- **Handling**:
  - Catch parsing exceptions
  - Validate file format and schema
  - Suggest re-downloading data


## Testing Strategy

### Unit Tests

**Data Pipeline Tests**
- Test each data collector independently with mock API responses
- Verify data validation catches invalid values
- Test interpolation and gap-filling logic
- Coverage target: >90%

```python
def test_ercot_lmp_collector():
    """Test ERCOT LMP data collection"""
    # Mock API response
    mock_response = create_mock_ercot_response()
    
    # Test data extraction
    df = parse_ercot_response(mock_response)
    
    # Assertions
    assert len(df) == 8760
    assert df['lmp'].min() >= -100
    assert df['lmp'].max() <= 5000
    assert df['lmp'].isna().sum() == 0
```

**Optimization Model Tests**
- Test model construction with small datasets (24 hours)
- Verify constraint generation (count and structure)
- Test objective function calculation
- Test solution extraction and validation

```python
def test_model_construction():
    """Test Pyomo model builds correctly"""
    # Create small test dataset
    test_data = create_test_market_data(hours=24)
    
    # Build model
    model = build_optimization_model(test_data, load_mw=10)
    
    # Verify structure
    assert len(model.hours) == 24
    assert hasattr(model, 'energy_balance')
    assert len(model.energy_balance) == 24
```

**Data Model Tests**
- Test dataclass validation methods
- Test serialization/deserialization
- Test metric calculations

### Integration Tests

**End-to-End Optimization**
- Test complete pipeline from data loading to solution
- Use small realistic dataset (1 week)
- Verify solution satisfies all constraints
- Check solution quality (objective value in expected range)

```python
def test_end_to_end_optimization():
    """Test complete optimization pipeline"""
    # Load test data
    data = load_test_data("test_week.csv")
    
    # Run optimization
    solution = run_optimization(data, facility_params)
    
    # Verify solution
    assert solution.metrics.reliability_pct >= 99.99
    assert solution.metrics.total_curtailment_mwh <= 0.67  # 1 hour for 1 week
    assert solution.capacity.grid_mw > 0  # Should always have some grid
```

**Scenario Analysis**
- Test batch solving with multiple scenarios
- Verify Pareto frontier calculation
- Test sensitivity analysis

### Validation Tests

**Physical Constraints**
- Energy balance: Supply = Demand at every hour
- Capacity limits: No dispatch exceeds capacity
- Battery dynamics: SOC evolution is correct
- Ramping: Gas generation respects ramp rates

```python
def test_energy_balance(solution: OptimizationSolution):
    """Verify energy balance at every hour"""
    dispatch = solution.dispatch.to_dataframe()
    
    for hour in range(8760):
        supply = (dispatch.loc[hour, 'Grid (MW)'] +
                 dispatch.loc[hour, 'Gas (MW)'] +
                 dispatch.loc[hour, 'Solar (MW)'] -
                 dispatch.loc[hour, 'Battery (MW)'] +
                 dispatch.loc[hour, 'Curtailment (MW)'])
        
        demand = 285  # MW
        
        assert abs(supply - demand) < 1e-3  # Tolerance for numerical precision
```

**Economic Validation**
- Cost calculations match manual computation
- NPV calculation is correct
- LCOE is reasonable (compare to industry benchmarks)

**Benchmark Tests**
- Grid-only solution should have zero CAPEX for BTM assets
- 100% reliable solution should have zero curtailment
- No-carbon-constraint should be cheaper than carbon-constrained

### Performance Tests

**Solve Time**
- Full year optimization completes in <30 minutes
- Weekly optimization completes in <2 minutes
- Measure solve time scaling with problem size

**Memory Usage**
- Monitor peak memory during solving
- Ensure <8GB for full year problem (laptop compatible)

**Dashboard Responsiveness**
- Page load time <2 seconds
- Visualization rendering <1 second
- Input changes trigger re-render <500ms

### Regression Tests

**Solution Stability**
- Same inputs produce same outputs (deterministic)
- Small input changes produce small output changes (continuity)
- Known test cases produce expected results

```python
def test_baseline_case():
    """Regression test for baseline scenario"""
    solution = run_optimization(baseline_params)
    
    # Compare to known good solution
    assert abs(solution.metrics.total_npv - 2_450_000_000) < 50_000_000  # Within 2%
    assert abs(solution.capacity.grid_mw - 200) < 10
    assert abs(solution.capacity.gas_mw - 150) < 10
```

### Test Data

**Synthetic Data**
- Generate realistic but controlled test data
- Include edge cases: price spikes, zero solar, extreme weather

**Historical Data Subset**
- Use 1 week of real data for fast tests
- Use 1 month for integration tests
- Use full year only for final validation

### Continuous Integration

**GitHub Actions Workflow**
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/unit --cov=src
      - name: Run integration tests
        run: pytest tests/integration
      - name: Check coverage
        run: coverage report --fail-under=80
```

**Test Organization**
```
tests/
├── unit/
│   ├── test_data_pipeline.py
│   ├── test_optimization_model.py
│   ├── test_data_models.py
│   └── test_visualization.py
├── integration/
│   ├── test_end_to_end.py
│   ├── test_scenario_analysis.py
│   └── test_dashboard.py
├── validation/
│   ├── test_physical_constraints.py
│   ├── test_economic_validation.py
│   └── test_benchmarks.py
└── fixtures/
    ├── test_data_1week.csv
    ├── test_data_1month.csv
    └── baseline_solution.json
```


## Design Decisions and Rationales

### 1. Linear Programming vs. Mixed-Integer Programming

**Decision**: Use LP formulation where possible, add binary variables only if needed for unit commitment

**Rationale**:
- LP solves much faster (seconds vs. minutes)
- Gas peaker ramping can be modeled with continuous variables
- Binary variables needed only for minimum up/down time constraints
- For this application, continuous relaxation is acceptable

**Trade-off**: Slightly less realistic gas operation, but 10-100x faster solve time

### 2. Hourly vs. Sub-Hourly Resolution

**Decision**: Use hourly time steps (8760 hours/year)

**Rationale**:
- Matches market data granularity (LMP is hourly)
- Sufficient for capacity planning decisions
- Keeps problem size manageable
- Data center load is constant, so sub-hourly detail not critical

**Alternative Considered**: 15-minute intervals (35,040 intervals/year)
- Would better capture battery arbitrage opportunities
- But 4x larger problem, much slower solving
- Marginal benefit for capacity planning

### 3. Deterministic vs. Stochastic Optimization

**Decision**: Use deterministic optimization with historical data

**Rationale**:
- Simpler to implement and explain
- Historical data provides realistic scenarios
- Scenario analysis captures uncertainty
- Stochastic programming requires scenario generation (complex)

**Future Enhancement**: Two-stage stochastic program
- Stage 1: Capacity decisions under uncertainty
- Stage 2: Dispatch decisions after uncertainty resolves
- Requires Monte Carlo scenario generation

### 4. Single-Year vs. Multi-Year Optimization

**Decision**: Optimize single representative year, extrapolate to 20 years

**Rationale**:
- Capacity decisions are long-lived (20+ years)
- Annual dispatch patterns repeat
- Forecasting 20 years of hourly prices is unrealistic
- NPV calculation accounts for time value of money

**Assumption**: Market conditions remain similar over planning horizon
- Reasonable for capacity planning
- Sensitivity analysis explores price variations

### 5. Pyomo vs. Direct Solver API

**Decision**: Use Pyomo algebraic modeling language

**Rationale**:
- Declarative syntax (easier to read and maintain)
- Solver-agnostic (can switch from Gurobi to CPLEX/HiGHS)
- Automatic constraint generation
- Better for prototyping and iteration

**Alternative**: Direct Gurobi Python API
- Faster model building (no translation layer)
- More control over solver features
- But less readable, harder to modify

### 6. Streamlit vs. Plotly Dash

**Decision**: Use Streamlit for dashboard

**Rationale**:
- Faster development (pure Python, minimal boilerplate)
- Good enough for portfolio project
- Free deployment on Streamlit Cloud
- Automatic reactivity

**Trade-off**: Less customization than Dash
- Acceptable for this use case
- Can migrate to Dash later if needed

### 7. Battery Model Complexity

**Decision**: Simple linear battery model with efficiency and SOC limits

**Rationale**:
- Captures key economics (round-trip efficiency, degradation)
- Avoids complexity of detailed electrochemical models
- Sufficient for capacity planning

**Omitted Details**:
- Temperature effects
- Cycle-dependent degradation
- Power-dependent efficiency
- These matter for detailed operation, not capacity planning

### 8. Solar Model

**Decision**: Use NREL PVWatts capacity factors (exogenous)

**Rationale**:
- Solar generation is deterministic given capacity
- No operational decisions (can't curtail solar in this model)
- Capacity factors from NREL are high quality

**Simplification**: No solar curtailment
- In reality, might curtail solar when battery is full and load is low
- But data centers have constant load, so rarely an issue

### 9. Grid Reliability Modeling

**Decision**: Model grid as always available up to capacity, use curtailment penalty

**Rationale**:
- Simplifies model (no stochastic outages)
- Curtailment penalty incentivizes reliability
- Captures economic trade-off

**Limitation**: Doesn't model specific outage events
- Could enhance with forced outage scenarios
- But adds complexity without much benefit for capacity planning

### 10. Carbon Accounting

**Decision**: Use hourly grid carbon intensity from EIA

**Rationale**:
- Captures time-varying grid emissions
- More accurate than annual average
- Enables 24/7 carbon-free energy analysis

**Alternative**: Annual average carbon intensity
- Simpler but less accurate
- Misses opportunity for temporal matching

### 11. Discount Rate

**Decision**: Use 7% real discount rate

**Rationale**:
- Typical for corporate infrastructure investments
- Between risk-free rate (~3%) and equity return (~10%)
- Sensitivity analysis explores 5-10% range

### 12. Planning Horizon

**Decision**: 20-year planning horizon

**Rationale**:
- Matches typical asset lifetime
- Grid interconnection: 30+ years
- Gas peakers: 25+ years
- Batteries: 10-15 years (assume replacement at year 10)
- Solar: 25+ years

**NPV Calculation**: Discount all future costs to present value

### 13. Demand Charges

**Decision**: Model demand charges as monthly peak grid draw

**Rationale**:
- Reflects actual utility tariff structure
- Major driver of battery value
- Requires tracking max grid draw across all hours

**Implementation**: Add variable for monthly peak, constrain to be ≥ hourly draw

### 14. Gas Ramping

**Decision**: 50% of capacity per hour ramp rate

**Rationale**:
- Typical for simple-cycle gas turbines
- Prevents unrealistic instantaneous ramping
- Adds realism without excessive complexity

**Alternative**: Unit commitment with minimum up/down times
- More realistic but requires binary variables
- Slower solving, marginal benefit

### 15. Curtailment Penalty

**Decision**: $10,000/MWh penalty

**Rationale**:
- Represents value of lost compute time
- 300 MW × $10,000/MWh = $3M/hour
- Aligns with industry estimates of AI training downtime cost
- High enough to strongly incentivize reliability

**Sensitivity**: Explore $5,000-$20,000/MWh range


## Implementation Phases

### Phase 1: Data Pipeline (Week 1)

**Objective**: Collect and validate all required data

**Deliverables**:
- `src/data_pipeline/ercot_collector.py`
- `src/data_pipeline/solar_collector.py`
- `src/data_pipeline/gas_collector.py`
- `src/data_pipeline/carbon_collector.py`
- `src/data_pipeline/validator.py`
- `data/ercot_lmp_hourly_2022_2024.csv`
- `data/solar_cf_west_texas.csv`
- `data/gas_prices_hourly.csv`
- `data/grid_carbon_intensity.csv`
- `data/tech_costs.json`
- `notebooks/01_data_exploration.ipynb`

**Success Criteria**:
- All data files generated and validated
- No more than 1% missing values
- Data exploration notebook shows reasonable patterns
- Unit tests pass for all collectors

### Phase 2: Optimization Model (Week 2-3)

**Objective**: Build and validate optimization model

**Deliverables**:
- `src/optimization/model_builder.py`
- `src/optimization/solver.py`
- `src/optimization/solution_extractor.py`
- `src/optimization/validator.py`
- `src/models/data_models.py`
- `notebooks/02_baseline_case.ipynb`
- `tests/unit/test_optimization_model.py`
- `tests/integration/test_end_to_end.py`

**Success Criteria**:
- Model solves for full year in <30 minutes
- Solution satisfies all constraints
- Baseline case (grid-only) produces sensible results
- Optimal portfolio case shows cost savings
- All tests pass

### Phase 3: Scenario Analysis (Week 3-4)

**Objective**: Explore parameter sensitivity and trade-offs

**Deliverables**:
- `src/analysis/scenario_generator.py`
- `src/analysis/batch_solver.py`
- `src/analysis/pareto_calculator.py`
- `src/analysis/sensitivity_analyzer.py`
- `notebooks/03_sensitivity_analysis.ipynb`
- `results/scenario_results.csv`
- `results/pareto_frontiers.json`

**Success Criteria**:
- 20+ scenarios solved successfully
- Pareto frontiers identified for cost-reliability and cost-carbon
- Sensitivity analysis shows which parameters matter most
- Key insights documented in notebook

### Phase 4: Visualization (Week 4)

**Objective**: Create compelling visualizations

**Deliverables**:
- `src/visualization/capacity_viz.py`
- `src/visualization/dispatch_viz.py`
- `src/visualization/cost_viz.py`
- `src/visualization/pareto_viz.py`
- `src/visualization/reliability_viz.py`
- `notebooks/04_visualization.ipynb`
- `results/figures/` (10+ publication-quality figures)

**Success Criteria**:
- All visualization functions work with real data
- Figures are clear, informative, and publication-ready
- Interactive plots work in Jupyter notebook

### Phase 5: Dashboard (Week 4-5)

**Objective**: Build interactive web dashboard

**Deliverables**:
- `dashboard/app.py`
- `dashboard/pages/setup.py`
- `dashboard/pages/portfolio.py`
- `dashboard/pages/dispatch.py`
- `dashboard/pages/scenarios.py`
- `dashboard/pages/case_study.py`
- `dashboard/utils.py`
- Deployed app on Streamlit Cloud

**Success Criteria**:
- Dashboard runs locally without errors
- All pages render correctly
- Input widgets work and trigger re-computation
- Dashboard deployed and accessible via URL

### Phase 6: Case Study (Week 5)

**Objective**: Analyze 300MW West Texas facility

**Deliverables**:
- `docs/case_study.md`
- `results/case_study_results.json`
- `results/figures/case_study_*.png`

**Success Criteria**:
- Baseline and optimal scenarios computed
- Clear recommendations with cost-benefit analysis
- Comparison to alternative strategies
- Narrative suitable for executive presentation

### Phase 7: Documentation and Polish (Week 5-6)

**Objective**: Finalize project for portfolio

**Deliverables**:
- `README.md` (comprehensive project overview)
- `docs/model_formulation.pdf` (LaTeX document)
- `docs/user_guide.md`
- `docs/api_reference.md`
- `requirements.txt`
- `.gitignore`
- `LICENSE`
- GitHub repository with clean commit history

**Success Criteria**:
- README clearly explains project and how to run
- Mathematical formulation documented in LaTeX
- Code is well-commented and follows PEP 8
- All tests pass
- Repository is public and presentable


## Project Structure

```
datacenter-energy-optimization/
├── README.md                          # Project overview and setup instructions
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore patterns
├── LICENSE                           # MIT or Apache 2.0
│
├── data/                             # Data files (not in git, download via scripts)
│   ├── raw/                          # Raw downloaded data
│   │   ├── ercot_lmp_2022.csv
│   │   ├── ercot_lmp_2023.csv
│   │   └── ercot_lmp_2024.csv
│   ├── processed/                    # Cleaned and validated data
│   │   ├── ercot_lmp_hourly_2022_2024.csv
│   │   ├── solar_cf_west_texas.csv
│   │   ├── gas_prices_hourly.csv
│   │   └── grid_carbon_intensity.csv
│   └── tech_costs.json               # Technology cost parameters
│
├── src/                              # Source code
│   ├── __init__.py
│   │
│   ├── data_pipeline/                # Data collection and preprocessing
│   │   ├── __init__.py
│   │   ├── ercot_collector.py       # ERCOT LMP data collection
│   │   ├── solar_collector.py       # NREL solar profile generation
│   │   ├── gas_collector.py         # EIA gas price collection
│   │   ├── carbon_collector.py      # EIA grid carbon intensity
│   │   └── validator.py             # Data validation utilities
│   │
│   ├── models/                       # Data models and schemas
│   │   ├── __init__.py
│   │   ├── market_data.py           # MarketData dataclass
│   │   ├── technology.py            # TechnologyCosts, FacilityParams
│   │   └── solution.py              # Solution dataclasses
│   │
│   ├── optimization/                 # Optimization model
│   │   ├── __init__.py
│   │   ├── model_builder.py         # Pyomo model construction
│   │   ├── solver.py                # Solver interface and configuration
│   │   ├── solution_extractor.py    # Extract results from solved model
│   │   └── validator.py             # Solution validation
│   │
│   ├── analysis/                     # Scenario and sensitivity analysis
│   │   ├── __init__.py
│   │   ├── scenario_generator.py    # Generate parameter combinations
│   │   ├── batch_solver.py          # Parallel scenario solving
│   │   ├── pareto_calculator.py     # Pareto frontier identification
│   │   └── sensitivity_analyzer.py  # Sensitivity analysis
│   │
│   ├── visualization/                # Plotting and visualization
│   │   ├── __init__.py
│   │   ├── capacity_viz.py          # Capacity mix visualizations
│   │   ├── dispatch_viz.py          # Dispatch heatmaps
│   │   ├── cost_viz.py              # Cost breakdown charts
│   │   ├── pareto_viz.py            # Pareto frontier plots
│   │   └── reliability_viz.py       # Reliability analysis plots
│   │
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       └── logging_config.py        # Logging setup
│
├── dashboard/                        # Streamlit dashboard
│   ├── app.py                       # Main dashboard application
│   ├── pages/                       # Dashboard pages
│   │   ├── setup.py                # Optimization setup page
│   │   ├── portfolio.py            # Optimal portfolio page
│   │   ├── dispatch.py             # Hourly dispatch page
│   │   ├── scenarios.py            # Scenario comparison page
│   │   └── case_study.py           # Case study page
│   └── utils.py                     # Dashboard utilities
│
├── notebooks/                        # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb    # Explore collected data
│   ├── 02_baseline_case.ipynb       # Grid-only baseline
│   ├── 03_optimal_portfolio.ipynb   # Optimal solution analysis
│   ├── 04_sensitivity_analysis.ipynb # Parameter sensitivity
│   └── 05_visualization.ipynb       # Create publication figures
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── unit/                        # Unit tests
│   │   ├── test_data_pipeline.py
│   │   ├── test_optimization_model.py
│   │   ├── test_data_models.py
│   │   └── test_visualization.py
│   ├── integration/                 # Integration tests
│   │   ├── test_end_to_end.py
│   │   ├── test_scenario_analysis.py
│   │   └── test_dashboard.py
│   ├── validation/                  # Validation tests
│   │   ├── test_physical_constraints.py
│   │   ├── test_economic_validation.py
│   │   └── test_benchmarks.py
│   └── fixtures/                    # Test data
│       ├── test_data_1week.csv
│       ├── test_data_1month.csv
│       └── baseline_solution.json
│
├── results/                          # Optimization results (not in git)
│   ├── solutions/                   # Individual solution files
│   │   ├── baseline_grid_only.json
│   │   ├── optimal_portfolio.json
│   │   └── scenario_*.json
│   ├── scenario_results.csv         # Summary of all scenarios
│   ├── pareto_frontiers.json        # Pareto frontier data
│   └── figures/                     # Generated figures
│       ├── capacity_mix.png
│       ├── dispatch_heatmap.png
│       ├── cost_breakdown.png
│       ├── pareto_cost_reliability.png
│       └── sensitivity_tornado.png
│
├── docs/                             # Documentation
│   ├── model_formulation.pdf        # Mathematical model (LaTeX)
│   ├── model_formulation.tex        # LaTeX source
│   ├── case_study.md                # 300MW West Texas case study
│   ├── user_guide.md                # How to use the system
│   ├── api_reference.md             # API documentation
│   └── design_decisions.md          # Design rationale
│
└── scripts/                          # Utility scripts
    ├── download_all_data.py         # Download all required data
    ├── run_baseline.py              # Run baseline optimization
    ├── run_scenarios.py             # Run all scenario analyses
    └── generate_report.py           # Generate case study report
```

## Key Files Description

**README.md**: Project overview, setup instructions, quick start guide, links to documentation

**requirements.txt**: 
```
pyomo>=6.7.0
gurobipy>=11.0.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
matplotlib>=3.8.0
streamlit>=1.28.0
requests>=2.31.0
beautifulsoup4>=4.12.0
pytest>=7.4.0
pytest-cov>=4.1.0
jupyter>=1.0.0
```

**src/optimization/model_builder.py**: Core optimization model (500-800 lines)
- Most complex file in project
- Pyomo model construction
- All constraints and objective function

**dashboard/app.py**: Streamlit dashboard entry point (200-300 lines)
- Page navigation
- Session state management
- Main layout

**notebooks/03_optimal_portfolio.ipynb**: Key analysis notebook
- Run optimization
- Analyze results
- Generate insights
- Create visualizations


## Performance Considerations

### Optimization Model Performance

**Problem Size**:
- Variables: ~35,000 (4 capacity + 8760 × 4 hourly)
- Constraints: ~70,000 (energy balance, capacity limits, battery dynamics, etc.)
- Problem type: Linear Program (LP) or Mixed-Integer Linear Program (MILP)

**Expected Solve Time**:
- LP formulation: 2-10 minutes on laptop
- MILP with unit commitment: 10-30 minutes
- Target: <30 minutes for full year

**Optimization Strategies**:
1. **Warm Start**: Use previous solution as starting point for similar scenarios
2. **Decomposition**: Solve weekly sub-problems, then coordinate (if needed)
3. **Constraint Reduction**: Remove redundant constraints
4. **Variable Bounds**: Tight bounds improve solver performance
5. **Scaling**: Scale variables to similar magnitudes (e.g., use GW instead of MW)

**Solver Configuration**:
```python
solver_options = {
    'MIPGap': 0.005,           # 0.5% optimality gap
    'TimeLimit': 1800,         # 30 minutes
    'Threads': 0,              # Use all available cores
    'Method': 2,               # Barrier method for LP
    'Crossover': 0,            # Skip crossover for faster solve
    'NumericFocus': 1,         # Moderate numerical precision
    'Presolve': 2,             # Aggressive presolve
}
```

### Data Pipeline Performance

**Data Volume**:
- ERCOT LMP: 3 years × 8760 hours = 26,280 rows
- Solar CF: 8760 rows
- Gas prices: 3 years × 365 days = 1,095 rows → interpolate to 26,280
- Total: ~100 MB of CSV data

**Optimization**:
1. **Caching**: Cache downloaded data, don't re-download
2. **Parallel Downloads**: Use asyncio for concurrent API calls
3. **Chunked Processing**: Process large files in chunks
4. **Efficient Storage**: Use Parquet instead of CSV for faster loading

### Dashboard Performance

**Challenges**:
- Large dispatch heatmap (8760 hours)
- Real-time optimization (5-30 minutes)
- Multiple concurrent users (if deployed)

**Solutions**:

1. **Pre-computation**: 
   - Pre-solve common scenarios
   - Cache results in session state
   - Load pre-computed results on page load

2. **Data Downsampling**:
   - Show daily averages instead of hourly for overview
   - Allow zooming to hourly detail
   - Use Plotly's built-in downsampling

3. **Lazy Loading**:
   - Load visualizations only when page is viewed
   - Use Streamlit's caching decorators

4. **Async Solving**:
   - Run optimization in background
   - Show progress bar
   - Allow user to continue exploring while solving

5. **Deployment Optimization**:
   - Use Streamlit Cloud's caching
   - Consider Redis for shared cache across users
   - Limit concurrent optimizations

**Caching Strategy**:
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_market_data():
    return pd.read_csv("data/ercot_lmp_hourly.csv")

@st.cache_resource
def get_solver():
    return pyo.SolverFactory('gurobi')

@st.cache_data
def solve_optimization(_model):  # Leading underscore prevents hashing
    solver = get_solver()
    results = solver.solve(_model)
    return extract_solution(_model)
```

### Memory Management

**Potential Issues**:
- Large Pyomo models consume memory
- Multiple scenarios in memory simultaneously
- Dashboard session state accumulation

**Mitigation**:
1. **Model Cleanup**: Delete model after extracting solution
2. **Batch Processing**: Solve scenarios sequentially, not all at once
3. **Session State Limits**: Clear old results from session state
4. **Garbage Collection**: Explicitly call `gc.collect()` after large operations

### Scalability Considerations

**Current Design**: Single-year, single-location optimization

**Future Scaling**:
1. **Multi-Year**: Optimize 5-10 years with technology evolution
   - Challenge: 5x-10x more variables
   - Solution: Rolling horizon optimization

2. **Multi-Location**: Optimize portfolio of data centers
   - Challenge: Coupling constraints, larger problem
   - Solution: Decomposition methods (Benders, Lagrangian)

3. **Stochastic**: Incorporate uncertainty in prices, demand, weather
   - Challenge: Scenario tree explosion
   - Solution: Sample average approximation, limited scenarios

4. **Real-Time**: Update dispatch decisions based on latest forecasts
   - Challenge: Need fast re-optimization (<1 minute)
   - Solution: Model predictive control, warm starts


## Security and Privacy Considerations

### Data Security

**Public Data Sources**:
- All data sources (ERCOT, NREL, EIA) are publicly available
- No authentication required
- No sensitive or proprietary data

**API Keys**:
- NREL PVWatts: Free API, no key required for basic usage
- ERCOT: Public data, no authentication
- EIA: Free API key (optional, increases rate limits)

**Best Practices**:
- Store API keys in environment variables, not in code
- Use `.env` file for local development (add to `.gitignore`)
- Document required environment variables in README

### Code Security

**Dependencies**:
- Use `requirements.txt` with pinned versions
- Regularly update dependencies for security patches
- Use `pip-audit` to check for known vulnerabilities

**Input Validation**:
- Validate all user inputs in dashboard
- Sanitize file paths to prevent directory traversal
- Limit input ranges to reasonable values

### Deployment Security

**Streamlit Cloud**:
- HTTPS by default
- No sensitive data stored
- Public repository (open source project)

**Secrets Management**:
- Use Streamlit secrets for API keys (if needed)
- Never commit secrets to git
- Document secrets setup in deployment guide

### Privacy

**No User Data Collection**:
- Dashboard doesn't collect personal information
- No user accounts or authentication
- No analytics or tracking (unless explicitly added)

**Data Retention**:
- User inputs stored only in session state (temporary)
- No persistent storage of user scenarios
- Results can be downloaded by user

## Deployment Strategy

### Local Development

**Setup**:
```bash
# Clone repository
git clone https://github.com/username/datacenter-energy-optimization.git
cd datacenter-energy-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download data
python scripts/download_all_data.py

# Run tests
pytest

# Run dashboard
streamlit run dashboard/app.py
```

### Streamlit Cloud Deployment

**Steps**:
1. Push code to GitHub repository
2. Connect Streamlit Cloud to GitHub
3. Select repository and branch
4. Configure Python version (3.10+)
5. Add secrets (if any API keys needed)
6. Deploy

**Configuration** (`streamlit/config.toml`):
```toml
[server]
maxUploadSize = 200
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

**Limitations**:
- Free tier: 1 GB RAM, 1 CPU core
- May need to reduce problem size or use pre-computed results
- Consider paid tier for full optimization capability

### Alternative Deployment Options

**Heroku**:
- More resources than Streamlit Cloud free tier
- Requires Procfile and setup.sh
- Gurobi license may be issue (academic license is node-locked)

**AWS/GCP/Azure**:
- Full control over resources
- Can use larger instances for faster solving
- More complex setup and cost

**Docker**:
- Containerize application for consistent deployment
- Useful for local testing and cloud deployment
- Include Gurobi license handling in Dockerfile

### Continuous Deployment

**GitHub Actions** (`.github/workflows/deploy.yml`):
```yaml
name: Deploy to Streamlit Cloud
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest
      - name: Trigger Streamlit Cloud deployment
        # Streamlit Cloud auto-deploys on push to main
        run: echo "Deployment triggered"
```

## Maintenance and Future Enhancements

### Maintenance Tasks

**Data Updates**:
- Update ERCOT LMP data quarterly
- Update technology costs annually (NREL ATB releases)
- Update solar profiles if location changes

**Dependency Updates**:
- Update Python packages quarterly
- Test compatibility with new versions
- Update Gurobi when new versions release

**Bug Fixes**:
- Monitor GitHub issues
- Fix critical bugs within 1 week
- Document known issues in README

### Future Enhancements

**Priority 1 (High Value, Moderate Effort)**:
1. **Additional Locations**: Add MISO, PJM, CAISO regions
2. **Nuclear Option**: Add small modular reactors (SMRs) as technology choice
3. **Demand Response**: Model flexible data center load
4. **Multi-Year**: Optimize technology evolution over time

**Priority 2 (High Value, High Effort)**:
1. **Stochastic Optimization**: Two-stage model with uncertainty
2. **Real-Time Dispatch**: Model predictive control for operations
3. **Multi-Site**: Optimize portfolio of data centers
4. **Machine Learning**: Forecast prices using ML models

**Priority 3 (Nice to Have)**:
1. **Mobile Dashboard**: Responsive design for mobile devices
2. **PDF Report Generation**: Automated report creation
3. **Email Notifications**: Alert when optimization completes
4. **User Accounts**: Save and share scenarios
5. **API**: RESTful API for programmatic access

### Research Extensions

**Academic Papers**:
- "Optimal Energy Portfolios for AI Data Centers Under Uncertainty"
- "Behind-the-Meter vs. Grid-Scale: A Comparative Analysis"
- "24/7 Carbon-Free Energy for Data Centers: Cost and Feasibility"

**Conference Presentations**:
- INFORMS Annual Meeting (Operations Research)
- IEEE Power & Energy Society General Meeting
- ACM e-Energy Conference

**Industry Engagement**:
- Share findings with data center operators
- Consult with utilities on interconnection policies
- Advise policymakers on grid reliability and data centers

