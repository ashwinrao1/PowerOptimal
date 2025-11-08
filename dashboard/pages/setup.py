"""
Optimization Setup Page

This page allows users to configure facility parameters and run the optimization model.
Users can adjust facility size, reliability targets, carbon reduction goals, and select
available technologies before triggering the optimization solve.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

# Add src directory to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import after path is set
# Import directly from modules to avoid __init__.py relative import issues
from models.market_data import MarketData
from models.technology import TechnologyCosts, FacilityParams
from optimization.model_builder import build_optimization_model
from optimization.solution_extractor import extract_solution
# Import solver and all error classes directly from solver module
import optimization.solver as solver_module
solve_model = solver_module.solve_model
SolverError = solver_module.SolverError
InfeasibleError = solver_module.InfeasibleError
UnboundedError = solver_module.UnboundedError
NumericalError = solver_module.NumericalError
TimeoutError = solver_module.TimeoutError


@st.cache_data(ttl=3600)
def load_market_data_for_year(year: str, location: str) -> MarketData:
    """
    Load market data for specified year and location with caching.
    Cache expires after 1 hour.
    
    Args:
        year: Year scenario (e.g., "2024")
        location: Location name (e.g., "West Texas")
        
    Returns:
        MarketData object with hourly data
    """
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    
    # Load data files
    lmp_df = pd.read_csv(data_dir / "ercot_lmp_hourly_2022_2024.csv")
    solar_df = pd.read_csv(data_dir / "solar_cf_west_texas.csv")
    gas_df = pd.read_csv(data_dir / "gas_prices_hourly.csv")
    carbon_df = pd.read_csv(data_dir / "grid_carbon_intensity.csv")
    
    # Filter by year if timestamp column exists
    if 'timestamp' in lmp_df.columns:
        lmp_df['timestamp'] = pd.to_datetime(lmp_df['timestamp'])
        lmp_df = lmp_df[lmp_df['timestamp'].dt.year == int(year)]
    
    # Take first 8760 hours and extract LMP data
    # Try different possible column names
    if 'lmp' in lmp_df.columns:
        lmp_data = lmp_df['lmp'].values[:8760]
    elif 'lmp_dam' in lmp_df.columns:
        lmp_data = lmp_df['lmp_dam'].values[:8760]
    elif 'lmp_rtm' in lmp_df.columns:
        lmp_data = lmp_df['lmp_rtm'].values[:8760]
    else:
        lmp_data = lmp_df.iloc[:8760, 1].values
    
    # Load other data with flexible column names
    if 'capacity_factor' in solar_df.columns:
        solar_cf = solar_df['capacity_factor'].values[:8760]
    elif 'cf' in solar_df.columns:
        solar_cf = solar_df['cf'].values[:8760]
    else:
        solar_cf = solar_df.iloc[:8760, 1].values
    
    if 'price_mmbtu' in gas_df.columns:
        gas_prices = gas_df['price_mmbtu'].values[:8760]
    elif 'price' in gas_df.columns:
        gas_prices = gas_df['price'].values[:8760]
    else:
        gas_prices = gas_df.iloc[:8760, 1].values
    
    if 'carbon_intensity_kg_per_mwh' in carbon_df.columns:
        carbon_intensity = carbon_df['carbon_intensity_kg_per_mwh'].values[:8760]
    elif 'carbon_intensity' in carbon_df.columns:
        carbon_intensity = carbon_df['carbon_intensity'].values[:8760]
    else:
        carbon_intensity = carbon_df.iloc[:8760, 1].values
    
    # Create timestamp index
    timestamps = pd.date_range(start=f'{year}-01-01', periods=8760, freq='H')
    
    return MarketData(
        timestamp=timestamps,
        lmp=lmp_data,
        gas_price=gas_prices,
        solar_cf=solar_cf,
        grid_carbon_intensity=carbon_intensity
    )


def run_optimization_workflow(
    facility_size_mw: float,
    reliability_target: float,
    carbon_reduction_pct: float,
    location: str,
    year_scenario: str,
    available_technologies: dict,
    progress_callback=None
) -> dict:
    """
    Run the complete optimization workflow.
    
    Args:
        facility_size_mw: Facility size in MW
        reliability_target: Reliability target as percentage (e.g., 99.99)
        carbon_reduction_pct: Carbon reduction target as percentage (0-100)
        location: Location name
        year_scenario: Year scenario
        available_technologies: Dict of technology availability
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dictionary with optimization results or error information
    """
    try:
        # Update progress
        if progress_callback:
            progress_callback("Loading market data...", 0.1)
        
        # Load market data
        market_data = load_market_data_for_year(year_scenario, location)
        
        # Update progress
        if progress_callback:
            progress_callback("Configuring optimization parameters...", 0.2)
        
        # Create technology costs
        tech_costs = TechnologyCosts()
        
        # Create facility parameters
        facility_params = FacilityParams(
            it_load_mw=facility_size_mw / 1.05,  # Adjust for PUE
            pue=1.05,
            reliability_target=reliability_target / 100.0,  # Convert percentage to fraction
            carbon_budget=None,  # Will set based on carbon_reduction_pct
            planning_horizon_years=20,
            discount_rate=0.07,
            curtailment_penalty=10000
        )
        
        # Calculate carbon budget if carbon reduction is specified
        if carbon_reduction_pct > 0:
            # Calculate baseline emissions (grid-only)
            baseline_emissions_kg = np.sum(facility_params.total_load_mw * market_data.grid_carbon_intensity)
            baseline_emissions_tons = baseline_emissions_kg / 1000
            
            # Set carbon budget based on reduction target
            reduction_factor = (100 - carbon_reduction_pct) / 100.0
            facility_params.carbon_budget = baseline_emissions_tons * reduction_factor
        
        # Update progress
        if progress_callback:
            progress_callback("Building optimization model...", 0.3)
        
        # Build optimization model
        model = build_optimization_model(
            market_data=market_data,
            tech_costs=tech_costs,
            facility_params=facility_params,
            allow_gas=available_technologies.get('gas', True),
            allow_battery=available_technologies.get('battery', True),
            allow_solar=available_technologies.get('solar', True)
        )
        
        # Update progress
        if progress_callback:
            progress_callback("Solving optimization model (this may take several minutes)...", 0.4)
        
        # Solve model
        results, solve_time = solve_model(
            model=model,
            time_limit=1800,  # 30 minutes
            mip_gap=0.005,
            verbose=False
        )
        
        # Update progress
        if progress_callback:
            progress_callback("Extracting solution...", 0.9)
        
        # Extract solution
        solution = extract_solution(
            model=model,
            market_data=market_data,
            tech_costs=tech_costs,
            facility_params=facility_params,
            solve_time=solve_time,
            optimality_gap=0.0,
            scenario_params={
                "facility_size_mw": facility_size_mw,
                "reliability_target": reliability_target,
                "carbon_reduction_pct": carbon_reduction_pct,
                "location": location,
                "year_scenario": year_scenario,
                "available_technologies": available_technologies
            }
        )
        
        # Update progress
        if progress_callback:
            progress_callback("Optimization complete!", 1.0)
        
        return {
            "success": True,
            "solution": solution,
            "solve_time": solve_time
        }
        
    except InfeasibleError as e:
        return {
            "success": False,
            "error_type": "Infeasible",
            "error_message": str(e)
        }
    except TimeoutError as e:
        return {
            "success": False,
            "error_type": "Timeout",
            "error_message": str(e)
        }
    except SolverError as e:
        return {
            "success": False,
            "error_type": "Solver Error",
            "error_message": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "error_type": "Unexpected Error",
            "error_message": f"An unexpected error occurred: {str(e)}"
        }


def render():
    """Render the optimization setup page."""
    st.title("Optimization Setup")
    st.markdown("Configure facility parameters and run the optimization model to determine the optimal energy portfolio.")
    
    st.markdown("---")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Facility Parameters")
        
        # Facility size slider
        facility_size = st.slider(
            "Facility Size (MW)",
            min_value=100,
            max_value=500,
            value=st.session_state.facility_size_mw,
            step=10,
            help="Total facility power requirement including IT load and overhead (PUE)"
        )
        st.session_state.facility_size_mw = facility_size
        
        # Reliability target slider
        reliability_target = st.slider(
            "Reliability Target (%)",
            min_value=99.0,
            max_value=99.999,
            value=st.session_state.reliability_target,
            step=0.001,
            format="%.3f",
            help="Target uptime percentage. 99.99% = 1 hour downtime per year"
        )
        st.session_state.reliability_target = reliability_target
        
        # Carbon reduction slider
        carbon_reduction = st.slider(
            "Carbon Reduction Target (%)",
            min_value=0,
            max_value=100,
            value=st.session_state.carbon_reduction_pct,
            step=5,
            help="Target carbon reduction vs grid-only baseline. 0% = no constraint"
        )
        st.session_state.carbon_reduction_pct = carbon_reduction
        
        st.markdown("---")
        
        st.subheader("Location and Scenario")
        
        # Location dropdown
        location_col, year_col = st.columns(2)
        
        with location_col:
            location = st.selectbox(
                "Location",
                options=["West Texas"],
                index=0,
                help="Geographic location for market data"
            )
            st.session_state.location = location
        
        with year_col:
            year_scenario = st.selectbox(
                "Year Scenario",
                options=["2022", "2023", "2024"],
                index=2,
                help="Historical year for market data"
            )
            st.session_state.year_scenario = year_scenario
        
        st.markdown("---")
        
        st.subheader("Available Technologies")
        st.markdown("Select which technologies are available for investment:")
        
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            grid_enabled = st.checkbox(
                "Grid Connection",
                value=st.session_state.available_technologies['grid'],
                help="Utility grid interconnection (always recommended)",
                disabled=True  # Grid should always be available
            )
            st.session_state.available_technologies['grid'] = grid_enabled
            
            battery_enabled = st.checkbox(
                "Battery Storage",
                value=st.session_state.available_technologies['battery'],
                help="Lithium-ion battery energy storage system"
            )
            st.session_state.available_technologies['battery'] = battery_enabled
        
        with tech_col2:
            gas_enabled = st.checkbox(
                "Natural Gas Peakers",
                value=st.session_state.available_technologies['gas'],
                help="On-site natural gas generation"
            )
            st.session_state.available_technologies['gas'] = gas_enabled
            
            solar_enabled = st.checkbox(
                "Solar PV",
                value=st.session_state.available_technologies['solar'],
                help="On-site solar photovoltaic generation"
            )
            st.session_state.available_technologies['solar'] = solar_enabled
    
    with col2:
        st.subheader("Configuration Summary")
        
        # Display current configuration
        st.metric("Facility Size", f"{facility_size} MW")
        st.metric("Reliability", f"{reliability_target:.3f}%")
        st.metric("Carbon Reduction", f"{carbon_reduction}%")
        
        st.markdown("**Location:**")
        st.write(f"{location}, {year_scenario}")
        
        st.markdown("**Technologies:**")
        tech_list = []
        if st.session_state.available_technologies['grid']:
            tech_list.append("Grid")
        if st.session_state.available_technologies['gas']:
            tech_list.append("Gas")
        if st.session_state.available_technologies['battery']:
            tech_list.append("Battery")
        if st.session_state.available_technologies['solar']:
            tech_list.append("Solar")
        
        for tech in tech_list:
            st.write(f"- {tech}")
        
        st.markdown("---")
        
        # Estimated solve time
        st.info("""
        **Estimated Solve Time:**
        
        5-30 minutes depending on problem complexity and available computing resources.
        """)
    
    st.markdown("---")
    
    # Run optimization button
    col_button1, col_button2, col_button3 = st.columns([1, 1, 1])
    
    with col_button2:
        run_button = st.button(
            "🚀 Run Optimization",
            type="primary",
            use_container_width=True,
            help="Start the optimization solve"
        )
    
    # Handle optimization execution
    if run_button:
        # Validate inputs
        if not any(st.session_state.available_technologies.values()):
            st.error("Please select at least one technology option.")
            return
        
        # Update status
        st.session_state.optimization_status = "Running"
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message: str, progress: float):
            """Callback to update progress indicators."""
            progress_bar.progress(progress)
            status_text.text(message)
        
        # Run optimization
        with st.spinner("Initializing optimization..."):
            result = run_optimization_workflow(
                facility_size_mw=facility_size,
                reliability_target=reliability_target,
                carbon_reduction_pct=carbon_reduction,
                location=location,
                year_scenario=year_scenario,
                available_technologies=st.session_state.available_technologies,
                progress_callback=update_progress
            )
        
        # Handle results
        if result["success"]:
            # Store results in session state
            st.session_state.optimization_result = result["solution"]
            st.session_state.solve_time = result["solve_time"]
            st.session_state.optimization_status = "Complete"
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Show success message
            st.success(f"Optimization completed successfully in {result['solve_time']:.1f} seconds!")
            
            # Display key results
            st.markdown("### Optimization Results")
            
            solution = result["solution"]
            
            # Display capacity results
            st.markdown("#### Optimal Capacity Mix")
            capacity_col1, capacity_col2, capacity_col3, capacity_col4 = st.columns(4)
            
            with capacity_col1:
                st.metric("Grid", f"{solution.capacity.grid_mw:.1f} MW")
            with capacity_col2:
                st.metric("Gas", f"{solution.capacity.gas_mw:.1f} MW")
            with capacity_col3:
                st.metric("Battery", f"{solution.capacity.battery_mwh:.1f} MWh")
            with capacity_col4:
                st.metric("Solar", f"{solution.capacity.solar_mw:.1f} MW")
            
            # Display key metrics
            st.markdown("#### Key Metrics")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    "Total NPV",
                    f"${solution.metrics.total_npv/1e9:.2f}B",
                    help="20-year net present value"
                )
                st.metric(
                    "LCOE",
                    f"${solution.metrics.lcoe:.2f}/MWh",
                    help="Levelized cost of energy"
                )
            
            with metric_col2:
                st.metric(
                    "Reliability",
                    f"{solution.metrics.reliability_pct:.4f}%",
                    help="Percentage of load served"
                )
                st.metric(
                    "Carbon Intensity",
                    f"{solution.metrics.carbon_intensity_g_per_kwh:.1f} g/kWh",
                    help="Carbon emissions per kWh"
                )
            
            with metric_col3:
                st.metric(
                    "Grid Dependence",
                    f"{solution.metrics.grid_dependence_pct:.1f}%",
                    help="Percentage of energy from grid"
                )
                st.metric(
                    "Carbon Reduction",
                    f"{solution.metrics.carbon_reduction_pct:.1f}%",
                    help="Reduction vs grid-only baseline"
                )
            
            st.markdown("---")
            
            st.info("Navigate to **Optimal Portfolio** page to view detailed results and visualizations.")
            
        else:
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Update status
            st.session_state.optimization_status = "Failed"
            
            # Show error message
            st.error(f"**{result['error_type']}**")
            st.error(result['error_message'])
            
            # Provide suggestions based on error type
            if result['error_type'] == "Infeasible":
                st.warning("""
                **Suggestions to resolve infeasibility:**
                - Reduce the reliability target
                - Increase the carbon reduction target (allow more emissions)
                - Enable more technology options
                - Increase facility size to allow more flexibility
                """)
            elif result['error_type'] == "Timeout":
                st.warning("""
                **Suggestions to resolve timeout:**
                - The problem may be too complex for the time limit
                - Try reducing the number of enabled technologies
                - Try a less strict reliability target
                - The solver may still have found a feasible solution (check results)
                """)
    
    # Show current optimization status if not running
    elif st.session_state.optimization_status == "Complete":
        st.success("Previous optimization completed successfully!")
        st.info("Modify parameters above and click 'Run Optimization' to solve a new scenario, or navigate to other pages to view results.")
    
    elif st.session_state.optimization_status == "Failed":
        st.warning("Previous optimization failed. Adjust parameters and try again.")
    
    else:
        st.info("Configure parameters above and click 'Run Optimization' to start.")
