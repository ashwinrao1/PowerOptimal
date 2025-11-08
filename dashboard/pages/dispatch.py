"""
Hourly Dispatch Page

This page visualizes hourly operational decisions including dispatch heatmaps,
time range selection for detailed analysis, and operational statistics.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.models.solution import OptimizationSolution

# Add src directory to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import after path is set
try:
    from visualization.dispatch_viz import plot_dispatch_heatmap, plot_dispatch_stacked_area
    from models.solution import OptimizationSolution
    from models.market_data import MarketData
except ImportError as e:
    import warnings
    warnings.warn(f"Import error: {e}. Some functionality may not be available.")
    OptimizationSolution = None
    MarketData = None


@st.cache_data(ttl=1800)
def calculate_operational_statistics(_solution: Any) -> dict:
    """
    Calculate operational statistics from dispatch solution with caching.
    Cache expires after 30 minutes.
    
    Args:
        _solution: OptimizationSolution object (underscore prevents hashing)
        
    Returns:
        Dictionary with operational statistics
    """
    dispatch_df = _solution.dispatch.to_dataframe()
    
    # Gas utilization hours (hours with non-zero gas generation)
    gas_utilization_hours = (dispatch_df["Gas (MW)"] > 0.01).sum()
    
    # Battery cycles per year
    # A cycle is a full charge-discharge, so sum absolute battery power and divide by 2x capacity
    if _solution.capacity.battery_mwh > 0:
        total_battery_throughput = dispatch_df["Battery (MW)"].abs().sum()
        battery_cycles = total_battery_throughput / (2 * _solution.capacity.battery_mwh)
    else:
        battery_cycles = 0
    
    # Solar capacity factor
    if _solution.capacity.solar_mw > 0:
        solar_energy = dispatch_df["Solar (MW)"].sum()
        max_solar_energy = _solution.capacity.solar_mw * len(dispatch_df)
        solar_capacity_factor = (solar_energy / max_solar_energy) * 100 if max_solar_energy > 0 else 0
    else:
        solar_capacity_factor = 0
    
    # Peak grid draw
    peak_grid_draw = dispatch_df["Grid (MW)"].max()
    
    # Average grid draw
    avg_grid_draw = dispatch_df["Grid (MW)"].mean()
    
    # Gas capacity factor
    if _solution.capacity.gas_mw > 0:
        gas_energy = dispatch_df["Gas (MW)"].sum()
        max_gas_energy = _solution.capacity.gas_mw * len(dispatch_df)
        gas_capacity_factor = (gas_energy / max_gas_energy) * 100 if max_gas_energy > 0 else 0
    else:
        gas_capacity_factor = 0
    
    # Battery statistics
    if _solution.capacity.battery_mwh > 0:
        avg_soc = dispatch_df["Battery SOC (MWh)"].mean()
        min_soc = dispatch_df["Battery SOC (MWh)"].min()
        max_soc = dispatch_df["Battery SOC (MWh)"].max()
        avg_soc_pct = (avg_soc / _solution.capacity.battery_mwh) * 100
    else:
        avg_soc = 0
        min_soc = 0
        max_soc = 0
        avg_soc_pct = 0
    
    # Curtailment statistics
    curtailment_hours = (dispatch_df["Curtailment (MW)"] > 0.01).sum()
    total_curtailment = dispatch_df["Curtailment (MW)"].sum()
    max_curtailment = dispatch_df["Curtailment (MW)"].max()
    
    return {
        "gas_utilization_hours": gas_utilization_hours,
        "battery_cycles": battery_cycles,
        "solar_capacity_factor": solar_capacity_factor,
        "peak_grid_draw": peak_grid_draw,
        "avg_grid_draw": avg_grid_draw,
        "gas_capacity_factor": gas_capacity_factor,
        "avg_soc": avg_soc,
        "min_soc": min_soc,
        "max_soc": max_soc,
        "avg_soc_pct": avg_soc_pct,
        "curtailment_hours": curtailment_hours,
        "total_curtailment": total_curtailment,
        "max_curtailment": max_curtailment
    }


def render_time_range_selector() -> Optional[tuple]:
    """Render time range selector controls.
    
    Returns:
        Tuple of (start_hour, end_hour) or None for full year
    """
    st.subheader("Time Range Selection")
    
    # Quick selection buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("Full Year", use_container_width=True):
            st.session_state.time_range_mode = "full"
    
    with col2:
        if st.button("First Week", use_container_width=True):
            st.session_state.time_range_mode = "week1"
    
    with col3:
        if st.button("Summer Week", use_container_width=True):
            st.session_state.time_range_mode = "summer"
    
    with col4:
        if st.button("Winter Week", use_container_width=True):
            st.session_state.time_range_mode = "winter"
    
    with col5:
        if st.button("Custom Range", use_container_width=True):
            st.session_state.time_range_mode = "custom"
    
    # Initialize time range mode if not set
    if 'time_range_mode' not in st.session_state:
        st.session_state.time_range_mode = "full"
    
    # Return appropriate time range based on mode
    mode = st.session_state.time_range_mode
    
    if mode == "full":
        st.info("Showing full year (8760 hours)")
        return None
    
    elif mode == "week1":
        st.info("Showing first week of year (Hours 1-168)")
        return (1, 168)
    
    elif mode == "summer":
        # Mid-July (approximately hour 4920)
        st.info("Showing summer week (Mid-July, Hours 4920-5088)")
        return (4920, 5088)
    
    elif mode == "winter":
        # Mid-January (approximately hour 360)
        st.info("Showing winter week (Mid-January, Hours 360-528)")
        return (360, 528)
    
    elif mode == "custom":
        st.markdown("#### Custom Time Range")
        col1, col2 = st.columns(2)
        
        with col1:
            start_hour = st.number_input(
                "Start Hour",
                min_value=1,
                max_value=8760,
                value=1,
                step=1,
                help="Starting hour (1-8760)"
            )
        
        with col2:
            end_hour = st.number_input(
                "End Hour",
                min_value=1,
                max_value=8760,
                value=168,
                step=1,
                help="Ending hour (1-8760)"
            )
        
        if start_hour >= end_hour:
            st.error("End hour must be greater than start hour")
            return None
        
        st.info(f"Showing custom range (Hours {start_hour}-{end_hour})")
        return (start_hour, end_hour)
    
    return None


@st.cache_data(ttl=1800)
def create_dispatch_visualization(_solution: Any, viz_type: str, time_range: Optional[tuple], _market_data: Optional[dict]):
    """
    Create dispatch visualization with caching.
    Cache expires after 30 minutes.
    
    Args:
        _solution: OptimizationSolution object (underscore prevents hashing)
        viz_type: Visualization type ("Heatmap" or "Stacked Area Chart")
        time_range: Optional tuple of (start_hour, end_hour)
        _market_data: Optional market data dictionary (underscore prevents hashing)
        
    Returns:
        Plotly figure or None if error
    """
    try:
        if viz_type == "Heatmap":
            fig = plot_dispatch_heatmap(
                solution=_solution,
                market_data=_market_data,
                time_range=time_range,
                height=600
            )
        else:  # Stacked Area Chart
            fig = plot_dispatch_stacked_area(
                solution=_solution,
                time_range=time_range,
                height=500
            )
        return fig
    except Exception as e:
        return None


def render_dispatch_visualization(solution: Any, time_range: Optional[tuple]):
    """Render dispatch heatmap and stacked area visualizations.
    
    Args:
        solution: OptimizationSolution object
        time_range: Optional tuple of (start_hour, end_hour)
    """
    st.subheader("Dispatch Visualization")
    
    # Visualization type selector
    viz_type = st.radio(
        "Visualization Type",
        options=["Heatmap", "Stacked Area Chart"],
        horizontal=True,
        help="Choose visualization format"
    )
    
    # Get market data from session state if available
    market_data = None
    if st.session_state.get('lmp_data') is not None:
        try:
            market_data = {
                'lmp': st.session_state.lmp_data,
                'gas_price': st.session_state.gas_prices,
                'solar_cf': st.session_state.solar_cf
            }
        except:
            pass
    
    # Use cached visualization
    fig = create_dispatch_visualization(solution, viz_type, time_range, market_data)
    
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        
        if viz_type == "Heatmap":
            st.info("""
            **Heatmap Guide:**
            - Color intensity shows power contribution (MW) from each source
            - Hover over cells to see detailed information including market prices
            - Use the range slider below the chart to zoom into specific periods
            - Battery charge (positive) and discharge (negative) are shown separately
            """)
        else:
            st.info("""
            **Stacked Area Guide:**
            - Each colored area represents power contribution from a source
            - Total height shows total generation meeting the load
            - Hover over the chart to see power breakdown at each hour
            - This view is useful for seeing overall generation patterns
            """)
    else:
        st.error("Error creating dispatch visualization")
        st.info("Please ensure optimization has been run successfully.")


def render_statistics_panel(solution: Any, stats: dict):
    """Render operational statistics panel.
    
    Args:
        solution: OptimizationSolution object
        stats: Dictionary with operational statistics
    """
    st.subheader("Operational Statistics")
    st.markdown("Key operational metrics from hourly dispatch decisions:")
    
    # Create tabs for different statistic categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "Grid & Gas",
        "Battery",
        "Solar",
        "Reliability"
    ])
    
    with tab1:
        st.markdown("#### Grid and Gas Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Peak Grid Draw",
                f"{stats['peak_grid_draw']:.1f} MW",
                help="Maximum power drawn from grid in any hour"
            )
            st.metric(
                "Average Grid Draw",
                f"{stats['avg_grid_draw']:.1f} MW",
                help="Average hourly power from grid"
            )
            st.metric(
                "Grid Capacity",
                f"{solution.capacity.grid_mw:.1f} MW",
                help="Installed grid interconnection capacity"
            )
        
        with col2:
            st.metric(
                "Gas Utilization Hours",
                f"{stats['gas_utilization_hours']} hours",
                help="Number of hours with gas generation"
            )
            st.metric(
                "Gas Capacity Factor",
                f"{stats['gas_capacity_factor']:.1f}%",
                help="Percentage of maximum possible gas generation"
            )
            st.metric(
                "Gas Capacity",
                f"{solution.capacity.gas_mw:.1f} MW",
                help="Installed gas peaker capacity"
            )
    
    with tab2:
        st.markdown("#### Battery Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Battery Cycles/Year",
                f"{stats['battery_cycles']:.1f}",
                help="Number of full charge-discharge cycles per year"
            )
            st.metric(
                "Average State of Charge",
                f"{stats['avg_soc']:.1f} MWh ({stats['avg_soc_pct']:.1f}%)",
                help="Average battery energy level"
            )
        
        with col2:
            st.metric(
                "Battery Capacity",
                f"{solution.capacity.battery_mwh:.1f} MWh",
                help="Installed battery storage capacity"
            )
            st.metric(
                "SOC Range",
                f"{stats['min_soc']:.1f} - {stats['max_soc']:.1f} MWh",
                help="Minimum and maximum state of charge"
            )
        
        if solution.capacity.battery_mwh > 0:
            st.info("""
            **Battery Health:**
            - SOC constrained between 10% and 90% of capacity
            - Typical battery lifetime: 10-15 years at this cycle rate
            - Degradation costs included in optimization
            """)
        else:
            st.warning("No battery storage in optimal portfolio")
    
    with tab3:
        st.markdown("#### Solar Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Solar Capacity Factor",
                f"{stats['solar_capacity_factor']:.1f}%",
                help="Percentage of maximum possible solar generation"
            )
        
        with col2:
            st.metric(
                "Solar Capacity",
                f"{solution.capacity.solar_mw:.1f} MW",
                help="Installed solar PV capacity"
            )
        
        if solution.capacity.solar_mw > 0:
            # Calculate annual solar generation
            dispatch_df = solution.dispatch.to_dataframe()
            annual_solar_mwh = dispatch_df["Solar (MW)"].sum()
            
            st.metric(
                "Annual Solar Generation",
                f"{annual_solar_mwh:,.0f} MWh",
                help="Total solar energy generated per year"
            )
            
            st.info("""
            **Solar Performance:**
            - Capacity factor varies by location and weather
            - West Texas has excellent solar resource
            - No fuel costs, low O&M costs
            """)
        else:
            st.warning("No solar PV in optimal portfolio")
    
    with tab4:
        st.markdown("#### Reliability Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Curtailment Hours",
                f"{stats['curtailment_hours']} hours",
                help="Number of hours with load curtailment"
            )
            st.metric(
                "Total Curtailment",
                f"{stats['total_curtailment']:.2f} MWh",
                help="Total annual energy curtailed"
            )
        
        with col2:
            st.metric(
                "Max Curtailment",
                f"{stats['max_curtailment']:.2f} MW",
                help="Maximum curtailment in any single hour"
            )
            st.metric(
                "Reliability",
                f"{solution.metrics.reliability_pct:.4f}%",
                help="Percentage of load served without curtailment"
            )
        
        if stats['curtailment_hours'] > 0:
            st.warning(f"""
            **Reliability Events:**
            - {stats['curtailment_hours']} hours with curtailment
            - Equivalent to {stats['total_curtailment'] / solution.scenario_params.get('facility_size_mw', 300):.2f} hours of full outage
            - Review dispatch heatmap to identify when curtailment occurs
            """)
        else:
            st.success("Perfect reliability - no curtailment events!")


def render_dispatch_table(solution: Any, time_range: Optional[tuple]):
    """Render detailed dispatch data table.
    
    Args:
        solution: OptimizationSolution object
        time_range: Optional tuple of (start_hour, end_hour)
    """
    with st.expander("View Detailed Dispatch Data"):
        st.markdown("#### Hourly Dispatch Data Table")
        
        dispatch_df = solution.dispatch.to_dataframe()
        
        # Apply time range filter if specified
        if time_range is not None:
            start_hour, end_hour = time_range
            dispatch_df = dispatch_df[
                (dispatch_df["Hour"] >= start_hour) & 
                (dispatch_df["Hour"] <= end_hour)
            ].copy()
        
        # Format numeric columns
        numeric_cols = ["Grid (MW)", "Gas (MW)", "Solar (MW)", "Battery (MW)", 
                       "Curtailment (MW)", "Battery SOC (MWh)"]
        for col in numeric_cols:
            if col in dispatch_df.columns:
                dispatch_df[col] = dispatch_df[col].round(2)
        
        # Display table with pagination
        st.dataframe(
            dispatch_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = dispatch_df.to_csv(index=False)
        st.download_button(
            label="Download Dispatch Data (CSV)",
            data=csv,
            file_name="hourly_dispatch.csv",
            mime="text/csv",
            help="Download complete hourly dispatch data"
        )


def render():
    """Render the hourly dispatch page."""
    st.title("Hourly Dispatch")
    st.markdown("Explore hour-by-hour operational decisions and system performance.")
    
    # Check if optimization has been run
    if st.session_state.optimization_result is None:
        st.warning("No optimization results available.")
        st.info("""
        Please run an optimization first:
        1. Navigate to the **Optimization Setup** page
        2. Configure facility parameters
        3. Click **Run Optimization**
        """)
        return
    
    # Get solution from session state
    solution = st.session_state.optimization_result
    
    st.markdown("---")
    
    # Time range selector
    time_range = render_time_range_selector()
    
    st.markdown("---")
    
    # Dispatch visualization
    render_dispatch_visualization(solution, time_range)
    
    st.markdown("---")
    
    # Calculate and display statistics
    stats = calculate_operational_statistics(solution)
    render_statistics_panel(solution, stats)
    
    st.markdown("---")
    
    # Detailed dispatch table
    render_dispatch_table(solution, time_range)
    
    st.markdown("---")
    
    # Navigation hints
    st.info("""
    **Analysis Tips:**
    - Use time range selection to zoom into specific periods
    - Look for patterns in dispatch during high/low price hours
    - Check battery charging during low-price hours and discharging during high-price hours
    - Identify when gas peakers are dispatched (typically during price spikes)
    - Review curtailment events to understand reliability constraints
    
    **Next Steps:**
    - View **Scenario Comparison** to analyze trade-offs and sensitivity
    - Review **Case Study** for detailed analysis of 300MW West Texas facility
    """)
