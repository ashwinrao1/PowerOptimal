"""
Optimal Portfolio Page

This page displays the optimization results including capacity mix visualization,
cost breakdown, and key performance metrics. Users can view detailed results
and export data to CSV or JSON formats.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.models.solution import OptimizationSolution

# Add src directory to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import after path is set
try:
    from visualization.capacity_viz import plot_capacity_mix
    from visualization.cost_viz import plot_cost_breakdown
    from models.solution import OptimizationSolution
    from models.technology import TechnologyCosts
except ImportError as e:
    import warnings
    warnings.warn(f"Import error: {e}. Some functionality may not be available.")
    # Define placeholder types for when imports fail
    OptimizationSolution = None
    TechnologyCosts = None


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a number as currency with appropriate units.
    
    Args:
        value: Numeric value to format
        decimals: Number of decimal places
    
    Returns:
        Formatted currency string
    """
    if value >= 1e9:
        return f"${value/1e9:.{decimals}f}B"
    elif value >= 1e6:
        return f"${value/1e6:.{decimals}f}M"
    elif value >= 1e3:
        return f"${value/1e3:.{decimals}f}K"
    else:
        return f"${value:.{decimals}f}"


def export_to_csv(solution: Any) -> str:
    """Export solution to CSV format.
    
    Args:
        solution: OptimizationSolution object
        
    Returns:
        CSV string for download
    """
    # Create summary DataFrame
    summary_data = {
        "Metric": [],
        "Value": [],
        "Unit": []
    }
    
    # Add capacity data
    summary_data["Metric"].extend([
        "Grid Connection Capacity",
        "Gas Peaker Capacity",
        "Battery Storage Capacity",
        "Solar PV Capacity"
    ])
    summary_data["Value"].extend([
        solution.capacity.grid_mw,
        solution.capacity.gas_mw,
        solution.capacity.battery_mwh,
        solution.capacity.solar_mw
    ])
    summary_data["Unit"].extend(["MW", "MW", "MWh", "MW"])
    
    # Add key metrics
    summary_data["Metric"].extend([
        "Total NPV (20-year)",
        "Capital Expenditure",
        "Annual Operating Expenditure",
        "Levelized Cost of Energy",
        "Reliability",
        "Total Annual Curtailment",
        "Number of Curtailment Hours",
        "Annual Carbon Emissions",
        "Carbon Intensity",
        "Carbon Reduction vs Baseline",
        "Grid Dependence",
        "Gas Capacity Factor",
        "Battery Cycles per Year",
        "Solar Capacity Factor",
        "Solve Time",
        "Optimality Gap"
    ])
    summary_data["Value"].extend([
        solution.metrics.total_npv,
        solution.metrics.capex,
        solution.metrics.opex_annual,
        solution.metrics.lcoe,
        solution.metrics.reliability_pct,
        solution.metrics.total_curtailment_mwh,
        solution.metrics.num_curtailment_hours,
        solution.metrics.carbon_tons_annual,
        solution.metrics.carbon_intensity_g_per_kwh,
        solution.metrics.carbon_reduction_pct,
        solution.metrics.grid_dependence_pct,
        solution.metrics.gas_capacity_factor,
        solution.metrics.battery_cycles_per_year,
        solution.metrics.solar_capacity_factor,
        solution.metrics.solve_time_seconds,
        solution.metrics.optimality_gap_pct
    ])
    summary_data["Unit"].extend([
        "$", "$", "$/year", "$/MWh", "%", "MWh", "hours",
        "tons CO2", "g CO2/kWh", "%", "%", "%", "cycles/year",
        "%", "seconds", "%"
    ])
    
    df = pd.DataFrame(summary_data)
    return df.to_csv(index=False)


def export_to_json(solution: Any) -> str:
    """Export solution to JSON format.
    
    Args:
        solution: OptimizationSolution object
        
    Returns:
        JSON string for download
    """
    # Create summary dictionary
    summary = solution.to_summary_dict()
    return json.dumps(summary, indent=2)


@st.cache_data(ttl=1800)
def create_capacity_visualization(_solution: Any, viz_format: str):
    """
    Create capacity visualization with caching.
    Cache expires after 30 minutes.
    
    Args:
        _solution: OptimizationSolution object (underscore prevents hashing)
        viz_format: Visualization format ("bar", "pie", "waterfall")
        
    Returns:
        Plotly figure
    """
    try:
        fig = plot_capacity_mix(
            solution=_solution,
            format=viz_format,
            show_values=True
        )
        return fig
    except Exception as e:
        return None


def render_capacity_section(solution: Any):
    """Render the capacity mix section.
    
    Args:
        solution: OptimizationSolution object
    """
    st.subheader("Optimal Capacity Mix")
    st.markdown("The optimization determined the following capacity investments:")
    
    # Display capacity metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Grid Connection",
            f"{solution.capacity.grid_mw:.1f} MW",
            help="Grid interconnection capacity"
        )
    
    with col2:
        st.metric(
            "Gas Peakers",
            f"{solution.capacity.gas_mw:.1f} MW",
            help="On-site natural gas generation capacity"
        )
    
    with col3:
        st.metric(
            "Battery Storage",
            f"{solution.capacity.battery_mwh:.1f} MWh",
            help="Battery energy storage capacity"
        )
    
    with col4:
        st.metric(
            "Solar PV",
            f"{solution.capacity.solar_mw:.1f} MW",
            help="On-site solar photovoltaic capacity"
        )
    
    st.markdown("---")
    
    # Capacity visualization
    st.markdown("#### Capacity Mix Visualization")
    
    # Format selector
    viz_format = st.radio(
        "Visualization Format",
        options=["Bar Chart", "Pie Chart", "Waterfall Chart"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Map format to function parameter
    format_map = {
        "Bar Chart": "bar",
        "Pie Chart": "pie",
        "Waterfall Chart": "waterfall"
    }
    
    # Use cached visualization
    fig = create_capacity_visualization(solution, format_map[viz_format])
    
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Error creating capacity visualization")


@st.cache_data(ttl=1800)
def create_cost_visualization(_solution: Any, _tech_costs: Any, cost_format: str):
    """
    Create cost visualization with caching.
    Cache expires after 30 minutes.
    
    Args:
        _solution: OptimizationSolution object (underscore prevents hashing)
        _tech_costs: TechnologyCosts object (underscore prevents hashing)
        cost_format: Visualization format ("waterfall", "stacked_bar")
        
    Returns:
        Plotly figure
    """
    try:
        fig = plot_cost_breakdown(
            solution=_solution,
            tech_costs=_tech_costs,
            format=cost_format,
            show_values=True
        )
        return fig
    except Exception as e:
        return None


def render_cost_section(solution: Any):
    """Render the cost breakdown section.
    
    Args:
        solution: OptimizationSolution object
    """
    st.subheader("Cost Breakdown")
    st.markdown("Detailed breakdown of capital and operating expenses:")
    
    # Display cost metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total NPV (20-year)",
            format_currency(solution.metrics.total_npv, 2),
            help="Total net present value over 20-year planning horizon"
        )
    
    with col2:
        st.metric(
            "Capital Expenditure",
            format_currency(solution.metrics.capex, 2),
            help="Upfront investment in capacity"
        )
    
    with col3:
        st.metric(
            "Annual OPEX",
            format_currency(solution.metrics.opex_annual, 2),
            help="Annual operating expenditure"
        )
    
    st.markdown("---")
    
    # Cost visualization
    st.markdown("#### Cost Breakdown Visualization")
    
    # Format selector
    cost_format = st.radio(
        "Cost Visualization Format",
        options=["Waterfall Chart", "Stacked Bar Chart"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Map format to function parameter
    cost_format_map = {
        "Waterfall Chart": "waterfall",
        "Stacked Bar Chart": "stacked_bar"
    }
    
    # Use cached visualization
    tech_costs = TechnologyCosts() if TechnologyCosts is not None else None
    
    if tech_costs is not None:
        fig = create_cost_visualization(solution, tech_costs, cost_format_map[cost_format])
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Error creating cost visualization")
    else:
        st.error("Technology costs not available")


def render_metrics_section(solution: Any):
    """Render the key metrics section.
    
    Args:
        solution: OptimizationSolution object
    """
    st.subheader("Key Performance Metrics")
    st.markdown("Summary of optimization results and system performance:")
    
    # Create tabs for different metric categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "Economic Metrics",
        "Reliability Metrics",
        "Carbon Metrics",
        "Operational Metrics"
    ])
    
    with tab1:
        st.markdown("#### Economic Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Total NPV",
                format_currency(solution.metrics.total_npv, 2),
                help="20-year net present value"
            )
            st.metric(
                "CAPEX",
                format_currency(solution.metrics.capex, 2),
                help="Capital expenditure"
            )
            st.metric(
                "Annual OPEX",
                format_currency(solution.metrics.opex_annual, 2),
                help="Annual operating expenditure"
            )
        
        with col2:
            st.metric(
                "LCOE",
                f"${solution.metrics.lcoe:.2f}/MWh",
                help="Levelized cost of energy"
            )
            
            # Calculate payback period estimate
            if solution.metrics.capex > 0 and solution.metrics.opex_annual > 0:
                # Simple payback calculation (not accounting for time value)
                # This is a rough estimate
                st.info("""
                **Cost Structure:**
                - Upfront investment required for capacity
                - Ongoing operational costs for energy and maintenance
                - Total cost optimized over 20-year horizon
                """)
    
    with tab2:
        st.markdown("#### Reliability Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Reliability",
                f"{solution.metrics.reliability_pct:.4f}%",
                help="Percentage of load served without curtailment"
            )
            st.metric(
                "Total Curtailment",
                f"{solution.metrics.total_curtailment_mwh:.2f} MWh",
                help="Total annual energy curtailed"
            )
        
        with col2:
            st.metric(
                "Curtailment Hours",
                f"{solution.metrics.num_curtailment_hours}",
                help="Number of hours with non-zero curtailment"
            )
            
            # Calculate equivalent downtime
            facility_size = solution.scenario_params.get("facility_size_mw", 300)
            if facility_size > 0:
                downtime_hours = solution.metrics.total_curtailment_mwh / facility_size
                st.metric(
                    "Equivalent Downtime",
                    f"{downtime_hours:.2f} hours/year",
                    help="Curtailment expressed as equivalent full outage hours"
                )
    
    with tab3:
        st.markdown("#### Carbon Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Annual Emissions",
                f"{solution.metrics.carbon_tons_annual:,.0f} tons CO2",
                help="Total annual carbon emissions"
            )
            st.metric(
                "Carbon Intensity",
                f"{solution.metrics.carbon_intensity_g_per_kwh:.1f} g CO2/kWh",
                help="Carbon emissions per kWh of energy consumed"
            )
        
        with col2:
            st.metric(
                "Carbon Reduction",
                f"{solution.metrics.carbon_reduction_pct:.1f}%",
                delta=f"{solution.metrics.carbon_reduction_pct:.1f}%",
                delta_color="normal",
                help="Reduction vs grid-only baseline"
            )
            
            # Show carbon context
            st.info("""
            **Carbon Context:**
            - Baseline assumes 100% grid power
            - Reduction achieved through behind-the-meter generation
            - Solar and battery enable cleaner energy mix
            """)
    
    with tab4:
        st.markdown("#### Operational Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Grid Dependence",
                f"{solution.metrics.grid_dependence_pct:.1f}%",
                help="Percentage of energy from grid"
            )
            st.metric(
                "Gas Capacity Factor",
                f"{solution.metrics.gas_capacity_factor:.1f}%",
                help="Gas peaker utilization"
            )
        
        with col2:
            st.metric(
                "Battery Cycles/Year",
                f"{solution.metrics.battery_cycles_per_year:.1f}",
                help="Number of full charge-discharge cycles per year"
            )
            st.metric(
                "Solar Capacity Factor",
                f"{solution.metrics.solar_capacity_factor:.1f}%",
                help="Solar PV utilization"
            )


def render_export_section(solution: Any):
    """Render the export section with download buttons.
    
    Args:
        solution: OptimizationSolution object
    """
    st.subheader("Export Results")
    st.markdown("Download optimization results in CSV or JSON format:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        csv_data = export_to_csv(solution)
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name="optimization_results.csv",
            mime="text/csv",
            help="Download summary results as CSV file"
        )
    
    with col2:
        # JSON export
        json_data = export_to_json(solution)
        st.download_button(
            label="Download as JSON",
            data=json_data,
            file_name="optimization_results.json",
            mime="application/json",
            help="Download complete results as JSON file"
        )


def render():
    """Render the optimal portfolio page."""
    st.title("Optimal Portfolio")
    st.markdown("View the optimal energy portfolio and detailed optimization results.")
    
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
    
    # Display optimization status
    if st.session_state.solve_time:
        st.success(
            f"Optimization completed successfully in "
            f"{st.session_state.solve_time:.1f} seconds"
        )
    
    st.markdown("---")
    
    # Render capacity section
    render_capacity_section(solution)
    
    st.markdown("---")
    
    # Render cost section
    render_cost_section(solution)
    
    st.markdown("---")
    
    # Render metrics section
    render_metrics_section(solution)
    
    st.markdown("---")
    
    # Render export section
    render_export_section(solution)
    
    st.markdown("---")
    
    # Navigation hints
    st.info("""
    **Next Steps:**
    - View **Hourly Dispatch** to see hour-by-hour operational decisions
    - Explore **Scenario Comparison** to analyze trade-offs and sensitivity
    - Review **Case Study** for detailed analysis of 300MW West Texas facility
    """)
