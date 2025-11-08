"""
Data Center Energy Optimization Dashboard

Main entry point for the Streamlit dashboard application.
Provides navigation between different analysis pages and manages session state.
"""

import streamlit as st
from pathlib import Path
import sys

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import caching utilities
try:
    from cache_config import (
        load_precomputed_baseline,
        load_precomputed_optimal,
        load_precomputed_pareto_frontiers,
        clear_all_caches
    )
except ImportError:
    # Fallback if cache_config not available
    load_precomputed_baseline = lambda: None
    load_precomputed_optimal = lambda: None
    load_precomputed_pareto_frontiers = lambda: None
    clear_all_caches = lambda: None


def load_precomputed_results():
    """
    Load pre-computed results if available.
    This improves initial dashboard experience by providing example results.
    """
    # Try to load pre-computed baseline
    if st.session_state.get('precomputed_baseline') is None:
        baseline = load_precomputed_baseline()
        if baseline is not None:
            st.session_state.precomputed_baseline = baseline
    
    # Try to load pre-computed optimal solution
    if st.session_state.get('precomputed_optimal') is None:
        optimal = load_precomputed_optimal()
        if optimal is not None:
            st.session_state.precomputed_optimal = optimal
    
    # Try to load pre-computed Pareto frontiers
    if st.session_state.get('precomputed_pareto') is None:
        pareto = load_precomputed_pareto_frontiers()
        if pareto is not None:
            st.session_state.precomputed_pareto = pareto


def initialize_session_state():
    """Initialize session state variables for persisting data across page changes."""
    
    # Load pre-computed results on first run
    load_precomputed_results()
    
    # Optimization inputs
    if 'facility_size_mw' not in st.session_state:
        st.session_state.facility_size_mw = 300
    
    if 'reliability_target' not in st.session_state:
        st.session_state.reliability_target = 99.99
    
    if 'carbon_reduction_pct' not in st.session_state:
        st.session_state.carbon_reduction_pct = 0
    
    if 'location' not in st.session_state:
        st.session_state.location = "West Texas"
    
    if 'year_scenario' not in st.session_state:
        st.session_state.year_scenario = "2024"
    
    if 'available_technologies' not in st.session_state:
        st.session_state.available_technologies = {
            'grid': True,
            'gas': True,
            'battery': True,
            'solar': True
        }
    
    # Optimization results
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    
    if 'optimization_status' not in st.session_state:
        st.session_state.optimization_status = "Not started"
    
    if 'solve_time' not in st.session_state:
        st.session_state.solve_time = None
    
    # Scenario analysis results
    if 'scenario_results' not in st.session_state:
        st.session_state.scenario_results = None
    
    if 'pareto_frontiers' not in st.session_state:
        st.session_state.pareto_frontiers = None
    
    # Market data cache
    if 'market_data_loaded' not in st.session_state:
        st.session_state.market_data_loaded = False
    
    if 'lmp_data' not in st.session_state:
        st.session_state.lmp_data = None
    
    if 'solar_cf' not in st.session_state:
        st.session_state.solar_cf = None
    
    if 'gas_prices' not in st.session_state:
        st.session_state.gas_prices = None
    
    if 'grid_carbon' not in st.session_state:
        st.session_state.grid_carbon = None


def configure_page():
    """Configure page layout and theme settings."""
    st.set_page_config(
        page_title="Data Center Energy Optimizer",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/username/datacenter-energy-optimization',
            'Report a bug': 'https://github.com/username/datacenter-energy-optimization/issues',
            'About': """
            # Data Center Energy Optimization
            
            This tool optimizes energy portfolios for AI training data centers,
            balancing cost, reliability, and carbon emissions.
            
            Built with Pyomo, Gurobi, and Streamlit.
            """
        }
    )


def render_sidebar():
    """Render sidebar with page navigation and key information."""
    with st.sidebar:
        st.title("⚡ Energy Optimizer")
        st.markdown("---")
        
        # Page navigation
        st.subheader("Navigation")
        page = st.radio(
            "Select Page",
            [
                "Optimization Setup",
                "Optimal Portfolio",
                "Hourly Dispatch",
                "Scenario Comparison",
                "Case Study"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Current configuration summary
        st.subheader("Current Configuration")
        st.metric("Facility Size", f"{st.session_state.facility_size_mw} MW")
        st.metric("Reliability Target", f"{st.session_state.reliability_target}%")
        st.metric("Carbon Reduction", f"{st.session_state.carbon_reduction_pct}%")
        
        st.markdown("---")
        
        # Optimization status
        st.subheader("Optimization Status")
        status = st.session_state.optimization_status
        
        if status == "Not started":
            st.info("No optimization run yet")
        elif status == "Running":
            st.warning("Optimization in progress...")
        elif status == "Complete":
            st.success("Optimization complete")
            if st.session_state.solve_time:
                st.caption(f"Solve time: {st.session_state.solve_time:.1f}s")
        elif status == "Failed":
            st.error("Optimization failed")
        
        st.markdown("---")
        
        # About section
        with st.expander("About This Tool"):
            st.markdown("""
            This dashboard helps optimize energy portfolios for large-scale
            AI training data centers. It considers:
            
            - **Grid connection**: Utility electricity
            - **Gas peakers**: On-site natural gas generation
            - **Battery storage**: Energy storage for arbitrage
            - **Solar PV**: On-site renewable generation
            
            The optimization minimizes 20-year total cost while meeting
            reliability and carbon constraints.
            """)
        
        # Cache management (for debugging/development)
        with st.expander("Advanced Options"):
            if st.button("Clear All Caches"):
                clear_all_caches()
                st.success("All caches cleared!")
                st.info("Caches will be rebuilt on next data access.")
        
        return page


def render_home_page():
    """Render the home/welcome page."""
    st.title("Data Center Energy Optimization")
    st.markdown("### Optimize energy portfolios for AI training facilities")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Minimize Cost")
        st.markdown("""
        Optimize capital and operating expenses over a 20-year planning horizon.
        Balance upfront investments with long-term savings.
        """)
    
    with col2:
        st.markdown("#### 🔌 Ensure Reliability")
        st.markdown("""
        Meet 99.99% uptime requirements for mission-critical AI training workloads.
        Minimize expensive compute interruptions.
        """)
    
    with col3:
        st.markdown("#### 🌱 Reduce Carbon")
        st.markdown("""
        Track and optionally constrain carbon emissions.
        Explore pathways to 24/7 carbon-free energy.
        """)
    
    st.markdown("---")
    
    st.markdown("### Getting Started")
    st.markdown("""
    1. **Optimization Setup**: Configure facility parameters and run optimization
    2. **Optimal Portfolio**: View recommended capacity mix and key metrics
    3. **Hourly Dispatch**: Explore hour-by-hour operational decisions
    4. **Scenario Comparison**: Analyze sensitivity and trade-offs
    5. **Case Study**: Review detailed analysis for 300MW West Texas facility
    """)
    
    st.markdown("---")
    
    st.info("👈 Use the sidebar to navigate between pages")


def main():
    """Main application entry point."""
    # Configure page settings
    configure_page()
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Route to appropriate page
    if page == "Optimization Setup":
        # Import here to avoid circular imports
        try:
            from pages import setup
            setup.render()
        except ImportError:
            st.error("Setup page not yet implemented")
            st.info("This page will allow you to configure optimization parameters and run the model.")
    
    elif page == "Optimal Portfolio":
        try:
            from pages import portfolio
            portfolio.render()
        except ImportError:
            st.error("Portfolio page not yet implemented")
            st.info("This page will display the optimal capacity mix and cost breakdown.")
    
    elif page == "Hourly Dispatch":
        try:
            from pages import dispatch
            dispatch.render()
        except ImportError:
            st.error("Dispatch page not yet implemented")
            st.info("This page will show hourly operational decisions with interactive heatmaps.")
    
    elif page == "Scenario Comparison":
        try:
            from pages import scenarios
            scenarios.render()
        except ImportError:
            st.error("Scenarios page not yet implemented")
            st.info("This page will display Pareto frontiers and sensitivity analysis.")
    
    elif page == "Case Study":
        try:
            from pages import case_study
            case_study.render()
        except ImportError:
            st.error("Case Study page not yet implemented")
            st.info("This page will present the detailed 300MW West Texas analysis.")


if __name__ == "__main__":
    main()
