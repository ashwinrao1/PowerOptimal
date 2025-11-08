"""
Dashboard Utilities

Shared utility functions for the Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


@st.cache_data(ttl=3600)
def load_cached_data():
    """
    Load market data with caching for performance.
    Cache expires after 1 hour.
    
    Returns:
        tuple: (lmp_data, solar_cf, gas_prices, grid_carbon) DataFrames
    """
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    
    try:
        lmp_data = pd.read_csv(data_dir / "ercot_lmp_hourly_2022_2024.csv")
        solar_cf = pd.read_csv(data_dir / "solar_cf_west_texas.csv")
        gas_prices = pd.read_csv(data_dir / "gas_prices_hourly.csv")
        grid_carbon = pd.read_csv(data_dir / "grid_carbon_intensity.csv")
        
        return lmp_data, solar_cf, gas_prices, grid_carbon
    
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.info("Please run data collection scripts first: `python scripts/download_all_data.py`")
        return None, None, None, None


@st.cache_data(ttl=3600)
def load_market_data_cached():
    """
    Cached version of market data loading.
    Cache expires after 1 hour.
    
    Returns:
        tuple: (lmp_data, solar_cf, gas_prices, grid_carbon) DataFrames
    """
    return load_cached_data()


def format_currency(value: float, decimals: int = 0) -> str:
    """
    Format a number as currency.
    
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


def format_energy(value: float, decimals: int = 1) -> str:
    """
    Format energy value with appropriate units.
    
    Args:
        value: Energy value in MWh
        decimals: Number of decimal places
    
    Returns:
        Formatted energy string
    """
    if value >= 1e6:
        return f"{value/1e6:.{decimals}f} TWh"
    elif value >= 1e3:
        return f"{value/1e3:.{decimals}f} GWh"
    else:
        return f"{value:.{decimals}f} MWh"


def format_power(value: float, decimals: int = 1) -> str:
    """
    Format power value with appropriate units.
    
    Args:
        value: Power value in MW
        decimals: Number of decimal places
    
    Returns:
        Formatted power string
    """
    if value >= 1e3:
        return f"{value/1e3:.{decimals}f} GW"
    else:
        return f"{value:.{decimals}f} MW"


def display_metric_card(label: str, value: Any, delta: Optional[Any] = None, 
                       help_text: Optional[str] = None):
    """
    Display a metric card with consistent styling.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta/change value
        help_text: Optional help text tooltip
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        help=help_text
    )


def show_optimization_progress(status_text: str, progress: float):
    """
    Display optimization progress indicator.
    
    Args:
        status_text: Status message to display
        progress: Progress value between 0 and 1
    """
    progress_bar = st.progress(progress)
    status_placeholder = st.empty()
    status_placeholder.text(status_text)
    
    return progress_bar, status_placeholder


def export_results_to_csv(results: Dict[str, Any], filename: str = "optimization_results.csv"):
    """
    Export optimization results to CSV format for download.
    
    Args:
        results: Dictionary containing optimization results
        filename: Output filename
    
    Returns:
        CSV string for download
    """
    # Convert results to DataFrame format
    if 'capacity' in results:
        capacity_df = pd.DataFrame([results['capacity']])
        return capacity_df.to_csv(index=False)
    
    return ""


def export_results_to_json(results: Dict[str, Any]) -> str:
    """
    Export optimization results to JSON format for download.
    
    Args:
        results: Dictionary containing optimization results
    
    Returns:
        JSON string for download
    """
    import json
    return json.dumps(results, indent=2)


def validate_input_parameters(facility_size: float, reliability_target: float, 
                              carbon_reduction: float) -> tuple[bool, str]:
    """
    Validate user input parameters.
    
    Args:
        facility_size: Facility size in MW
        reliability_target: Reliability target in %
        carbon_reduction: Carbon reduction target in %
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if facility_size < 100 or facility_size > 500:
        return False, "Facility size must be between 100 and 500 MW"
    
    if reliability_target < 99.0 or reliability_target > 99.999:
        return False, "Reliability target must be between 99.0% and 99.999%"
    
    if carbon_reduction < 0 or carbon_reduction > 100:
        return False, "Carbon reduction must be between 0% and 100%"
    
    return True, ""


def get_color_scheme() -> Dict[str, str]:
    """
    Get consistent color scheme for visualizations.
    
    Returns:
        Dictionary mapping technology names to colors
    """
    return {
        'grid': '#1f77b4',      # Blue
        'gas': '#ff7f0e',       # Orange
        'battery': '#2ca02c',   # Green
        'solar': '#ffd700',     # Gold
        'curtailment': '#d62728' # Red
    }


def show_data_quality_warning():
    """Display warning if data quality issues are detected."""
    st.warning("""
    ⚠️ Data Quality Notice
    
    Some data files may be missing or incomplete. Please ensure you have run
    the data collection scripts before using the optimization features.
    
    Run: `python scripts/download_all_data.py`
    """)


def create_download_button(data: str, filename: str, label: str, 
                          file_type: str = "text/csv"):
    """
    Create a download button for results.
    
    Args:
        data: Data to download
        filename: Suggested filename
        label: Button label
        file_type: MIME type
    """
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=file_type
    )


@st.cache_data(ttl=7200)
def load_technology_costs():
    """
    Load technology cost data with caching.
    Cache expires after 2 hours.
    
    Returns:
        Dictionary with technology cost parameters
    """
    import json
    data_dir = Path(__file__).parent.parent / "data"
    
    try:
        with open(data_dir / "tech_costs.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default values if file not found
        return {
            "grid_interconnection": {"capex_per_kw": 3000, "fixed_om_per_kw_year": 0},
            "gas_peaker": {"capex_per_kw": 1000, "variable_om_per_mwh": 15, "heat_rate_mmbtu_per_mwh": 10},
            "battery": {"capex_per_kwh": 350, "degradation_per_mwh": 5, "efficiency": 0.85},
            "solar": {"capex_per_kw": 1200, "fixed_om_per_kw_year": 20}
        }


@st.cache_data(ttl=86400)
def load_precomputed_scenarios():
    """
    Load pre-computed scenario results with caching.
    Cache expires after 24 hours.
    
    Returns:
        List of pre-computed scenario results or None if not available
    """
    results_dir = Path(__file__).parent.parent / "results" / "scenarios"
    
    try:
        import json
        scenario_files = list(results_dir.glob("*.json"))
        
        if not scenario_files:
            return None
        
        scenarios = []
        for file in scenario_files:
            with open(file, 'r') as f:
                scenarios.append(json.load(f))
        
        return scenarios
    except Exception:
        return None


@st.cache_resource
def get_solver_instance():
    """
    Get a cached solver instance.
    This uses cache_resource to persist the solver across reruns.
    
    Returns:
        Solver instance or None if not available
    """
    try:
        import pyomo.environ as pyo
        solver = pyo.SolverFactory('gurobi')
        
        # Check if solver is available
        if solver.available():
            return solver
        else:
            return None
    except Exception:
        return None
