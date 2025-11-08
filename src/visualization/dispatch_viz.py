"""Dispatch heatmap visualization functions."""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np

try:
    from ..models.solution import DispatchSolution, OptimizationSolution
    from ..models.market_data import MarketData
except ImportError:
    from models.solution import DispatchSolution, OptimizationSolution
    from models.market_data import MarketData


def plot_dispatch_heatmap(
    solution: Union[DispatchSolution, OptimizationSolution, Dict],
    market_data: Optional[Union[MarketData, pd.DataFrame, Dict]] = None,
    time_range: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    height: int = 600
) -> go.Figure:
    """Create interactive 2D heatmap of hourly dispatch decisions.
    
    Visualizes power contribution from each source across all hours of the year.
    The heatmap shows hours on the x-axis and power sources on the y-axis, with
    color intensity representing MW contribution.
    
    Args:
        solution: DispatchSolution object, OptimizationSolution object, or dict with dispatch data
        market_data: Optional MarketData object, DataFrame, or dict with LMP, gas prices, and solar CF
                    for enhanced hover tooltips
        time_range: Optional tuple (start_hour, end_hour) for zooming into specific period.
                   Hours are 1-indexed (1-8760). If None, shows full year.
        title: Custom title for the plot (optional)
        height: Figure height in pixels (default: 600)
        
    Returns:
        Plotly Figure object with interactive heatmap
        
    Raises:
        ValueError: If solution format is invalid or time_range is out of bounds
    """
    # Extract dispatch data from solution
    if isinstance(solution, OptimizationSolution):
        dispatch = solution.dispatch
    elif isinstance(solution, DispatchSolution):
        dispatch = solution
    elif isinstance(solution, dict):
        # Handle dict format from JSON
        if "dispatch" in solution:
            dispatch_dict = solution["dispatch"]
        else:
            dispatch_dict = solution
        
        # Create DispatchSolution from dict
        dispatch = DispatchSolution(
            hour=np.array(dispatch_dict["hour"]),
            grid_power=np.array(dispatch_dict["grid_power"]),
            gas_power=np.array(dispatch_dict["gas_power"]),
            solar_power=np.array(dispatch_dict["solar_power"]),
            battery_power=np.array(dispatch_dict["battery_power"]),
            curtailment=np.array(dispatch_dict["curtailment"]),
            battery_soc=np.array(dispatch_dict["battery_soc"])
        )
    else:
        raise ValueError(
            "solution must be DispatchSolution, OptimizationSolution, or dict"
        )
    
    # Convert to DataFrame for easier manipulation
    dispatch_df = dispatch.to_dataframe()
    
    # Apply time range filter if specified
    if time_range is not None:
        start_hour, end_hour = time_range
        if start_hour < 1 or end_hour > 8760 or start_hour >= end_hour:
            raise ValueError(
                f"Invalid time_range ({start_hour}, {end_hour}). "
                "Must be 1 <= start_hour < end_hour <= 8760"
            )
        # Filter to specified range (hours are 1-indexed)
        dispatch_df = dispatch_df[
            (dispatch_df["Hour"] >= start_hour) & 
            (dispatch_df["Hour"] <= end_hour)
        ].copy()
    
    # Extract market data if provided
    lmp_data = None
    gas_price_data = None
    solar_cf_data = None
    
    if market_data is not None:
        if isinstance(market_data, MarketData):
            lmp_data = market_data.lmp
            gas_price_data = market_data.gas_price
            solar_cf_data = market_data.solar_cf
        elif isinstance(market_data, pd.DataFrame):
            if "lmp" in market_data.columns:
                lmp_data = market_data["lmp"].values
            if "gas_price" in market_data.columns:
                gas_price_data = market_data["gas_price"].values
            if "solar_cf" in market_data.columns:
                solar_cf_data = market_data["solar_cf"].values
        elif isinstance(market_data, dict):
            lmp_data = market_data.get("lmp")
            gas_price_data = market_data.get("gas_price")
            solar_cf_data = market_data.get("solar_cf")
        
        # Apply time range filter to market data if specified
        if time_range is not None and lmp_data is not None:
            start_idx = start_hour - 1  # Convert to 0-indexed
            end_idx = end_hour
            lmp_data = lmp_data[start_idx:end_idx]
            if gas_price_data is not None:
                gas_price_data = gas_price_data[start_idx:end_idx]
            if solar_cf_data is not None:
                solar_cf_data = solar_cf_data[start_idx:end_idx]
    
    # Prepare data for heatmap
    # We'll create a matrix where rows are power sources and columns are hours
    power_sources = ["Grid", "Gas", "Solar", "Battery Discharge", "Battery Charge"]
    hours = dispatch_df["Hour"].values
    
    # Separate battery into charge and discharge for clearer visualization
    battery_discharge = np.where(dispatch_df["Battery (MW)"] < 0, 
                                  -dispatch_df["Battery (MW)"], 0)
    battery_charge = np.where(dispatch_df["Battery (MW)"] > 0, 
                              dispatch_df["Battery (MW)"], 0)
    
    # Create data matrix (rows = power sources, columns = hours)
    data_matrix = np.array([
        dispatch_df["Grid (MW)"].values,
        dispatch_df["Gas (MW)"].values,
        dispatch_df["Solar (MW)"].values,
        battery_discharge,
        battery_charge
    ])
    
    # Create custom hover text with market data if available
    hover_text = _create_hover_text(
        dispatch_df, hours, power_sources,
        lmp_data, gas_price_data, solar_cf_data
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data_matrix,
        x=hours,
        y=power_sources,
        colorscale="Viridis",
        hovertext=hover_text,
        hoverinfo="text",
        colorbar=dict(
            title="Power (MW)",
            titleside="right"
        )
    ))
    
    # Update layout
    default_title = "Hourly Dispatch Heatmap"
    if time_range is not None:
        default_title += f" (Hours {time_range[0]}-{time_range[1]})"
    
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Hour of Year",
        yaxis_title="Power Source",
        template="plotly_white",
        height=height,
        font=dict(size=12),
        hovermode="closest",
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            type="linear"
        )
    )
    
    return fig


def _create_hover_text(
    dispatch_df: pd.DataFrame,
    hours: np.ndarray,
    power_sources: list,
    lmp_data: Optional[np.ndarray],
    gas_price_data: Optional[np.ndarray],
    solar_cf_data: Optional[np.ndarray]
) -> list:
    """Create custom hover text for heatmap with market data.
    
    Args:
        dispatch_df: DataFrame with dispatch data
        hours: Array of hour indices
        power_sources: List of power source names
        lmp_data: Optional LMP data array
        gas_price_data: Optional gas price data array
        solar_cf_data: Optional solar capacity factor data array
        
    Returns:
        List of lists with hover text for each cell
    """
    hover_text = []
    
    for i, source in enumerate(power_sources):
        source_hover = []
        for j, hour in enumerate(hours):
            # Base hover text with power source and hour
            text_parts = [f"<b>{source}</b>", f"Hour: {hour}"]
            
            # Add power value
            if source == "Grid":
                power = dispatch_df.iloc[j]["Grid (MW)"]
            elif source == "Gas":
                power = dispatch_df.iloc[j]["Gas (MW)"]
            elif source == "Solar":
                power = dispatch_df.iloc[j]["Solar (MW)"]
            elif source == "Battery Discharge":
                power = max(0, -dispatch_df.iloc[j]["Battery (MW)"])
            elif source == "Battery Charge":
                power = max(0, dispatch_df.iloc[j]["Battery (MW)"])
            else:
                power = 0
            
            text_parts.append(f"Power: {power:.2f} MW")
            
            # Add market data if available
            if lmp_data is not None and j < len(lmp_data):
                text_parts.append(f"LMP: ${lmp_data[j]:.2f}/MWh")
            
            if gas_price_data is not None and j < len(gas_price_data):
                text_parts.append(f"Gas Price: ${gas_price_data[j]:.2f}/MMBtu")
            
            if solar_cf_data is not None and j < len(solar_cf_data):
                text_parts.append(f"Solar CF: {solar_cf_data[j]:.2%}")
            
            # Add battery SOC if this is a battery row
            if "Battery" in source:
                soc = dispatch_df.iloc[j]["Battery SOC (MWh)"]
                text_parts.append(f"Battery SOC: {soc:.2f} MWh")
            
            # Add curtailment if non-zero
            curtailment = dispatch_df.iloc[j]["Curtailment (MW)"]
            if curtailment > 0.01:
                text_parts.append(f"<b>Curtailment: {curtailment:.2f} MW</b>")
            
            source_hover.append("<br>".join(text_parts))
        
        hover_text.append(source_hover)
    
    return hover_text


def plot_dispatch_stacked_area(
    solution: Union[DispatchSolution, OptimizationSolution, Dict],
    time_range: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    height: int = 500
) -> go.Figure:
    """Create stacked area chart of hourly dispatch decisions.
    
    Alternative visualization showing power contribution as stacked areas over time.
    Useful for seeing the overall generation mix and how it changes throughout the year.
    
    Args:
        solution: DispatchSolution object, OptimizationSolution object, or dict with dispatch data
        time_range: Optional tuple (start_hour, end_hour) for zooming into specific period
        title: Custom title for the plot (optional)
        height: Figure height in pixels (default: 500)
        
    Returns:
        Plotly Figure object with stacked area chart
    """
    # Extract dispatch data
    if isinstance(solution, OptimizationSolution):
        dispatch = solution.dispatch
    elif isinstance(solution, DispatchSolution):
        dispatch = solution
    elif isinstance(solution, dict):
        if "dispatch" in solution:
            dispatch_dict = solution["dispatch"]
        else:
            dispatch_dict = solution
        dispatch = DispatchSolution(
            hour=np.array(dispatch_dict["hour"]),
            grid_power=np.array(dispatch_dict["grid_power"]),
            gas_power=np.array(dispatch_dict["gas_power"]),
            solar_power=np.array(dispatch_dict["solar_power"]),
            battery_power=np.array(dispatch_dict["battery_power"]),
            curtailment=np.array(dispatch_dict["curtailment"]),
            battery_soc=np.array(dispatch_dict["battery_soc"])
        )
    else:
        raise ValueError(
            "solution must be DispatchSolution, OptimizationSolution, or dict"
        )
    
    # Convert to DataFrame
    dispatch_df = dispatch.to_dataframe()
    
    # Apply time range filter if specified
    if time_range is not None:
        start_hour, end_hour = time_range
        if start_hour < 1 or end_hour > 8760 or start_hour >= end_hour:
            raise ValueError(
                f"Invalid time_range ({start_hour}, {end_hour}). "
                "Must be 1 <= start_hour < end_hour <= 8760"
            )
        dispatch_df = dispatch_df[
            (dispatch_df["Hour"] >= start_hour) & 
            (dispatch_df["Hour"] <= end_hour)
        ].copy()
    
    # Separate battery into discharge only (negative values become positive)
    dispatch_df["Battery Discharge (MW)"] = np.where(
        dispatch_df["Battery (MW)"] < 0,
        -dispatch_df["Battery (MW)"],
        0
    )
    
    # Create stacked area chart
    fig = go.Figure()
    
    # Add traces in order (bottom to top of stack)
    colors = {
        "Grid (MW)": "#1f77b4",
        "Gas (MW)": "#ff7f0e",
        "Solar (MW)": "#2ca02c",
        "Battery Discharge (MW)": "#d62728"
    }
    
    for column in ["Grid (MW)", "Gas (MW)", "Solar (MW)", "Battery Discharge (MW)"]:
        if column in dispatch_df.columns or column == "Battery Discharge (MW)":
            if column == "Battery Discharge (MW)":
                values = dispatch_df["Battery Discharge (MW)"]
                name = "Battery Discharge"
            else:
                values = dispatch_df[column]
                name = column.replace(" (MW)", "")
            
            fig.add_trace(go.Scatter(
                x=dispatch_df["Hour"],
                y=values,
                name=name,
                mode="lines",
                stackgroup="one",
                fillcolor=colors.get(column, "#999999"),
                line=dict(width=0.5, color=colors.get(column, "#999999")),
                hovertemplate=f"{name}<br>Hour: %{{x}}<br>Power: %{{y:.2f}} MW<extra></extra>"
            ))
    
    # Update layout
    default_title = "Hourly Dispatch (Stacked Area)"
    if time_range is not None:
        default_title += f" (Hours {time_range[0]}-{time_range[1]})"
    
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Hour of Year",
        yaxis_title="Power (MW)",
        template="plotly_white",
        height=height,
        font=dict(size=12),
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig
