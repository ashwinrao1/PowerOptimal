"""Reliability analysis visualization functions."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np

try:
    from ..models.solution import DispatchSolution, OptimizationSolution, SolutionMetrics
except ImportError:
    from models.solution import DispatchSolution, OptimizationSolution, SolutionMetrics


def plot_reliability_analysis(
    solution: Union[OptimizationSolution, Dict],
    title: Optional[str] = None,
    height: int = 800
) -> go.Figure:
    """Create comprehensive reliability analysis visualization.
    
    Multi-panel visualization showing:
    1. Histogram of hourly curtailment events
    2. Time series of reserve margin over the year
    3. Top 10 worst-case reliability events
    4. Summary statistics panel
    
    Args:
        solution: OptimizationSolution object or dict with solution data
        title: Custom title for the plot (optional)
        height: Figure height in pixels (default: 800)
        
    Returns:
        Plotly Figure object with multi-panel reliability analysis
        
    Raises:
        ValueError: If solution format is invalid
    """
    # Extract solution data
    if isinstance(solution, OptimizationSolution):
        dispatch = solution.dispatch
        metrics = solution.metrics
        capacity = solution.capacity
    elif isinstance(solution, dict):
        # Handle dict format from JSON
        if "dispatch" in solution and "metrics" in solution:
            dispatch_dict = solution["dispatch"]
            metrics_dict = solution["metrics"]
            capacity_dict = solution.get("capacity", {})
        else:
            raise ValueError(
                "Dict must contain 'dispatch' and 'metrics' keys"
            )
        
        # Create objects from dict
        dispatch = DispatchSolution(
            hour=np.array(dispatch_dict["hour"]),
            grid_power=np.array(dispatch_dict["grid_power"]),
            gas_power=np.array(dispatch_dict["gas_power"]),
            solar_power=np.array(dispatch_dict["solar_power"]),
            battery_power=np.array(dispatch_dict["battery_power"]),
            curtailment=np.array(dispatch_dict["curtailment"]),
            battery_soc=np.array(dispatch_dict["battery_soc"])
        )
        
        metrics = SolutionMetrics(**metrics_dict)
    else:
        raise ValueError(
            "solution must be OptimizationSolution or dict"
        )
    
    # Convert dispatch to DataFrame
    dispatch_df = dispatch.to_dataframe()
    
    # Calculate reserve margin for each hour
    # Reserve margin = (Total Available Capacity - Load) / Load
    # For simplicity, we'll calculate it as (Total Generation - Curtailment) / Load
    # A negative reserve margin indicates insufficient capacity
    load_mw = 285  # Default data center load
    total_generation = (dispatch_df["Grid (MW)"] + 
                       dispatch_df["Gas (MW)"] + 
                       dispatch_df["Solar (MW)"] - 
                       dispatch_df["Battery (MW)"])  # Battery negative = discharge
    reserve_margin_pct = ((total_generation - dispatch_df["Curtailment (MW)"]) / load_mw - 1) * 100
    
    # Create subplots: 2 rows, 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Curtailment Event Histogram",
            "Reserve Margin Over Time",
            "Top 10 Worst Reliability Events",
            "Reliability Statistics"
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "table"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # Panel 1: Histogram of curtailment events
    curtailment_events = dispatch_df[dispatch_df["Curtailment (MW)"] > 0.01]
    
    if len(curtailment_events) > 0:
        # Create histogram bins
        hist_data = curtailment_events["Curtailment (MW)"]
        
        fig.add_trace(
            go.Histogram(
                x=hist_data,
                nbinsx=20,
                marker_color="#d62728",
                name="Curtailment Events",
                hovertemplate="Curtailment: %{x:.2f} MW<br>Count: %{y}<extra></extra>"
            ),
            row=1, col=1
        )
    else:
        # No curtailment events - add annotation
        fig.add_annotation(
            text="No curtailment events",
            xref="x1", yref="y1",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#2ca02c"),
            row=1, col=1
        )
    
    # Panel 2: Reserve margin time series
    # Downsample for better visualization (show weekly averages)
    hours_per_week = 168
    num_weeks = len(dispatch_df) // hours_per_week
    
    if num_weeks > 0:
        weekly_reserve = []
        week_numbers = []
        
        for week in range(num_weeks):
            start_idx = week * hours_per_week
            end_idx = start_idx + hours_per_week
            weekly_avg = reserve_margin_pct.iloc[start_idx:end_idx].mean()
            weekly_reserve.append(weekly_avg)
            week_numbers.append(week + 1)
        
        # Add color based on reserve margin (red if negative, green if positive)
        colors = ["#d62728" if rm < 0 else "#2ca02c" for rm in weekly_reserve]
        
        fig.add_trace(
            go.Scatter(
                x=week_numbers,
                y=weekly_reserve,
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=6, color=colors),
                name="Weekly Avg Reserve Margin",
                hovertemplate="Week %{x}<br>Reserve Margin: %{y:.2f}%<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Add zero line
        fig.add_hline(
            y=0, line_dash="dash", line_color="gray",
            row=1, col=2
        )
    
    # Panel 3: Top 10 worst reliability events
    # Find hours with highest curtailment
    top_events = dispatch_df.nlargest(10, "Curtailment (MW)")
    
    if len(top_events) > 0 and top_events["Curtailment (MW)"].max() > 0.01:
        event_labels = [f"Hour {int(h)}" for h in top_events["Hour"]]
        event_curtailment = top_events["Curtailment (MW)"].values
        
        fig.add_trace(
            go.Bar(
                x=event_labels,
                y=event_curtailment,
                marker_color="#d62728",
                name="Worst Events",
                text=[f"{c:.2f} MW" for c in event_curtailment],
                textposition="outside",
                hovertemplate="Hour: %{x}<br>Curtailment: %{y:.2f} MW<extra></extra>"
            ),
            row=2, col=1
        )
    else:
        # No significant curtailment events
        fig.add_annotation(
            text="No significant curtailment events",
            xref="x3", yref="y3",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#2ca02c"),
            row=2, col=1
        )
    
    # Panel 4: Statistics table
    stats_data = _create_statistics_table(dispatch_df, metrics)
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Value</b>"],
                fill_color="#1f77b4",
                font=dict(color="white", size=12),
                align="left"
            ),
            cells=dict(
                values=[stats_data["metrics"], stats_data["values"]],
                fill_color=[["#f0f0f0", "white"] * len(stats_data["metrics"])],
                font=dict(size=11),
                align="left",
                height=25
            )
        ),
        row=2, col=2
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Curtailment (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Number of Events", row=1, col=1)
    
    fig.update_xaxes(title_text="Week of Year", row=1, col=2)
    fig.update_yaxes(title_text="Reserve Margin (%)", row=1, col=2)
    
    fig.update_xaxes(title_text="Event", row=2, col=1, tickangle=-45)
    fig.update_yaxes(title_text="Curtailment (MW)", row=2, col=1)
    
    # Update layout
    default_title = "Reliability Analysis"
    fig.update_layout(
        title=dict(
            text=title or default_title,
            x=0.5,
            xanchor="center",
            font=dict(size=16)
        ),
        template="plotly_white",
        height=height,
        showlegend=False,
        font=dict(size=11)
    )
    
    return fig


def _create_statistics_table(
    dispatch_df: pd.DataFrame,
    metrics: SolutionMetrics
) -> Dict[str, list]:
    """Create statistics data for table display.
    
    Args:
        dispatch_df: DataFrame with dispatch data
        metrics: Solution metrics
        
    Returns:
        Dictionary with metric names and values
    """
    # Calculate statistics
    curtailment_events = dispatch_df[dispatch_df["Curtailment (MW)"] > 0.01]
    num_curtailment_hours = len(curtailment_events)
    total_curtailment_mwh = dispatch_df["Curtailment (MW)"].sum()
    max_curtailment_mw = dispatch_df["Curtailment (MW)"].max()
    avg_curtailment_when_occurs = (
        curtailment_events["Curtailment (MW)"].mean() 
        if len(curtailment_events) > 0 else 0
    )
    
    # Calculate reliability percentage
    reliability_pct = metrics.reliability_pct
    
    # Calculate uptime
    total_hours = len(dispatch_df)
    uptime_hours = total_hours - num_curtailment_hours
    
    metric_names = [
        "Reliability Target",
        "Actual Reliability",
        "Total Uptime Hours",
        "Total Curtailment Hours",
        "Total Curtailment Energy",
        "Max Single-Hour Curtailment",
        "Avg Curtailment (when occurs)",
        "Curtailment Events"
    ]
    
    metric_values = [
        "99.99%",
        f"{reliability_pct:.4f}%",
        f"{uptime_hours:,} hours",
        f"{num_curtailment_hours} hours",
        f"{total_curtailment_mwh:.2f} MWh",
        f"{max_curtailment_mw:.2f} MW",
        f"{avg_curtailment_when_occurs:.2f} MW",
        f"{num_curtailment_hours} events"
    ]
    
    return {
        "metrics": metric_names,
        "values": metric_values
    }


def plot_curtailment_histogram(
    solution: Union[OptimizationSolution, Dict],
    title: Optional[str] = None,
    bins: int = 30,
    height: int = 500
) -> go.Figure:
    """Create histogram of hourly curtailment events.
    
    Shows the distribution of curtailment magnitudes across all hours
    where curtailment occurred.
    
    Args:
        solution: OptimizationSolution object or dict with solution data
        title: Custom title for the plot (optional)
        bins: Number of histogram bins (default: 30)
        height: Figure height in pixels (default: 500)
        
    Returns:
        Plotly Figure object with curtailment histogram
    """
    # Extract dispatch data
    if isinstance(solution, OptimizationSolution):
        dispatch = solution.dispatch
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
            "solution must be OptimizationSolution or dict"
        )
    
    # Get curtailment data
    dispatch_df = dispatch.to_dataframe()
    curtailment_events = dispatch_df[dispatch_df["Curtailment (MW)"] > 0.01]
    
    # Create histogram
    fig = go.Figure()
    
    if len(curtailment_events) > 0:
        fig.add_trace(go.Histogram(
            x=curtailment_events["Curtailment (MW)"],
            nbinsx=bins,
            marker_color="#d62728",
            name="Curtailment Events",
            hovertemplate="Curtailment: %{x:.2f} MW<br>Count: %{y}<extra></extra>"
        ))
        
        # Add statistics annotation
        mean_curtailment = curtailment_events["Curtailment (MW)"].mean()
        max_curtailment = curtailment_events["Curtailment (MW)"].max()
        num_events = len(curtailment_events)
        
        annotation_text = (
            f"Total Events: {num_events}<br>"
            f"Mean: {mean_curtailment:.2f} MW<br>"
            f"Max: {max_curtailment:.2f} MW"
        )
        
        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#d62728",
            borderwidth=2,
            borderpad=4
        )
    else:
        # No curtailment events
        fig.add_annotation(
            text="No curtailment events detected<br>100% reliability achieved",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#2ca02c")
        )
    
    # Update layout
    default_title = "Distribution of Curtailment Events"
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Curtailment (MW)",
        yaxis_title="Number of Events",
        template="plotly_white",
        height=height,
        font=dict(size=12),
        showlegend=False
    )
    
    return fig


def plot_reserve_margin_timeseries(
    solution: Union[OptimizationSolution, Dict],
    load_mw: float = 285,
    time_range: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    height: int = 500
) -> go.Figure:
    """Create time series plot showing reserve margin over the year.
    
    Reserve margin indicates how much excess capacity is available beyond
    the required load. Negative values indicate insufficient capacity.
    
    Args:
        solution: OptimizationSolution object or dict with solution data
        load_mw: Data center load in MW (default: 285)
        time_range: Optional tuple (start_hour, end_hour) for zooming
        title: Custom title for the plot (optional)
        height: Figure height in pixels (default: 500)
        
    Returns:
        Plotly Figure object with reserve margin time series
    """
    # Extract dispatch data
    if isinstance(solution, OptimizationSolution):
        dispatch = solution.dispatch
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
            "solution must be OptimizationSolution or dict"
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
    
    # Calculate reserve margin
    # Reserve margin = (Available Generation - Load) / Load * 100
    total_generation = (dispatch_df["Grid (MW)"] + 
                       dispatch_df["Gas (MW)"] + 
                       dispatch_df["Solar (MW)"] - 
                       dispatch_df["Battery (MW)"])  # Battery negative = discharge
    
    actual_supply = total_generation - dispatch_df["Curtailment (MW)"]
    reserve_margin_pct = (actual_supply / load_mw - 1) * 100
    
    # Create figure
    fig = go.Figure()
    
    # Add reserve margin line
    # Color based on whether reserve is positive or negative
    colors = np.where(reserve_margin_pct >= 0, "#2ca02c", "#d62728")
    
    fig.add_trace(go.Scatter(
        x=dispatch_df["Hour"],
        y=reserve_margin_pct,
        mode="lines",
        line=dict(color="#1f77b4", width=1.5),
        name="Reserve Margin",
        hovertemplate="Hour: %{x}<br>Reserve Margin: %{y:.2f}%<extra></extra>"
    ))
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Zero Reserve",
        annotation_position="right"
    )
    
    # Highlight curtailment events
    curtailment_hours = dispatch_df[dispatch_df["Curtailment (MW)"] > 0.01]
    if len(curtailment_hours) > 0:
        fig.add_trace(go.Scatter(
            x=curtailment_hours["Hour"],
            y=reserve_margin_pct[dispatch_df["Curtailment (MW)"] > 0.01],
            mode="markers",
            marker=dict(size=8, color="#d62728", symbol="x"),
            name="Curtailment Events",
            hovertemplate="Hour: %{x}<br>Curtailment Event<extra></extra>"
        ))
    
    # Update layout
    default_title = "Reserve Margin Over Time"
    if time_range is not None:
        default_title += f" (Hours {time_range[0]}-{time_range[1]})"
    
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Hour of Year",
        yaxis_title="Reserve Margin (%)",
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


def plot_worst_reliability_events(
    solution: Union[OptimizationSolution, Dict],
    n_events: int = 10,
    title: Optional[str] = None,
    height: int = 500
) -> go.Figure:
    """Identify and visualize top N worst-case reliability events.
    
    Shows the hours with the highest curtailment, helping identify
    when the system is most stressed.
    
    Args:
        solution: OptimizationSolution object or dict with solution data
        n_events: Number of worst events to show (default: 10)
        title: Custom title for the plot (optional)
        height: Figure height in pixels (default: 500)
        
    Returns:
        Plotly Figure object with worst events bar chart
    """
    # Extract dispatch data
    if isinstance(solution, OptimizationSolution):
        dispatch = solution.dispatch
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
            "solution must be OptimizationSolution or dict"
        )
    
    # Get dispatch DataFrame
    dispatch_df = dispatch.to_dataframe()
    
    # Find top N worst events
    top_events = dispatch_df.nlargest(n_events, "Curtailment (MW)")
    
    # Create figure
    fig = go.Figure()
    
    if len(top_events) > 0 and top_events["Curtailment (MW)"].max() > 0.01:
        # Create labels with hour and day of year
        event_labels = []
        for hour in top_events["Hour"]:
            day = int((hour - 1) // 24) + 1
            hour_of_day = int((hour - 1) % 24)
            event_labels.append(f"Hour {int(hour)}<br>(Day {day}, {hour_of_day}:00)")
        
        event_curtailment = top_events["Curtailment (MW)"].values
        
        fig.add_trace(go.Bar(
            x=event_labels,
            y=event_curtailment,
            marker_color="#d62728",
            text=[f"{c:.2f} MW" for c in event_curtailment],
            textposition="outside",
            hovertemplate="%{x}<br>Curtailment: %{y:.2f} MW<extra></extra>"
        ))
        
        # Add mean line
        mean_curtailment = event_curtailment.mean()
        fig.add_hline(
            y=mean_curtailment,
            line_dash="dash",
            line_color="#ff7f0e",
            annotation_text=f"Mean: {mean_curtailment:.2f} MW",
            annotation_position="right"
        )
    else:
        # No significant curtailment events
        fig.add_annotation(
            text="No significant curtailment events detected<br>System maintains high reliability",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#2ca02c")
        )
    
    # Update layout
    default_title = f"Top {n_events} Worst Reliability Events"
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Event (Hour of Year)",
        yaxis_title="Curtailment (MW)",
        template="plotly_white",
        height=height,
        font=dict(size=12),
        showlegend=False,
        xaxis=dict(tickangle=-45)
    )
    
    return fig
