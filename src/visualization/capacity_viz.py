"""Capacity mix visualization functions."""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional, Union
import pandas as pd

from ..models.solution import CapacitySolution, OptimizationSolution


def plot_capacity_mix(
    solution: Union[CapacitySolution, OptimizationSolution, Dict],
    format: str = "bar",
    title: Optional[str] = None,
    show_values: bool = True
) -> go.Figure:
    """Create interactive capacity mix visualization.
    
    Args:
        solution: CapacitySolution object, OptimizationSolution object, or dict with capacity data
        format: Visualization format - "bar", "pie", or "waterfall"
        title: Custom title for the plot (optional)
        show_values: Whether to show values on the plot
        
    Returns:
        Plotly Figure object
        
    Raises:
        ValueError: If format is not supported or solution format is invalid
    """
    # Extract capacity data from solution
    if isinstance(solution, OptimizationSolution):
        capacity = solution.capacity
    elif isinstance(solution, CapacitySolution):
        capacity = solution
    elif isinstance(solution, dict):
        # Handle dict format from JSON
        if "capacity" in solution:
            capacity_dict = solution["capacity"]
        else:
            capacity_dict = solution
        
        # Create CapacitySolution from dict
        capacity = CapacitySolution(
            grid_mw=capacity_dict.get("Grid Connection (MW)", 0),
            gas_mw=capacity_dict.get("Gas Peakers (MW)", 0),
            battery_mwh=capacity_dict.get("Battery Storage (MWh)", 0),
            solar_mw=capacity_dict.get("Solar PV (MW)", 0)
        )
    else:
        raise ValueError(
            "solution must be CapacitySolution, OptimizationSolution, or dict"
        )
    
    # Dispatch to appropriate visualization function
    if format == "bar":
        return _plot_bar_chart(capacity, title, show_values)
    elif format == "pie":
        return _plot_pie_chart(capacity, title, show_values)
    elif format == "waterfall":
        return _plot_waterfall_chart(capacity, title, show_values)
    else:
        raise ValueError(
            f"Unsupported format '{format}'. Choose from: 'bar', 'pie', 'waterfall'"
        )


def _plot_bar_chart(
    capacity: CapacitySolution,
    title: Optional[str],
    show_values: bool
) -> go.Figure:
    """Create stacked bar chart showing capacity for each technology.
    
    Args:
        capacity: CapacitySolution object
        title: Custom title for the plot
        show_values: Whether to show values on bars
        
    Returns:
        Plotly Figure object
    """
    # Prepare data
    technologies = ["Grid Connection", "Gas Peakers", "Solar PV", "Battery Storage"]
    values = [capacity.grid_mw, capacity.gas_mw, capacity.solar_mw, capacity.battery_mwh]
    units = ["MW", "MW", "MW", "MWh"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    # Create bar chart
    fig = go.Figure()
    
    for tech, value, unit, color in zip(technologies, values, units, colors):
        if value > 0:  # Only show technologies with non-zero capacity
            hover_text = f"{tech}<br>{value:.1f} {unit}"
            text_display = f"{value:.1f} {unit}" if show_values else None
            
            fig.add_trace(go.Bar(
                x=[tech],
                y=[value],
                name=tech,
                marker_color=color,
                text=text_display,
                textposition="auto",
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=True
            ))
    
    # Update layout
    default_title = "Optimal Capacity Mix"
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Technology",
        yaxis_title="Capacity",
        barmode="group",
        hovermode="closest",
        template="plotly_white",
        height=500,
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def _plot_pie_chart(
    capacity: CapacitySolution,
    title: Optional[str],
    show_values: bool
) -> go.Figure:
    """Create pie chart showing percentage breakdown of capacity.
    
    Note: Battery storage (MWh) is converted to equivalent MW assuming 4-hour duration
    for comparison with power capacity.
    
    Args:
        capacity: CapacitySolution object
        title: Custom title for the plot
        show_values: Whether to show values on pie slices
        
    Returns:
        Plotly Figure object
    """
    # Convert battery MWh to MW equivalent (assuming 4-hour battery)
    battery_mw_equiv = capacity.battery_mwh / 4.0
    
    # Prepare data
    technologies = []
    values = []
    colors = []
    
    tech_data = [
        ("Grid Connection", capacity.grid_mw, "#1f77b4"),
        ("Gas Peakers", capacity.gas_mw, "#ff7f0e"),
        ("Solar PV", capacity.solar_mw, "#2ca02c"),
        ("Battery Storage", battery_mw_equiv, "#d62728")
    ]
    
    # Only include technologies with non-zero capacity
    for tech, value, color in tech_data:
        if value > 0:
            technologies.append(tech)
            values.append(value)
            colors.append(color)
    
    # Create custom hover text
    hover_text = []
    for tech, value in zip(technologies, values):
        if tech == "Battery Storage":
            hover_text.append(
                f"{tech}<br>{value:.1f} MW (equiv.)<br>{value*4:.1f} MWh"
            )
        else:
            hover_text.append(f"{tech}<br>{value:.1f} MW")
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=technologies,
        values=values,
        marker=dict(colors=colors),
        hovertext=hover_text,
        hoverinfo="label+percent+text",
        textinfo="label+percent" if show_values else "label",
        textposition="auto"
    )])
    
    # Update layout
    default_title = "Capacity Mix Distribution (MW Equivalent)"
    fig.update_layout(
        title=title or default_title,
        template="plotly_white",
        height=500,
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def _plot_waterfall_chart(
    capacity: CapacitySolution,
    title: Optional[str],
    show_values: bool
) -> go.Figure:
    """Create waterfall chart showing cumulative capacity buildup.
    
    Args:
        capacity: CapacitySolution object
        title: Custom title for the plot
        show_values: Whether to show values on bars
        
    Returns:
        Plotly Figure object
    """
    # Prepare data - show how capacity builds up
    technologies = ["Grid Connection", "Gas Peakers", "Solar PV", "Battery Storage"]
    values = [capacity.grid_mw, capacity.gas_mw, capacity.solar_mw, capacity.battery_mwh]
    units = ["MW", "MW", "MW", "MWh"]
    
    # Filter out zero values
    filtered_data = [
        (tech, val, unit) 
        for tech, val, unit in zip(technologies, values, units) 
        if val > 0
    ]
    
    if not filtered_data:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No capacity data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=title or "Capacity Buildup",
            template="plotly_white",
            height=500
        )
        return fig
    
    # Prepare waterfall data
    x_labels = [tech for tech, _, _ in filtered_data] + ["Total"]
    y_values = [val for _, val, _ in filtered_data]
    text_labels = [
        f"{val:.1f} {unit}" if show_values else ""
        for _, val, unit in filtered_data
    ]
    
    # Calculate total (note: battery is in MWh, others in MW)
    total_mw = sum(val for tech, val, unit in filtered_data if unit == "MW")
    total_mwh = sum(val for tech, val, unit in filtered_data if unit == "MWh")
    
    # Create measure types for waterfall
    measures = ["relative"] * len(filtered_data) + ["total"]
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        x=x_labels,
        y=y_values + [0],  # Add 0 for total (it will be calculated)
        measure=measures,
        text=text_labels + [f"{total_mw:.1f} MW<br>{total_mwh:.1f} MWh" if show_values else ""],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2ca02c"}},
        decreasing={"marker": {"color": "#d62728"}},
        totals={"marker": {"color": "#1f77b4"}},
        hovertemplate="%{x}<br>%{y:.1f}<extra></extra>"
    ))
    
    # Update layout
    default_title = "Capacity Buildup"
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Technology",
        yaxis_title="Capacity",
        template="plotly_white",
        height=500,
        font=dict(size=12),
        showlegend=False
    )
    
    return fig


def plot_capacity_comparison(
    solutions: Dict[str, Union[CapacitySolution, OptimizationSolution, Dict]],
    title: Optional[str] = None,
    show_values: bool = True
) -> go.Figure:
    """Create grouped bar chart comparing capacity across multiple scenarios.
    
    Args:
        solutions: Dictionary mapping scenario names to solution objects
        title: Custom title for the plot
        show_values: Whether to show values on bars
        
    Returns:
        Plotly Figure object
    """
    # Extract capacity data from all solutions
    scenarios = []
    grid_capacities = []
    gas_capacities = []
    solar_capacities = []
    battery_capacities = []
    
    for scenario_name, solution in solutions.items():
        # Extract capacity
        if isinstance(solution, OptimizationSolution):
            capacity = solution.capacity
        elif isinstance(solution, CapacitySolution):
            capacity = solution
        elif isinstance(solution, dict):
            if "capacity" in solution:
                capacity_dict = solution["capacity"]
            else:
                capacity_dict = solution
            capacity = CapacitySolution(
                grid_mw=capacity_dict.get("Grid Connection (MW)", 0),
                gas_mw=capacity_dict.get("Gas Peakers (MW)", 0),
                battery_mwh=capacity_dict.get("Battery Storage (MWh)", 0),
                solar_mw=capacity_dict.get("Solar PV (MW)", 0)
            )
        else:
            continue
        
        scenarios.append(scenario_name)
        grid_capacities.append(capacity.grid_mw)
        gas_capacities.append(capacity.gas_mw)
        solar_capacities.append(capacity.solar_mw)
        battery_capacities.append(capacity.battery_mwh)
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Grid Connection",
        x=scenarios,
        y=grid_capacities,
        marker_color="#1f77b4",
        text=[f"{val:.1f} MW" if show_values and val > 0 else "" for val in grid_capacities],
        textposition="auto",
        hovertemplate="Grid Connection<br>%{y:.1f} MW<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        name="Gas Peakers",
        x=scenarios,
        y=gas_capacities,
        marker_color="#ff7f0e",
        text=[f"{val:.1f} MW" if show_values and val > 0 else "" for val in gas_capacities],
        textposition="auto",
        hovertemplate="Gas Peakers<br>%{y:.1f} MW<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        name="Solar PV",
        x=scenarios,
        y=solar_capacities,
        marker_color="#2ca02c",
        text=[f"{val:.1f} MW" if show_values and val > 0 else "" for val in solar_capacities],
        textposition="auto",
        hovertemplate="Solar PV<br>%{y:.1f} MW<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        name="Battery Storage",
        x=scenarios,
        y=battery_capacities,
        marker_color="#d62728",
        text=[f"{val:.1f} MWh" if show_values and val > 0 else "" for val in battery_capacities],
        textposition="auto",
        hovertemplate="Battery Storage<br>%{y:.1f} MWh<extra></extra>"
    ))
    
    # Update layout
    default_title = "Capacity Mix Comparison Across Scenarios"
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Scenario",
        yaxis_title="Capacity",
        barmode="group",
        template="plotly_white",
        height=500,
        font=dict(size=12),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        hovermode="closest"
    )
    
    return fig
