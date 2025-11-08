"""Cost breakdown visualization functions."""

import plotly.graph_objects as go
from typing import Dict, Optional, Union
import numpy as np

try:
    from ..models.solution import OptimizationSolution, SolutionMetrics, CapacitySolution
    from ..models.technology import TechnologyCosts
except ImportError:
    from models.solution import OptimizationSolution, SolutionMetrics, CapacitySolution
    from models.technology import TechnologyCosts


def plot_cost_breakdown(
    solution: Union[OptimizationSolution, Dict],
    tech_costs: Optional[TechnologyCosts] = None,
    title: Optional[str] = None,
    show_values: bool = True,
    format: str = "waterfall"
) -> go.Figure:
    """Create cost breakdown visualization showing CAPEX and OPEX components.
    
    Visualizes the breakdown of costs by technology (grid, gas, battery, solar)
    and by cost type (CAPEX vs OPEX). Shows total NPV and annual costs.
    
    Args:
        solution: OptimizationSolution object or dict with solution data
        tech_costs: TechnologyCosts object for calculating component costs.
                   If None, uses default costs.
        title: Custom title for the plot (optional)
        show_values: Whether to show values on the chart
        format: Visualization format - "waterfall" or "stacked_bar"
        
    Returns:
        Plotly Figure object
        
    Raises:
        ValueError: If solution format is invalid or format is not supported
    """
    # Extract solution data
    if isinstance(solution, OptimizationSolution):
        capacity = solution.capacity
        metrics = solution.metrics
    elif isinstance(solution, dict):
        # Handle dict format from JSON
        if "capacity" in solution and "metrics" in solution:
            capacity_dict = solution["capacity"]
            metrics_dict = solution["metrics"]
        else:
            raise ValueError(
                "Dict must contain 'capacity' and 'metrics' keys"
            )
        
        # Create objects from dict
        capacity = CapacitySolution(
            grid_mw=capacity_dict.get("Grid Connection (MW)", 0),
            gas_mw=capacity_dict.get("Gas Peakers (MW)", 0),
            battery_mwh=capacity_dict.get("Battery Storage (MWh)", 0),
            solar_mw=capacity_dict.get("Solar PV (MW)", 0)
        )
        
        metrics = SolutionMetrics(**metrics_dict)
    else:
        raise ValueError(
            "solution must be OptimizationSolution or dict"
        )
    
    # Use default tech costs if not provided
    if tech_costs is None:
        tech_costs = TechnologyCosts()
    
    # Calculate cost components
    cost_components = _calculate_cost_components(capacity, metrics, tech_costs)
    
    # Dispatch to appropriate visualization function
    if format == "waterfall":
        return _plot_waterfall_breakdown(cost_components, title, show_values)
    elif format == "stacked_bar":
        return _plot_stacked_bar_breakdown(cost_components, title, show_values)
    else:
        raise ValueError(
            f"Unsupported format '{format}'. Choose from: 'waterfall', 'stacked_bar'"
        )


def _calculate_cost_components(
    capacity: CapacitySolution,
    metrics: SolutionMetrics,
    tech_costs: TechnologyCosts
) -> Dict[str, float]:
    """Calculate individual cost components by technology and type.
    
    Args:
        capacity: Capacity investment decisions
        metrics: Solution metrics with total costs
        tech_costs: Technology cost parameters
        
    Returns:
        Dictionary with cost breakdown
    """
    # Calculate CAPEX components
    grid_capex = capacity.grid_mw * tech_costs.grid_capex_per_kw
    gas_capex = capacity.gas_mw * tech_costs.gas_capex_per_kw
    battery_capex = capacity.battery_mwh * tech_costs.battery_capex_per_kwh
    solar_capex = capacity.solar_mw * tech_costs.solar_capex_per_kw
    
    total_capex = grid_capex + gas_capex + battery_capex + solar_capex
    
    # Estimate OPEX components from annual OPEX
    # This is an approximation since we don't have detailed OPEX breakdown
    # We'll use capacity factors and typical cost structures
    annual_opex = metrics.opex_annual
    
    # Estimate grid OPEX (typically largest component)
    # Assume grid provides base load, so high utilization
    grid_energy_fraction = metrics.grid_dependence_pct / 100.0
    
    # Estimate gas OPEX based on capacity factor
    gas_cf = metrics.gas_capacity_factor
    
    # Estimate solar OPEX (fixed O&M only)
    solar_opex = capacity.solar_mw * tech_costs.solar_fixed_om
    
    # Estimate battery OPEX (degradation costs)
    # Battery cycles per year * capacity * 2 (charge + discharge) * degradation cost
    battery_opex = (metrics.battery_cycles_per_year * 
                   capacity.battery_mwh * 2 * 
                   tech_costs.battery_degradation)
    
    # Remaining OPEX goes to grid and gas
    remaining_opex = annual_opex - solar_opex - battery_opex
    
    # Split remaining between grid and gas based on energy contribution
    # Grid typically has higher costs due to demand charges and energy purchases
    if grid_energy_fraction > 0:
        grid_opex = remaining_opex * 0.7  # Grid typically 70% of variable costs
        gas_opex = remaining_opex * 0.3   # Gas typically 30% of variable costs
    else:
        grid_opex = 0
        gas_opex = remaining_opex
    
    return {
        "grid_capex": grid_capex,
        "gas_capex": gas_capex,
        "battery_capex": battery_capex,
        "solar_capex": solar_capex,
        "total_capex": total_capex,
        "grid_opex": grid_opex,
        "gas_opex": gas_opex,
        "battery_opex": battery_opex,
        "solar_opex": solar_opex,
        "total_opex_annual": annual_opex,
        "total_npv": metrics.total_npv
    }


def _plot_waterfall_breakdown(
    cost_components: Dict[str, float],
    title: Optional[str],
    show_values: bool
) -> go.Figure:
    """Create waterfall chart showing cost buildup.
    
    Args:
        cost_components: Dictionary with cost breakdown
        title: Custom title for the plot
        show_values: Whether to show values on bars
        
    Returns:
        Plotly Figure object
    """
    # Prepare waterfall data
    # Show CAPEX components first, then OPEX components, then total
    
    x_labels = [
        "Grid CAPEX",
        "Gas CAPEX",
        "Battery CAPEX",
        "Solar CAPEX",
        "Total CAPEX",
        "Grid OPEX (Annual)",
        "Gas OPEX (Annual)",
        "Battery OPEX (Annual)",
        "Solar OPEX (Annual)",
        "Total Annual OPEX"
    ]
    
    y_values = [
        cost_components["grid_capex"],
        cost_components["gas_capex"],
        cost_components["battery_capex"],
        cost_components["solar_capex"],
        0,  # Total CAPEX (calculated by waterfall)
        cost_components["grid_opex"],
        cost_components["gas_opex"],
        cost_components["battery_opex"],
        cost_components["solar_opex"],
        0   # Total OPEX (calculated by waterfall)
    ]
    
    # Create measure types for waterfall
    # "relative" for individual components, "total" for subtotals
    measures = [
        "relative",  # Grid CAPEX
        "relative",  # Gas CAPEX
        "relative",  # Battery CAPEX
        "relative",  # Solar CAPEX
        "total",     # Total CAPEX
        "relative",  # Grid OPEX
        "relative",  # Gas OPEX
        "relative",  # Battery OPEX
        "relative",  # Solar OPEX
        "total"      # Total OPEX
    ]
    
    # Create text labels
    text_labels = []
    for i, (label, value) in enumerate(zip(x_labels, y_values)):
        if measures[i] == "total":
            if "CAPEX" in label:
                actual_value = cost_components["total_capex"]
            else:
                actual_value = cost_components["total_opex_annual"]
            text_labels.append(f"${actual_value/1e6:.1f}M" if show_values else "")
        else:
            text_labels.append(f"${value/1e6:.1f}M" if show_values and value > 0 else "")
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        x=x_labels,
        y=y_values,
        measure=measures,
        text=text_labels,
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2ca02c"}},
        decreasing={"marker": {"color": "#d62728"}},
        totals={"marker": {"color": "#1f77b4"}},
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>"
    ))
    
    # Update layout
    default_title = "Cost Breakdown by Technology and Type"
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Cost Component",
        yaxis_title="Cost ($)",
        template="plotly_white",
        height=600,
        font=dict(size=11),
        showlegend=False,
        xaxis=dict(
            tickangle=-45
        ),
        yaxis=dict(
            tickformat="$,.0f"
        )
    )
    
    # Add annotation with total NPV
    fig.add_annotation(
        text=f"Total 20-Year NPV: ${cost_components['total_npv']/1e6:.1f}M",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.05,
        showarrow=False,
        font=dict(size=14, color="#1f77b4"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#1f77b4",
        borderwidth=2,
        borderpad=4
    )
    
    return fig


def _plot_stacked_bar_breakdown(
    cost_components: Dict[str, float],
    title: Optional[str],
    show_values: bool
) -> go.Figure:
    """Create stacked bar chart showing CAPEX and OPEX by technology.
    
    Args:
        cost_components: Dictionary with cost breakdown
        title: Custom title for the plot
        show_values: Whether to show values on bars
        
    Returns:
        Plotly Figure object
    """
    # Prepare data for stacked bars
    technologies = ["Grid", "Gas", "Battery", "Solar"]
    
    capex_values = [
        cost_components["grid_capex"],
        cost_components["gas_capex"],
        cost_components["battery_capex"],
        cost_components["solar_capex"]
    ]
    
    opex_values = [
        cost_components["grid_opex"],
        cost_components["gas_opex"],
        cost_components["battery_opex"],
        cost_components["solar_opex"]
    ]
    
    # Create stacked bar chart
    fig = go.Figure()
    
    # Add CAPEX bars
    fig.add_trace(go.Bar(
        name="CAPEX",
        x=technologies,
        y=capex_values,
        marker_color="#1f77b4",
        text=[f"${val/1e6:.1f}M" if show_values and val > 0 else "" for val in capex_values],
        textposition="inside",
        hovertemplate="CAPEX<br>%{x}<br>$%{y:,.0f}<extra></extra>"
    ))
    
    # Add OPEX bars (annual)
    fig.add_trace(go.Bar(
        name="Annual OPEX",
        x=technologies,
        y=opex_values,
        marker_color="#ff7f0e",
        text=[f"${val/1e6:.1f}M" if show_values and val > 0 else "" for val in opex_values],
        textposition="inside",
        hovertemplate="Annual OPEX<br>%{x}<br>$%{y:,.0f}<extra></extra>"
    ))
    
    # Update layout
    default_title = "Cost Breakdown by Technology (CAPEX vs Annual OPEX)"
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Technology",
        yaxis_title="Cost ($)",
        barmode="stack",
        template="plotly_white",
        height=600,
        font=dict(size=12),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        yaxis=dict(
            tickformat="$,.0f"
        )
    )
    
    # Add annotation with totals
    total_capex = cost_components["total_capex"]
    total_opex = cost_components["total_opex_annual"]
    total_npv = cost_components["total_npv"]
    
    annotation_text = (
        f"Total CAPEX: ${total_capex/1e6:.1f}M<br>"
        f"Total Annual OPEX: ${total_opex/1e6:.1f}M<br>"
        f"Total 20-Year NPV: ${total_npv/1e6:.1f}M"
    )
    
    fig.add_annotation(
        text=annotation_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font=dict(size=11, color="#1f77b4"),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#1f77b4",
        borderwidth=2,
        borderpad=4,
        align="left"
    )
    
    return fig


def plot_cost_comparison(
    solutions: Dict[str, Union[OptimizationSolution, Dict]],
    tech_costs: Optional[TechnologyCosts] = None,
    title: Optional[str] = None,
    metric: str = "total_npv"
) -> go.Figure:
    """Create bar chart comparing costs across multiple scenarios.
    
    Args:
        solutions: Dictionary mapping scenario names to solution objects
        tech_costs: TechnologyCosts object for calculating component costs
        title: Custom title for the plot
        metric: Cost metric to compare - "total_npv", "capex", or "opex_annual"
        
    Returns:
        Plotly Figure object
        
    Raises:
        ValueError: If metric is not supported
    """
    if metric not in ["total_npv", "capex", "opex_annual"]:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            "Choose from: 'total_npv', 'capex', 'opex_annual'"
        )
    
    # Use default tech costs if not provided
    if tech_costs is None:
        tech_costs = TechnologyCosts()
    
    # Extract cost data from all solutions
    scenarios = []
    values = []
    
    for scenario_name, solution in solutions.items():
        # Extract metrics
        if isinstance(solution, OptimizationSolution):
            metrics = solution.metrics
        elif isinstance(solution, dict):
            if "metrics" in solution:
                metrics = SolutionMetrics(**solution["metrics"])
            else:
                continue
        else:
            continue
        
        scenarios.append(scenario_name)
        
        if metric == "total_npv":
            values.append(metrics.total_npv)
        elif metric == "capex":
            values.append(metrics.capex)
        elif metric == "opex_annual":
            values.append(metrics.opex_annual)
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=scenarios,
        y=values,
        marker_color="#1f77b4",
        text=[f"${val/1e6:.1f}M" for val in values],
        textposition="outside",
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>"
    ))
    
    # Update layout
    metric_labels = {
        "total_npv": "Total 20-Year NPV",
        "capex": "Capital Expenditure (CAPEX)",
        "opex_annual": "Annual Operating Expenditure (OPEX)"
    }
    
    default_title = f"{metric_labels[metric]} Comparison Across Scenarios"
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Scenario",
        yaxis_title=f"{metric_labels[metric]} ($)",
        template="plotly_white",
        height=500,
        font=dict(size=12),
        showlegend=False,
        yaxis=dict(
            tickformat="$,.0f"
        )
    )
    
    return fig
