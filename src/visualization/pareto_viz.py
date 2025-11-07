"""Pareto frontier visualization functions."""

import plotly.graph_objects as go
from typing import Dict, Optional, Union, List, Any
import pandas as pd
import numpy as np

from ..models.solution import OptimizationSolution


def plot_pareto_frontier(
    solutions: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, OptimizationSolution]],
    objective1: str,
    objective2: str,
    title: Optional[str] = None,
    baseline_solution: Optional[Union[Dict[str, Any], OptimizationSolution]] = None,
    optimal_solution: Optional[Union[Dict[str, Any], OptimizationSolution]] = None,
    show_all_solutions: bool = True,
    height: int = 600
) -> go.Figure:
    """Create scatter plot showing Pareto frontier with two objectives.
    
    Visualizes the trade-off between two competing objectives, highlighting
    Pareto-optimal solutions (non-dominated) with different colors and markers.
    Can annotate key solutions like baseline and optimal.
    
    Args:
        solutions: One of:
            - DataFrame with Pareto frontier data (from pareto_calculator)
            - List of solution dictionaries from batch_solver
            - Dict mapping scenario names to OptimizationSolution objects
        objective1: Name of first objective (x-axis)
            Common values: 'total_npv', 'grid_dependence_pct', 'carbon_tons_annual'
        objective2: Name of second objective (y-axis)
            Common values: 'reliability_pct', 'carbon_tons_annual', 'carbon_intensity_g_per_kwh'
        title: Custom title for the plot (optional)
        baseline_solution: Optional baseline solution to annotate (e.g., grid-only)
        optimal_solution: Optional optimal solution to annotate
        show_all_solutions: If True, show all solutions; if False, only show Pareto-optimal
        height: Figure height in pixels (default: 600)
        
    Returns:
        Plotly Figure object with interactive scatter plot
        
    Raises:
        ValueError: If solutions format is invalid or objectives not found
        
    Example:
        >>> from src.analysis.pareto_calculator import calculate_pareto_frontier
        >>> pareto_df = calculate_pareto_frontier(
        ...     solutions, 'total_npv', 'carbon_tons_annual'
        ... )
        >>> fig = plot_pareto_frontier(
        ...     pareto_df,
        ...     objective1='total_npv',
        ...     objective2='carbon_tons_annual',
        ...     baseline_solution=baseline,
        ...     optimal_solution=optimal
        ... )
        >>> fig.show()
    """
    # Convert solutions to DataFrame format
    df = _prepare_dataframe(solutions, objective1, objective2)
    
    if df.empty:
        raise ValueError("No valid solutions to plot")
    
    # Check if objectives exist
    if objective1 not in df.columns or objective2 not in df.columns:
        raise ValueError(
            f"Objectives {objective1} and/or {objective2} not found in solutions. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Identify Pareto-optimal solutions if not already marked
    if 'is_pareto_optimal' not in df.columns:
        df['is_pareto_optimal'] = False
    
    # Create figure
    fig = go.Figure()
    
    # Plot non-Pareto solutions if requested
    if show_all_solutions:
        non_pareto_df = df[~df['is_pareto_optimal']]
        if not non_pareto_df.empty:
            fig.add_trace(go.Scatter(
                x=non_pareto_df[objective1],
                y=non_pareto_df[objective2],
                mode='markers',
                name='Non-Pareto Solutions',
                marker=dict(
                    size=8,
                    color='lightgray',
                    symbol='circle',
                    line=dict(width=1, color='gray')
                ),
                hovertemplate=(
                    f"<b>%{{customdata[0]}}</b><br>"
                    f"{_format_objective_label(objective1)}: %{{x:,.2f}}<br>"
                    f"{_format_objective_label(objective2)}: %{{y:,.2f}}<br>"
                    "<extra></extra>"
                ),
                customdata=non_pareto_df[['scenario_name']].values if 'scenario_name' in non_pareto_df.columns else [['Solution']] * len(non_pareto_df)
            ))
    
    # Plot Pareto-optimal solutions
    pareto_df = df[df['is_pareto_optimal']]
    if not pareto_df.empty:
        # Sort by objective1 for connecting line
        pareto_df_sorted = pareto_df.sort_values(by=objective1)
        
        # Add connecting line for Pareto frontier
        fig.add_trace(go.Scatter(
            x=pareto_df_sorted[objective1],
            y=pareto_df_sorted[objective2],
            mode='lines',
            name='Pareto Frontier',
            line=dict(color='#1f77b4', width=2, dash='dash'),
            hoverinfo='skip',
            showlegend=True
        ))
        
        # Add Pareto-optimal points
        fig.add_trace(go.Scatter(
            x=pareto_df[objective1],
            y=pareto_df[objective2],
            mode='markers',
            name='Pareto-Optimal Solutions',
            marker=dict(
                size=12,
                color='#1f77b4',
                symbol='diamond',
                line=dict(width=2, color='darkblue')
            ),
            hovertemplate=(
                f"<b>%{{customdata[0]}}</b><br>"
                f"{_format_objective_label(objective1)}: %{{x:,.2f}}<br>"
                f"{_format_objective_label(objective2)}: %{{y:,.2f}}<br>"
                "<extra></extra>"
            ),
            customdata=pareto_df[['scenario_name']].values if 'scenario_name' in pareto_df.columns else [['Pareto Solution']] * len(pareto_df)
        ))
    
    # Add baseline solution annotation if provided
    if baseline_solution is not None:
        _add_solution_annotation(
            fig, baseline_solution, objective1, objective2,
            name='Baseline', color='#d62728', symbol='square'
        )
    
    # Add optimal solution annotation if provided
    if optimal_solution is not None:
        _add_solution_annotation(
            fig, optimal_solution, objective1, objective2,
            name='Optimal', color='#2ca02c', symbol='star'
        )
    
    # Identify and annotate extreme points on Pareto frontier
    if not pareto_df.empty:
        _add_extreme_point_annotations(fig, pareto_df, objective1, objective2)
    
    # Update layout
    default_title = f"Pareto Frontier: {_format_objective_label(objective1)} vs {_format_objective_label(objective2)}"
    
    fig.update_layout(
        title=title or default_title,
        xaxis_title=_format_objective_label(objective1),
        yaxis_title=_format_objective_label(objective2),
        template="plotly_white",
        height=height,
        font=dict(size=12),
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        showlegend=True
    )
    
    # Format axes based on objective types
    _format_axes(fig, objective1, objective2)
    
    return fig


def plot_multiple_pareto_frontiers(
    frontiers: Dict[str, pd.DataFrame],
    objective_pairs: Optional[List[tuple]] = None,
    title: Optional[str] = None,
    height: int = 500
) -> go.Figure:
    """Create subplots showing multiple Pareto frontiers.
    
    Displays multiple objective trade-offs in a grid layout for easy comparison.
    
    Args:
        frontiers: Dictionary mapping frontier names to DataFrames
            Expected keys: 'cost_reliability', 'cost_carbon', 'grid_reliability'
        objective_pairs: Optional list of (obj1, obj2) tuples for each frontier.
            If None, uses standard pairs based on frontier names.
        title: Custom title for the overall figure
        height: Figure height in pixels per subplot row
        
    Returns:
        Plotly Figure object with subplots
        
    Example:
        >>> from src.analysis.pareto_calculator import calculate_all_pareto_frontiers
        >>> frontiers = calculate_all_pareto_frontiers(solutions)
        >>> fig = plot_multiple_pareto_frontiers(frontiers)
        >>> fig.show()
    """
    from plotly.subplots import make_subplots
    
    # Define standard objective pairs
    standard_pairs = {
        'cost_reliability': ('total_npv', 'reliability_pct'),
        'cost_carbon': ('total_npv', 'carbon_tons_annual'),
        'grid_reliability': ('grid_dependence_pct', 'reliability_pct')
    }
    
    # Filter to available frontiers
    available_frontiers = {k: v for k, v in frontiers.items() if not v.empty}
    
    if not available_frontiers:
        raise ValueError("No non-empty frontiers to plot")
    
    n_frontiers = len(available_frontiers)
    n_cols = min(2, n_frontiers)
    n_rows = (n_frontiers + n_cols - 1) // n_cols
    
    # Create subplot titles
    subplot_titles = []
    for name in available_frontiers.keys():
        if name in standard_pairs:
            obj1, obj2 = standard_pairs[name]
            subplot_titles.append(
                f"{_format_objective_label(obj1)} vs {_format_objective_label(obj2)}"
            )
        else:
            subplot_titles.append(name.replace('_', ' ').title())
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Plot each frontier
    for idx, (name, df) in enumerate(available_frontiers.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Get objectives for this frontier
        if objective_pairs and idx < len(objective_pairs):
            obj1, obj2 = objective_pairs[idx]
        elif name in standard_pairs:
            obj1, obj2 = standard_pairs[name]
        else:
            continue
        
        # Sort by first objective
        df_sorted = df.sort_values(by=obj1)
        
        # Add Pareto frontier line
        fig.add_trace(
            go.Scatter(
                x=df_sorted[obj1],
                y=df_sorted[obj2],
                mode='lines+markers',
                name=name.replace('_', ' ').title(),
                line=dict(width=2),
                marker=dict(size=8, symbol='diamond'),
                showlegend=False,
                hovertemplate=(
                    f"<b>%{{customdata[0]}}</b><br>"
                    f"{_format_objective_label(obj1)}: %{{x:,.2f}}<br>"
                    f"{_format_objective_label(obj2)}: %{{y:,.2f}}<br>"
                    "<extra></extra>"
                ),
                customdata=df_sorted[['scenario_name']].values if 'scenario_name' in df_sorted.columns else [['Solution']] * len(df_sorted)
            ),
            row=row,
            col=col
        )
        
        # Update axes labels
        fig.update_xaxes(
            title_text=_format_objective_label(obj1),
            row=row,
            col=col
        )
        fig.update_yaxes(
            title_text=_format_objective_label(obj2),
            row=row,
            col=col
        )
    
    # Update layout
    default_title = "Pareto Frontier Analysis: Multiple Objective Trade-offs"
    fig.update_layout(
        title=title or default_title,
        template="plotly_white",
        height=height * n_rows,
        font=dict(size=11),
        hovermode="closest"
    )
    
    return fig


def _prepare_dataframe(
    solutions: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, OptimizationSolution]],
    objective1: str,
    objective2: str
) -> pd.DataFrame:
    """Convert various solution formats to DataFrame.
    
    Args:
        solutions: Solutions in various formats
        objective1: First objective name
        objective2: Second objective name
        
    Returns:
        DataFrame with solution data
    """
    if isinstance(solutions, pd.DataFrame):
        return solutions.copy()
    
    elif isinstance(solutions, list):
        # List of solution dictionaries from batch_solver
        rows = []
        for sol in solutions:
            if sol.get('status') != 'success':
                continue
            
            metrics = sol.get('metrics', {})
            row = {
                'scenario_name': sol.get('scenario_name', 'Unknown'),
                'scenario_index': sol.get('scenario_index', 0),
                'is_pareto_optimal': sol.get('is_pareto_optimal', False)
            }
            row.update(metrics)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    elif isinstance(solutions, dict):
        # Dict mapping scenario names to OptimizationSolution objects
        rows = []
        for name, sol in solutions.items():
            if isinstance(sol, OptimizationSolution):
                metrics = sol.metrics.to_dict()
            elif isinstance(sol, dict):
                metrics = sol.get('metrics', {})
            else:
                continue
            
            row = {
                'scenario_name': name,
                'is_pareto_optimal': False
            }
            row.update(metrics)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    else:
        raise ValueError(
            "solutions must be DataFrame, list of dicts, or dict of OptimizationSolution objects"
        )


def _add_solution_annotation(
    fig: go.Figure,
    solution: Union[Dict[str, Any], OptimizationSolution],
    objective1: str,
    objective2: str,
    name: str,
    color: str,
    symbol: str
) -> None:
    """Add a specific solution as an annotated point.
    
    Args:
        fig: Plotly figure to add annotation to
        solution: Solution to annotate
        objective1: First objective name
        objective2: Second objective name
        name: Label for the solution
        color: Marker color
        symbol: Marker symbol
    """
    # Extract objective values
    if isinstance(solution, OptimizationSolution):
        metrics = solution.metrics.to_dict()
    elif isinstance(solution, dict):
        metrics = solution.get('metrics', solution)
    else:
        return
    
    obj1_value = metrics.get(objective1)
    obj2_value = metrics.get(objective2)
    
    if obj1_value is None or obj2_value is None:
        return
    
    # Add marker
    fig.add_trace(go.Scatter(
        x=[obj1_value],
        y=[obj2_value],
        mode='markers',
        name=name,
        marker=dict(
            size=14,
            color=color,
            symbol=symbol,
            line=dict(width=2, color='white')
        ),
        hovertemplate=(
            f"<b>{name}</b><br>"
            f"{_format_objective_label(objective1)}: %{{x:,.2f}}<br>"
            f"{_format_objective_label(objective2)}: %{{y:,.2f}}<br>"
            "<extra></extra>"
        )
    ))
    
    # Add text annotation
    fig.add_annotation(
        x=obj1_value,
        y=obj2_value,
        text=name,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=color,
        ax=40,
        ay=-40,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=color,
        borderwidth=2,
        borderpad=4,
        font=dict(size=11, color=color)
    )


def _add_extreme_point_annotations(
    fig: go.Figure,
    pareto_df: pd.DataFrame,
    objective1: str,
    objective2: str
) -> None:
    """Add annotations for extreme points on Pareto frontier.
    
    Args:
        fig: Plotly figure to add annotations to
        pareto_df: DataFrame with Pareto-optimal solutions
        objective1: First objective name
        objective2: Second objective name
    """
    if len(pareto_df) < 2:
        return
    
    # Find extreme points
    obj1_min_idx = pareto_df[objective1].idxmin()
    obj1_max_idx = pareto_df[objective1].idxmax()
    obj2_min_idx = pareto_df[objective2].idxmin()
    obj2_max_idx = pareto_df[objective2].idxmax()
    
    # Only annotate if extreme points are different
    extreme_indices = set([obj1_min_idx, obj1_max_idx, obj2_min_idx, obj2_max_idx])
    
    if len(extreme_indices) <= 2:
        return
    
    # Annotate extreme points with small labels
    for idx in extreme_indices:
        scenario_name = pareto_df.loc[idx, 'scenario_name'] if 'scenario_name' in pareto_df.columns else 'Extreme'
        
        # Determine which extreme this is
        labels = []
        if idx == obj1_min_idx:
            labels.append(f"Min {_format_objective_label(objective1, short=True)}")
        if idx == obj1_max_idx:
            labels.append(f"Max {_format_objective_label(objective1, short=True)}")
        if idx == obj2_min_idx:
            labels.append(f"Min {_format_objective_label(objective2, short=True)}")
        if idx == obj2_max_idx:
            labels.append(f"Max {_format_objective_label(objective2, short=True)}")
        
        label_text = "<br>".join(labels)
        
        fig.add_annotation(
            x=pareto_df.loc[idx, objective1],
            y=pareto_df.loc[idx, objective2],
            text=label_text,
            showarrow=True,
            arrowhead=1,
            arrowsize=0.8,
            arrowwidth=1,
            arrowcolor='gray',
            ax=30,
            ay=30,
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor='gray',
            borderwidth=1,
            borderpad=2,
            font=dict(size=9, color='gray')
        )


def _format_objective_label(objective: str, short: bool = False) -> str:
    """Format objective name for display.
    
    Args:
        objective: Objective column name
        short: If True, return abbreviated label
        
    Returns:
        Formatted label string
    """
    labels = {
        'total_npv': ('Total NPV ($)', 'NPV'),
        'capex': ('CAPEX ($)', 'CAPEX'),
        'opex_annual': ('Annual OPEX ($)', 'OPEX'),
        'lcoe': ('LCOE ($/MWh)', 'LCOE'),
        'reliability_pct': ('Reliability (%)', 'Reliability'),
        'total_curtailment_mwh': ('Total Curtailment (MWh)', 'Curtailment'),
        'carbon_tons_annual': ('Annual Carbon Emissions (tons CO2)', 'Carbon'),
        'carbon_intensity_g_per_kwh': ('Carbon Intensity (g CO2/kWh)', 'Carbon Int.'),
        'carbon_reduction_pct': ('Carbon Reduction (%)', 'Carbon Red.'),
        'grid_dependence_pct': ('Grid Dependence (%)', 'Grid Dep.'),
        'gas_capacity_factor': ('Gas Capacity Factor', 'Gas CF'),
        'battery_cycles_per_year': ('Battery Cycles/Year', 'Battery'),
        'solar_capacity_factor': ('Solar Capacity Factor', 'Solar CF')
    }
    
    if objective in labels:
        return labels[objective][1] if short else labels[objective][0]
    else:
        return objective.replace('_', ' ').title()


def _format_axes(fig: go.Figure, objective1: str, objective2: str) -> None:
    """Format axes based on objective types.
    
    Args:
        fig: Plotly figure to format
        objective1: First objective name
        objective2: Second objective name
    """
    # Format x-axis
    if 'npv' in objective1.lower() or 'capex' in objective1.lower() or 'opex' in objective1.lower():
        fig.update_xaxes(tickformat='$,.0f')
    elif 'pct' in objective1.lower() or 'factor' in objective1.lower():
        fig.update_xaxes(tickformat='.1f', ticksuffix='%')
    elif 'lcoe' in objective1.lower():
        fig.update_xaxes(tickformat='$.2f')
    
    # Format y-axis
    if 'npv' in objective2.lower() or 'capex' in objective2.lower() or 'opex' in objective2.lower():
        fig.update_yaxes(tickformat='$,.0f')
    elif 'pct' in objective2.lower() or 'factor' in objective2.lower():
        fig.update_yaxes(tickformat='.1f', ticksuffix='%')
    elif 'lcoe' in objective2.lower():
        fig.update_yaxes(tickformat='$.2f')
    elif 'tons' in objective2.lower():
        fig.update_yaxes(tickformat=',.0f')
