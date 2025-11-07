"""Sensitivity analysis visualization functions."""

import plotly.graph_objects as go
from typing import Dict, Optional, Any, List
import numpy as np


def plot_sensitivity_tornado(
    sensitivity_results: Dict[str, Dict[str, Any]],
    metric: str = 'total_npv',
    title: Optional[str] = None,
    top_n: Optional[int] = None,
    show_values: bool = True,
    height: int = 600
) -> go.Figure:
    """Create tornado chart showing parameter impacts on NPV or other metrics.
    
    A tornado chart displays horizontal bars showing how each parameter affects
    the objective metric. Parameters are sorted by magnitude of impact (largest
    at top), with bars extending left for negative variations and right for
    positive variations.
    
    Args:
        sensitivity_results: Dictionary of sensitivity analysis results from
                           analyze_multiple_parameters() or similar
        metric: Metric being analyzed (default: 'total_npv')
        title: Custom title for the plot (optional)
        top_n: Show only top N most impactful parameters (optional)
        show_values: Whether to show values on the bars
        height: Figure height in pixels (default: 600)
        
    Returns:
        Plotly Figure object with interactive tornado chart
        
    Raises:
        ValueError: If sensitivity_results is empty or invalid
        
    Example:
        >>> from src.analysis.sensitivity_analyzer import analyze_multiple_parameters
        >>> sensitivity = analyze_multiple_parameters(
        ...     base_solution,
        ...     scenario_results,
        ...     ['gas_price_multiplier', 'lmp_multiplier', 'battery_cost_per_kwh']
        ... )
        >>> fig = plot_sensitivity_tornado(sensitivity)
        >>> fig.show()
    """
    if not sensitivity_results:
        raise ValueError("sensitivity_results cannot be empty")
    
    # Extract data for tornado chart
    tornado_data = _prepare_tornado_data(sensitivity_results, metric)
    
    if not tornado_data:
        raise ValueError("No valid sensitivity data to plot")
    
    # Sort by impact magnitude (largest at top)
    tornado_data.sort(key=lambda x: abs(x['impact_range']), reverse=True)
    
    # Limit to top N if specified
    if top_n is not None and top_n > 0:
        tornado_data = tornado_data[:top_n]
    
    # Create tornado chart
    fig = _create_tornado_figure(
        tornado_data,
        metric,
        title,
        show_values,
        height
    )
    
    return fig


def _prepare_tornado_data(
    sensitivity_results: Dict[str, Dict[str, Any]],
    metric: str
) -> List[Dict[str, Any]]:
    """Prepare data for tornado chart from sensitivity results.
    
    Args:
        sensitivity_results: Dictionary of sensitivity analysis results
        metric: Metric being analyzed
        
    Returns:
        List of dictionaries with tornado chart data
    """
    tornado_data = []
    
    for param_name, analysis in sensitivity_results.items():
        if analysis.get('metric') != metric:
            continue
        
        # Get base value and metric values
        base_metric = analysis.get('base_metric_value')
        metric_values = analysis.get('metric_values', [])
        param_values = analysis.get('parameter_values', [])
        base_param = analysis.get('base_parameter_value')
        
        if not metric_values or base_metric is None:
            continue
        
        # Calculate min and max metric values
        min_metric = min(metric_values)
        max_metric = max(metric_values)
        
        # Calculate percentage changes from base
        if base_metric != 0:
            min_pct_change = ((min_metric - base_metric) / abs(base_metric)) * 100
            max_pct_change = ((max_metric - base_metric) / abs(base_metric)) * 100
        else:
            min_pct_change = 0
            max_pct_change = 0
        
        # Find parameter values corresponding to min and max metrics
        min_idx = metric_values.index(min_metric)
        max_idx = metric_values.index(max_metric)
        
        min_param = param_values[min_idx] if min_idx < len(param_values) else base_param
        max_param = param_values[max_idx] if max_idx < len(param_values) else base_param
        
        # Calculate parameter percentage changes
        if base_param != 0:
            min_param_pct = ((min_param - base_param) / abs(base_param)) * 100
            max_param_pct = ((max_param - base_param) / abs(base_param)) * 100
        else:
            min_param_pct = 0
            max_param_pct = 0
        
        tornado_data.append({
            'parameter': _format_parameter_name(param_name),
            'parameter_raw': param_name,
            'base_metric': base_metric,
            'min_metric': min_metric,
            'max_metric': max_metric,
            'min_pct_change': min_pct_change,
            'max_pct_change': max_pct_change,
            'impact_range': max_pct_change - min_pct_change,
            'min_param': min_param,
            'max_param': max_param,
            'min_param_pct': min_param_pct,
            'max_param_pct': max_param_pct,
            'elasticity': analysis.get('elasticity', 0),
            'impact_score': analysis.get('impact_score', 0)
        })
    
    return tornado_data


def _create_tornado_figure(
    tornado_data: List[Dict[str, Any]],
    metric: str,
    title: Optional[str],
    show_values: bool,
    height: int
) -> go.Figure:
    """Create the tornado chart figure.
    
    Args:
        tornado_data: Prepared tornado chart data
        metric: Metric being analyzed
        title: Custom title
        show_values: Whether to show values on bars
        height: Figure height
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Extract data for plotting
    parameters = [d['parameter'] for d in tornado_data]
    min_changes = [d['min_pct_change'] for d in tornado_data]
    max_changes = [d['max_pct_change'] for d in tornado_data]
    
    # Create bars for negative variations (left side)
    # These represent the minimum metric values
    negative_bars = []
    positive_bars = []
    
    for i, data in enumerate(tornado_data):
        min_change = data['min_pct_change']
        max_change = data['max_pct_change']
        
        # Determine which is negative and which is positive
        # The bar should extend from 0 to the change value
        if min_change < 0:
            negative_bars.append(min_change)
        else:
            negative_bars.append(0)
        
        if max_change > 0:
            positive_bars.append(max_change)
        else:
            positive_bars.append(0)
    
    # Add negative variation bars (left side, typically lower parameter values)
    fig.add_trace(go.Bar(
        name='Low Parameter Value',
        y=parameters,
        x=negative_bars,
        orientation='h',
        marker=dict(
            color='#d62728',
            line=dict(color='darkred', width=1)
        ),
        text=[f"{val:.1f}%" if show_values and val != 0 else "" for val in negative_bars],
        textposition='inside',
        textangle=0,
        insidetextanchor='middle',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Impact: %{x:.2f}%<br>"
            "Parameter: %{customdata[0]:.2f} (%{customdata[1]:+.1f}%)<br>"
            "Metric: %{customdata[2]:,.0f}<br>"
            "<extra></extra>"
        ),
        customdata=[[d['min_param'], d['min_param_pct'], d['min_metric']] 
                    for d in tornado_data]
    ))
    
    # Add positive variation bars (right side, typically higher parameter values)
    fig.add_trace(go.Bar(
        name='High Parameter Value',
        y=parameters,
        x=positive_bars,
        orientation='h',
        marker=dict(
            color='#2ca02c',
            line=dict(color='darkgreen', width=1)
        ),
        text=[f"+{val:.1f}%" if show_values and val != 0 else "" for val in positive_bars],
        textposition='inside',
        textangle=0,
        insidetextanchor='middle',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Impact: +%{x:.2f}%<br>"
            "Parameter: %{customdata[0]:.2f} (%{customdata[1]:+.1f}%)<br>"
            "Metric: %{customdata[2]:,.0f}<br>"
            "<extra></extra>"
        ),
        customdata=[[d['max_param'], d['max_param_pct'], d['max_metric']] 
                    for d in tornado_data]
    ))
    
    # Add vertical line at x=0 (base case)
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="black",
        line_width=2,
        annotation_text="Base Case",
        annotation_position="top"
    )
    
    # Update layout
    metric_label = _format_metric_label(metric)
    default_title = f"Sensitivity Analysis: Impact on {metric_label}"
    
    fig.update_layout(
        title=title or default_title,
        xaxis_title=f"Change in {metric_label} (%)",
        yaxis_title="Parameter",
        template="plotly_white",
        height=height,
        font=dict(size=11),
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(
            ticksuffix="%",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        ),
        yaxis=dict(
            autorange="reversed"  # Largest impact at top
        ),
        hovermode="y unified"
    )
    
    # Add annotation with explanation
    fig.add_annotation(
        text="Bars show % change in metric when parameter varies from base case",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.1,
        xanchor="center",
        yanchor="top",
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="center"
    )
    
    return fig


def plot_sensitivity_comparison(
    sensitivity_results: Dict[str, Dict[str, Any]],
    metrics: List[str] = ['total_npv', 'carbon_tons_annual', 'reliability_pct'],
    title: Optional[str] = None,
    height: int = 400
) -> go.Figure:
    """Create comparison chart showing parameter impacts across multiple metrics.
    
    Displays how each parameter affects different metrics in a grouped bar chart.
    
    Args:
        sensitivity_results: Dictionary of sensitivity analysis results
        metrics: List of metrics to compare
        title: Custom title for the plot
        height: Figure height in pixels per metric
        
    Returns:
        Plotly Figure object with subplots
        
    Example:
        >>> fig = plot_sensitivity_comparison(
        ...     sensitivity,
        ...     metrics=['total_npv', 'carbon_tons_annual']
        ... )
        >>> fig.show()
    """
    from plotly.subplots import make_subplots
    
    if not sensitivity_results:
        raise ValueError("sensitivity_results cannot be empty")
    
    # Create subplots for each metric
    n_metrics = len(metrics)
    fig = make_subplots(
        rows=n_metrics,
        cols=1,
        subplot_titles=[_format_metric_label(m) for m in metrics],
        vertical_spacing=0.12
    )
    
    # Plot tornado chart for each metric
    for idx, metric in enumerate(metrics):
        row = idx + 1
        
        # Prepare data for this metric
        tornado_data = _prepare_tornado_data(sensitivity_results, metric)
        
        if not tornado_data:
            continue
        
        # Sort by impact
        tornado_data.sort(key=lambda x: abs(x['impact_range']), reverse=True)
        
        parameters = [d['parameter'] for d in tornado_data]
        impact_scores = [d['impact_score'] for d in tornado_data]
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=impact_scores,
                y=parameters,
                orientation='h',
                marker=dict(color='#1f77b4'),
                showlegend=False,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Impact Score: %{x:.2f}<br>"
                    "<extra></extra>"
                )
            ),
            row=row,
            col=1
        )
        
        # Update axes
        fig.update_xaxes(title_text="Impact Score", row=row, col=1)
        fig.update_yaxes(autorange="reversed", row=row, col=1)
    
    # Update layout
    default_title = "Parameter Impact Comparison Across Metrics"
    fig.update_layout(
        title=title or default_title,
        template="plotly_white",
        height=height * n_metrics,
        font=dict(size=11)
    )
    
    return fig


def _format_parameter_name(param_name: str) -> str:
    """Format parameter name for display.
    
    Args:
        param_name: Raw parameter name
        
    Returns:
        Formatted parameter name
    """
    # Common parameter name mappings
    name_map = {
        'gas_price_multiplier': 'Gas Price',
        'lmp_multiplier': 'Grid LMP',
        'battery_cost_per_kwh': 'Battery Cost',
        'solar_cost_per_kw': 'Solar Cost',
        'reliability_target': 'Reliability Target',
        'carbon_budget': 'Carbon Budget',
        'discount_rate': 'Discount Rate',
        'curtailment_penalty': 'Curtailment Penalty'
    }
    
    if param_name in name_map:
        return name_map[param_name]
    else:
        # Convert snake_case to Title Case
        return param_name.replace('_', ' ').title()


def _format_metric_label(metric: str) -> str:
    """Format metric name for display.
    
    Args:
        metric: Metric name
        
    Returns:
        Formatted metric label
    """
    labels = {
        'total_npv': 'Total NPV',
        'capex': 'CAPEX',
        'opex_annual': 'Annual OPEX',
        'lcoe': 'LCOE',
        'reliability_pct': 'Reliability',
        'total_curtailment_mwh': 'Total Curtailment',
        'carbon_tons_annual': 'Annual Carbon Emissions',
        'carbon_intensity_g_per_kwh': 'Carbon Intensity',
        'carbon_reduction_pct': 'Carbon Reduction',
        'grid_dependence_pct': 'Grid Dependence',
        'gas_capacity_factor': 'Gas Capacity Factor',
        'battery_cycles_per_year': 'Battery Cycles',
        'solar_capacity_factor': 'Solar Capacity Factor'
    }
    
    return labels.get(metric, metric.replace('_', ' ').title())
