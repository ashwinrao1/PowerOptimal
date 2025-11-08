"""
Scenario Comparison Page

This page enables multi-scenario analysis including Pareto frontier visualization,
scenario comparison tables, and sensitivity tornado charts. Users can configure
and run multiple scenarios to explore trade-offs between cost, reliability, and
carbon emissions.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Add src directory to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import after path is set
try:
    from analysis.scenario_generator import (
        generate_scenarios,
        generate_gas_price_scenarios,
        generate_lmp_scenarios,
        generate_battery_cost_scenarios,
        generate_reliability_scenarios,
        generate_carbon_scenarios,
        generate_pareto_scenarios
    )
    from analysis.batch_solver import solve_scenarios
    from analysis.pareto_calculator import (
        calculate_pareto_frontier,
        calculate_cost_reliability_frontier,
        calculate_cost_carbon_frontier,
        calculate_grid_reliability_frontier,
        calculate_all_pareto_frontiers
    )
    from analysis.sensitivity_analyzer import (
        analyze_multiple_parameters,
        rank_parameters_by_impact
    )
    from visualization.pareto_viz import (
        plot_pareto_frontier,
        plot_multiple_pareto_frontiers
    )
    from visualization.sensitivity_viz import (
        plot_sensitivity_tornado,
        plot_sensitivity_comparison
    )
    from models.market_data import MarketData
    from models.technology import TechnologyCosts, FacilityParams
except ImportError as e:
    import warnings
    warnings.warn(f"Import error: {e}. Some functionality may not be available.")


def render_scenario_configuration() -> Dict[str, Any]:
    """Render scenario configuration controls.
    
    Returns:
        Dictionary with scenario configuration parameters
    """
    st.subheader("Scenario Configuration")
    st.markdown("Configure which parameters to vary for scenario analysis:")
    
    # Create tabs for different scenario types
    tab1, tab2, tab3 = st.tabs([
        "Quick Scenarios",
        "Custom Scenarios",
        "Pareto Analysis"
    ])
    
    with tab1:
        st.markdown("#### Quick Scenario Sets")
        st.markdown("Pre-configured scenario sets for common analyses:")
        
        scenario_type = st.selectbox(
            "Select Scenario Type",
            options=[
                "Gas Price Sensitivity",
                "Grid LMP Sensitivity",
                "Battery Cost Sensitivity",
                "Reliability Sensitivity",
                "Carbon Constraint Sensitivity"
            ],
            help="Choose a pre-configured set of scenarios"
        )
        
        # Show details based on selection
        if scenario_type == "Gas Price Sensitivity":
            st.info("Varies gas prices from -50% to +50% of baseline")
            variations = st.multiselect(
                "Gas Price Variations",
                options=[0.5, 0.75, 1.0, 1.25, 1.5],
                default=[0.5, 1.0, 1.5],
                format_func=lambda x: f"{int(x*100)}% of baseline"
            )
            config = {
                'type': 'gas_price',
                'variations': variations
            }
        
        elif scenario_type == "Grid LMP Sensitivity":
            st.info("Varies grid electricity prices from -30% to +30% of baseline")
            variations = st.multiselect(
                "LMP Variations",
                options=[0.7, 0.85, 1.0, 1.15, 1.3],
                default=[0.7, 1.0, 1.3],
                format_func=lambda x: f"{int(x*100)}% of baseline"
            )
            config = {
                'type': 'lmp',
                'variations': variations
            }
        
        elif scenario_type == "Battery Cost Sensitivity":
            st.info("Varies battery costs from $200/kWh to $500/kWh")
            variations = st.multiselect(
                "Battery Cost Variations ($/kWh)",
                options=[200, 275, 350, 425, 500],
                default=[200, 350, 500]
            )
            config = {
                'type': 'battery_cost',
                'variations': variations
            }
        
        elif scenario_type == "Reliability Sensitivity":
            st.info("Varies reliability targets from 99.9% to 99.999%")
            variations = st.multiselect(
                "Reliability Targets",
                options=[0.999, 0.9999, 0.99999],
                default=[0.999, 0.9999, 0.99999],
                format_func=lambda x: f"{x*100:.3f}%"
            )
            config = {
                'type': 'reliability',
                'variations': variations
            }
        
        else:  # Carbon Constraint Sensitivity
            st.info("Varies carbon reduction targets from 0% to 100%")
            variations = st.multiselect(
                "Carbon Reduction Targets",
                options=[None, 50, 80, 100],
                default=[None, 50, 100],
                format_func=lambda x: "No constraint" if x is None else f"{x}% reduction"
            )
            config = {
                'type': 'carbon',
                'variations': variations
            }
        
        return config
    
    with tab2:
        st.markdown("#### Custom Scenario Configuration")
        st.markdown("Configure multiple parameter variations simultaneously:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            vary_gas = st.checkbox("Vary Gas Prices", value=False)
            if vary_gas:
                gas_variations = st.multiselect(
                    "Gas Price Multipliers",
                    options=[0.5, 0.75, 1.0, 1.25, 1.5],
                    default=[0.5, 1.0, 1.5]
                )
            else:
                gas_variations = [1.0]
            
            vary_battery = st.checkbox("Vary Battery Costs", value=False)
            if vary_battery:
                battery_variations = st.multiselect(
                    "Battery Costs ($/kWh)",
                    options=[200, 275, 350, 425, 500],
                    default=[200, 350, 500]
                )
            else:
                battery_variations = [350]
        
        with col2:
            vary_lmp = st.checkbox("Vary Grid LMP", value=False)
            if vary_lmp:
                lmp_variations = st.multiselect(
                    "LMP Multipliers",
                    options=[0.7, 0.85, 1.0, 1.15, 1.3],
                    default=[0.7, 1.0, 1.3]
                )
            else:
                lmp_variations = [1.0]
            
            vary_reliability = st.checkbox("Vary Reliability", value=False)
            if vary_reliability:
                reliability_variations = st.multiselect(
                    "Reliability Targets",
                    options=[0.999, 0.9999, 0.99999],
                    default=[0.9999],
                    format_func=lambda x: f"{x*100:.3f}%"
                )
            else:
                reliability_variations = [0.9999]
        
        # Calculate total scenarios
        total_scenarios = (len(gas_variations) * len(lmp_variations) * 
                          len(battery_variations) * len(reliability_variations))
        
        st.info(f"Total scenarios to run: {total_scenarios}")
        
        if total_scenarios > 50:
            st.warning("Large number of scenarios may take significant time to solve.")
        
        config = {
            'type': 'custom',
            'gas_variations': gas_variations,
            'lmp_variations': lmp_variations,
            'battery_variations': battery_variations,
            'reliability_variations': reliability_variations
        }
        
        return config
    
    with tab3:
        st.markdown("#### Pareto Frontier Analysis")
        st.markdown("Generate scenarios optimized for exploring trade-offs:")
        
        objective_pair = st.selectbox(
            "Select Objective Pair",
            options=[
                "Cost vs Reliability",
                "Cost vs Carbon",
                "Grid Dependence vs Reliability"
            ],
            help="Choose which trade-off to explore"
        )
        
        if objective_pair == "Cost vs Reliability":
            st.info("Varies reliability targets to explore cost-reliability trade-off")
            config = {
                'type': 'pareto',
                'objective_pair': 'cost_reliability'
            }
        elif objective_pair == "Cost vs Carbon":
            st.info("Varies carbon constraints to explore cost-carbon trade-off")
            config = {
                'type': 'pareto',
                'objective_pair': 'cost_carbon'
            }
        else:
            st.info("Varies reliability and gas prices to explore grid dependence trade-off")
            config = {
                'type': 'pareto',
                'objective_pair': 'grid_reliability'
            }
        
        return config


@st.cache_data(ttl=3600)
def generate_scenarios_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate scenario list from configuration with caching.
    Cache expires after 1 hour.
    
    Args:
        config: Scenario configuration dictionary
        
    Returns:
        List of scenario parameter dictionaries
    """
    scenario_type = config.get('type')
    
    if scenario_type == 'gas_price':
        return generate_gas_price_scenarios(variations=config.get('variations'))
    
    elif scenario_type == 'lmp':
        return generate_lmp_scenarios(variations=config.get('variations'))
    
    elif scenario_type == 'battery_cost':
        return generate_battery_cost_scenarios(variations=config.get('variations'))
    
    elif scenario_type == 'reliability':
        return generate_reliability_scenarios(variations=config.get('variations'))
    
    elif scenario_type == 'carbon':
        return generate_carbon_scenarios(variations=config.get('variations'))
    
    elif scenario_type == 'custom':
        return generate_scenarios(
            gas_price_variations=config.get('gas_variations', [1.0]),
            lmp_variations=config.get('lmp_variations', [1.0]),
            battery_cost_variations=config.get('battery_variations', [350]),
            reliability_variations=config.get('reliability_variations', [0.9999])
        )
    
    elif scenario_type == 'pareto':
        return generate_pareto_scenarios(
            objective_pair=config.get('objective_pair', 'cost_reliability')
        )
    
    else:
        return []


@st.cache_data(ttl=1800)
def calculate_pareto_frontiers_cached(_scenario_results: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """
    Calculate Pareto frontiers with caching.
    Cache expires after 30 minutes.
    
    Args:
        _scenario_results: List of scenario results (underscore prevents hashing)
        
    Returns:
        Dictionary of Pareto frontier DataFrames
    """
    try:
        return calculate_all_pareto_frontiers(_scenario_results)
    except Exception:
        return {}


def render_pareto_frontiers(scenario_results: List[Dict[str, Any]]):
    """Render Pareto frontier visualizations.
    
    Args:
        scenario_results: List of scenario results from batch solver
    """
    st.subheader("Pareto Frontier Analysis")
    st.markdown("Explore trade-offs between competing objectives:")
    
    # Calculate Pareto frontiers with caching
    try:
        with st.spinner("Calculating Pareto frontiers..."):
            frontiers = calculate_pareto_frontiers_cached(scenario_results)
        
        if not frontiers:
            st.warning("No Pareto frontiers could be calculated from the scenario results.")
            return
        
        # Display frontier selector
        frontier_names = list(frontiers.keys())
        frontier_labels = {
            'cost_reliability': 'Cost vs Reliability',
            'cost_carbon': 'Cost vs Carbon',
            'grid_reliability': 'Grid Dependence vs Reliability'
        }
        
        selected_frontier = st.selectbox(
            "Select Pareto Frontier",
            options=frontier_names,
            format_func=lambda x: frontier_labels.get(x, x)
        )
        
        frontier_df = frontiers[selected_frontier]
        
        if frontier_df.empty:
            st.warning(f"No Pareto-optimal solutions found for {frontier_labels.get(selected_frontier, selected_frontier)}")
            return
        
        # Display Pareto frontier plot
        st.markdown(f"#### {frontier_labels.get(selected_frontier, selected_frontier)}")
        
        # Determine objectives based on frontier type
        if selected_frontier == 'cost_reliability':
            obj1, obj2 = 'total_npv', 'reliability_pct'
        elif selected_frontier == 'cost_carbon':
            obj1, obj2 = 'total_npv', 'carbon_tons_annual'
        else:  # grid_reliability
            obj1, obj2 = 'grid_dependence_pct', 'reliability_pct'
        
        # Get baseline and optimal solutions if available
        baseline_sol = None
        optimal_sol = None
        
        if st.session_state.optimization_result is not None:
            optimal_sol = st.session_state.optimization_result
        
        # Create Pareto plot
        fig = plot_pareto_frontier(
            solutions=frontier_df,
            objective1=obj1,
            objective2=obj2,
            baseline_solution=baseline_sol,
            optimal_solution=optimal_sol,
            show_all_solutions=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display Pareto frontier summary
        st.markdown("#### Pareto Frontier Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Pareto-Optimal Solutions",
                len(frontier_df),
                help="Number of non-dominated solutions"
            )
        
        with col2:
            obj1_range = frontier_df[obj1].max() - frontier_df[obj1].min()
            st.metric(
                f"{obj1.replace('_', ' ').title()} Range",
                f"{obj1_range:,.0f}",
                help=f"Range of {obj1} values on frontier"
            )
        
        with col3:
            obj2_range = frontier_df[obj2].max() - frontier_df[obj2].min()
            st.metric(
                f"{obj2.replace('_', ' ').title()} Range",
                f"{obj2_range:,.2f}",
                help=f"Range of {obj2} values on frontier"
            )
        
        # Display Pareto frontier table
        with st.expander("View Pareto Frontier Data"):
            st.markdown("#### Pareto-Optimal Solutions")
            
            # Select columns to display
            display_cols = ['scenario_name', obj1, obj2]
            
            # Add capacity columns if available
            capacity_cols = [c for c in frontier_df.columns if c.startswith('capacity_')]
            display_cols.extend(capacity_cols[:4])  # Show first 4 capacity columns
            
            # Add key metrics
            metric_cols = ['lcoe', 'carbon_intensity_g_per_kwh', 'grid_dependence_pct']
            for col in metric_cols:
                if col in frontier_df.columns and col not in display_cols:
                    display_cols.append(col)
            
            # Filter and format DataFrame
            display_df = frontier_df[display_cols].copy()
            
            # Format numeric columns
            for col in display_df.columns:
                if col != 'scenario_name' and display_df[col].dtype in [np.float64, np.int64]:
                    display_df[col] = display_df[col].round(2)
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download Pareto Frontier (CSV)",
                data=csv,
                file_name=f"pareto_frontier_{selected_frontier}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error calculating Pareto frontiers: {str(e)}")
        st.info("Ensure scenarios have been run successfully before calculating Pareto frontiers.")


def render_scenario_comparison_table(scenario_results: List[Dict[str, Any]]):
    """Render scenario comparison table.
    
    Args:
        scenario_results: List of scenario results from batch solver
    """
    st.subheader("Scenario Comparison Table")
    st.markdown("Compare key metrics across all scenarios:")
    
    # Filter successful scenarios
    successful_scenarios = [s for s in scenario_results if s.get('status') == 'success']
    
    if not successful_scenarios:
        st.warning("No successful scenarios to compare.")
        return
    
    # Build comparison DataFrame
    rows = []
    for scenario in successful_scenarios:
        row = {
            'Scenario': scenario.get('scenario_name', 'Unknown'),
            'Status': scenario.get('status', 'Unknown')
        }
        
        # Add key metrics
        metrics = scenario.get('metrics', {})
        row['Total NPV ($M)'] = metrics.get('total_npv', 0) / 1e6
        row['LCOE ($/MWh)'] = metrics.get('lcoe', 0)
        row['Reliability (%)'] = metrics.get('reliability_pct', 0)
        row['Carbon (tons/yr)'] = metrics.get('carbon_tons_annual', 0)
        row['Grid Dep. (%)'] = metrics.get('grid_dependence_pct', 0)
        
        # Add capacity information
        capacity = scenario.get('capacity', {})
        row['Grid (MW)'] = capacity.get('grid_mw', 0)
        row['Gas (MW)'] = capacity.get('gas_mw', 0)
        row['Battery (MWh)'] = capacity.get('battery_mwh', 0)
        row['Solar (MW)'] = capacity.get('solar_mw', 0)
        
        # Add solve time
        row['Solve Time (s)'] = scenario.get('solve_time', 0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Format numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'Scenario':
            df[col] = df[col].round(2)
    
    # Display table
    st.dataframe(df, use_container_width=True, height=400)
    
    # Summary statistics
    st.markdown("#### Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Scenarios",
            len(successful_scenarios),
            help="Number of successfully solved scenarios"
        )
    
    with col2:
        avg_npv = df['Total NPV ($M)'].mean()
        st.metric(
            "Avg NPV",
            f"${avg_npv:.1f}M",
            help="Average total NPV across scenarios"
        )
    
    with col3:
        avg_reliability = df['Reliability (%)'].mean()
        st.metric(
            "Avg Reliability",
            f"{avg_reliability:.3f}%",
            help="Average reliability across scenarios"
        )
    
    with col4:
        avg_carbon = df['Carbon (tons/yr)'].mean()
        st.metric(
            "Avg Carbon",
            f"{avg_carbon:,.0f} tons",
            help="Average annual carbon emissions"
        )
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Scenario Comparison (CSV)",
        data=csv,
        file_name="scenario_comparison.csv",
        mime="text/csv",
        help="Download complete scenario comparison table"
    )


@st.cache_data(ttl=1800)
def perform_sensitivity_analysis_cached(_base_solution: Dict[str, Any], _scenario_results: List[Dict[str, Any]], 
                                        varied_params: List[str], metric: str) -> Dict[str, Any]:
    """
    Perform sensitivity analysis with caching.
    Cache expires after 30 minutes.
    
    Args:
        _base_solution: Base solution dictionary (underscore prevents hashing)
        _scenario_results: List of scenario results (underscore prevents hashing)
        varied_params: List of parameter names that were varied
        metric: Metric to analyze
        
    Returns:
        Dictionary with sensitivity analysis results
    """
    try:
        return analyze_multiple_parameters(
            base_solution=_base_solution,
            scenario_results=_scenario_results,
            parameters=varied_params,
            metric=metric
        )
    except Exception:
        return {}


def render_sensitivity_analysis(scenario_results: List[Dict[str, Any]]):
    """Render sensitivity analysis tornado chart.
    
    Args:
        scenario_results: List of scenario results from batch solver
    """
    st.subheader("Sensitivity Analysis")
    st.markdown("Identify which parameters have the largest impact on optimization results:")
    
    # Check if we have a base solution
    if st.session_state.optimization_result is None:
        st.warning("Base solution required for sensitivity analysis. Please run an optimization first.")
        return
    
    base_solution = {
        'status': 'success',
        'scenario_name': 'base',
        'scenario_params': st.session_state.optimization_result.scenario_params,
        'metrics': st.session_state.optimization_result.metrics.to_dict(),
        'capacity': st.session_state.optimization_result.capacity.to_dict()
    }
    
    # Identify which parameters were varied
    varied_params = set()
    for scenario in scenario_results:
        if scenario.get('status') != 'success':
            continue
        params = scenario.get('scenario_params', {})
        for key in params.keys():
            if key not in ['scenario_name']:
                varied_params.add(key)
    
    varied_params = list(varied_params)
    
    if not varied_params:
        st.warning("No parameter variations found in scenario results.")
        return
    
    # Perform sensitivity analysis with caching
    try:
        with st.spinner("Analyzing parameter sensitivity..."):
            sensitivity_results = perform_sensitivity_analysis_cached(
                base_solution,
                scenario_results,
                varied_params,
                'total_npv'
            )
        
        if not sensitivity_results:
            st.warning("Could not perform sensitivity analysis on the scenario results.")
            return
        
        # Display tornado chart
        st.markdown("#### Sensitivity Tornado Chart")
        
        # Options for tornado chart
        col1, col2 = st.columns([3, 1])
        
        with col1:
            metric_options = ['total_npv', 'lcoe', 'carbon_tons_annual', 'reliability_pct']
            selected_metric = st.selectbox(
                "Select Metric",
                options=metric_options,
                format_func=lambda x: x.replace('_', ' ').title(),
                help="Choose which metric to analyze"
            )
        
        with col2:
            top_n = st.number_input(
                "Top N Parameters",
                min_value=1,
                max_value=len(sensitivity_results),
                value=min(5, len(sensitivity_results)),
                help="Show only top N most impactful parameters"
            )
        
        # Create tornado chart
        fig = plot_sensitivity_tornado(
            sensitivity_results=sensitivity_results,
            metric=selected_metric,
            top_n=int(top_n),
            show_values=True,
            height=400 + top_n * 40
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display parameter ranking table
        st.markdown("#### Parameter Impact Ranking")
        
        ranking_df = rank_parameters_by_impact(
            sensitivity_results=sensitivity_results,
            metric='impact_score'
        )
        
        if not ranking_df.empty:
            # Format the DataFrame for display
            display_df = ranking_df[['rank', 'parameter', 'impact_score', 'elasticity', 'r_squared']].copy()
            display_df.columns = ['Rank', 'Parameter', 'Impact Score', 'Elasticity', 'R²']
            
            # Format numeric columns
            display_df['Impact Score'] = display_df['Impact Score'].round(2)
            display_df['Elasticity'] = display_df['Elasticity'].round(3)
            display_df['R²'] = display_df['R²'].round(3)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.info("""
            **Interpretation:**
            - **Impact Score**: Percentage range of metric variation (higher = more impactful)
            - **Elasticity**: % change in metric per % change in parameter
            - **R²**: How well a linear model fits the relationship (1.0 = perfect fit)
            """)
        
    except Exception as e:
        st.error(f"Error performing sensitivity analysis: {str(e)}")
        st.info("Ensure scenarios vary parameters systematically for sensitivity analysis.")


def render():
    """Render the scenario comparison page."""
    st.title("Scenario Comparison")
    st.markdown("Explore trade-offs and sensitivity through multi-scenario analysis.")
    
    st.markdown("---")
    
    # Check if scenario results exist in session state
    if 'scenario_results' not in st.session_state or st.session_state.scenario_results is None:
        st.info("""
        **No scenario results available yet.**
        
        Configure and run scenarios below to:
        - Explore Pareto frontiers showing trade-offs between objectives
        - Compare scenarios across key metrics
        - Analyze parameter sensitivity with tornado charts
        """)
        
        st.markdown("---")
        
        # Scenario configuration
        config = render_scenario_configuration()
        
        st.markdown("---")
        
        # Run scenarios button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            run_scenarios_button = st.button(
                "🚀 Run Scenarios",
                type="primary",
                use_container_width=True,
                help="Generate and solve all configured scenarios"
            )
        
        if run_scenarios_button:
            st.warning("""
            **Scenario solving not yet implemented in this demo.**
            
            In a full implementation, this would:
            1. Generate scenarios based on your configuration
            2. Solve each scenario using the optimization model
            3. Store results for analysis and visualization
            
            For now, please run individual optimizations from the Setup page.
            """)
            
            st.info("""
            **Implementation Note:**
            
            Scenario solving requires:
            - Market data loading for each scenario
            - Parallel optimization solving (can take 10-60 minutes)
            - Result storage and caching
            
            This functionality is available in the full application.
            """)
    
    else:
        # Display scenario analysis results
        scenario_results = st.session_state.scenario_results
        
        st.success(f"Loaded {len(scenario_results)} scenario results")
        
        st.markdown("---")
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs([
            "Pareto Frontiers",
            "Scenario Comparison",
            "Sensitivity Analysis"
        ])
        
        with tab1:
            render_pareto_frontiers(scenario_results)
        
        with tab2:
            render_scenario_comparison_table(scenario_results)
        
        with tab3:
            render_sensitivity_analysis(scenario_results)
        
        st.markdown("---")
        
        # Option to clear results and run new scenarios
        if st.button("Clear Results and Configure New Scenarios"):
            st.session_state.scenario_results = None
            st.rerun()
    
    st.markdown("---")
    
    # Navigation hints
    st.info("""
    **Analysis Tips:**
    - **Pareto Frontiers**: Identify optimal trade-offs between competing objectives
    - **Scenario Comparison**: Compare capacity mixes and metrics across scenarios
    - **Sensitivity Analysis**: Understand which parameters drive optimization results
    
    **Next Steps:**
    - Review **Case Study** for detailed analysis of 300MW West Texas facility
    - Return to **Optimization Setup** to run additional scenarios
    """)
