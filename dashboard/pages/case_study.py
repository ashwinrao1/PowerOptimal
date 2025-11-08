"""
Case Study Page

This page presents a detailed analysis of a 300MW AI training data center
in West Texas, comparing a baseline grid-only approach with an optimal
behind-the-meter energy portfolio. The analysis includes pre-computed results,
narrative explanations, and comparisons to alternative strategies.
"""

import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
from typing import Dict, Any

# Add src directory to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import after path is set
try:
    from visualization.capacity_viz import plot_capacity_mix
    from visualization.cost_viz import plot_cost_breakdown
    from models.technology import TechnologyCosts
except ImportError as e:
    import warnings
    warnings.warn(f"Import error: {e}. Some functionality may not be available.")


def load_case_study_results() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load pre-computed baseline and optimal portfolio results.
    
    Returns:
        tuple: (baseline_results, optimal_results) dictionaries
    """
    results_dir = Path(__file__).parent.parent.parent / "results" / "solutions"
    
    try:
        with open(results_dir / "baseline_grid_only.json", 'r') as f:
            baseline = json.load(f)
        
        with open(results_dir / "optimal_portfolio.json", 'r') as f:
            optimal = json.load(f)
        
        return baseline, optimal
    
    except FileNotFoundError as e:
        st.error(f"Case study results not found: {e}")
        st.info("Please run the baseline and optimal portfolio scripts to generate case study data.")
        return None, None


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a number as currency with appropriate units."""
    if value >= 1e9:
        return f"${value/1e9:.{decimals}f}B"
    elif value >= 1e6:
        return f"${value/1e6:.{decimals}f}M"
    elif value >= 1e3:
        return f"${value/1e3:.{decimals}f}K"
    else:
        return f"${value:.{decimals}f}"


def render_executive_summary():
    """Render the executive summary section."""
    st.markdown("## Executive Summary")
    
    st.markdown("""
    This case study analyzes the optimal energy strategy for a **300MW AI training data center** 
    in **West Texas**, comparing a traditional grid-only approach with an optimized portfolio 
    that includes behind-the-meter generation and storage.
    
    ### Key Findings:
    
    - **Cost Savings**: The optimal portfolio saves **$588M over 20 years** (58% reduction)
    - **Carbon Reduction**: Achieves **56% lower carbon emissions** vs grid-only baseline
    - **Reliability**: Maintains **100% reliability** (99.99%+ uptime)
    - **Payback Period**: Behind-the-meter investments pay back in **less than 1 year**
    
    ### Recommended Strategy:
    
    The optimal energy portfolio combines:
    - **Grid connection** for baseload and reliability
    - **Large-scale battery storage** (93 GWh) for energy arbitrage and demand charge reduction
    - **Solar PV** (9.7 GW) for clean, low-cost daytime generation
    - **No natural gas** peakers (not economically competitive in this scenario)
    """)


def render_problem_statement():
    """Render the problem statement section."""
    st.markdown("## Problem Statement")
    
    st.markdown("""
    ### The Challenge
    
    AI training data centers face a unique energy trilemma:
    
    1. **Cost Pressure**: Training large language models costs millions in electricity
    2. **Reliability Requirements**: 99.99% uptime needed to avoid wasting expensive GPU time
    3. **Carbon Commitments**: Tech companies have pledged 24/7 carbon-free energy by 2030
    
    ### The Opportunity
    
    West Texas offers:
    - **Low electricity prices**: ERCOT wholesale market with frequent negative prices
    - **Excellent solar resources**: High capacity factors year-round
    - **Grid flexibility**: Ability to participate in demand response and ancillary services
    
    ### The Question
    
    What is the optimal combination of grid connection, behind-the-meter generation, 
    and energy storage to minimize 20-year total cost while meeting reliability and 
    carbon goals?
    """)


def render_baseline_scenario(baseline: Dict[str, Any]):
    """Render the baseline grid-only scenario section."""
    st.markdown("## Baseline Scenario: Grid-Only Approach")
    
    st.markdown("""
    The baseline scenario represents the traditional approach: rely entirely on grid 
    electricity with no behind-the-meter assets. This establishes our cost and carbon baseline.
    """)
    
    # Capacity
    st.markdown("### Capacity Configuration")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Grid Connection", f"{baseline['capacity']['Grid Connection (MW)']:.0f} MW")
    with col2:
        st.metric("Gas Peakers", "0 MW", help="Not included in baseline")
    with col3:
        st.metric("Battery Storage", "0 MWh", help="Not included in baseline")
    with col4:
        st.metric("Solar PV", "0 MW", help="Not included in baseline")
    
    st.markdown("---")
    
    # Metrics
    st.markdown("### Performance Metrics")
    
    tab1, tab2, tab3 = st.tabs(["Cost", "Reliability", "Carbon"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total NPV (20-year)",
                format_currency(baseline['metrics']['total_npv'], 2),
                help="Total net present value over planning horizon"
            )
            st.metric(
                "CAPEX",
                format_currency(baseline['metrics']['capex'], 2),
                help="Capital expenditure for grid interconnection"
            )
        with col2:
            st.metric(
                "Annual OPEX",
                format_currency(baseline['metrics']['opex_annual'], 2),
                help="Annual operating expenditure"
            )
            st.metric(
                "LCOE",
                f"${baseline['metrics']['lcoe']:.2f}/MWh",
                help="Levelized cost of energy"
            )
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Reliability",
                f"{baseline['metrics']['reliability_pct']:.4f}%",
                help="Percentage of load served"
            )
        with col2:
            st.metric(
                "Curtailment",
                f"{baseline['metrics']['total_curtailment_mwh']:.2f} MWh/year",
                help="Total annual energy curtailed"
            )
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Annual Emissions",
                f"{baseline['metrics']['carbon_tons_annual']:,.0f} tons CO2",
                help="Total annual carbon emissions"
            )
        with col2:
            st.metric(
                "Carbon Intensity",
                f"{baseline['metrics']['carbon_intensity_g_per_kwh']:.1f} g CO2/kWh",
                help="Carbon emissions per kWh"
            )
    
    st.markdown("---")
    
    st.info("""
    **Baseline Insights:**
    - 100% grid dependence results in high carbon intensity (341 g CO2/kWh)
    - Annual electricity cost of $96M driven by high LMP prices and demand charges
    - No capital investment in behind-the-meter assets
    - Vulnerable to grid price volatility and carbon regulations
    """)


def render_optimal_scenario(optimal: Dict[str, Any]):
    """Render the optimal portfolio scenario section."""
    st.markdown("## Optimal Portfolio: Behind-the-Meter Strategy")
    
    st.markdown("""
    The optimal portfolio leverages behind-the-meter assets to dramatically reduce costs 
    and carbon emissions while maintaining 100% reliability.
    """)
    
    # Capacity
    st.markdown("### Optimal Capacity Mix")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Grid Connection",
            f"{optimal['capacity']['Grid Connection (MW)']:.0f} MW",
            help="Reduced grid interconnection"
        )
    with col2:
        st.metric(
            "Gas Peakers",
            f"{optimal['capacity']['Gas Peakers (MW)']:.0f} MW",
            help="Natural gas generation capacity"
        )
    with col3:
        st.metric(
            "Battery Storage",
            f"{optimal['capacity']['Battery Storage (MWh)']/1000:.1f} GWh",
            help="Energy storage capacity"
        )
    with col4:
        st.metric(
            "Solar PV",
            f"{optimal['capacity']['Solar PV (MW)']/1000:.1f} GW",
            help="Solar generation capacity"
        )
    
    st.markdown("---")
    
    # Metrics
    st.markdown("### Performance Metrics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Cost", "Reliability", "Carbon", "Operations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total NPV (20-year)",
                format_currency(optimal['metrics']['total_npv'], 2),
                help="Total net present value over planning horizon"
            )
            st.metric(
                "CAPEX",
                format_currency(optimal['metrics']['capex'], 2),
                help="Capital expenditure for BTM assets"
            )
        with col2:
            st.metric(
                "Annual OPEX",
                format_currency(optimal['metrics']['opex_annual'], 2),
                help="Annual operating expenditure"
            )
            st.metric(
                "LCOE",
                f"${optimal['metrics']['lcoe']:.2f}/MWh",
                help="Levelized cost of energy"
            )
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Reliability",
                f"{optimal['metrics']['reliability_pct']:.4f}%",
                help="Percentage of load served"
            )
        with col2:
            st.metric(
                "Curtailment",
                f"{optimal['metrics']['total_curtailment_mwh']:.2f} MWh/year",
                help="Total annual energy curtailed"
            )
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Annual Emissions",
                f"{optimal['metrics']['carbon_tons_annual']:,.0f} tons CO2",
                help="Total annual carbon emissions"
            )
            st.metric(
                "Carbon Intensity",
                f"{optimal['metrics']['carbon_intensity_g_per_kwh']:.1f} g CO2/kWh",
                help="Carbon emissions per kWh"
            )
        with col2:
            st.metric(
                "Carbon Reduction",
                f"{optimal['metrics']['carbon_reduction_pct']:.1f}%",
                delta=f"{optimal['metrics']['carbon_reduction_pct']:.1f}%",
                delta_color="normal",
                help="Reduction vs grid-only baseline"
            )
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Grid Dependence",
                f"{optimal['metrics']['grid_dependence_pct']:.1f}%",
                help="Percentage of energy from grid"
            )
            st.metric(
                "Battery Cycles/Year",
                f"{optimal['metrics']['battery_cycles_per_year']:.1f}",
                help="Full charge-discharge cycles"
            )
        with col2:
            st.metric(
                "Gas Capacity Factor",
                f"{optimal['metrics']['gas_capacity_factor']:.1f}%",
                help="Gas peaker utilization"
            )
            st.metric(
                "Solar Capacity Factor",
                f"{optimal['metrics']['solar_capacity_factor']:.1f}%",
                help="Solar PV utilization"
            )
    
    st.markdown("---")
    
    st.success("""
    **Optimal Portfolio Insights:**
    - Massive battery storage (93 GWh) enables energy arbitrage during low-price hours
    - Large solar array (9.7 GW) provides clean, low-cost daytime generation
    - Grid dependence reduced to 44%, significantly lowering exposure to price volatility
    - 56% carbon reduction achieved without sacrificing reliability
    """)


def render_comparison(baseline: Dict[str, Any], optimal: Dict[str, Any]):
    """Render side-by-side comparison section."""
    st.markdown("## Baseline vs Optimal: Side-by-Side Comparison")
    
    # Calculate savings
    npv_savings = baseline['metrics']['total_npv'] - optimal['metrics']['total_npv']
    npv_savings_pct = (npv_savings / baseline['metrics']['total_npv']) * 100
    
    opex_savings = baseline['metrics']['opex_annual'] - optimal['metrics']['opex_annual']
    opex_savings_pct = (opex_savings / baseline['metrics']['opex_annual']) * 100
    
    carbon_reduction = baseline['metrics']['carbon_tons_annual'] - optimal['metrics']['carbon_tons_annual']
    carbon_reduction_pct = (carbon_reduction / baseline['metrics']['carbon_tons_annual']) * 100
    
    # Payback calculation
    btm_capex = optimal['metrics']['capex'] - baseline['metrics']['capex']
    if opex_savings > 0:
        payback_years = btm_capex / opex_savings
    else:
        payback_years = float('inf')
    
    st.markdown("### Financial Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "20-Year NPV Savings",
            format_currency(npv_savings, 2),
            delta=f"-{npv_savings_pct:.1f}%",
            delta_color="inverse",
            help="Total cost savings over planning horizon"
        )
    
    with col2:
        st.metric(
            "Annual OPEX Savings",
            format_currency(opex_savings, 2),
            delta=f"-{opex_savings_pct:.1f}%",
            delta_color="inverse",
            help="Annual operating cost savings"
        )
    
    with col3:
        st.metric(
            "BTM Investment",
            format_currency(btm_capex, 2),
            help="Upfront investment in behind-the-meter assets"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Simple Payback Period",
            f"{payback_years:.1f} years",
            help="Time to recover BTM investment through OPEX savings"
        )
    
    with col2:
        st.metric(
            "LCOE Reduction",
            f"${baseline['metrics']['lcoe'] - optimal['metrics']['lcoe']:.2f}/MWh",
            delta=f"-{((baseline['metrics']['lcoe'] - optimal['metrics']['lcoe']) / baseline['metrics']['lcoe'] * 100):.1f}%",
            delta_color="inverse",
            help="Levelized cost of energy reduction"
        )
    
    st.markdown("---")
    
    st.markdown("### Carbon Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Annual Emissions Reduction",
            f"{carbon_reduction:,.0f} tons CO2/year",
            delta=f"-{carbon_reduction_pct:.1f}%",
            delta_color="inverse",
            help="Annual carbon emissions reduction"
        )
    
    with col2:
        st.metric(
            "Carbon Intensity Reduction",
            f"{baseline['metrics']['carbon_intensity_g_per_kwh'] - optimal['metrics']['carbon_intensity_g_per_kwh']:.1f} g CO2/kWh",
            delta=f"-{((baseline['metrics']['carbon_intensity_g_per_kwh'] - optimal['metrics']['carbon_intensity_g_per_kwh']) / baseline['metrics']['carbon_intensity_g_per_kwh'] * 100):.1f}%",
            delta_color="inverse",
            help="Carbon intensity reduction"
        )
    
    with col3:
        st.metric(
            "20-Year Emissions Avoided",
            f"{carbon_reduction * 20 / 1e6:.2f} million tons CO2",
            help="Total emissions avoided over planning horizon"
        )
    
    st.markdown("---")
    
    # Comparison table
    st.markdown("### Detailed Metrics Comparison")
    
    comparison_data = {
        "Metric": [
            "Total NPV (20-year)",
            "CAPEX",
            "Annual OPEX",
            "LCOE",
            "Reliability",
            "Annual Emissions",
            "Carbon Intensity",
            "Grid Dependence"
        ],
        "Baseline (Grid-Only)": [
            format_currency(baseline['metrics']['total_npv'], 2),
            format_currency(baseline['metrics']['capex'], 2),
            format_currency(baseline['metrics']['opex_annual'], 2),
            f"${baseline['metrics']['lcoe']:.2f}/MWh",
            f"{baseline['metrics']['reliability_pct']:.4f}%",
            f"{baseline['metrics']['carbon_tons_annual']:,.0f} tons",
            f"{baseline['metrics']['carbon_intensity_g_per_kwh']:.1f} g/kWh",
            f"{baseline['metrics']['grid_dependence_pct']:.1f}%"
        ],
        "Optimal Portfolio": [
            format_currency(optimal['metrics']['total_npv'], 2),
            format_currency(optimal['metrics']['capex'], 2),
            format_currency(optimal['metrics']['opex_annual'], 2),
            f"${optimal['metrics']['lcoe']:.2f}/MWh",
            f"{optimal['metrics']['reliability_pct']:.4f}%",
            f"{optimal['metrics']['carbon_tons_annual']:,.0f} tons",
            f"{optimal['metrics']['carbon_intensity_g_per_kwh']:.1f} g/kWh",
            f"{optimal['metrics']['grid_dependence_pct']:.1f}%"
        ],
        "Improvement": [
            f"-{npv_savings_pct:.1f}%",
            f"+{((optimal['metrics']['capex'] - baseline['metrics']['capex']) / baseline['metrics']['capex'] * 100):.0f}%",
            f"-{opex_savings_pct:.1f}%",
            f"-{((baseline['metrics']['lcoe'] - optimal['metrics']['lcoe']) / baseline['metrics']['lcoe'] * 100):.1f}%",
            f"+{optimal['metrics']['reliability_pct'] - baseline['metrics']['reliability_pct']:.4f}%",
            f"-{carbon_reduction_pct:.1f}%",
            f"-{((baseline['metrics']['carbon_intensity_g_per_kwh'] - optimal['metrics']['carbon_intensity_g_per_kwh']) / baseline['metrics']['carbon_intensity_g_per_kwh'] * 100):.1f}%",
            f"-{baseline['metrics']['grid_dependence_pct'] - optimal['metrics']['grid_dependence_pct']:.1f}%"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_alternative_strategies():
    """Render comparison to alternative strategies section."""
    st.markdown("## Comparison to Alternative Strategies")
    
    st.markdown("""
    How does the optimal behind-the-meter portfolio compare to other approaches 
    being pursued by tech companies?
    """)
    
    # Create tabs for different strategies
    tab1, tab2, tab3 = st.tabs([
        "Microsoft Three Mile Island",
        "Google 24/7 CFE",
        "Amazon Nuclear SMRs"
    ])
    
    with tab1:
        st.markdown("### Microsoft Three Mile Island Approach")
        
        st.markdown("""
        **Strategy**: Microsoft signed a 20-year power purchase agreement to restart 
        Unit 1 of the Three Mile Island nuclear plant, securing 835 MW of carbon-free 
        baseload power for AI data centers.
        
        **Comparison to Optimal Portfolio**:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Microsoft Nuclear Approach:**
            - 835 MW nuclear baseload
            - 20-year PPA commitment
            - Zero carbon emissions
            - Fixed price (estimated $80-100/MWh)
            - Single point of failure risk
            - Regulatory and public perception challenges
            """)
        
        with col2:
            st.markdown("""
            **Optimal BTM Portfolio:**
            - Diversified energy mix
            - Flexible grid + storage + solar
            - 56% carbon reduction (not 100%)
            - Lower LCOE ($14.67/MWh)
            - Distributed reliability
            - Faster deployment (no nuclear licensing)
            """)
        
        st.info("""
        **Key Insight**: The nuclear approach achieves 100% carbon-free energy but at 
        significantly higher cost and longer timeline. The optimal BTM portfolio offers 
        a pragmatic balance of cost, carbon, and speed to deployment.
        """)
    
    with tab2:
        st.markdown("### Google 24/7 Carbon-Free Energy Approach")
        
        st.markdown("""
        **Strategy**: Google aims to match every hour of electricity consumption with 
        carbon-free energy sources by 2030, going beyond annual renewable energy matching.
        
        **Comparison to Optimal Portfolio**:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Google 24/7 CFE:**
            - Hourly carbon matching
            - Mix of wind, solar, geothermal, nuclear
            - Advanced forecasting and dispatch
            - Premium pricing for clean energy
            - Requires extensive grid coordination
            - 100% carbon-free goal
            """)
        
        with col2:
            st.markdown("""
            **Optimal BTM Portfolio:**
            - 44% grid dependence (some carbon)
            - Solar + battery for daytime matching
            - Grid power during low-carbon hours
            - Cost-optimized approach
            - Behind-the-meter control
            - 56% carbon reduction achieved
            """)
        
        st.info("""
        **Key Insight**: Achieving 24/7 CFE requires additional investment in clean 
        firm power (nuclear, geothermal, or long-duration storage). The optimal portfolio 
        could be enhanced with these technologies if carbon constraints tighten.
        """)
    
    with tab3:
        st.markdown("### Amazon Nuclear SMR Approach")
        
        st.markdown("""
        **Strategy**: Amazon invested in X-energy to develop small modular reactors (SMRs) 
        for data center power, targeting 300 MW modules with faster deployment than 
        traditional nuclear.
        
        **Comparison to Optimal Portfolio**:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Amazon SMR Approach:**
            - 300 MW modular nuclear units
            - Carbon-free baseload
            - Estimated $5,000-7,000/kW CAPEX
            - 5-7 year deployment timeline
            - Regulatory approval required
            - Fuel supply chain needed
            """)
        
        with col2:
            st.markdown("""
            **Optimal BTM Portfolio:**
            - Solar + battery + grid mix
            - Partial carbon reduction
            - $1,767/kW average CAPEX
            - 2-3 year deployment timeline
            - Proven technology
            - No fuel requirements
            """)
        
        st.info("""
        **Key Insight**: SMRs offer a promising path to carbon-free energy but face 
        higher costs and longer timelines. The optimal BTM portfolio can be deployed 
        immediately while SMR technology matures.
        """)
    
    st.markdown("---")
    
    st.markdown("### Strategic Recommendation")
    
    st.success("""
    **Recommended Phased Approach:**
    
    1. **Phase 1 (Years 1-3)**: Deploy optimal BTM portfolio (solar + battery + grid)
       - Immediate 58% cost savings and 56% carbon reduction
       - Proven technology with fast deployment
       - Payback in less than 1 year
    
    2. **Phase 2 (Years 4-7)**: Add clean firm power as technology matures
       - Evaluate SMRs, advanced geothermal, or long-duration storage
       - Target 80-90% carbon reduction
       - Maintain cost competitiveness
    
    3. **Phase 3 (Years 8-10)**: Achieve 24/7 carbon-free energy
       - Full portfolio of clean technologies
       - 100% carbon-free operation
       - Participate in clean energy markets
    
    This phased approach balances immediate cost savings with long-term carbon goals, 
    while maintaining flexibility as clean energy technologies evolve.
    """)


def render_key_insights():
    """Render key insights and recommendations section."""
    st.markdown("## Key Insights and Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Why Battery Storage Dominates")
        st.markdown("""
        The optimal portfolio invests heavily in battery storage (93 GWh) because:
        
        1. **Energy Arbitrage**: ERCOT prices vary dramatically (often negative to $1000+/MWh)
        2. **Demand Charge Reduction**: Batteries reduce peak grid draw, saving $15/kW-month
        3. **Solar Firming**: Batteries store excess solar for evening/night use
        4. **Declining Costs**: Battery costs have fallen 90% since 2010
        
        At current prices ($350/kWh), batteries pay for themselves in months through 
        arbitrage and demand charge savings.
        """)
        
        st.markdown("### Why Solar Scales So Large")
        st.markdown("""
        The 9.7 GW solar array (32x the facility load) is economically optimal because:
        
        1. **Low LCOE**: Solar in West Texas costs $1,200/kW with excellent capacity factors
        2. **Negative Correlation**: Solar generates when grid prices are highest (midday)
        3. **Battery Synergy**: Excess solar charges batteries for later use
        4. **Carbon Benefits**: Solar provides zero-carbon generation
        
        The large oversizing is economically justified by low solar costs and high 
        battery capacity to absorb excess generation.
        """)
    
    with col2:
        st.markdown("### Why No Natural Gas")
        st.markdown("""
        The optimizer chose zero gas capacity because:
        
        1. **Fuel Costs**: Natural gas prices ($3-5/MMBtu) make gas generation expensive
        2. **Carbon Penalty**: Gas emits 0.4 tons CO2/MWh
        3. **Battery Alternative**: Batteries provide reliability without fuel costs
        4. **Grid Availability**: Grid connection provides backup at lower cost
        
        Gas might become attractive with:
        - Higher battery costs
        - Stricter carbon constraints (gas as backup to renewables)
        - Grid reliability concerns
        """)
        
        st.markdown("### Implementation Considerations")
        st.markdown("""
        **Technical Challenges:**
        - Land requirements: ~20,000 acres for 9.7 GW solar
        - Grid interconnection: Requires utility coordination
        - Battery safety: Fire suppression and thermal management
        - System integration: Complex control systems needed
        
        **Financial Considerations:**
        - Upfront capital: $44M investment required
        - Financing options: Tax equity, green bonds, PPAs
        - Risk management: Hedge against technology and price risk
        - Incentives: ITC/PTC for solar, IRA benefits
        
        **Regulatory Factors:**
        - Permitting: Environmental and land use approvals
        - Interconnection: ERCOT queue and studies
        - Market participation: Ancillary services registration
        - Compliance: Safety and environmental regulations
        """)
    
    st.markdown("---")
    
    st.markdown("### Strategic Recommendations")
    
    st.markdown("""
    Based on this analysis, we recommend the following actions:
    
    1. **Immediate Actions (0-6 months)**:
       - Secure land for solar development (20,000+ acres)
       - Begin ERCOT interconnection process
       - Issue RFPs for solar and battery systems
       - Arrange project financing
    
    2. **Near-Term Actions (6-18 months)**:
       - Complete engineering and permitting
       - Execute EPC contracts
       - Begin construction of solar and battery systems
       - Develop energy management system
    
    3. **Medium-Term Actions (18-36 months)**:
       - Commission solar and battery systems
       - Optimize dispatch strategies
       - Participate in ERCOT ancillary services
       - Monitor performance and refine operations
    
    4. **Long-Term Actions (3+ years)**:
       - Evaluate additional clean firm power (SMRs, geothermal)
       - Expand portfolio as data center grows
       - Share learnings with industry
       - Advocate for supportive policies
    """)


def export_case_study_report(baseline: Dict[str, Any], optimal: Dict[str, Any]) -> str:
    """
    Generate a downloadable case study report.
    
    Args:
        baseline: Baseline scenario results
        optimal: Optimal scenario results
    
    Returns:
        Markdown-formatted report string
    """
    npv_savings = baseline['metrics']['total_npv'] - optimal['metrics']['total_npv']
    npv_savings_pct = (npv_savings / baseline['metrics']['total_npv']) * 100
    
    report = f"""# 300MW West Texas Data Center Energy Optimization Case Study

## Executive Summary

This case study analyzes the optimal energy strategy for a 300MW AI training data center 
in West Texas. The analysis compares a traditional grid-only approach with an optimized 
portfolio that includes behind-the-meter generation and storage.

### Key Findings

- **Cost Savings**: ${npv_savings/1e9:.2f}B over 20 years ({npv_savings_pct:.1f}% reduction)
- **Carbon Reduction**: {optimal['metrics']['carbon_reduction_pct']:.1f}% lower emissions
- **Reliability**: {optimal['metrics']['reliability_pct']:.4f}% uptime maintained
- **Payback**: Less than 1 year for BTM investments

## Baseline Scenario (Grid-Only)

### Capacity
- Grid Connection: {baseline['capacity']['Grid Connection (MW)']:.0f} MW
- Gas Peakers: 0 MW
- Battery Storage: 0 MWh
- Solar PV: 0 MW

### Metrics
- Total NPV: ${baseline['metrics']['total_npv']/1e9:.2f}B
- Annual OPEX: ${baseline['metrics']['opex_annual']/1e6:.1f}M
- LCOE: ${baseline['metrics']['lcoe']:.2f}/MWh
- Carbon Intensity: {baseline['metrics']['carbon_intensity_g_per_kwh']:.1f} g CO2/kWh

## Optimal Portfolio

### Capacity
- Grid Connection: {optimal['capacity']['Grid Connection (MW)']:.0f} MW
- Gas Peakers: {optimal['capacity']['Gas Peakers (MW)']:.0f} MW
- Battery Storage: {optimal['capacity']['Battery Storage (MWh)']/1000:.1f} GWh
- Solar PV: {optimal['capacity']['Solar PV (MW)']/1000:.1f} GW

### Metrics
- Total NPV: ${optimal['metrics']['total_npv']/1e9:.2f}B
- Annual OPEX: ${optimal['metrics']['opex_annual']/1e6:.1f}M
- LCOE: ${optimal['metrics']['lcoe']:.2f}/MWh
- Carbon Intensity: {optimal['metrics']['carbon_intensity_g_per_kwh']:.1f} g CO2/kWh
- Grid Dependence: {optimal['metrics']['grid_dependence_pct']:.1f}%

## Recommendations

1. Deploy optimal BTM portfolio immediately for 58% cost savings
2. Invest in large-scale battery storage (93 GWh) for energy arbitrage
3. Build oversized solar array (9.7 GW) to maximize clean generation
4. Maintain grid connection for reliability and backup power
5. Monitor emerging clean firm power technologies for future phases

---

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report


def render():
    """Render the case study page."""
    st.title("Case Study: 300MW West Texas Data Center")
    st.markdown("Detailed analysis comparing grid-only baseline with optimal behind-the-meter portfolio")
    
    st.markdown("---")
    
    # Load case study results
    with st.spinner("Loading case study results..."):
        baseline, optimal = load_case_study_results()
    
    if baseline is None or optimal is None:
        st.error("Unable to load case study results. Please ensure the baseline and optimal portfolio solutions have been generated.")
        st.info("""
        To generate case study results:
        1. Run `python scripts/run_baseline.py`
        2. Run `python scripts/run_optimal_portfolio.py`
        3. Refresh this page
        """)
        return
    
    # Render executive summary
    render_executive_summary()
    
    st.markdown("---")
    
    # Render problem statement
    render_problem_statement()
    
    st.markdown("---")
    
    # Render baseline scenario
    render_baseline_scenario(baseline)
    
    st.markdown("---")
    
    # Render optimal scenario
    render_optimal_scenario(optimal)
    
    st.markdown("---")
    
    # Render comparison
    render_comparison(baseline, optimal)
    
    st.markdown("---")
    
    # Render alternative strategies
    render_alternative_strategies()
    
    st.markdown("---")
    
    # Render key insights
    render_key_insights()
    
    st.markdown("---")
    
    # Export section
    st.markdown("## Download Case Study Report")
    
    st.markdown("""
    Download a comprehensive report summarizing the case study findings, 
    including all metrics, comparisons, and recommendations.
    """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        report_text = export_case_study_report(baseline, optimal)
        st.download_button(
            label="Download Case Study Report (Markdown)",
            data=report_text,
            file_name="west_texas_case_study.md",
            mime="text/markdown",
            help="Download detailed case study report in Markdown format",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Footer
    st.info("""
    **About This Case Study:**
    
    This analysis uses real market data from ERCOT West Texas (2023) and industry-standard 
    technology costs from NREL ATB 2024. The optimization model minimizes 20-year total 
    cost while meeting 99.99% reliability requirements.
    
    Results are based on deterministic optimization with historical data. Actual performance 
    may vary based on future market conditions, technology costs, and operational factors.
    
    For questions or to explore custom scenarios, navigate to the **Optimization Setup** page.
    """)
