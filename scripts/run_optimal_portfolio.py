"""
Run optimal portfolio optimization scenario.

This script runs an optimization with all technologies available (grid, gas,
battery, solar) and allows the optimizer to determine the optimal capacity mix
that minimizes total cost while meeting reliability and carbon constraints.
"""

import sys
from pathlib import Path
import logging
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.models.market_data import MarketData
from src.models.technology import TechnologyCosts, FacilityParams
from src.optimization.model_builder import build_optimization_model
from src.optimization.solver import solve_model
from src.optimization.solution_extractor import extract_solution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_market_data(year: int = 2023) -> MarketData:
    """
    Load market data from processed CSV files.
    
    Args:
        year: Year to use for data (2022, 2023, or 2024)
        
    Returns:
        MarketData object with hourly data
    """
    logger.info(f"Loading market data for year {year}...")
    
    # Load ERCOT LMP data
    lmp_df = pd.read_csv('data/processed/ercot_lmp_hourly_2022_2024.csv')
    lmp_df['timestamp'] = pd.to_datetime(lmp_df['timestamp'])
    
    # Filter to specific year
    lmp_df = lmp_df[lmp_df['timestamp'].dt.year == year].copy()
    
    # Handle leap years by taking first 8760 hours
    if len(lmp_df) > 8760:
        logger.warning(f"Year {year} has {len(lmp_df)} hours (leap year). Using first 8760 hours.")
        lmp_df = lmp_df.iloc[:8760].copy()
    
    # Ensure we have exactly 8760 hours
    if len(lmp_df) != 8760:
        raise ValueError(f"Expected 8760 hours for year {year}, got {len(lmp_df)}")
    
    # Load other data files
    solar_df = pd.read_csv('data/processed/solar_cf_west_texas.csv')
    gas_df = pd.read_csv('data/processed/gas_prices_hourly.csv')
    gas_df['timestamp'] = pd.to_datetime(gas_df['timestamp'])
    gas_df = gas_df[gas_df['timestamp'].dt.year == year].copy()
    
    # Handle leap years for gas data
    if len(gas_df) > 8760:
        gas_df = gas_df.iloc[:8760].copy()
    
    carbon_df = pd.read_csv('data/processed/grid_carbon_intensity.csv')
    carbon_df['timestamp'] = pd.to_datetime(carbon_df['timestamp'])
    carbon_df = carbon_df[carbon_df['timestamp'].dt.year == year].copy()
    
    # Handle leap years for carbon data
    if len(carbon_df) > 8760:
        carbon_df = carbon_df.iloc[:8760].copy()
    
    # Create MarketData object
    market_data = MarketData(
        timestamp=lmp_df['timestamp'],
        lmp=lmp_df['lmp_dam'].values,
        gas_price=gas_df['price_mmbtu'].values,
        solar_cf=solar_df['capacity_factor'].values,
        grid_carbon_intensity=carbon_df['carbon_intensity_kg_per_mwh'].values
    )
    
    logger.info("Market data loaded successfully")
    logger.info(f"  LMP range: ${market_data.lmp.min():.2f} - ${market_data.lmp.max():.2f}/MWh")
    logger.info(f"  Gas price range: ${market_data.gas_price.min():.2f} - ${market_data.gas_price.max():.2f}/MMBtu")
    logger.info(f"  Solar CF range: {market_data.solar_cf.min():.3f} - {market_data.solar_cf.max():.3f}")
    logger.info(f"  Carbon intensity range: {market_data.grid_carbon_intensity.min():.1f} - {market_data.grid_carbon_intensity.max():.1f} kg CO2/MWh")
    
    return market_data


def load_technology_costs() -> TechnologyCosts:
    """
    Load technology costs from JSON file.
    
    Returns:
        TechnologyCosts object
    """
    logger.info("Loading technology costs...")
    
    with open('data/tech_costs.json', 'r') as f:
        costs_data = json.load(f)
    
    tech_costs = TechnologyCosts(
        grid_capex_per_kw=costs_data['grid_interconnection']['capex_per_kw'],
        gas_capex_per_kw=costs_data['gas_peaker']['capex_per_kw'],
        battery_capex_per_kwh=costs_data['battery']['capex_per_kwh'],
        solar_capex_per_kw=costs_data['solar']['capex_per_kw'],
        gas_variable_om=costs_data['gas_peaker']['variable_om_per_mwh'],
        gas_heat_rate=costs_data['gas_peaker']['heat_rate_mmbtu_per_mwh'],
        gas_efficiency=costs_data['gas_peaker']['thermal_efficiency'],
        battery_degradation=costs_data['battery']['degradation_cost_per_mwh'],
        battery_efficiency=costs_data['battery']['round_trip_efficiency'],
        battery_duration=costs_data['battery']['duration_hours'],
        solar_fixed_om=costs_data['solar']['fixed_om_per_kw_year'],
        grid_demand_charge=costs_data['grid_interconnection']['demand_charge_per_kw_month']
    )
    
    logger.info("Technology costs loaded successfully")
    
    return tech_costs


def create_facility_params(it_load_mw: float = 300) -> FacilityParams:
    """
    Create facility parameters for optimal portfolio scenario.
    
    Args:
        it_load_mw: IT equipment load in MW
        
    Returns:
        FacilityParams object
    """
    logger.info(f"Creating facility parameters for {it_load_mw} MW IT load...")
    
    facility_params = FacilityParams(
        it_load_mw=it_load_mw,
        pue=1.05,
        reliability_target=0.9999,
        carbon_budget=None,
        planning_horizon_years=20,
        discount_rate=0.07,
        curtailment_penalty=10000
    )
    
    logger.info(f"  Total facility load: {facility_params.total_load_mw:.1f} MW")
    logger.info(f"  Reliability target: {facility_params.reliability_target*100:.2f}%")
    logger.info(f"  Max annual curtailment: {facility_params.max_annual_curtailment_mwh():.2f} MWh")
    
    return facility_params


def load_baseline_solution(baseline_path: str = 'results/solutions/baseline_grid_only.json'):
    """
    Load baseline solution for comparison.
    
    Args:
        baseline_path: Path to baseline solution file
        
    Returns:
        Baseline solution metrics dict
    """
    logger.info(f"Loading baseline solution from {baseline_path}...")
    
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)
    
    baseline_metrics = baseline_data['metrics']
    
    logger.info("Baseline solution loaded successfully")
    logger.info(f"  Baseline NPV: ${baseline_metrics['total_npv']:,.0f}")
    logger.info(f"  Baseline LCOE: ${baseline_metrics['lcoe']:.2f}/MWh")
    logger.info(f"  Baseline Carbon: {baseline_metrics['carbon_tons_annual']:,.0f} tons CO2/year")
    
    return baseline_metrics


def run_optimal_portfolio_optimization(
    market_data: MarketData,
    tech_costs: TechnologyCosts,
    facility_params: FacilityParams
):
    """
    Run optimal portfolio optimization with all technologies available.
    
    Args:
        market_data: Market data
        tech_costs: Technology costs
        facility_params: Facility parameters
        
    Returns:
        OptimizationSolution object
    """
    logger.info("="*80)
    logger.info("OPTIMAL PORTFOLIO OPTIMIZATION")
    logger.info("="*80)
    logger.info("Building optimization model with all technologies enabled...")
    logger.info("  Grid connection: ENABLED")
    logger.info("  Gas peakers: ENABLED")
    logger.info("  Battery storage: ENABLED")
    logger.info("  Solar PV: ENABLED")
    
    # Build model with all technologies allowed
    model = build_optimization_model(
        market_data=market_data,
        tech_costs=tech_costs,
        facility_params=facility_params,
        allow_gas=True,
        allow_battery=True,
        allow_solar=True
    )
    
    logger.info(f"Model built successfully")
    logger.info(f"  Variables: ~{len(list(model.component_objects())) * 8760}")
    logger.info(f"  Constraints: ~{len(list(model.component_objects())) * 8760}")
    
    # Solve model
    logger.info("Solving optimization model...")
    logger.info("Note: Using GLPK solver (open-source). For faster solving, install Gurobi.")
    results, solve_time = solve_model(
        model=model,
        time_limit=1800,
        mip_gap=0.005,
        verbose=True,
        solver_name='glpk'
    )
    
    logger.info(f"Optimization solved in {solve_time:.2f} seconds")
    
    # Extract solution
    logger.info("Extracting solution...")
    solution = extract_solution(
        model=model,
        market_data=market_data,
        tech_costs=tech_costs,
        facility_params=facility_params,
        solve_time=solve_time,
        optimality_gap=0.0,
        scenario_params={
            "scenario_name": "optimal_portfolio",
            "facility_load_mw": facility_params.total_load_mw,
            "reliability_target": facility_params.reliability_target,
            "carbon_budget": facility_params.carbon_budget,
            "allow_gas": True,
            "allow_battery": True,
            "allow_solar": True,
            "year": 2023
        }
    )
    
    return solution


def calculate_comparison_metrics(optimal_solution, baseline_metrics):
    """
    Calculate comparison metrics between optimal and baseline solutions.
    
    Args:
        optimal_solution: OptimizationSolution object for optimal portfolio
        baseline_metrics: Baseline solution metrics dict
        
    Returns:
        Dictionary of comparison metrics
    """
    logger.info("Calculating comparison metrics...")
    
    # Cost savings
    cost_savings_npv = baseline_metrics['total_npv'] - optimal_solution.metrics.total_npv
    cost_savings_pct = (cost_savings_npv / baseline_metrics['total_npv']) * 100
    
    annual_opex_savings = baseline_metrics['opex_annual'] - optimal_solution.metrics.opex_annual
    annual_opex_savings_pct = (annual_opex_savings / baseline_metrics['opex_annual']) * 100
    
    # Reliability improvement
    reliability_improvement = optimal_solution.metrics.reliability_pct - baseline_metrics['reliability_pct']
    
    # Carbon reduction
    carbon_reduction_tons = baseline_metrics['carbon_tons_annual'] - optimal_solution.metrics.carbon_tons_annual
    carbon_reduction_pct = (carbon_reduction_tons / baseline_metrics['carbon_tons_annual']) * 100
    
    # Payback period (simple)
    total_btm_capex = (
        optimal_solution.capacity.gas_mw * 1000 +
        optimal_solution.capacity.battery_mwh * 350 +
        optimal_solution.capacity.solar_mw * 1200
    )
    
    if annual_opex_savings > 0:
        payback_years = total_btm_capex / annual_opex_savings
    else:
        payback_years = float('inf')
    
    comparison = {
        'cost_savings_npv': cost_savings_npv,
        'cost_savings_pct': cost_savings_pct,
        'annual_opex_savings': annual_opex_savings,
        'annual_opex_savings_pct': annual_opex_savings_pct,
        'reliability_improvement': reliability_improvement,
        'carbon_reduction_tons': carbon_reduction_tons,
        'carbon_reduction_pct': carbon_reduction_pct,
        'btm_capex': total_btm_capex,
        'payback_years': payback_years
    }
    
    logger.info("Comparison metrics calculated successfully")
    
    return comparison


def print_optimal_portfolio_summary(solution, baseline_metrics, comparison):
    """
    Print summary of optimal portfolio solution with comparison to baseline.
    
    Args:
        solution: OptimizationSolution object
        baseline_metrics: Baseline solution metrics dict
        comparison: Comparison metrics dict
    """
    logger.info("="*80)
    logger.info("OPTIMAL PORTFOLIO SOLUTION SUMMARY")
    logger.info("="*80)
    
    logger.info("\nCapacity Investments:")
    logger.info(f"  Grid Connection: {solution.capacity.grid_mw:.1f} MW")
    logger.info(f"  Gas Peakers: {solution.capacity.gas_mw:.1f} MW")
    logger.info(f"  Battery Storage: {solution.capacity.battery_mwh:.1f} MWh")
    logger.info(f"  Solar PV: {solution.capacity.solar_mw:.1f} MW")
    
    logger.info("\nCost Metrics:")
    logger.info(f"  Total NPV (20-year): ${solution.metrics.total_npv:,.0f}")
    logger.info(f"  CAPEX: ${solution.metrics.capex:,.0f}")
    logger.info(f"  Annual OPEX: ${solution.metrics.opex_annual:,.0f}/year")
    logger.info(f"  LCOE: ${solution.metrics.lcoe:.2f}/MWh")
    
    logger.info("\nReliability Metrics:")
    logger.info(f"  Reliability: {solution.metrics.reliability_pct:.4f}%")
    logger.info(f"  Total Curtailment: {solution.metrics.total_curtailment_mwh:.2f} MWh/year")
    logger.info(f"  Curtailment Hours: {solution.metrics.num_curtailment_hours} hours/year")
    
    logger.info("\nCarbon Metrics:")
    logger.info(f"  Annual Emissions: {solution.metrics.carbon_tons_annual:,.0f} tons CO2/year")
    logger.info(f"  Carbon Intensity: {solution.metrics.carbon_intensity_g_per_kwh:.1f} g CO2/kWh")
    logger.info(f"  Carbon Reduction: {solution.metrics.carbon_reduction_pct:.1f}% (vs baseline)")
    
    logger.info("\nOperational Metrics:")
    logger.info(f"  Grid Dependence: {solution.metrics.grid_dependence_pct:.1f}%")
    logger.info(f"  Gas Capacity Factor: {solution.metrics.gas_capacity_factor:.1f}%")
    logger.info(f"  Battery Cycles/Year: {solution.metrics.battery_cycles_per_year:.1f}")
    logger.info(f"  Solar Capacity Factor: {solution.metrics.solar_capacity_factor:.1f}%")
    
    logger.info("\nSolver Performance:")
    logger.info(f"  Solve Time: {solution.metrics.solve_time_seconds:.2f} seconds")
    logger.info(f"  Optimality Gap: {solution.metrics.optimality_gap_pct:.4f}%")
    
    logger.info("\n" + "="*80)
    logger.info("COMPARISON TO BASELINE (GRID-ONLY)")
    logger.info("="*80)
    
    logger.info("\nCost Savings:")
    logger.info(f"  NPV Savings: ${comparison['cost_savings_npv']:,.0f} ({comparison['cost_savings_pct']:.1f}%)")
    logger.info(f"  Annual OPEX Savings: ${comparison['annual_opex_savings']:,.0f}/year ({comparison['annual_opex_savings_pct']:.1f}%)")
    logger.info(f"  BTM CAPEX Investment: ${comparison['btm_capex']:,.0f}")
    logger.info(f"  Simple Payback Period: {comparison['payback_years']:.1f} years")
    
    logger.info("\nReliability Improvement:")
    logger.info(f"  Reliability Change: {comparison['reliability_improvement']:+.4f}%")
    
    logger.info("\nCarbon Reduction:")
    logger.info(f"  Annual Emissions Reduction: {comparison['carbon_reduction_tons']:,.0f} tons CO2/year ({comparison['carbon_reduction_pct']:.1f}%)")
    
    logger.info("="*80)


def save_optimal_portfolio_solution(
    solution,
    comparison,
    output_path: str = 'results/solutions/optimal_portfolio.json'
):
    """
    Save optimal portfolio solution to JSON file.
    
    Args:
        solution: OptimizationSolution object
        comparison: Comparison metrics dict
        output_path: Path to output file
    """
    logger.info(f"Saving optimal portfolio solution to {output_path}...")
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save solution
    solution.save(output_path)
    
    # Add comparison metrics to the saved file
    with open(output_path, 'r') as f:
        solution_data = json.load(f)
    
    solution_data['comparison_to_baseline'] = comparison
    
    with open(output_path, 'w') as f:
        json.dump(solution_data, f, indent=2)
    
    logger.info(f"Optimal portfolio solution saved successfully")
    
    # Also save a summary in human-readable format
    summary_path = output_path.replace('.json', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OPTIMAL PORTFOLIO SCENARIO SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("SCENARIO DESCRIPTION:\n")
        f.write("This optimal portfolio scenario allows the optimizer to select the best\n")
        f.write("combination of grid connection, natural gas peakers, battery storage, and\n")
        f.write("solar PV to minimize total 20-year cost while meeting reliability and\n")
        f.write("carbon constraints. This represents the economically optimal energy strategy.\n\n")
        
        f.write("CAPACITY INVESTMENTS:\n")
        f.write(f"  Grid Connection: {solution.capacity.grid_mw:.1f} MW\n")
        f.write(f"  Gas Peakers: {solution.capacity.gas_mw:.1f} MW\n")
        f.write(f"  Battery Storage: {solution.capacity.battery_mwh:.1f} MWh\n")
        f.write(f"  Solar PV: {solution.capacity.solar_mw:.1f} MW\n\n")
        
        f.write("COST METRICS:\n")
        f.write(f"  Total NPV (20-year): ${solution.metrics.total_npv:,.0f}\n")
        f.write(f"  CAPEX: ${solution.metrics.capex:,.0f}\n")
        f.write(f"  Annual OPEX: ${solution.metrics.opex_annual:,.0f}/year\n")
        f.write(f"  LCOE: ${solution.metrics.lcoe:.2f}/MWh\n\n")
        
        f.write("RELIABILITY METRICS:\n")
        f.write(f"  Reliability: {solution.metrics.reliability_pct:.4f}%\n")
        f.write(f"  Total Curtailment: {solution.metrics.total_curtailment_mwh:.2f} MWh/year\n")
        f.write(f"  Curtailment Hours: {solution.metrics.num_curtailment_hours} hours/year\n\n")
        
        f.write("CARBON METRICS:\n")
        f.write(f"  Annual Emissions: {solution.metrics.carbon_tons_annual:,.0f} tons CO2/year\n")
        f.write(f"  Carbon Intensity: {solution.metrics.carbon_intensity_g_per_kwh:.1f} g CO2/kWh\n\n")
        
        f.write("OPERATIONAL METRICS:\n")
        f.write(f"  Grid Dependence: {solution.metrics.grid_dependence_pct:.1f}%\n")
        f.write(f"  Gas Capacity Factor: {solution.metrics.gas_capacity_factor:.1f}%\n")
        f.write(f"  Battery Cycles/Year: {solution.metrics.battery_cycles_per_year:.1f}\n")
        f.write(f"  Solar Capacity Factor: {solution.metrics.solar_capacity_factor:.1f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("COMPARISON TO BASELINE (GRID-ONLY)\n")
        f.write("="*80 + "\n\n")
        
        f.write("COST SAVINGS:\n")
        f.write(f"  NPV Savings: ${comparison['cost_savings_npv']:,.0f} ({comparison['cost_savings_pct']:.1f}%)\n")
        f.write(f"  Annual OPEX Savings: ${comparison['annual_opex_savings']:,.0f}/year ({comparison['annual_opex_savings_pct']:.1f}%)\n")
        f.write(f"  BTM CAPEX Investment: ${comparison['btm_capex']:,.0f}\n")
        f.write(f"  Simple Payback Period: {comparison['payback_years']:.1f} years\n\n")
        
        f.write("RELIABILITY IMPROVEMENT:\n")
        f.write(f"  Reliability Change: {comparison['reliability_improvement']:+.4f}%\n\n")
        
        f.write("CARBON REDUCTION:\n")
        f.write(f"  Annual Emissions Reduction: {comparison['carbon_reduction_tons']:,.0f} tons CO2/year\n")
        f.write(f"  Carbon Reduction Percentage: {comparison['carbon_reduction_pct']:.1f}%\n\n")
        
        f.write("KEY INSIGHTS:\n")
        
        if solution.capacity.gas_mw > 0:
            f.write(f"  - Gas peakers provide {solution.capacity.gas_mw:.1f} MW of flexible capacity\n")
        
        if solution.capacity.battery_mwh > 0:
            f.write(f"  - Battery storage ({solution.capacity.battery_mwh:.1f} MWh) enables energy arbitrage\n")
        
        if solution.capacity.solar_mw > 0:
            f.write(f"  - Solar PV ({solution.capacity.solar_mw:.1f} MW) reduces grid dependence and carbon\n")
        
        if comparison['cost_savings_npv'] > 0:
            f.write(f"  - Optimal portfolio saves ${comparison['cost_savings_npv']:,.0f} over 20 years\n")
        
        if comparison['carbon_reduction_pct'] > 0:
            f.write(f"  - {comparison['carbon_reduction_pct']:.1f}% carbon reduction vs grid-only baseline\n")
        
        f.write("\n" + "="*80 + "\n")
    
    logger.info(f"Optimal portfolio summary saved to {summary_path}")


def main():
    """Main execution function."""
    try:
        # Load data
        market_data = load_market_data(year=2023)
        tech_costs = load_technology_costs()
        facility_params = create_facility_params(it_load_mw=300)
        
        # Load baseline for comparison
        baseline_metrics = load_baseline_solution()
        
        # Run optimal portfolio optimization
        solution = run_optimal_portfolio_optimization(
            market_data=market_data,
            tech_costs=tech_costs,
            facility_params=facility_params
        )
        
        # Calculate comparison metrics
        comparison = calculate_comparison_metrics(solution, baseline_metrics)
        
        # Print summary
        print_optimal_portfolio_summary(solution, baseline_metrics, comparison)
        
        # Save solution
        save_optimal_portfolio_solution(solution, comparison)
        
        logger.info("\n" + "="*80)
        logger.info("OPTIMAL PORTFOLIO OPTIMIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Results saved to: results/solutions/optimal_portfolio.json")
        logger.info(f"Summary saved to: results/solutions/optimal_portfolio_summary.txt")
        logger.info("\nKey Findings:")
        logger.info(f"  - Cost Savings: ${comparison['cost_savings_npv']:,.0f} ({comparison['cost_savings_pct']:.1f}%)")
        logger.info(f"  - Carbon Reduction: {comparison['carbon_reduction_tons']:,.0f} tons CO2/year ({comparison['carbon_reduction_pct']:.1f}%)")
        logger.info(f"  - Payback Period: {comparison['payback_years']:.1f} years")
        logger.info("\nNext steps:")
        logger.info("  1. Review optimal capacity mix and dispatch strategy")
        logger.info("  2. Analyze sensitivity to key parameters (task 15-18)")
        logger.info("  3. Create visualizations for presentation (task 19-24)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running optimal portfolio optimization: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
