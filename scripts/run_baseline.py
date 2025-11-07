"""
Run baseline grid-only optimization scenario.

This script runs an optimization with zero capacity for gas, battery, and solar,
allowing only grid connection to meet the full data center load. This establishes
a baseline for comparison with optimal portfolio solutions.
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
    Create facility parameters for baseline scenario.
    
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


def run_baseline_optimization(
    market_data: MarketData,
    tech_costs: TechnologyCosts,
    facility_params: FacilityParams
):
    """
    Run baseline grid-only optimization.
    
    Args:
        market_data: Market data
        tech_costs: Technology costs
        facility_params: Facility parameters
        
    Returns:
        OptimizationSolution object
    """
    logger.info("="*80)
    logger.info("BASELINE GRID-ONLY OPTIMIZATION")
    logger.info("="*80)
    logger.info("Building optimization model with grid-only configuration...")
    logger.info("  Gas capacity: DISABLED")
    logger.info("  Battery capacity: DISABLED")
    logger.info("  Solar capacity: DISABLED")
    
    # Build model with only grid allowed
    model = build_optimization_model(
        market_data=market_data,
        tech_costs=tech_costs,
        facility_params=facility_params,
        allow_gas=False,
        allow_battery=False,
        allow_solar=False
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
            "scenario_name": "baseline_grid_only",
            "facility_load_mw": facility_params.total_load_mw,
            "reliability_target": facility_params.reliability_target,
            "carbon_budget": facility_params.carbon_budget,
            "allow_gas": False,
            "allow_battery": False,
            "allow_solar": False,
            "year": 2023
        }
    )
    
    return solution


def print_baseline_summary(solution):
    """
    Print summary of baseline solution.
    
    Args:
        solution: OptimizationSolution object
    """
    logger.info("="*80)
    logger.info("BASELINE SOLUTION SUMMARY")
    logger.info("="*80)
    
    logger.info("\nCapacity Investments:")
    logger.info(f"  Grid Connection: {solution.capacity.grid_mw:.1f} MW")
    logger.info(f"  Gas Peakers: {solution.capacity.gas_mw:.1f} MW (disabled)")
    logger.info(f"  Battery Storage: {solution.capacity.battery_mwh:.1f} MWh (disabled)")
    logger.info(f"  Solar PV: {solution.capacity.solar_mw:.1f} MW (disabled)")
    
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
    
    logger.info("="*80)


def save_baseline_solution(solution, output_path: str = 'results/solutions/baseline_grid_only.json'):
    """
    Save baseline solution to JSON file.
    
    Args:
        solution: OptimizationSolution object
        output_path: Path to output file
    """
    logger.info(f"Saving baseline solution to {output_path}...")
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save solution
    solution.save(output_path)
    
    logger.info(f"Baseline solution saved successfully")
    
    # Also save a summary in human-readable format
    summary_path = output_path.replace('.json', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE GRID-ONLY SCENARIO SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("SCENARIO DESCRIPTION:\n")
        f.write("This baseline scenario represents a data center that relies entirely on\n")
        f.write("grid electricity with no behind-the-meter generation or storage. This\n")
        f.write("establishes a cost and carbon baseline for comparison with optimal\n")
        f.write("portfolio solutions that include gas peakers, battery storage, and solar PV.\n\n")
        
        f.write("CAPACITY INVESTMENTS:\n")
        f.write(f"  Grid Connection: {solution.capacity.grid_mw:.1f} MW\n")
        f.write(f"  Gas Peakers: {solution.capacity.gas_mw:.1f} MW (disabled)\n")
        f.write(f"  Battery Storage: {solution.capacity.battery_mwh:.1f} MWh (disabled)\n")
        f.write(f"  Solar PV: {solution.capacity.solar_mw:.1f} MW (disabled)\n\n")
        
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
        
        f.write("KEY INSIGHTS:\n")
        f.write(f"  - Grid-only approach requires {solution.capacity.grid_mw:.1f} MW interconnection\n")
        f.write(f"  - 100% grid dependence results in high carbon intensity\n")
        f.write(f"  - Annual electricity cost: ${solution.metrics.opex_annual:,.0f}\n")
        f.write(f"  - This baseline will be compared to optimal portfolio with BTM assets\n\n")
        
        f.write("="*80 + "\n")
    
    logger.info(f"Baseline summary saved to {summary_path}")


def main():
    """Main execution function."""
    try:
        # Load data
        market_data = load_market_data(year=2023)
        tech_costs = load_technology_costs()
        facility_params = create_facility_params(it_load_mw=300)
        
        # Run baseline optimization
        solution = run_baseline_optimization(
            market_data=market_data,
            tech_costs=tech_costs,
            facility_params=facility_params
        )
        
        # Print summary
        print_baseline_summary(solution)
        
        # Save solution
        save_baseline_solution(solution)
        
        logger.info("\n" + "="*80)
        logger.info("BASELINE OPTIMIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Results saved to: results/solutions/baseline_grid_only.json")
        logger.info(f"Summary saved to: results/solutions/baseline_grid_only_summary.txt")
        logger.info("\nNext steps:")
        logger.info("  1. Review baseline metrics in the summary file")
        logger.info("  2. Run optimal portfolio optimization (task 14)")
        logger.info("  3. Compare baseline vs optimal to quantify BTM asset value")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running baseline optimization: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
