"""
Run scenario analysis with parameter variations.

This script executes multiple optimization scenarios with varied parameters
to analyze sensitivity and trade-offs between cost, reliability, and carbon.
"""

import sys
from pathlib import Path
import logging
import argparse
import json
from datetime import datetime
from typing import List, Dict, Any
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.models.market_data import MarketData
from src.models.technology import TechnologyCosts, FacilityParams
from src.optimization.model_builder import build_optimization_model
from src.optimization.solver import solve_model
from src.optimization.solution_extractor import extract_solution
from src.analysis.scenario_generator import generate_scenarios
from src.analysis.pareto_calculator import calculate_pareto_frontier
from src.analysis.sensitivity_analyzer import analyze_sensitivity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_market_data(year: int = 2023) -> MarketData:
    """Load market data from processed CSV files."""
    logger.info(f"Loading market data for year {year}...")
    
    lmp_df = pd.read_csv('data/processed/ercot_lmp_hourly_2022_2024.csv')
    lmp_df['timestamp'] = pd.to_datetime(lmp_df['timestamp'])
    lmp_df = lmp_df[lmp_df['timestamp'].dt.year == year].copy()
    
    if len(lmp_df) > 8760:
        logger.warning(f"Year {year} has {len(lmp_df)} hours (leap year). Using first 8760 hours.")
        lmp_df = lmp_df.iloc[:8760].copy()
    
    if len(lmp_df) != 8760:
        raise ValueError(f"Expected 8760 hours for year {year}, got {len(lmp_df)}")
    
    solar_df = pd.read_csv('data/processed/solar_cf_west_texas.csv')
    gas_df = pd.read_csv('data/processed/gas_prices_hourly.csv')
    gas_df['timestamp'] = pd.to_datetime(gas_df['timestamp'])
    gas_df = gas_df[gas_df['timestamp'].dt.year == year].copy()
    
    if len(gas_df) > 8760:
        gas_df = gas_df.iloc[:8760].copy()
    
    carbon_df = pd.read_csv('data/processed/grid_carbon_intensity.csv')
    carbon_df['timestamp'] = pd.to_datetime(carbon_df['timestamp'])
    carbon_df = carbon_df[carbon_df['timestamp'].dt.year == year].copy()
    
    if len(carbon_df) > 8760:
        carbon_df = carbon_df.iloc[:8760].copy()
    
    market_data = MarketData(
        timestamp=lmp_df['timestamp'],
        lmp=lmp_df['lmp_dam'].values,
        gas_price=gas_df['price_mmbtu'].values,
        solar_cf=solar_df['capacity_factor'].values,
        grid_carbon_intensity=carbon_df['carbon_intensity_kg_per_mwh'].values
    )
    
    logger.info("Market data loaded successfully")
    return market_data


def load_technology_costs() -> TechnologyCosts:
    """Load technology costs from JSON file."""
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


def run_single_scenario(
    scenario_params: Dict[str, Any],
    market_data: MarketData,
    base_tech_costs: TechnologyCosts,
    base_facility_params: FacilityParams
) -> Dict[str, Any]:
    """
    Run optimization for a single scenario.
    
    Args:
        scenario_params: Dictionary of scenario parameters
        market_data: Market data
        base_tech_costs: Base technology costs
        base_facility_params: Base facility parameters
        
    Returns:
        Dictionary with scenario results
    """
    scenario_name = scenario_params.get('name', 'unnamed')
    logger.info(f"Running scenario: {scenario_name}")
    
    try:
        tech_costs = TechnologyCosts(
            grid_capex_per_kw=base_tech_costs.grid_capex_per_kw,
            gas_capex_per_kw=base_tech_costs.gas_capex_per_kw,
            battery_capex_per_kwh=scenario_params.get('battery_cost', base_tech_costs.battery_capex_per_kwh),
            solar_capex_per_kw=base_tech_costs.solar_capex_per_kw,
            gas_variable_om=base_tech_costs.gas_variable_om,
            gas_heat_rate=base_tech_costs.gas_heat_rate,
            gas_efficiency=base_tech_costs.gas_efficiency,
            battery_degradation=base_tech_costs.battery_degradation,
            battery_efficiency=base_tech_costs.battery_efficiency,
            battery_duration=base_tech_costs.battery_duration,
            solar_fixed_om=base_tech_costs.solar_fixed_om,
            grid_demand_charge=base_tech_costs.grid_demand_charge
        )
        
        adjusted_market_data = MarketData(
            timestamp=market_data.timestamp,
            lmp=market_data.lmp * scenario_params.get('lmp_multiplier', 1.0),
            gas_price=market_data.gas_price * scenario_params.get('gas_price_multiplier', 1.0),
            solar_cf=market_data.solar_cf,
            grid_carbon_intensity=market_data.grid_carbon_intensity
        )
        
        facility_params = FacilityParams(
            it_load_mw=base_facility_params.it_load_mw,
            pue=base_facility_params.pue,
            reliability_target=scenario_params.get('reliability_target', base_facility_params.reliability_target),
            carbon_budget=scenario_params.get('carbon_budget', base_facility_params.carbon_budget),
            planning_horizon_years=base_facility_params.planning_horizon_years,
            discount_rate=base_facility_params.discount_rate,
            curtailment_penalty=base_facility_params.curtailment_penalty
        )
        
        model = build_optimization_model(
            market_data=adjusted_market_data,
            tech_costs=tech_costs,
            facility_params=facility_params,
            allow_gas=True,
            allow_battery=True,
            allow_solar=True
        )
        
        results, solve_time = solve_model(
            model=model,
            time_limit=1800,
            mip_gap=0.01,
            verbose=False,
            solver_name='glpk'
        )
        
        solution = extract_solution(
            model=model,
            market_data=adjusted_market_data,
            tech_costs=tech_costs,
            facility_params=facility_params,
            solve_time=solve_time,
            optimality_gap=0.0,
            scenario_params=scenario_params
        )
        
        result = {
            'scenario_name': scenario_name,
            'scenario_params': scenario_params,
            'capacity': {
                'grid_mw': solution.capacity.grid_mw,
                'gas_mw': solution.capacity.gas_mw,
                'battery_mwh': solution.capacity.battery_mwh,
                'solar_mw': solution.capacity.solar_mw
            },
            'metrics': {
                'total_npv': solution.metrics.total_npv,
                'capex': solution.metrics.capex,
                'opex_annual': solution.metrics.opex_annual,
                'lcoe': solution.metrics.lcoe,
                'reliability_pct': solution.metrics.reliability_pct,
                'carbon_tons_annual': solution.metrics.carbon_tons_annual,
                'carbon_intensity_g_per_kwh': solution.metrics.carbon_intensity_g_per_kwh,
                'grid_dependence_pct': solution.metrics.grid_dependence_pct,
                'solve_time_seconds': solution.metrics.solve_time_seconds
            },
            'status': 'success'
        }
        
        logger.info(f"Scenario {scenario_name} completed successfully")
        logger.info(f"  NPV: ${result['metrics']['total_npv']:,.0f}")
        logger.info(f"  Reliability: {result['metrics']['reliability_pct']:.4f}%")
        logger.info(f"  Carbon: {result['metrics']['carbon_tons_annual']:,.0f} tons/year")
        
        return result
        
    except Exception as e:
        logger.error(f"Scenario {scenario_name} failed: {e}", exc_info=True)
        return {
            'scenario_name': scenario_name,
            'scenario_params': scenario_params,
            'status': 'failed',
            'error': str(e)
        }


def run_gas_price_sensitivity(
    market_data: MarketData,
    tech_costs: TechnologyCosts,
    facility_params: FacilityParams
) -> List[Dict[str, Any]]:
    """Run scenarios with varied gas prices."""
    logger.info("="*80)
    logger.info("GAS PRICE SENSITIVITY ANALYSIS")
    logger.info("="*80)
    
    multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    scenarios = []
    
    for mult in multipliers:
        scenario = {
            'name': f'gas_price_{int(mult*100)}pct',
            'gas_price_multiplier': mult,
            'lmp_multiplier': 1.0,
            'battery_cost': tech_costs.battery_capex_per_kwh,
            'reliability_target': facility_params.reliability_target,
            'carbon_budget': None
        }
        scenarios.append(scenario)
    
    results = []
    for scenario in scenarios:
        result = run_single_scenario(scenario, market_data, tech_costs, facility_params)
        results.append(result)
    
    return results


def run_lmp_sensitivity(
    market_data: MarketData,
    tech_costs: TechnologyCosts,
    facility_params: FacilityParams
) -> List[Dict[str, Any]]:
    """Run scenarios with varied LMP prices."""
    logger.info("="*80)
    logger.info("LMP SENSITIVITY ANALYSIS")
    logger.info("="*80)
    
    multipliers = [0.7, 0.85, 1.0, 1.15, 1.3]
    scenarios = []
    
    for mult in multipliers:
        scenario = {
            'name': f'lmp_{int(mult*100)}pct',
            'gas_price_multiplier': 1.0,
            'lmp_multiplier': mult,
            'battery_cost': tech_costs.battery_capex_per_kwh,
            'reliability_target': facility_params.reliability_target,
            'carbon_budget': None
        }
        scenarios.append(scenario)
    
    results = []
    for scenario in scenarios:
        result = run_single_scenario(scenario, market_data, tech_costs, facility_params)
        results.append(result)
    
    return results


def run_battery_cost_sensitivity(
    market_data: MarketData,
    tech_costs: TechnologyCosts,
    facility_params: FacilityParams
) -> List[Dict[str, Any]]:
    """Run scenarios with varied battery costs."""
    logger.info("="*80)
    logger.info("BATTERY COST SENSITIVITY ANALYSIS")
    logger.info("="*80)
    
    battery_costs = [200, 275, 350, 425, 500]
    scenarios = []
    
    for cost in battery_costs:
        scenario = {
            'name': f'battery_{cost}usd_per_kwh',
            'gas_price_multiplier': 1.0,
            'lmp_multiplier': 1.0,
            'battery_cost': cost,
            'reliability_target': facility_params.reliability_target,
            'carbon_budget': None
        }
        scenarios.append(scenario)
    
    results = []
    for scenario in scenarios:
        result = run_single_scenario(scenario, market_data, tech_costs, facility_params)
        results.append(result)
    
    return results


def run_reliability_sensitivity(
    market_data: MarketData,
    tech_costs: TechnologyCosts,
    facility_params: FacilityParams
) -> List[Dict[str, Any]]:
    """Run scenarios with varied reliability targets."""
    logger.info("="*80)
    logger.info("RELIABILITY TARGET SENSITIVITY ANALYSIS")
    logger.info("="*80)
    
    reliability_targets = [0.999, 0.9999, 0.99999]
    scenarios = []
    
    for target in reliability_targets:
        scenario = {
            'name': f'reliability_{target*100:.3f}pct',
            'gas_price_multiplier': 1.0,
            'lmp_multiplier': 1.0,
            'battery_cost': tech_costs.battery_capex_per_kwh,
            'reliability_target': target,
            'carbon_budget': None
        }
        scenarios.append(scenario)
    
    results = []
    for scenario in scenarios:
        result = run_single_scenario(scenario, market_data, tech_costs, facility_params)
        results.append(result)
    
    return results


def save_scenario_results(results: List[Dict[str, Any]], output_path: str):
    """Save scenario results to JSON file."""
    logger.info(f"Saving scenario results to {output_path}...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'total_scenarios': len(results),
        'successful_scenarios': sum(1 for r in results if r['status'] == 'success'),
        'failed_scenarios': sum(1 for r in results if r['status'] == 'failed'),
        'scenarios': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Scenario results saved successfully")


def print_scenario_summary(results: List[Dict[str, Any]]):
    """Print summary of scenario results."""
    logger.info("\n" + "="*80)
    logger.info("SCENARIO ANALYSIS SUMMARY")
    logger.info("="*80)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    logger.info(f"\nTotal scenarios: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if successful:
        logger.info("\nSuccessful Scenarios:")
        logger.info(f"{'Scenario':<40} {'NPV ($M)':<15} {'Reliability (%)':<18} {'Carbon (tons/yr)':<20}")
        logger.info("-"*93)
        
        for result in successful:
            name = result['scenario_name']
            npv = result['metrics']['total_npv'] / 1e6
            reliability = result['metrics']['reliability_pct']
            carbon = result['metrics']['carbon_tons_annual']
            logger.info(f"{name:<40} {npv:>13.2f}  {reliability:>16.4f}  {carbon:>18,.0f}")
    
    if failed:
        logger.info("\nFailed Scenarios:")
        for result in failed:
            logger.info(f"  - {result['scenario_name']}: {result.get('error', 'Unknown error')}")
    
    logger.info("="*80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run scenario analysis for datacenter energy optimization"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        choices=[2022, 2023, 2024],
        help="Year for market data (default: 2023)"
    )
    parser.add_argument(
        "--facility-size",
        type=float,
        default=300.0,
        help="IT load in MW (default: 300)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scenarios",
        help="Output directory for scenario results (default: results/scenarios)"
    )
    parser.add_argument(
        "--analysis-type",
        type=str,
        choices=['all', 'gas', 'lmp', 'battery', 'reliability'],
        default='all',
        help="Type of sensitivity analysis to run (default: all)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("="*80)
        logger.info("SCENARIO ANALYSIS SCRIPT")
        logger.info("="*80)
        logger.info(f"Year: {args.year}")
        logger.info(f"Facility size: {args.facility_size} MW IT load")
        logger.info(f"Analysis type: {args.analysis_type}")
        logger.info("="*80)
        
        market_data = load_market_data(year=args.year)
        tech_costs = load_technology_costs()
        
        facility_params = FacilityParams(
            it_load_mw=args.facility_size,
            pue=1.05,
            reliability_target=0.9999,
            carbon_budget=None,
            planning_horizon_years=20,
            discount_rate=0.07,
            curtailment_penalty=10000
        )
        
        all_results = []
        
        if args.analysis_type in ['all', 'gas']:
            gas_results = run_gas_price_sensitivity(market_data, tech_costs, facility_params)
            all_results.extend(gas_results)
        
        if args.analysis_type in ['all', 'lmp']:
            lmp_results = run_lmp_sensitivity(market_data, tech_costs, facility_params)
            all_results.extend(lmp_results)
        
        if args.analysis_type in ['all', 'battery']:
            battery_results = run_battery_cost_sensitivity(market_data, tech_costs, facility_params)
            all_results.extend(battery_results)
        
        if args.analysis_type in ['all', 'reliability']:
            reliability_results = run_reliability_sensitivity(market_data, tech_costs, facility_params)
            all_results.extend(reliability_results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output_dir) / f"scenario_results_{timestamp}.json"
        save_scenario_results(all_results, str(output_path))
        
        print_scenario_summary(all_results)
        
        logger.info(f"\nScenario analysis complete!")
        logger.info(f"Results saved to: {output_path}")
        logger.info("\nNext steps:")
        logger.info("  1. Review scenario results in the JSON file")
        logger.info("  2. Generate Pareto frontier plots")
        logger.info("  3. Create sensitivity tornado charts")
        logger.info("  4. Analyze trade-offs between cost, reliability, and carbon")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running scenario analysis: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
