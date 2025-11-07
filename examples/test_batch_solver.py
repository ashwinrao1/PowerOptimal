"""
Test script for batch solver functionality.

This script demonstrates the batch solver with a small set of test scenarios.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.scenario_generator import generate_scenarios
from src.analysis.batch_solver import (
    solve_scenarios,
    save_scenario_results_csv,
    get_batch_summary,
    get_successful_scenarios,
    get_failed_scenarios
)
from src.models.market_data import MarketData
from src.models.technology import TechnologyCosts, FacilityParams
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_test_market_data(hours: int = 168) -> MarketData:
    """Create synthetic market data for testing (1 week)."""
    timestamps = pd.date_range('2024-01-01', periods=hours, freq='H')
    
    # Synthetic LMP with daily pattern
    lmp = 30 + 20 * np.sin(np.arange(hours) * 2 * np.pi / 24) + np.random.randn(hours) * 5
    lmp = np.maximum(lmp, 10)  # Floor at $10/MWh
    
    # Synthetic gas prices
    gas_price = 3.0 + 0.5 * np.sin(np.arange(hours) * 2 * np.pi / 24) + np.random.randn(hours) * 0.2
    gas_price = np.maximum(gas_price, 2.0)  # Floor at $2/MMBtu
    
    # Synthetic solar capacity factors
    solar_cf = np.zeros(hours)
    for h in range(hours):
        hour_of_day = h % 24
        if 6 <= hour_of_day <= 18:
            solar_cf[h] = 0.3 + 0.4 * np.sin((hour_of_day - 6) * np.pi / 12)
    
    # Synthetic grid carbon intensity
    grid_carbon = 400 + 100 * np.sin(np.arange(hours) * 2 * np.pi / 24)
    
    return MarketData(
        timestamp=timestamps,
        lmp=lmp,
        gas_price=gas_price,
        solar_cf=solar_cf,
        grid_carbon_intensity=grid_carbon
    )


def main():
    """Run batch solver test."""
    logger.info("Starting batch solver test")
    
    # Create test data (1 week for faster testing)
    logger.info("Creating test market data (1 week)")
    market_data = create_test_market_data(hours=168)
    
    # Create technology costs
    tech_costs = TechnologyCosts()
    
    # Create facility parameters (smaller load for testing)
    facility_params = FacilityParams(
        it_load_mw=50,  # 50 MW for testing
        pue=1.05,
        reliability_target=0.999,  # Lower reliability for testing
        planning_horizon_years=20,
        discount_rate=0.07
    )
    
    # Generate test scenarios (small set)
    logger.info("Generating test scenarios")
    scenarios = generate_scenarios(
        gas_price_variations=[0.8, 1.0, 1.2],
        reliability_variations=[0.99, 0.999]
    )
    
    logger.info(f"Generated {len(scenarios)} scenarios")
    for i, scenario in enumerate(scenarios):
        logger.info(f"  Scenario {i+1}: {scenario['scenario_name']}")
    
    # Solve scenarios in batch
    logger.info("Solving scenarios in batch (this may take a few minutes)...")
    try:
        results = solve_scenarios(
            scenarios=scenarios,
            market_data=market_data,
            base_tech_costs=tech_costs,
            base_facility_params=facility_params,
            n_workers=2,  # Use 2 workers for testing
            save_solutions=True,
            output_dir="results/scenarios/test",
            verbose=True
        )
        
        # Get summary
        summary = get_batch_summary(results)
        logger.info("\nBatch Solve Summary:")
        logger.info(f"  Total scenarios: {summary['total_scenarios']}")
        logger.info(f"  Successful: {summary['successful']}")
        logger.info(f"  Failed: {summary['failed']}")
        logger.info(f"  Success rate: {summary['success_rate']:.1%}")
        
        if 'solve_time' in summary:
            logger.info(f"  Total solve time: {summary['solve_time']['total']:.1f}s")
            logger.info(f"  Average solve time: {summary['solve_time']['mean']:.1f}s")
        
        # Show successful results
        successful = get_successful_scenarios(results)
        if successful:
            logger.info("\nSuccessful Scenarios:")
            for result in successful:
                logger.info(f"  {result['scenario_name']}:")
                logger.info(f"    NPV: ${result['metrics']['total_npv']:,.0f}")
                logger.info(f"    LCOE: ${result['metrics']['lcoe']:.2f}/MWh")
                logger.info(f"    Reliability: {result['metrics']['reliability_pct']:.2f}%")
        
        # Show failed results
        failed = get_failed_scenarios(results)
        if failed:
            logger.warning("\nFailed Scenarios:")
            for result in failed:
                logger.warning(f"  {result['scenario_name']}: {result['error']}")
        
        # Save results to CSV
        logger.info("\nSaving results to CSV...")
        save_scenario_results_csv(results, "results/scenario_results_test.csv")
        logger.info("Results saved to results/scenario_results_test.csv")
        
        logger.info("\nBatch solver test completed successfully!")
        
    except Exception as e:
        logger.error(f"Batch solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
