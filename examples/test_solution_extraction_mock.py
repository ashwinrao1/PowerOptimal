"""
Test script for solution extraction and validation with mock data.

This script tests the solution extraction and validation modules without
requiring a solver, using pre-constructed solution data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.solution import (
    CapacitySolution,
    DispatchSolution,
    SolutionMetrics,
    OptimizationSolution
)
from src.models.market_data import MarketData
from src.models.technology import TechnologyCosts, FacilityParams
from src.optimization.validator import (
    validate_solution,
    generate_validation_report
)


def create_mock_solution() -> OptimizationSolution:
    """Create a mock solution for testing."""
    # Create capacity solution
    capacity = CapacitySolution(
        grid_mw=200.0,
        gas_mw=100.0,
        battery_mwh=400.0,  # 4-hour battery at 100 MW
        solar_mw=150.0
    )
    
    # Create dispatch solution (simplified - just 24 hours repeated)
    hours = 8760
    hour_array = np.arange(1, hours + 1)
    
    # Simple dispatch pattern
    load_mw = 285.0
    
    # Grid provides base load
    grid_power = np.full(hours, 150.0)
    
    # Gas provides some peaking
    gas_power = np.full(hours, 50.0)
    
    # Solar follows daily pattern
    daily_solar = np.maximum(0, np.sin(np.linspace(0, 2*np.pi, 24))) * 100
    solar_power = np.tile(daily_solar, hours // 24)
    
    # Battery balances (positive = charge, negative = discharge)
    # Charge during high solar, discharge at night
    daily_battery = -np.maximum(0, np.sin(np.linspace(0, 2*np.pi, 24))) * 20 + 10
    battery_power = np.tile(daily_battery, hours // 24)
    
    # Calculate curtailment to balance
    curtailment = load_mw - (grid_power + gas_power + solar_power - battery_power)
    curtailment = np.maximum(0, curtailment)  # Can't be negative
    
    # Adjust to ensure balance
    actual_supply = grid_power + gas_power + solar_power - battery_power + curtailment
    
    # Battery SOC (simplified - oscillates between 20% and 80%)
    daily_soc = 200 + 100 * np.sin(np.linspace(0, 2*np.pi, 24))
    battery_soc = np.tile(daily_soc, hours // 24)
    
    dispatch = DispatchSolution(
        hour=hour_array,
        grid_power=grid_power,
        gas_power=gas_power,
        solar_power=solar_power,
        battery_power=battery_power,
        curtailment=curtailment,
        battery_soc=battery_soc
    )
    
    # Create metrics
    metrics = SolutionMetrics(
        total_npv=2_500_000_000,
        capex=500_000_000,
        opex_annual=100_000_000,
        lcoe=45.0,
        reliability_pct=99.99,
        total_curtailment_mwh=2.85,
        num_curtailment_hours=10,
        carbon_tons_annual=500_000,
        carbon_intensity_g_per_kwh=250,
        carbon_reduction_pct=30.0,
        grid_dependence_pct=60.0,
        gas_capacity_factor=50.0,
        battery_cycles_per_year=300,
        solar_capacity_factor=25.0,
        solve_time_seconds=120.0,
        optimality_gap_pct=0.5
    )
    
    scenario_params = {
        "facility_load_mw": 285.0,
        "reliability_target": 0.9999,
        "carbon_budget": None
    }
    
    return OptimizationSolution(
        capacity=capacity,
        dispatch=dispatch,
        metrics=metrics,
        scenario_params=scenario_params
    )


def create_test_market_data() -> MarketData:
    """Create test market data."""
    hours = 8760
    timestamp = pd.date_range('2024-01-01', periods=hours, freq='h')
    
    # Simple patterns
    daily_lmp = 30 + 20 * np.sin(np.linspace(0, 2*np.pi, 24))
    lmp = np.tile(daily_lmp, hours // 24)
    
    gas_price = np.full(hours, 3.0)
    
    daily_solar = np.maximum(0, np.sin(np.linspace(0, 2*np.pi, 24)))
    solar_cf = np.tile(daily_solar, hours // 24)
    
    daily_carbon = 400 + 100 * np.sin(np.linspace(0, 2*np.pi, 24))
    grid_carbon = np.tile(daily_carbon, hours // 24)
    
    return MarketData(
        timestamp=timestamp,
        lmp=lmp,
        gas_price=gas_price,
        solar_cf=solar_cf,
        grid_carbon_intensity=grid_carbon
    )


def main():
    """Run test of solution validation with mock data."""
    print("=" * 80)
    print("Testing Solution Validation with Mock Data")
    print("=" * 80)
    
    # Create mock solution
    print("\n1. Creating mock solution...")
    solution = create_mock_solution()
    print(f"   Mock solution created")
    print(f"\n   Capacity:")
    print(f"     Grid:    {solution.capacity.grid_mw:.2f} MW")
    print(f"     Gas:     {solution.capacity.gas_mw:.2f} MW")
    print(f"     Battery: {solution.capacity.battery_mwh:.2f} MWh")
    print(f"     Solar:   {solution.capacity.solar_mw:.2f} MW")
    
    # Create parameters
    print("\n2. Creating facility and technology parameters...")
    facility_params = FacilityParams(
        it_load_mw=271.43,  # Results in 285 MW total with PUE 1.05
        pue=1.05,
        reliability_target=0.9999,
        planning_horizon_years=20,
        discount_rate=0.07,
        curtailment_penalty=10000
    )
    tech_costs = TechnologyCosts()
    print(f"   Parameters created")
    
    # Validate solution
    print("\n3. Validating solution...")
    try:
        is_valid, violations = validate_solution(
            solution=solution,
            facility_params=facility_params,
            tech_costs=tech_costs,
            tolerance=1e-3  # Relaxed tolerance for mock data
        )
        
        if is_valid:
            print(f"   ✓ Solution is VALID - all constraints satisfied")
        else:
            print(f"   ✗ Solution has issues - {len(violations)} violations found")
            print("\n   First 10 violations:")
            for i, violation in enumerate(violations[:10], 1):
                print(f"     {i}. {violation}")
            if len(violations) > 10:
                print(f"     ... and {len(violations) - 10} more")
        
    except Exception as e:
        print(f"   ERROR: Failed to validate solution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate validation report
    print("\n4. Generating validation report...")
    try:
        report = generate_validation_report(
            solution=solution,
            facility_params=facility_params,
            tech_costs=tech_costs,
            tolerance=1e-3
        )
        
        print(f"   Validation Report Generated:")
        print(f"     Valid:            {report['is_valid']}")
        print(f"     Violations:       {report['num_violations']}")
        print(f"     Tolerance:        {report['tolerance']}")
        
        stats = report['statistics']
        print(f"\n   Energy Balance Statistics:")
        print(f"     Max Error:        {stats['energy_balance']['max_error']:.6f} MW")
        print(f"     Mean Error:       {stats['energy_balance']['mean_error']:.6f} MW")
        print(f"     Hours with Error: {stats['energy_balance']['num_hours_with_error']}")
        
        print(f"\n   Curtailment Statistics:")
        print(f"     Total:            {stats['curtailment']['total_mwh']:.4f} MWh")
        print(f"     Max Allowed:      {stats['curtailment']['max_allowed_mwh']:.4f} MWh")
        print(f"     Hours:            {stats['curtailment']['num_hours']}")
        print(f"     Max Hourly:       {stats['curtailment']['max_hourly_mw']:.4f} MW")
        
        print(f"\n   Battery Statistics:")
        print(f"     Min SOC:          {stats['battery']['min_soc_mwh']:.4f} MWh")
        print(f"     Max SOC:          {stats['battery']['max_soc_mwh']:.4f} MWh")
        print(f"     Capacity:         {stats['battery']['capacity_mwh']:.4f} MWh")
        
    except Exception as e:
        print(f"   ERROR: Failed to generate validation report: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test solution save/load
    print("\n5. Testing solution save/load...")
    try:
        test_file = Path(__file__).parent.parent / "results" / "solutions" / "test_mock_solution.json"
        solution.save(str(test_file))
        print(f"   Solution saved to: {test_file}")
        
        loaded_solution = OptimizationSolution.load(str(test_file))
        print(f"   Solution loaded successfully")
        
        # Verify loaded data matches
        assert loaded_solution.capacity.grid_mw == solution.capacity.grid_mw
        assert loaded_solution.metrics.lcoe == solution.metrics.lcoe
        assert len(loaded_solution.dispatch.hour) == len(solution.dispatch.hour)
        print(f"   ✓ Loaded solution matches original")
        
    except Exception as e:
        print(f"   ERROR: Failed to save/load solution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test solution summary
    print("\n6. Testing solution summary...")
    try:
        summary = solution.to_summary_dict()
        print(f"   Summary generated:")
        print(f"     Capacity entries: {len(summary['capacity'])}")
        print(f"     Key metrics:      {len(summary['key_metrics'])}")
        print(f"     Scenario params:  {len(summary['scenario'])}")
        
    except Exception as e:
        print(f"   ERROR: Failed to generate summary: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("All validation tests completed!")
    print("=" * 80)
    print("\nNote: This test uses mock data. Some constraint violations are expected")
    print("since the mock data is simplified and not from an actual optimization solve.")


if __name__ == "__main__":
    main()
