"""
Test script for solution extraction and validation.

This script demonstrates how to use the solution extraction and validation
modules with a small test case.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.market_data import MarketData
from src.models.technology import TechnologyCosts, FacilityParams
from src.optimization import (
    build_optimization_model,
    solve_model,
    extract_solution,
    validate_solution,
    generate_validation_report
)


def create_test_data(hours: int = 8760) -> MarketData:
    """Create simple test market data."""
    # Create test data for full year (8760 hours)
    timestamp = pd.date_range('2024-01-01', periods=hours, freq='h')
    
    # Simple patterns that repeat daily
    hours_per_day = 24
    num_days = hours // hours_per_day
    
    # Create daily patterns and repeat
    daily_lmp = 30 + 20 * np.sin(np.linspace(0, 2*np.pi, hours_per_day))
    lmp = np.tile(daily_lmp, num_days)[:hours]  # Varies 10-50 $/MWh
    
    gas_price = np.full(hours, 3.0)  # Constant 3 $/MMBtu
    
    daily_solar = np.maximum(0, np.sin(np.linspace(0, 2*np.pi, hours_per_day)))
    solar_cf = np.tile(daily_solar, num_days)[:hours]  # Day/night pattern
    
    daily_carbon = 400 + 100 * np.sin(np.linspace(0, 2*np.pi, hours_per_day))
    grid_carbon = np.tile(daily_carbon, num_days)[:hours]  # Varies 300-500 kg/MWh
    
    return MarketData(
        timestamp=timestamp,
        lmp=lmp,
        gas_price=gas_price,
        solar_cf=solar_cf,
        grid_carbon_intensity=grid_carbon
    )


def main():
    """Run test of solution extraction and validation."""
    print("=" * 80)
    print("Testing Solution Extraction and Validation")
    print("=" * 80)
    
    # Create test data (full year)
    print("\n1. Creating test data (8760 hours)...")
    market_data = create_test_data(hours=8760)
    print(f"   Market data created: {len(market_data.timestamp)} hours")
    
    # Create parameters
    print("\n2. Creating facility and technology parameters...")
    facility_params = FacilityParams(
        it_load_mw=10,  # Small 10 MW facility for testing
        pue=1.05,
        reliability_target=0.99,  # 99% for testing (allows some curtailment)
        planning_horizon_years=20,
        discount_rate=0.07,
        curtailment_penalty=10000
    )
    print(f"   Facility load: {facility_params.total_load_mw:.2f} MW")
    print(f"   Reliability target: {facility_params.reliability_target*100:.2f}%")
    
    tech_costs = TechnologyCosts()
    print(f"   Technology costs loaded (defaults)")
    
    # Build optimization model
    print("\n3. Building optimization model...")
    model = build_optimization_model(
        market_data=market_data,
        tech_costs=tech_costs,
        facility_params=facility_params,
        allow_gas=True,
        allow_battery=True,
        allow_solar=True
    )
    print(f"   Model built successfully")
    print(f"   Variables: ~{len(list(model.component_data_objects(pyo.Var)))}")
    print(f"   Constraints: ~{len(list(model.component_data_objects(pyo.Constraint)))}")
    
    # Solve model
    print("\n4. Solving optimization model...")
    print("   (This may take a few minutes for 8760 hours...)")
    try:
        results, solve_time = solve_model(
            model=model,
            time_limit=600,  # 10 minutes max
            mip_gap=0.01,  # 1% gap for testing
            verbose=False
        )
        print(f"   Model solved in {solve_time:.2f} seconds")
        print(f"   Objective value: ${pyo.value(model.total_cost):,.0f}")
    except Exception as e:
        print(f"   ERROR: Failed to solve model: {e}")
        return
    
    # Extract solution
    print("\n5. Extracting solution...")
    try:
        solution = extract_solution(
            model=model,
            market_data=market_data,
            tech_costs=tech_costs,
            facility_params=facility_params,
            solve_time=solve_time,
            optimality_gap=0.01
        )
        print(f"   Solution extracted successfully")
        
        # Display capacity results
        print("\n   Optimal Capacity:")
        print(f"     Grid:    {solution.capacity.grid_mw:8.2f} MW")
        print(f"     Gas:     {solution.capacity.gas_mw:8.2f} MW")
        print(f"     Battery: {solution.capacity.battery_mwh:8.2f} MWh")
        print(f"     Solar:   {solution.capacity.solar_mw:8.2f} MW")
        
        # Display key metrics
        print("\n   Key Metrics:")
        print(f"     Total NPV:        ${solution.metrics.total_npv:,.0f}")
        print(f"     CAPEX:            ${solution.metrics.capex:,.0f}")
        print(f"     Annual OPEX:      ${solution.metrics.opex_annual:,.0f}")
        print(f"     LCOE:             ${solution.metrics.lcoe:.2f}/MWh")
        print(f"     Reliability:      {solution.metrics.reliability_pct:.4f}%")
        print(f"     Curtailment:      {solution.metrics.total_curtailment_mwh:.4f} MWh")
        print(f"     Carbon Intensity: {solution.metrics.carbon_intensity_g_per_kwh:.2f} g CO2/kWh")
        print(f"     Grid Dependence:  {solution.metrics.grid_dependence_pct:.2f}%")
        
    except Exception as e:
        print(f"   ERROR: Failed to extract solution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Validate solution
    print("\n6. Validating solution...")
    try:
        is_valid, violations = validate_solution(
            solution=solution,
            facility_params=facility_params,
            tech_costs=tech_costs,
            tolerance=1e-4
        )
        
        if is_valid:
            print(f"   ✓ Solution is VALID - all constraints satisfied")
        else:
            print(f"   ✗ Solution is INVALID - {len(violations)} violations found")
            print("\n   First 5 violations:")
            for i, violation in enumerate(violations[:5], 1):
                print(f"     {i}. {violation}")
            if len(violations) > 5:
                print(f"     ... and {len(violations) - 5} more")
        
    except Exception as e:
        print(f"   ERROR: Failed to validate solution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate validation report
    print("\n7. Generating validation report...")
    try:
        report = generate_validation_report(
            solution=solution,
            facility_params=facility_params,
            tech_costs=tech_costs,
            tolerance=1e-4
        )
        
        print(f"   Validation Report:")
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
        
        if solution.capacity.battery_mwh > 0:
            print(f"\n   Battery Statistics:")
            print(f"     Min SOC:          {stats['battery']['min_soc_mwh']:.4f} MWh")
            print(f"     Max SOC:          {stats['battery']['max_soc_mwh']:.4f} MWh")
            print(f"     Capacity:         {stats['battery']['capacity_mwh']:.4f} MWh")
        
    except Exception as e:
        print(f"   ERROR: Failed to generate validation report: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test saving and loading
    print("\n8. Testing solution save/load...")
    try:
        test_file = Path(__file__).parent.parent / "results" / "solutions" / "test_solution.json"
        solution.save(str(test_file))
        print(f"   Solution saved to: {test_file}")
        
        from src.models.solution import OptimizationSolution
        loaded_solution = OptimizationSolution.load(str(test_file))
        print(f"   Solution loaded successfully")
        print(f"   Loaded LCOE: ${loaded_solution.metrics.lcoe:.2f}/MWh")
        
    except Exception as e:
        print(f"   ERROR: Failed to save/load solution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Import pyomo here to avoid issues if not installed
    try:
        import pyomo.environ as pyo
    except ImportError:
        print("ERROR: Pyomo is not installed. Please install it with:")
        print("  pip install pyomo")
        sys.exit(1)
    
    main()
