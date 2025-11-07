"""
Test script for Pareto frontier calculator.

This script demonstrates the usage of the Pareto calculator with synthetic
solution data to verify functionality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pareto_calculator import (
    calculate_pareto_frontier,
    calculate_cost_reliability_frontier,
    calculate_cost_carbon_frontier,
    calculate_grid_reliability_frontier,
    calculate_all_pareto_frontiers,
    save_pareto_frontiers,
    load_pareto_frontiers,
    get_pareto_summary,
    identify_knee_point,
    ParetoCalculatorError
)


def create_synthetic_solutions():
    """Create synthetic solution data for testing."""
    solutions = []
    
    # Create a set of solutions with varying cost, reliability, and carbon
    # Some will be Pareto-optimal, others will be dominated
    
    scenarios = [
        # (NPV, Reliability, Carbon, Grid Dependence)
        (2000e6, 99.9, 150000, 80),    # Low cost, low reliability, high carbon
        (2200e6, 99.95, 140000, 75),   # Medium cost, medium reliability
        (2400e6, 99.99, 130000, 70),   # Higher cost, high reliability (Pareto)
        (2600e6, 99.999, 120000, 65),  # Highest cost, highest reliability (Pareto)
        (2300e6, 99.98, 135000, 72),   # Dominated by scenario 3
        (2100e6, 99.92, 145000, 78),   # Dominated
        (2500e6, 99.995, 125000, 68),  # Pareto-optimal
        (2350e6, 99.985, 132000, 71),  # Dominated
        (2150e6, 99.94, 142000, 76),   # Dominated
        (2450e6, 99.992, 128000, 69),  # Pareto-optimal
    ]
    
    for i, (npv, reliability, carbon, grid_dep) in enumerate(scenarios):
        solution = {
            'status': 'success',
            'scenario_index': i,
            'scenario_name': f'scenario_{i}',
            'scenario_params': {
                'gas_price_multiplier': 1.0,
                'lmp_multiplier': 1.0,
                'battery_cost_per_kwh': 350.0,
                'reliability_target': reliability / 100.0,
                'carbon_reduction_pct': None
            },
            'capacity': {
                'Grid Connection (MW)': 200.0,
                'Gas Peakers (MW)': 150.0 - i * 5,
                'Battery Storage (MWh)': 400.0 + i * 20,
                'Solar PV (MW)': 100.0 + i * 10
            },
            'metrics': {
                'total_npv': npv,
                'capex': 850e6,
                'opex_annual': 95e6,
                'lcoe': 42.5,
                'reliability_pct': reliability,
                'total_curtailment_mwh': (100 - reliability) * 28.5,
                'num_curtailment_hours': int((100 - reliability) * 10),
                'carbon_tons_annual': carbon,
                'carbon_intensity_g_per_kwh': carbon / 2500000,
                'carbon_reduction_pct': (200000 - carbon) / 200000 * 100,
                'grid_dependence_pct': grid_dep,
                'gas_capacity_factor': 0.3,
                'battery_cycles_per_year': 250,
                'solar_capacity_factor': 0.25,
                'solve_time_seconds': 120.0,
                'optimality_gap_pct': 0.5
            },
            'solve_time': 120.0
        }
        solutions.append(solution)
    
    return solutions


def test_basic_pareto_calculation():
    """Test basic Pareto frontier calculation."""
    print("\n" + "="*70)
    print("TEST 1: Basic Pareto Frontier Calculation")
    print("="*70)
    
    solutions = create_synthetic_solutions()
    print(f"Created {len(solutions)} synthetic solutions")
    
    # Calculate cost-reliability frontier
    try:
        pareto_df = calculate_pareto_frontier(
            solutions=solutions,
            objective1='total_npv',
            objective2='reliability_pct',
            minimize_obj1=True,
            minimize_obj2=False  # Maximize reliability
        )
        
        print(f"\nFound {len(pareto_df)} Pareto-optimal solutions")
        print("\nPareto-optimal solutions:")
        print(pareto_df[['scenario_name', 'total_npv', 'reliability_pct']].to_string(index=False))
        
        # Expected: scenarios 0, 3, 6, 9 should be Pareto-optimal
        # (lowest cost for each reliability level)
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_specialized_frontiers():
    """Test specialized frontier calculation functions."""
    print("\n" + "="*70)
    print("TEST 2: Specialized Frontier Functions")
    print("="*70)
    
    solutions = create_synthetic_solutions()
    
    try:
        # Cost-reliability frontier
        cost_rel_df = calculate_cost_reliability_frontier(solutions)
        print(f"\nCost-Reliability Frontier: {len(cost_rel_df)} solutions")
        
        # Cost-carbon frontier
        cost_carbon_df = calculate_cost_carbon_frontier(solutions)
        print(f"Cost-Carbon Frontier: {len(cost_carbon_df)} solutions")
        
        # Grid-reliability frontier
        grid_rel_df = calculate_grid_reliability_frontier(solutions)
        print(f"Grid-Reliability Frontier: {len(grid_rel_df)} solutions")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_all_frontiers():
    """Test calculating all frontiers at once."""
    print("\n" + "="*70)
    print("TEST 3: Calculate All Frontiers")
    print("="*70)
    
    solutions = create_synthetic_solutions()
    
    try:
        frontiers = calculate_all_pareto_frontiers(solutions)
        
        print(f"\nCalculated {len(frontiers)} Pareto frontiers:")
        for name, df in frontiers.items():
            print(f"  {name}: {len(df)} solutions")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_save_load():
    """Test saving and loading Pareto frontiers."""
    print("\n" + "="*70)
    print("TEST 4: Save and Load Frontiers")
    print("="*70)
    
    solutions = create_synthetic_solutions()
    
    try:
        # Calculate frontiers
        frontiers = calculate_all_pareto_frontiers(solutions)
        
        # Save to file
        output_path = "results/test_pareto_frontiers.json"
        save_pareto_frontiers(frontiers, output_path)
        print(f"\nSaved frontiers to {output_path}")
        
        # Load from file
        loaded_frontiers = load_pareto_frontiers(output_path)
        print(f"Loaded {len(loaded_frontiers)} frontiers")
        
        # Verify data matches
        for name in frontiers.keys():
            if name in loaded_frontiers:
                original_len = len(frontiers[name])
                loaded_len = len(loaded_frontiers[name])
                if original_len == loaded_len:
                    print(f"  {name}: OK ({loaded_len} solutions)")
                else:
                    print(f"  {name}: MISMATCH (original: {original_len}, loaded: {loaded_len})")
                    return False
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pareto_summary():
    """Test Pareto frontier summary statistics."""
    print("\n" + "="*70)
    print("TEST 5: Pareto Frontier Summary")
    print("="*70)
    
    solutions = create_synthetic_solutions()
    
    try:
        pareto_df = calculate_cost_reliability_frontier(solutions)
        
        summary = get_pareto_summary(
            pareto_df,
            objective1='total_npv',
            objective2='reliability_pct'
        )
        
        print(f"\nSummary for Cost-Reliability Frontier:")
        print(f"  Number of solutions: {summary['num_solutions']}")
        print(f"  NPV range: ${summary['objective1']['min']:,.0f} - ${summary['objective1']['max']:,.0f}")
        print(f"  Reliability range: {summary['objective2']['min']:.3f}% - {summary['objective2']['max']:.3f}%")
        
        print(f"\nExtreme points:")
        for point_name, point_data in summary['extreme_points'].items():
            print(f"  {point_name}: {point_data['scenario_name']}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_knee_point():
    """Test knee point identification."""
    print("\n" + "="*70)
    print("TEST 6: Knee Point Identification")
    print("="*70)
    
    solutions = create_synthetic_solutions()
    
    try:
        pareto_df = calculate_cost_reliability_frontier(solutions)
        
        knee_point = identify_knee_point(
            pareto_df,
            objective1='total_npv',
            objective2='reliability_pct'
        )
        
        print(f"\nKnee point identified:")
        print(f"  Scenario: {knee_point['scenario_name']}")
        print(f"  NPV: ${knee_point['total_npv']:,.0f}")
        print(f"  Reliability: {knee_point['reliability_pct']:.3f}%")
        print(f"  Distance from line: {knee_point['distance_from_line']:.4f}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "="*70)
    print("TEST 7: Error Handling")
    print("="*70)
    
    # Test with empty solutions
    try:
        calculate_pareto_frontier([], 'total_npv', 'reliability_pct')
        print("ERROR: Should have raised ParetoCalculatorError for empty solutions")
        return False
    except ParetoCalculatorError as e:
        print(f"Correctly raised error for empty solutions: {e}")
    
    # Test with invalid objective names
    solutions = create_synthetic_solutions()
    try:
        calculate_pareto_frontier(solutions, 'invalid_obj', 'reliability_pct')
        print("ERROR: Should have raised ParetoCalculatorError for invalid objective")
        return False
    except ParetoCalculatorError as e:
        print(f"Correctly raised error for invalid objective: {e}")
    
    # Test with all failed solutions
    failed_solutions = [
        {'status': 'failed', 'scenario_name': 'failed_1', 'error': 'Test error'}
    ]
    try:
        calculate_pareto_frontier(failed_solutions, 'total_npv', 'reliability_pct')
        print("ERROR: Should have raised ParetoCalculatorError for all failed solutions")
        return False
    except ParetoCalculatorError as e:
        print(f"Correctly raised error for all failed solutions: {e}")
    
    print("\nAll error handling tests passed")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PARETO FRONTIER CALCULATOR TEST SUITE")
    print("="*70)
    
    tests = [
        ("Basic Pareto Calculation", test_basic_pareto_calculation),
        ("Specialized Frontiers", test_specialized_frontiers),
        ("All Frontiers", test_all_frontiers),
        ("Save and Load", test_save_load),
        ("Pareto Summary", test_pareto_summary),
        ("Knee Point", test_knee_point),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nUNEXPECTED ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed successfully!")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
