"""
Test script for scenario generator functionality.

This script demonstrates the usage of the scenario generator module
and validates that it produces the expected parameter combinations.
"""

import sys
sys.path.insert(0, 'src')

from analysis.scenario_generator import (
    generate_scenarios,
    generate_gas_price_scenarios,
    generate_lmp_scenarios,
    generate_battery_cost_scenarios,
    generate_reliability_scenarios,
    generate_carbon_scenarios,
    generate_full_sensitivity_scenarios,
    generate_pareto_scenarios,
    get_scenario_summary
)


def test_basic_scenario_generation():
    """Test basic scenario generation with default parameters."""
    print("Test 1: Basic scenario generation")
    print("-" * 60)
    
    scenarios = generate_scenarios()
    print(f"Generated {len(scenarios)} scenario(s)")
    print(f"Base scenario: {scenarios[0]}")
    print()


def test_gas_price_variations():
    """Test gas price variation scenarios."""
    print("Test 2: Gas price variations (±50%)")
    print("-" * 60)
    
    scenarios = generate_gas_price_scenarios()
    print(f"Generated {len(scenarios)} scenarios")
    for s in scenarios:
        print(f"  {s['scenario_name']}: gas_mult={s['gas_price_multiplier']}")
    print()


def test_lmp_variations():
    """Test LMP variation scenarios."""
    print("Test 3: LMP variations (±30%)")
    print("-" * 60)
    
    scenarios = generate_lmp_scenarios()
    print(f"Generated {len(scenarios)} scenarios")
    for s in scenarios:
        print(f"  {s['scenario_name']}: lmp_mult={s['lmp_multiplier']}")
    print()


def test_battery_cost_variations():
    """Test battery cost variation scenarios."""
    print("Test 4: Battery cost variations ($200-500/kWh)")
    print("-" * 60)
    
    scenarios = generate_battery_cost_scenarios()
    print(f"Generated {len(scenarios)} scenarios")
    for s in scenarios:
        print(f"  {s['scenario_name']}: battery_cost=${s['battery_cost_per_kwh']}/kWh")
    print()


def test_reliability_variations():
    """Test reliability target variation scenarios."""
    print("Test 5: Reliability variations (99.9%, 99.99%, 99.999%)")
    print("-" * 60)
    
    scenarios = generate_reliability_scenarios()
    print(f"Generated {len(scenarios)} scenarios")
    for s in scenarios:
        print(f"  {s['scenario_name']}: reliability={s['reliability_target']*100:.3f}%")
    print()


def test_carbon_variations():
    """Test carbon constraint variation scenarios."""
    print("Test 6: Carbon constraint variations")
    print("-" * 60)
    
    scenarios = generate_carbon_scenarios()
    print(f"Generated {len(scenarios)} scenarios")
    for s in scenarios:
        carbon = s['carbon_reduction_pct']
        carbon_str = f"{carbon}% reduction" if carbon is not None else "no constraint"
        print(f"  {s['scenario_name']}: {carbon_str}")
    print()


def test_combined_variations():
    """Test combined parameter variations."""
    print("Test 7: Combined variations")
    print("-" * 60)
    
    scenarios = generate_scenarios(
        gas_price_variations=[0.5, 1.0, 1.5],
        reliability_variations=[0.9999, 0.99999]
    )
    print(f"Generated {len(scenarios)} scenarios (3 gas × 2 reliability = 6)")
    for s in scenarios[:6]:
        print(f"  {s['scenario_name']}: gas={s['gas_price_multiplier']}, "
              f"rel={s['reliability_target']*100:.3f}%")
    print()


def test_full_sensitivity():
    """Test full sensitivity analysis scenario generation."""
    print("Test 8: Full sensitivity analysis")
    print("-" * 60)
    
    scenarios = generate_full_sensitivity_scenarios()
    print(f"Generated {len(scenarios)} scenarios")
    print(f"Expected: 3×3×3×3×4 = 324 scenarios")
    
    summary = get_scenario_summary(scenarios)
    print(f"\nSummary:")
    print(f"  Total scenarios: {summary['total_scenarios']}")
    print(f"  Gas price variations: {summary['parameter_ranges']['gas_price_multiplier']['unique_values']}")
    print(f"  LMP variations: {summary['parameter_ranges']['lmp_multiplier']['unique_values']}")
    print(f"  Battery cost variations: {summary['parameter_ranges']['battery_cost_per_kwh']['unique_values']}")
    print(f"  Reliability variations: {summary['parameter_ranges']['reliability_target']['unique_values']}")
    print(f"  Carbon constraint scenarios: {summary['parameter_ranges']['carbon_reduction_pct']['scenarios_with_constraint']}")
    print()


def test_pareto_scenarios():
    """Test Pareto frontier scenario generation."""
    print("Test 9: Pareto frontier scenarios")
    print("-" * 60)
    
    # Cost vs. Reliability
    scenarios_cr = generate_pareto_scenarios(objective_pair='cost_reliability')
    print(f"Cost-Reliability: {len(scenarios_cr)} scenarios")
    
    # Cost vs. Carbon
    scenarios_cc = generate_pareto_scenarios(objective_pair='cost_carbon')
    print(f"Cost-Carbon: {len(scenarios_cc)} scenarios")
    
    # Grid vs. Reliability
    scenarios_gr = generate_pareto_scenarios(objective_pair='grid_reliability')
    print(f"Grid-Reliability: {len(scenarios_gr)} scenarios")
    print()


def test_custom_base_params():
    """Test scenario generation with custom base parameters."""
    print("Test 10: Custom base parameters")
    print("-" * 60)
    
    custom_base = {
        'facility_size_mw': 500,
        'location': 'west_texas',
        'year': 2024,
        'gas_price_multiplier': 1.0,
        'lmp_multiplier': 1.0,
        'battery_cost_per_kwh': 350.0,
        'reliability_target': 0.9999,
        'carbon_reduction_pct': None
    }
    
    scenarios = generate_scenarios(
        base_params=custom_base,
        gas_price_variations=[0.5, 1.5]
    )
    
    print(f"Generated {len(scenarios)} scenarios with custom base")
    for s in scenarios:
        print(f"  {s['scenario_name']}: facility={s['facility_size_mw']}MW, "
              f"location={s['location']}, gas_mult={s['gas_price_multiplier']}")
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("SCENARIO GENERATOR TEST SUITE")
    print("=" * 60)
    print()
    
    test_basic_scenario_generation()
    test_gas_price_variations()
    test_lmp_variations()
    test_battery_cost_variations()
    test_reliability_variations()
    test_carbon_variations()
    test_combined_variations()
    test_full_sensitivity()
    test_pareto_scenarios()
    test_custom_base_params()
    
    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == '__main__':
    main()
