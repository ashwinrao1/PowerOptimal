"""
Test script for sensitivity analyzer functionality.

This script demonstrates how to use the sensitivity analyzer to:
1. Analyze sensitivity to individual parameters
2. Rank parameters by impact
3. Identify breakeven points
4. Generate sensitivity metrics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.sensitivity_analyzer import (
    analyze_sensitivity,
    analyze_multiple_parameters,
    rank_parameters_by_impact,
    generate_sensitivity_metrics,
    save_sensitivity_results,
    create_sensitivity_dataframe,
    identify_critical_parameters,
    compare_parameter_impacts
)


def create_mock_solution(
    scenario_name: str,
    gas_mult: float = 1.0,
    lmp_mult: float = 1.0,
    battery_cost: float = 350.0,
    npv: float = 2_500_000_000,
    grid_mw: float = 200,
    gas_mw: float = 150,
    battery_mwh: float = 400,
    solar_mw: float = 100
) -> dict:
    """Create a mock solution for testing."""
    return {
        'status': 'success',
        'scenario_index': 0,
        'scenario_name': scenario_name,
        'scenario_params': {
            'gas_price_multiplier': gas_mult,
            'lmp_multiplier': lmp_mult,
            'battery_cost_per_kwh': battery_cost,
            'reliability_target': 0.9999,
            'carbon_reduction_pct': None
        },
        'capacity': {
            'Grid Connection (MW)': grid_mw,
            'Gas Peakers (MW)': gas_mw,
            'Battery Storage (MWh)': battery_mwh,
            'Solar PV (MW)': solar_mw
        },
        'metrics': {
            'total_npv': npv,
            'capex': 500_000_000,
            'opex_annual': 150_000_000,
            'lcoe': 65.0,
            'reliability_pct': 99.99,
            'total_curtailment_mwh': 2.5,
            'num_curtailment_hours': 10,
            'carbon_tons_annual': 500_000,
            'carbon_intensity_g_per_kwh': 200,
            'carbon_reduction_pct': 30,
            'grid_dependence_pct': 60,
            'gas_capacity_factor': 25,
            'battery_cycles_per_year': 250,
            'solar_capacity_factor': 28,
            'solve_time_seconds': 120,
            'optimality_gap_pct': 0.5
        },
        'solve_time': 120
    }


def test_gas_price_sensitivity():
    """Test sensitivity analysis for gas price variations."""
    print("\n" + "="*80)
    print("TEST 1: Gas Price Sensitivity Analysis")
    print("="*80)
    
    # Create base solution
    base_solution = create_mock_solution(
        'base',
        gas_mult=1.0,
        npv=2_500_000_000,
        gas_mw=150
    )
    
    # Create varied solutions with different gas prices
    varied_solutions = [
        create_mock_solution('gas_50', gas_mult=0.5, npv=2_300_000_000, gas_mw=200),
        create_mock_solution('gas_75', gas_mult=0.75, npv=2_400_000_000, gas_mw=175),
        create_mock_solution('gas_100', gas_mult=1.0, npv=2_500_000_000, gas_mw=150),
        create_mock_solution('gas_125', gas_mult=1.25, npv=2_600_000_000, gas_mw=125),
        create_mock_solution('gas_150', gas_mult=1.5, npv=2_700_000_000, gas_mw=100),
    ]
    
    # Analyze sensitivity
    sensitivity = analyze_sensitivity(
        base_solution=base_solution,
        varied_solutions=varied_solutions,
        parameter_name='gas_price_multiplier',
        metric='total_npv'
    )
    
    print(f"\nParameter: {sensitivity['parameter_name']}")
    print(f"Metric: {sensitivity['metric']}")
    print(f"Base parameter value: {sensitivity['base_parameter_value']}")
    print(f"Base metric value: ${sensitivity['base_metric_value']:,.0f}")
    print(f"\nElasticity: {sensitivity['elasticity']:.3f}")
    print(f"  (% change in NPV per % change in gas price)")
    print(f"\nImpact Score: {sensitivity['impact_score']:.2f}")
    print(f"  (Range of NPV as % of base NPV)")
    print(f"\nRegression:")
    print(f"  Slope: {sensitivity['regression']['slope']:,.0f}")
    print(f"  Intercept: {sensitivity['regression']['intercept']:,.0f}")
    print(f"  R-squared: {sensitivity['regression']['r_squared']:.4f}")
    print(f"\nBreakeven Points: {len(sensitivity['breakeven_points'])}")
    for i, bp in enumerate(sensitivity['breakeven_points'], 1):
        print(f"  {i}. At gas price multiplier ~{bp['parameter_value']:.2f}")
        print(f"     {bp['description']}")
    
    # Create DataFrame for plotting
    df = create_sensitivity_dataframe(sensitivity)
    print(f"\nSensitivity DataFrame shape: {df.shape}")
    print(df.head())
    
    return sensitivity


def test_multiple_parameters():
    """Test sensitivity analysis for multiple parameters."""
    print("\n" + "="*80)
    print("TEST 2: Multiple Parameter Sensitivity Analysis")
    print("="*80)
    
    # Create base solution
    base_solution = create_mock_solution(
        'base',
        gas_mult=1.0,
        lmp_mult=1.0,
        battery_cost=350.0,
        npv=2_500_000_000
    )
    
    # Create scenarios varying different parameters
    all_scenarios = []
    
    # Gas price variations
    for mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
        npv = 2_500_000_000 + (mult - 1.0) * 400_000_000
        all_scenarios.append(
            create_mock_solution(f'gas_{int(mult*100)}', gas_mult=mult, npv=npv)
        )
    
    # LMP variations
    for mult in [0.7, 0.85, 1.0, 1.15, 1.3]:
        npv = 2_500_000_000 + (mult - 1.0) * 600_000_000
        all_scenarios.append(
            create_mock_solution(f'lmp_{int(mult*100)}', lmp_mult=mult, npv=npv)
        )
    
    # Battery cost variations
    for cost in [200, 275, 350, 425, 500]:
        npv = 2_500_000_000 + (cost - 350) * 500_000
        all_scenarios.append(
            create_mock_solution(f'batt_{cost}', battery_cost=cost, npv=npv)
        )
    
    # Analyze all parameters
    parameters = ['gas_price_multiplier', 'lmp_multiplier', 'battery_cost_per_kwh']
    
    sensitivity_results = analyze_multiple_parameters(
        base_solution=base_solution,
        scenario_results=all_scenarios,
        parameters=parameters,
        metric='total_npv'
    )
    
    print(f"\nAnalyzed {len(sensitivity_results)} parameters")
    
    for param_name, analysis in sensitivity_results.items():
        print(f"\n{param_name}:")
        print(f"  Elasticity: {analysis['elasticity']:.3f}")
        print(f"  Impact Score: {analysis['impact_score']:.2f}")
        print(f"  R-squared: {analysis['regression']['r_squared']:.4f}")
        print(f"  Scenarios: {analysis['num_scenarios']}")
    
    return sensitivity_results


def test_parameter_ranking():
    """Test parameter ranking by impact."""
    print("\n" + "="*80)
    print("TEST 3: Parameter Ranking by Impact")
    print("="*80)
    
    # Create base solution
    base_solution = create_mock_solution('base', npv=2_500_000_000)
    
    # Create scenarios with different impacts
    all_scenarios = []
    
    # Gas price: moderate impact
    for mult in [0.5, 1.0, 1.5]:
        npv = 2_500_000_000 + (mult - 1.0) * 400_000_000
        all_scenarios.append(
            create_mock_solution(f'gas_{int(mult*100)}', gas_mult=mult, npv=npv)
        )
    
    # LMP: high impact
    for mult in [0.7, 1.0, 1.3]:
        npv = 2_500_000_000 + (mult - 1.0) * 800_000_000
        all_scenarios.append(
            create_mock_solution(f'lmp_{int(mult*100)}', lmp_mult=mult, npv=npv)
        )
    
    # Battery cost: low impact
    for cost in [200, 350, 500]:
        npv = 2_500_000_000 + (cost - 350) * 300_000
        all_scenarios.append(
            create_mock_solution(f'batt_{cost}', battery_cost=cost, npv=npv)
        )
    
    # Analyze parameters
    parameters = ['gas_price_multiplier', 'lmp_multiplier', 'battery_cost_per_kwh']
    sensitivity_results = analyze_multiple_parameters(
        base_solution, all_scenarios, parameters
    )
    
    # Rank by impact score
    ranking = rank_parameters_by_impact(sensitivity_results, metric='impact_score')
    
    print("\nParameter Ranking by Impact Score:")
    print(ranking.to_string(index=False))
    
    # Identify critical parameters
    critical = identify_critical_parameters(sensitivity_results, impact_threshold=10.0)
    print(f"\nCritical parameters (impact > 10%): {critical}")
    
    return ranking


def test_parameter_comparison():
    """Test comparison between two parameters."""
    print("\n" + "="*80)
    print("TEST 4: Parameter Comparison")
    print("="*80)
    
    # Create base solution
    base_solution = create_mock_solution('base', npv=2_500_000_000)
    
    # Create scenarios
    all_scenarios = []
    
    for mult in [0.5, 1.0, 1.5]:
        npv_gas = 2_500_000_000 + (mult - 1.0) * 400_000_000
        all_scenarios.append(
            create_mock_solution(f'gas_{int(mult*100)}', gas_mult=mult, npv=npv_gas)
        )
        
        npv_lmp = 2_500_000_000 + (mult - 1.0) * 800_000_000
        all_scenarios.append(
            create_mock_solution(f'lmp_{int(mult*100)}', lmp_mult=mult, npv=npv_lmp)
        )
    
    # Analyze parameters
    sensitivity_results = analyze_multiple_parameters(
        base_solution,
        all_scenarios,
        ['gas_price_multiplier', 'lmp_multiplier']
    )
    
    # Compare parameters
    comparison = compare_parameter_impacts(
        sensitivity_results,
        'gas_price_multiplier',
        'lmp_multiplier'
    )
    
    print(f"\nComparing {comparison['parameter1']} vs {comparison['parameter2']}:")
    print(f"  {comparison['parameter1']} impact: {comparison['parameter1_impact']:.2f}")
    print(f"  {comparison['parameter2']} impact: {comparison['parameter2_impact']:.2f}")
    print(f"  Impact ratio: {comparison['impact_score_ratio']:.2f}x")
    print(f"  More impactful: {comparison['more_impactful']}")
    
    return comparison


def test_sensitivity_metrics():
    """Test generation of sensitivity metrics."""
    print("\n" + "="*80)
    print("TEST 5: Sensitivity Metrics Generation")
    print("="*80)
    
    # Create base solution
    base_solution = create_mock_solution('base', npv=2_500_000_000)
    
    # Create scenarios
    all_scenarios = []
    for mult in [0.5, 1.0, 1.5]:
        all_scenarios.append(
            create_mock_solution(f'gas_{int(mult*100)}', gas_mult=mult, 
                               npv=2_500_000_000 + (mult - 1.0) * 400_000_000)
        )
        all_scenarios.append(
            create_mock_solution(f'lmp_{int(mult*100)}', lmp_mult=mult,
                               npv=2_500_000_000 + (mult - 1.0) * 600_000_000)
        )
    
    # Analyze parameters
    sensitivity_results = analyze_multiple_parameters(
        base_solution,
        all_scenarios,
        ['gas_price_multiplier', 'lmp_multiplier']
    )
    
    # Generate metrics
    metrics = generate_sensitivity_metrics(sensitivity_results)
    
    print("\nSensitivity Metrics Summary:")
    print(f"  Parameters analyzed: {metrics['num_parameters_analyzed']}")
    print(f"\n  Elasticity:")
    print(f"    Mean: {metrics['elasticity']['mean']:.3f}")
    print(f"    Std: {metrics['elasticity']['std']:.3f}")
    print(f"    Range: [{metrics['elasticity']['min']:.3f}, {metrics['elasticity']['max']:.3f}]")
    print(f"\n  Impact Score:")
    print(f"    Mean: {metrics['impact_score']['mean']:.2f}")
    print(f"    Std: {metrics['impact_score']['std']:.2f}")
    print(f"    Range: [{metrics['impact_score']['min']:.2f}, {metrics['impact_score']['max']:.2f}]")
    print(f"\n  R-squared:")
    print(f"    Mean: {metrics['r_squared']['mean']:.4f}")
    print(f"    Std: {metrics['r_squared']['std']:.4f}")
    print(f"    Range: [{metrics['r_squared']['min']:.4f}, {metrics['r_squared']['max']:.4f}]")
    
    return metrics


def test_save_load():
    """Test saving and loading sensitivity results."""
    print("\n" + "="*80)
    print("TEST 6: Save and Load Sensitivity Results")
    print("="*80)
    
    # Create base solution
    base_solution = create_mock_solution('base', npv=2_500_000_000)
    
    # Create scenarios
    all_scenarios = [
        create_mock_solution(f'gas_{int(m*100)}', gas_mult=m, 
                           npv=2_500_000_000 + (m - 1.0) * 400_000_000)
        for m in [0.5, 1.0, 1.5]
    ]
    
    # Analyze
    sensitivity_results = analyze_multiple_parameters(
        base_solution,
        all_scenarios,
        ['gas_price_multiplier']
    )
    
    # Save
    output_path = "results/test_sensitivity_analysis.json"
    save_sensitivity_results(sensitivity_results, output_path)
    print(f"\nSaved results to {output_path}")
    
    # Load
    from src.analysis.sensitivity_analyzer import load_sensitivity_results
    loaded_results = load_sensitivity_results(output_path)
    print(f"Loaded {len(loaded_results)} parameter analyses")
    
    # Verify
    for param_name in sensitivity_results.keys():
        original = sensitivity_results[param_name]
        loaded = loaded_results[param_name]
        print(f"\n{param_name}:")
        print(f"  Original elasticity: {original['elasticity']:.3f}")
        print(f"  Loaded elasticity: {loaded['elasticity']:.3f}")
        print(f"  Match: {abs(original['elasticity'] - loaded['elasticity']) < 1e-6}")
    
    return loaded_results


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SENSITIVITY ANALYZER TEST SUITE")
    print("="*80)
    
    try:
        # Run tests
        test_gas_price_sensitivity()
        test_multiple_parameters()
        test_parameter_ranking()
        test_parameter_comparison()
        test_sensitivity_metrics()
        test_save_load()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
