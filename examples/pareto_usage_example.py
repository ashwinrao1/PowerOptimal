"""
Example usage of Pareto frontier calculator.

This script demonstrates how to use the Pareto calculator with
optimization results to identify trade-offs between objectives.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import (
    calculate_pareto_frontier,
    calculate_cost_reliability_frontier,
    calculate_cost_carbon_frontier,
    calculate_all_pareto_frontiers,
    save_pareto_frontiers,
    load_pareto_frontiers,
    get_pareto_summary,
    identify_knee_point
)


def example_basic_usage():
    """Example: Basic Pareto frontier calculation."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Pareto Frontier Calculation")
    print("="*70)
    
    # Assume we have optimization results from batch_solver
    # In practice, these would come from:
    # results = solve_scenarios(scenarios, market_data, tech_costs, facility_params)
    
    # For this example, we'll use mock data
    results = [
        {
            'status': 'success',
            'scenario_name': 'low_cost',
            'metrics': {'total_npv': 2000e6, 'reliability_pct': 99.9, 'carbon_tons_annual': 150000}
        },
        {
            'status': 'success',
            'scenario_name': 'balanced',
            'metrics': {'total_npv': 2400e6, 'reliability_pct': 99.99, 'carbon_tons_annual': 130000}
        },
        {
            'status': 'success',
            'scenario_name': 'high_reliability',
            'metrics': {'total_npv': 2600e6, 'reliability_pct': 99.999, 'carbon_tons_annual': 120000}
        }
    ]
    
    # Calculate Pareto frontier for cost vs. reliability
    pareto_df = calculate_pareto_frontier(
        solutions=results,
        objective1='total_npv',
        objective2='reliability_pct',
        minimize_obj1=True,
        minimize_obj2=False  # Maximize reliability
    )
    
    print(f"\nFound {len(pareto_df)} Pareto-optimal solutions")
    print("\nSolutions:")
    for _, row in pareto_df.iterrows():
        print(f"  {row['scenario_name']}: "
              f"${row['total_npv']/1e9:.2f}B NPV, "
              f"{row['reliability_pct']:.3f}% reliability")


def example_specialized_frontiers():
    """Example: Using specialized frontier functions."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Specialized Frontier Functions")
    print("="*70)
    
    # Mock results with multiple objectives
    results = [
        {
            'status': 'success',
            'scenario_name': f'scenario_{i}',
            'metrics': {
                'total_npv': 2000e6 + i * 100e6,
                'reliability_pct': 99.9 + i * 0.02,
                'carbon_tons_annual': 150000 - i * 5000,
                'grid_dependence_pct': 80 - i * 2
            }
        }
        for i in range(5)
    ]
    
    # Calculate different frontiers
    cost_rel = calculate_cost_reliability_frontier(results)
    print(f"\nCost-Reliability Frontier: {len(cost_rel)} solutions")
    
    cost_carbon = calculate_cost_carbon_frontier(results)
    print(f"Cost-Carbon Frontier: {len(cost_carbon)} solutions")


def example_all_frontiers():
    """Example: Calculate all frontiers at once."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Calculate All Frontiers")
    print("="*70)
    
    # Mock results
    results = [
        {
            'status': 'success',
            'scenario_name': f'scenario_{i}',
            'metrics': {
                'total_npv': 2000e6 + i * 100e6,
                'reliability_pct': 99.9 + i * 0.02,
                'carbon_tons_annual': 150000 - i * 5000,
                'grid_dependence_pct': 80 - i * 2
            }
        }
        for i in range(5)
    ]
    
    # Calculate all standard frontiers
    frontiers = calculate_all_pareto_frontiers(results)
    
    print(f"\nCalculated {len(frontiers)} Pareto frontiers:")
    for name, df in frontiers.items():
        print(f"  {name}: {len(df)} solutions")


def example_save_and_load():
    """Example: Save and load Pareto frontiers."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Save and Load Frontiers")
    print("="*70)
    
    # Mock results
    results = [
        {
            'status': 'success',
            'scenario_name': f'scenario_{i}',
            'metrics': {
                'total_npv': 2000e6 + i * 100e6,
                'reliability_pct': 99.9 + i * 0.02,
                'carbon_tons_annual': 150000 - i * 5000,
                'grid_dependence_pct': 80 - i * 2
            }
        }
        for i in range(5)
    ]
    
    # Calculate frontiers
    frontiers = calculate_all_pareto_frontiers(results)
    
    # Save to file
    output_path = "results/example_pareto_frontiers.json"
    save_pareto_frontiers(frontiers, output_path)
    print(f"\nSaved frontiers to {output_path}")
    
    # Load from file
    loaded_frontiers = load_pareto_frontiers(output_path)
    print(f"Loaded {len(loaded_frontiers)} frontiers from file")


def example_analysis():
    """Example: Analyze Pareto frontier."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Analyze Pareto Frontier")
    print("="*70)
    
    # Mock results with more variation
    results = [
        {
            'status': 'success',
            'scenario_index': i,
            'scenario_name': f'scenario_{i}',
            'metrics': {
                'total_npv': 2000e6 + i * 100e6,
                'reliability_pct': 99.9 + i * 0.02,
                'carbon_tons_annual': 150000 - i * 5000
            }
        }
        for i in range(10)
    ]
    
    # Calculate frontier
    pareto_df = calculate_cost_reliability_frontier(results)
    
    # Get summary statistics
    summary = get_pareto_summary(
        pareto_df,
        objective1='total_npv',
        objective2='reliability_pct'
    )
    
    print(f"\nFrontier Summary:")
    print(f"  Number of solutions: {summary['num_solutions']}")
    print(f"  NPV range: ${summary['objective1']['min']/1e9:.2f}B - "
          f"${summary['objective1']['max']/1e9:.2f}B")
    print(f"  Reliability range: {summary['objective2']['min']:.3f}% - "
          f"{summary['objective2']['max']:.3f}%")
    
    # Identify knee point
    knee = identify_knee_point(
        pareto_df,
        objective1='total_npv',
        objective2='reliability_pct'
    )
    
    print(f"\nKnee Point (Best Trade-off):")
    print(f"  Scenario: {knee['scenario_name']}")
    print(f"  NPV: ${knee['total_npv']/1e9:.2f}B")
    print(f"  Reliability: {knee['reliability_pct']:.3f}%")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PARETO FRONTIER CALCULATOR - USAGE EXAMPLES")
    print("="*70)
    
    example_basic_usage()
    example_specialized_frontiers()
    example_all_frontiers()
    example_save_and_load()
    example_analysis()
    
    print("\n" + "="*70)
    print("Examples completed successfully!")
    print("="*70)
    print("\nFor real usage, replace mock data with actual optimization results:")
    print("  from src.analysis import solve_scenarios, generate_pareto_scenarios")
    print("  scenarios = generate_pareto_scenarios(objective_pair='cost_reliability')")
    print("  results = solve_scenarios(scenarios, market_data, tech_costs, facility_params)")
    print("  frontiers = calculate_all_pareto_frontiers(results)")
    print()


if __name__ == "__main__":
    main()
