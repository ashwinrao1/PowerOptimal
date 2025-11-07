"""Example usage of Pareto frontier visualization functions.

This script demonstrates how to use the Pareto frontier visualization
functions with optimization results.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.pareto_viz import (
    plot_pareto_frontier,
    plot_multiple_pareto_frontiers
)
from src.analysis.pareto_calculator import (
    calculate_pareto_frontier,
    calculate_all_pareto_frontiers,
    load_pareto_frontiers
)


def example_1_basic_pareto_plot():
    """Example 1: Basic Pareto frontier plot from existing data."""
    print("\nExample 1: Basic Pareto frontier plot")
    print("-" * 50)
    
    # Load existing Pareto frontier data
    frontiers = load_pareto_frontiers("results/example_pareto_frontiers.json")
    
    # Get cost-reliability frontier
    cost_reliability_df = frontiers['cost_reliability']
    
    # Create plot
    fig = plot_pareto_frontier(
        cost_reliability_df,
        objective1='total_npv',
        objective2='reliability_pct',
        title='Cost vs Reliability Trade-off'
    )
    
    # Save or display
    fig.write_html("results/figures/example_pareto_basic.html")
    print("Saved to: results/figures/example_pareto_basic.html")


def example_2_with_annotations():
    """Example 2: Pareto plot with baseline and optimal annotations."""
    print("\nExample 2: Pareto plot with annotations")
    print("-" * 50)
    
    # Load Pareto frontier
    frontiers = load_pareto_frontiers("results/example_pareto_frontiers.json")
    cost_carbon_df = frontiers['cost_carbon']
    
    # Define baseline (grid-only) and optimal solutions
    baseline = {
        'metrics': {
            'total_npv': 2.5e9,
            'carbon_tons_annual': 600000
        }
    }
    
    optimal = {
        'metrics': {
            'total_npv': 2.2e9,
            'carbon_tons_annual': 350000
        }
    }
    
    # Create plot with annotations
    fig = plot_pareto_frontier(
        cost_carbon_df,
        objective1='total_npv',
        objective2='carbon_tons_annual',
        title='Cost vs Carbon: Baseline and Optimal Solutions',
        baseline_solution=baseline,
        optimal_solution=optimal
    )
    
    fig.write_html("results/figures/example_pareto_annotated.html")
    print("Saved to: results/figures/example_pareto_annotated.html")


def example_3_multiple_frontiers():
    """Example 3: Multiple Pareto frontiers in subplots."""
    print("\nExample 3: Multiple Pareto frontiers")
    print("-" * 50)
    
    # Load all frontiers
    frontiers = load_pareto_frontiers("results/example_pareto_frontiers.json")
    
    # Create multi-panel plot
    fig = plot_multiple_pareto_frontiers(
        frontiers,
        title='Comprehensive Pareto Frontier Analysis'
    )
    
    fig.write_html("results/figures/example_pareto_multiple.html")
    print("Saved to: results/figures/example_pareto_multiple.html")


def example_4_from_batch_results():
    """Example 4: Create Pareto frontier from batch solver results."""
    print("\nExample 4: Pareto frontier from batch results")
    print("-" * 50)
    
    # Simulate batch solver results
    # In practice, these would come from batch_solver.solve_scenarios()
    batch_results = [
        {
            'scenario_name': 'Scenario 1',
            'scenario_index': 0,
            'status': 'success',
            'metrics': {
                'total_npv': 2.0e9,
                'reliability_pct': 99.9,
                'carbon_tons_annual': 500000,
                'grid_dependence_pct': 80
            }
        },
        {
            'scenario_name': 'Scenario 2',
            'scenario_index': 1,
            'status': 'success',
            'metrics': {
                'total_npv': 2.3e9,
                'reliability_pct': 99.99,
                'carbon_tons_annual': 350000,
                'grid_dependence_pct': 60
            }
        },
        {
            'scenario_name': 'Scenario 3',
            'scenario_index': 2,
            'status': 'success',
            'metrics': {
                'total_npv': 2.8e9,
                'reliability_pct': 99.999,
                'carbon_tons_annual': 200000,
                'grid_dependence_pct': 40
            }
        }
    ]
    
    # Calculate Pareto frontier
    pareto_df = calculate_pareto_frontier(
        batch_results,
        objective1='total_npv',
        objective2='reliability_pct',
        minimize_obj1=True,
        minimize_obj2=False  # Maximize reliability
    )
    
    print(f"Found {len(pareto_df)} Pareto-optimal solutions")
    
    # Visualize
    fig = plot_pareto_frontier(
        pareto_df,
        objective1='total_npv',
        objective2='reliability_pct',
        title='Pareto Frontier from Batch Results',
        show_all_solutions=False
    )
    
    fig.write_html("results/figures/example_pareto_from_batch.html")
    print("Saved to: results/figures/example_pareto_from_batch.html")


def example_5_show_all_solutions():
    """Example 5: Show both Pareto and non-Pareto solutions."""
    print("\nExample 5: Show all solutions")
    print("-" * 50)
    
    # Simulate results with dominated solutions
    all_results = [
        {
            'scenario_name': f'Scenario {i}',
            'scenario_index': i,
            'status': 'success',
            'metrics': {
                'total_npv': 2.0e9 + i * 0.2e9,
                'reliability_pct': 99.9 + i * 0.02,
                'carbon_tons_annual': 500000 - i * 50000
            }
        }
        for i in range(10)
    ]
    
    # Calculate Pareto frontier
    pareto_df = calculate_pareto_frontier(
        all_results,
        objective1='total_npv',
        objective2='carbon_tons_annual',
        minimize_obj1=True,
        minimize_obj2=True
    )
    
    # Add non-Pareto solutions to DataFrame
    import pandas as pd
    all_df = pd.DataFrame([
        {
            'scenario_name': r['scenario_name'],
            'scenario_index': r['scenario_index'],
            'is_pareto_optimal': r['scenario_index'] in pareto_df['scenario_index'].values,
            **r['metrics']
        }
        for r in all_results
    ])
    
    # Visualize with all solutions
    fig = plot_pareto_frontier(
        all_df,
        objective1='total_npv',
        objective2='carbon_tons_annual',
        title='All Solutions: Pareto-Optimal and Dominated',
        show_all_solutions=True
    )
    
    fig.write_html("results/figures/example_pareto_all_solutions.html")
    print("Saved to: results/figures/example_pareto_all_solutions.html")
    print(f"Total solutions: {len(all_df)}")
    print(f"Pareto-optimal: {len(pareto_df)}")
    print(f"Dominated: {len(all_df) - len(pareto_df)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Pareto Frontier Visualization Examples")
    print("=" * 60)
    
    # Create output directory
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    # Run examples
    try:
        example_1_basic_pareto_plot()
        example_2_with_annotations()
        example_3_multiple_frontiers()
        example_4_from_batch_results()
        example_5_show_all_solutions()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
