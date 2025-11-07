"""Test script for Pareto frontier visualization."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.pareto_viz import plot_pareto_frontier, plot_multiple_pareto_frontiers
from src.analysis.pareto_calculator import load_pareto_frontiers
import pandas as pd


def test_with_existing_data():
    """Test visualization with existing Pareto frontier data."""
    print("Testing Pareto frontier visualization with existing data...")
    
    # Try to load existing Pareto frontier data
    pareto_file = Path("results/example_pareto_frontiers.json")
    
    if not pareto_file.exists():
        print(f"Pareto frontier file not found: {pareto_file}")
        print("Creating mock data for testing...")
        test_with_mock_data()
        return
    
    try:
        # Load Pareto frontiers
        frontiers = load_pareto_frontiers(str(pareto_file))
        print(f"Loaded {len(frontiers)} Pareto frontiers")
        
        # Test individual frontier plots
        for name, df in frontiers.items():
            print(f"\nTesting {name} frontier ({len(df)} solutions)...")
            
            # Determine objectives based on frontier name
            if name == 'cost_reliability':
                obj1, obj2 = 'total_npv', 'reliability_pct'
            elif name == 'cost_carbon':
                obj1, obj2 = 'total_npv', 'carbon_tons_annual'
            elif name == 'grid_reliability':
                obj1, obj2 = 'grid_dependence_pct', 'reliability_pct'
            else:
                print(f"Unknown frontier type: {name}")
                continue
            
            # Create plot
            fig = plot_pareto_frontier(
                df,
                objective1=obj1,
                objective2=obj2,
                show_all_solutions=False
            )
            
            # Save to file
            output_file = f"results/figures/pareto_{name}.html"
            fig.write_html(output_file)
            print(f"Saved plot to {output_file}")
        
        # Test multiple frontiers plot
        print("\nTesting multiple frontiers plot...")
        fig_multi = plot_multiple_pareto_frontiers(frontiers)
        output_file = "results/figures/pareto_all_frontiers.html"
        fig_multi.write_html(output_file)
        print(f"Saved multi-frontier plot to {output_file}")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


def test_with_mock_data():
    """Test visualization with mock data."""
    print("\nTesting with mock data...")
    
    # Create mock Pareto frontier data
    mock_data = {
        'scenario_name': [
            'Low Cost', 'Balanced', 'High Reliability', 
            'Very High Reliability', 'Ultra Reliable'
        ],
        'scenario_index': [0, 1, 2, 3, 4],
        'is_pareto_optimal': [True, True, True, True, True],
        'total_npv': [2.0e9, 2.2e9, 2.5e9, 2.8e9, 3.2e9],
        'reliability_pct': [99.9, 99.95, 99.99, 99.995, 99.999],
        'carbon_tons_annual': [500000, 400000, 300000, 250000, 200000],
        'grid_dependence_pct': [80, 70, 60, 50, 40],
        'capex': [500e6, 600e6, 750e6, 900e6, 1.1e9],
        'opex_annual': [150e6, 140e6, 130e6, 125e6, 120e6]
    }
    
    df = pd.DataFrame(mock_data)
    
    # Test cost vs reliability
    print("\nTesting cost vs reliability plot...")
    fig1 = plot_pareto_frontier(
        df,
        objective1='total_npv',
        objective2='reliability_pct',
        title='Mock Data: Cost vs Reliability Trade-off',
        show_all_solutions=False
    )
    output_file = "results/figures/pareto_mock_cost_reliability.html"
    fig1.write_html(output_file)
    print(f"Saved plot to {output_file}")
    
    # Test cost vs carbon
    print("\nTesting cost vs carbon plot...")
    fig2 = plot_pareto_frontier(
        df,
        objective1='total_npv',
        objective2='carbon_tons_annual',
        title='Mock Data: Cost vs Carbon Trade-off',
        show_all_solutions=False
    )
    output_file = "results/figures/pareto_mock_cost_carbon.html"
    fig2.write_html(output_file)
    print(f"Saved plot to {output_file}")
    
    # Test grid dependence vs reliability
    print("\nTesting grid dependence vs reliability plot...")
    fig3 = plot_pareto_frontier(
        df,
        objective1='grid_dependence_pct',
        objective2='reliability_pct',
        title='Mock Data: Grid Dependence vs Reliability Trade-off',
        show_all_solutions=False
    )
    output_file = "results/figures/pareto_mock_grid_reliability.html"
    fig3.write_html(output_file)
    print(f"Saved plot to {output_file}")
    
    # Test with baseline and optimal annotations
    print("\nTesting with baseline and optimal annotations...")
    baseline = {
        'metrics': {
            'total_npv': 2.5e9,
            'reliability_pct': 99.5,
            'carbon_tons_annual': 600000,
            'grid_dependence_pct': 100
        }
    }
    
    optimal = {
        'metrics': {
            'total_npv': 2.2e9,
            'reliability_pct': 99.95,
            'carbon_tons_annual': 400000,
            'grid_dependence_pct': 70
        }
    }
    
    fig4 = plot_pareto_frontier(
        df,
        objective1='total_npv',
        objective2='reliability_pct',
        title='Mock Data: With Baseline and Optimal Annotations',
        baseline_solution=baseline,
        optimal_solution=optimal,
        show_all_solutions=False
    )
    output_file = "results/figures/pareto_mock_annotated.html"
    fig4.write_html(output_file)
    print(f"Saved annotated plot to {output_file}")
    
    # Test multiple frontiers
    print("\nTesting multiple frontiers plot...")
    frontiers = {
        'cost_reliability': df.copy(),
        'cost_carbon': df.copy(),
        'grid_reliability': df.copy()
    }
    
    fig5 = plot_multiple_pareto_frontiers(
        frontiers,
        title='Mock Data: Multiple Pareto Frontiers'
    )
    output_file = "results/figures/pareto_mock_multiple.html"
    fig5.write_html(output_file)
    print(f"Saved multiple frontiers plot to {output_file}")
    
    print("\nMock data tests passed!")


def test_with_all_solutions():
    """Test showing both Pareto and non-Pareto solutions."""
    print("\nTesting with all solutions (Pareto and non-Pareto)...")
    
    # Create mock data with both Pareto and non-Pareto solutions
    mock_data = {
        'scenario_name': [
            'Pareto 1', 'Pareto 2', 'Pareto 3', 'Pareto 4',
            'Non-Pareto 1', 'Non-Pareto 2', 'Non-Pareto 3'
        ],
        'scenario_index': [0, 1, 2, 3, 4, 5, 6],
        'is_pareto_optimal': [True, True, True, True, False, False, False],
        'total_npv': [2.0e9, 2.2e9, 2.5e9, 2.8e9, 2.3e9, 2.6e9, 2.9e9],
        'reliability_pct': [99.9, 99.95, 99.99, 99.995, 99.92, 99.96, 99.98],
        'carbon_tons_annual': [500000, 400000, 300000, 250000, 450000, 380000, 320000]
    }
    
    df = pd.DataFrame(mock_data)
    
    fig = plot_pareto_frontier(
        df,
        objective1='total_npv',
        objective2='reliability_pct',
        title='All Solutions: Pareto and Non-Pareto',
        show_all_solutions=True
    )
    
    output_file = "results/figures/pareto_all_solutions.html"
    fig.write_html(output_file)
    print(f"Saved plot with all solutions to {output_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("Pareto Frontier Visualization Test")
    print("=" * 60)
    
    # Create results/figures directory if it doesn't exist
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    # Run tests
    test_with_existing_data()
    test_with_mock_data()
    test_with_all_solutions()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
