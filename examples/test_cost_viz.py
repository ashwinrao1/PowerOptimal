"""Test script for cost breakdown visualization."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.cost_viz import plot_cost_breakdown, plot_cost_comparison
from src.models.solution import OptimizationSolution
from src.models.technology import TechnologyCosts


def test_cost_breakdown_with_json():
    """Test cost breakdown visualization with JSON solution data."""
    print("Testing cost breakdown visualization with JSON data...")
    
    # Load a solution from JSON
    solution_path = "results/solutions/optimal_portfolio.json"
    
    try:
        solution = OptimizationSolution.load(solution_path)
        print(f"Loaded solution from {solution_path}")
        
        # Create waterfall chart
        print("\nCreating waterfall chart...")
        fig_waterfall = plot_cost_breakdown(
            solution,
            format="waterfall",
            title="Cost Breakdown - Optimal Portfolio (Waterfall)"
        )
        
        # Save to HTML
        output_path = "results/figures/cost_breakdown_waterfall.html"
        fig_waterfall.write_html(output_path)
        print(f"Saved waterfall chart to {output_path}")
        
        # Create stacked bar chart
        print("\nCreating stacked bar chart...")
        fig_stacked = plot_cost_breakdown(
            solution,
            format="stacked_bar",
            title="Cost Breakdown - Optimal Portfolio (Stacked Bar)"
        )
        
        # Save to HTML
        output_path = "results/figures/cost_breakdown_stacked.html"
        fig_stacked.write_html(output_path)
        print(f"Saved stacked bar chart to {output_path}")
        
        # Print cost summary
        print("\nCost Summary:")
        print(f"  Total NPV: ${solution.metrics.total_npv/1e6:.2f}M")
        print(f"  CAPEX: ${solution.metrics.capex/1e6:.2f}M")
        print(f"  Annual OPEX: ${solution.metrics.opex_annual/1e6:.2f}M")
        print(f"  LCOE: ${solution.metrics.lcoe:.2f}/MWh")
        
    except FileNotFoundError:
        print(f"Solution file not found: {solution_path}")
        print("Please run the optimization first to generate solution data.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_cost_comparison():
    """Test cost comparison across multiple scenarios."""
    print("\n" + "="*60)
    print("Testing cost comparison visualization...")
    
    # Load multiple solutions
    solutions = {}
    
    solution_files = [
        ("Baseline (Grid Only)", "results/solutions/baseline_grid_only.json"),
        ("Optimal Portfolio", "results/solutions/optimal_portfolio.json")
    ]
    
    for name, path in solution_files:
        try:
            solution = OptimizationSolution.load(path)
            solutions[name] = solution
            print(f"Loaded: {name}")
        except FileNotFoundError:
            print(f"File not found: {path}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    if len(solutions) < 2:
        print("Need at least 2 solutions for comparison. Skipping comparison test.")
        return False
    
    # Create comparison charts for different metrics
    metrics = ["total_npv", "capex", "opex_annual"]
    
    for metric in metrics:
        print(f"\nCreating comparison chart for {metric}...")
        fig = plot_cost_comparison(
            solutions,
            metric=metric,
            title=f"Cost Comparison: {metric.replace('_', ' ').title()}"
        )
        
        output_path = f"results/figures/cost_comparison_{metric}.html"
        fig.write_html(output_path)
        print(f"Saved to {output_path}")
    
    return True


def test_with_dict_format():
    """Test cost breakdown with dict format (simulating JSON load)."""
    print("\n" + "="*60)
    print("Testing cost breakdown with dict format...")
    
    # Create mock solution data
    mock_solution = {
        "capacity": {
            "Grid Connection (MW)": 200.0,
            "Gas Peakers (MW)": 150.0,
            "Battery Storage (MWh)": 400.0,
            "Solar PV (MW)": 100.0
        },
        "metrics": {
            "total_npv": 2500000000.0,
            "capex": 800000000.0,
            "opex_annual": 120000000.0,
            "lcoe": 85.5,
            "reliability_pct": 99.99,
            "total_curtailment_mwh": 2.5,
            "num_curtailment_hours": 10,
            "carbon_tons_annual": 450000.0,
            "carbon_intensity_g_per_kwh": 180.0,
            "carbon_reduction_pct": 35.0,
            "grid_dependence_pct": 45.0,
            "gas_capacity_factor": 25.0,
            "battery_cycles_per_year": 250.0,
            "solar_capacity_factor": 28.0,
            "solve_time_seconds": 450.0,
            "optimality_gap_pct": 0.3
        }
    }
    
    try:
        # Create waterfall chart
        fig = plot_cost_breakdown(
            mock_solution,
            format="waterfall",
            title="Cost Breakdown - Mock Data"
        )
        
        output_path = "results/figures/cost_breakdown_mock.html"
        fig.write_html(output_path)
        print(f"Saved mock data visualization to {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Cost Breakdown Visualization Test")
    print("="*60)
    
    # Run tests
    test1 = test_cost_breakdown_with_json()
    test2 = test_cost_comparison()
    test3 = test_with_dict_format()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print(f"  Cost breakdown with JSON: {'PASSED' if test1 else 'FAILED'}")
    print(f"  Cost comparison: {'PASSED' if test2 else 'FAILED'}")
    print(f"  Dict format support: {'PASSED' if test3 else 'FAILED'}")
    
    if test1 or test2 or test3:
        print("\nAt least one test passed successfully!")
        print("Check the results/figures/ directory for generated visualizations.")
    else:
        print("\nAll tests failed. Please check error messages above.")
