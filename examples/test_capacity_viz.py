"""Test script for capacity visualization functions."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.solution import OptimizationSolution
from src.visualization.capacity_viz import plot_capacity_mix, plot_capacity_comparison


def test_capacity_viz():
    """Test capacity visualization with real solution data."""
    print("Testing capacity visualization functions...")
    
    # Load existing solution
    solution_path = "results/solutions/optimal_portfolio.json"
    if not Path(solution_path).exists():
        print(f"Solution file not found: {solution_path}")
        print("Skipping test.")
        return
    
    print(f"\nLoading solution from {solution_path}...")
    solution = OptimizationSolution.load(solution_path)
    
    print("\nCapacity data:")
    print(f"  Grid Connection: {solution.capacity.grid_mw:.1f} MW")
    print(f"  Gas Peakers: {solution.capacity.gas_mw:.1f} MW")
    print(f"  Battery Storage: {solution.capacity.battery_mwh:.1f} MWh")
    print(f"  Solar PV: {solution.capacity.solar_mw:.1f} MW")
    
    # Test bar chart
    print("\n1. Creating bar chart...")
    fig_bar = plot_capacity_mix(solution, format="bar")
    print(f"   Bar chart created with {len(fig_bar.data)} traces")
    
    # Test pie chart
    print("\n2. Creating pie chart...")
    fig_pie = plot_capacity_mix(solution, format="pie")
    print(f"   Pie chart created with {len(fig_pie.data)} traces")
    
    # Test waterfall chart
    print("\n3. Creating waterfall chart...")
    fig_waterfall = plot_capacity_mix(solution, format="waterfall")
    print(f"   Waterfall chart created with {len(fig_waterfall.data)} traces")
    
    # Test with dict input
    print("\n4. Testing with dict input...")
    capacity_dict = solution.capacity.to_dict()
    fig_dict = plot_capacity_mix(capacity_dict, format="bar")
    print(f"   Bar chart from dict created with {len(fig_dict.data)} traces")
    
    # Test comparison plot if baseline exists
    baseline_path = "results/solutions/baseline_grid_only.json"
    if Path(baseline_path).exists():
        print(f"\n5. Creating comparison plot...")
        baseline = OptimizationSolution.load(baseline_path)
        
        solutions = {
            "Baseline (Grid Only)": baseline,
            "Optimal Portfolio": solution
        }
        
        fig_comparison = plot_capacity_comparison(solutions)
        print(f"   Comparison chart created with {len(fig_comparison.data)} traces")
    else:
        print(f"\n5. Baseline solution not found at {baseline_path}, skipping comparison test")
    
    print("\nAll tests completed successfully!")
    print("\nNote: Figures were created but not displayed.")
    print("To view figures, add fig.show() calls or save with fig.write_html()")


if __name__ == "__main__":
    test_capacity_viz()
