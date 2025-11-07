"""Demo script showing how to use dispatch heatmap visualization."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.solution import OptimizationSolution
from src.visualization.dispatch_viz import plot_dispatch_heatmap, plot_dispatch_stacked_area


def main():
    """Demonstrate dispatch visualization capabilities."""
    
    # Load solution
    solution_path = Path(__file__).parent.parent / "results" / "solutions" / "optimal_portfolio.json"
    solution = OptimizationSolution.load(str(solution_path))
    
    print("Dispatch Heatmap Visualization Demo")
    print("=" * 60)
    print(f"\nLoaded solution with:")
    print(f"  Grid: {solution.capacity.grid_mw:.1f} MW")
    print(f"  Gas: {solution.capacity.gas_mw:.1f} MW")
    print(f"  Solar: {solution.capacity.solar_mw:.1f} MW")
    print(f"  Battery: {solution.capacity.battery_mwh:.1f} MWh")
    
    # Example 1: Full year heatmap
    print("\n1. Creating full year dispatch heatmap...")
    fig1 = plot_dispatch_heatmap(solution, title="Annual Dispatch Pattern")
    fig1.show()
    
    # Example 2: Focus on a specific week
    print("\n2. Creating first week dispatch heatmap...")
    fig2 = plot_dispatch_heatmap(
        solution, 
        time_range=(1, 168),
        title="First Week Dispatch Pattern"
    )
    fig2.show()
    
    # Example 3: Stacked area chart for better time series view
    print("\n3. Creating stacked area chart for first month...")
    fig3 = plot_dispatch_stacked_area(
        solution,
        time_range=(1, 744),
        title="First Month Dispatch (Stacked Area)"
    )
    fig3.show()
    
    print("\nDemo complete! Check your browser for interactive visualizations.")
    print("\nKey features:")
    print("  - Hover over cells to see detailed information")
    print("  - Use the range slider at the bottom to zoom in/out")
    print("  - Click and drag to pan across time periods")
    print("  - Double-click to reset the view")


if __name__ == "__main__":
    main()
