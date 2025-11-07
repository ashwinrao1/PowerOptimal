"""Demo script showing cost breakdown visualization usage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.cost_viz import plot_cost_breakdown
from src.models.solution import OptimizationSolution


def main():
    """Demonstrate cost breakdown visualization."""
    
    # Load an optimization solution
    solution = OptimizationSolution.load("results/solutions/optimal_portfolio.json")
    
    print("Optimal Portfolio Cost Breakdown")
    print("="*60)
    print(f"\nCapacity Mix:")
    print(f"  Grid Connection: {solution.capacity.grid_mw:.1f} MW")
    print(f"  Gas Peakers: {solution.capacity.gas_mw:.1f} MW")
    print(f"  Battery Storage: {solution.capacity.battery_mwh:.1f} MWh")
    print(f"  Solar PV: {solution.capacity.solar_mw:.1f} MW")
    
    print(f"\nCost Metrics:")
    print(f"  Total NPV (20 years): ${solution.metrics.total_npv/1e6:.2f}M")
    print(f"  CAPEX: ${solution.metrics.capex/1e6:.2f}M")
    print(f"  Annual OPEX: ${solution.metrics.opex_annual/1e6:.2f}M")
    print(f"  LCOE: ${solution.metrics.lcoe:.2f}/MWh")
    
    # Create waterfall chart
    print("\nGenerating waterfall chart...")
    fig_waterfall = plot_cost_breakdown(
        solution,
        format="waterfall",
        title="Cost Breakdown - Waterfall Chart"
    )
    fig_waterfall.write_html("results/figures/demo_cost_waterfall.html")
    print("Saved: results/figures/demo_cost_waterfall.html")
    
    # Create stacked bar chart
    print("\nGenerating stacked bar chart...")
    fig_stacked = plot_cost_breakdown(
        solution,
        format="stacked_bar",
        title="Cost Breakdown - Stacked Bar Chart"
    )
    fig_stacked.write_html("results/figures/demo_cost_stacked.html")
    print("Saved: results/figures/demo_cost_stacked.html")
    
    print("\nVisualization complete! Open the HTML files in a browser to view.")


if __name__ == "__main__":
    main()
