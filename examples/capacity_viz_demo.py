"""Demonstration of capacity visualization features.

This script creates all supported visualization formats and saves them as HTML files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.solution import OptimizationSolution
from src.visualization.capacity_viz import plot_capacity_mix, plot_capacity_comparison


def main():
    """Create and save capacity visualizations."""
    print("Capacity Visualization Demo")
    print("=" * 50)
    
    # Load solutions
    optimal_path = "results/solutions/optimal_portfolio.json"
    baseline_path = "results/solutions/baseline_grid_only.json"
    
    if not Path(optimal_path).exists():
        print(f"Error: Solution file not found: {optimal_path}")
        return
    
    print(f"\nLoading optimal solution from {optimal_path}...")
    optimal = OptimizationSolution.load(optimal_path)
    
    print("\nOptimal Portfolio Capacity:")
    print(f"  Grid Connection: {optimal.capacity.grid_mw:,.1f} MW")
    print(f"  Gas Peakers: {optimal.capacity.gas_mw:,.1f} MW")
    print(f"  Battery Storage: {optimal.capacity.battery_mwh:,.1f} MWh")
    print(f"  Solar PV: {optimal.capacity.solar_mw:,.1f} MW")
    
    # Create output directory
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar chart
    print("\n1. Creating bar chart...")
    fig_bar = plot_capacity_mix(
        optimal,
        format="bar",
        title="Optimal Capacity Mix - Bar Chart"
    )
    output_file = output_dir / "capacity_bar_chart.html"
    fig_bar.write_html(str(output_file))
    print(f"   Saved to {output_file}")
    
    # 2. Pie chart
    print("\n2. Creating pie chart...")
    fig_pie = plot_capacity_mix(
        optimal,
        format="pie",
        title="Optimal Capacity Mix - Distribution"
    )
    output_file = output_dir / "capacity_pie_chart.html"
    fig_pie.write_html(str(output_file))
    print(f"   Saved to {output_file}")
    
    # 3. Waterfall chart
    print("\n3. Creating waterfall chart...")
    fig_waterfall = plot_capacity_mix(
        optimal,
        format="waterfall",
        title="Capacity Buildup - Waterfall Chart"
    )
    output_file = output_dir / "capacity_waterfall_chart.html"
    fig_waterfall.write_html(str(output_file))
    print(f"   Saved to {output_file}")
    
    # 4. Comparison chart (if baseline exists)
    if Path(baseline_path).exists():
        print(f"\n4. Creating comparison chart...")
        baseline = OptimizationSolution.load(baseline_path)
        
        print("\nBaseline (Grid Only) Capacity:")
        print(f"  Grid Connection: {baseline.capacity.grid_mw:,.1f} MW")
        print(f"  Gas Peakers: {baseline.capacity.gas_mw:,.1f} MW")
        print(f"  Battery Storage: {baseline.capacity.battery_mwh:,.1f} MWh")
        print(f"  Solar PV: {baseline.capacity.solar_mw:,.1f} MW")
        
        solutions = {
            "Baseline\n(Grid Only)": baseline,
            "Optimal\nPortfolio": optimal
        }
        
        fig_comparison = plot_capacity_comparison(
            solutions,
            title="Capacity Mix Comparison: Baseline vs Optimal"
        )
        output_file = output_dir / "capacity_comparison.html"
        fig_comparison.write_html(str(output_file))
        print(f"   Saved to {output_file}")
    else:
        print(f"\n4. Baseline solution not found, skipping comparison")
    
    # 5. Bar chart without values
    print("\n5. Creating bar chart without value labels...")
    fig_bar_no_values = plot_capacity_mix(
        optimal,
        format="bar",
        title="Optimal Capacity Mix (Clean)",
        show_values=False
    )
    output_file = output_dir / "capacity_bar_chart_clean.html"
    fig_bar_no_values.write_html(str(output_file))
    print(f"   Saved to {output_file}")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print(f"\nAll visualizations saved to: {output_dir.absolute()}")
    print("\nOpen the HTML files in a web browser to view interactive charts.")


if __name__ == "__main__":
    main()
