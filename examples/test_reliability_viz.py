"""Test script for reliability visualization functions."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.solution import OptimizationSolution
from src.visualization.reliability_viz import (
    plot_reliability_analysis,
    plot_curtailment_histogram,
    plot_reserve_margin_timeseries,
    plot_worst_reliability_events
)


def test_reliability_visualizations():
    """Test all reliability visualization functions with real solution data."""
    
    print("Loading solution data...")
    solution_path = "results/solutions/optimal_portfolio.json"
    
    try:
        solution = OptimizationSolution.load(solution_path)
        print(f"Loaded solution from {solution_path}")
        print(f"Reliability: {solution.metrics.reliability_pct:.4f}%")
        print(f"Total curtailment: {solution.metrics.total_curtailment_mwh:.2f} MWh")
        print(f"Curtailment hours: {solution.metrics.num_curtailment_hours}")
    except FileNotFoundError:
        print(f"Solution file not found: {solution_path}")
        print("Please run the optimization first to generate solution data.")
        return
    except Exception as e:
        print(f"Error loading solution: {e}")
        return
    
    print("\n" + "="*60)
    print("Testing reliability visualization functions...")
    print("="*60)
    
    # Test 1: Comprehensive reliability analysis
    print("\n1. Testing plot_reliability_analysis()...")
    try:
        fig = plot_reliability_analysis(solution)
        output_path = "results/figures/reliability_analysis.html"
        fig.write_html(output_path)
        print(f"   SUCCESS: Saved to {output_path}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 2: Curtailment histogram
    print("\n2. Testing plot_curtailment_histogram()...")
    try:
        fig = plot_curtailment_histogram(solution)
        output_path = "results/figures/curtailment_histogram.html"
        fig.write_html(output_path)
        print(f"   SUCCESS: Saved to {output_path}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 3: Reserve margin time series (full year)
    print("\n3. Testing plot_reserve_margin_timeseries() - Full Year...")
    try:
        fig = plot_reserve_margin_timeseries(solution)
        output_path = "results/figures/reserve_margin_full_year.html"
        fig.write_html(output_path)
        print(f"   SUCCESS: Saved to {output_path}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 4: Reserve margin time series (first week)
    print("\n4. Testing plot_reserve_margin_timeseries() - First Week...")
    try:
        fig = plot_reserve_margin_timeseries(
            solution,
            time_range=(1, 168),
            title="Reserve Margin - First Week"
        )
        output_path = "results/figures/reserve_margin_week1.html"
        fig.write_html(output_path)
        print(f"   SUCCESS: Saved to {output_path}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 5: Worst reliability events
    print("\n5. Testing plot_worst_reliability_events()...")
    try:
        fig = plot_worst_reliability_events(solution, n_events=10)
        output_path = "results/figures/worst_reliability_events.html"
        fig.write_html(output_path)
        print(f"   SUCCESS: Saved to {output_path}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 6: Test with dict format (from JSON)
    print("\n6. Testing with dict format (JSON compatibility)...")
    try:
        import json
        with open(solution_path, 'r') as f:
            solution_dict = json.load(f)
        
        fig = plot_reliability_analysis(solution_dict)
        output_path = "results/figures/reliability_analysis_from_dict.html"
        fig.write_html(output_path)
        print(f"   SUCCESS: Saved to {output_path}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    print("\nGenerated visualizations:")
    print("  - results/figures/reliability_analysis.html")
    print("  - results/figures/curtailment_histogram.html")
    print("  - results/figures/reserve_margin_full_year.html")
    print("  - results/figures/reserve_margin_week1.html")
    print("  - results/figures/worst_reliability_events.html")
    print("  - results/figures/reliability_analysis_from_dict.html")


if __name__ == "__main__":
    test_reliability_visualizations()
