"""Test reliability visualization with baseline solution."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.solution import OptimizationSolution
from src.visualization.reliability_viz import plot_reliability_analysis


def test_baseline_reliability():
    """Test reliability visualization with baseline solution."""
    
    print("Testing reliability visualization with baseline solution...")
    
    solution_path = "results/solutions/baseline_grid_only.json"
    
    try:
        solution = OptimizationSolution.load(solution_path)
        print(f"\nLoaded baseline solution:")
        print(f"  Reliability: {solution.metrics.reliability_pct:.4f}%")
        print(f"  Total curtailment: {solution.metrics.total_curtailment_mwh:.2f} MWh")
        print(f"  Curtailment hours: {solution.metrics.num_curtailment_hours}")
        
        # Generate visualization
        fig = plot_reliability_analysis(
            solution,
            title="Reliability Analysis - Baseline (Grid Only)"
        )
        
        output_path = "results/figures/reliability_analysis_baseline.html"
        fig.write_html(output_path)
        print(f"\nSUCCESS: Saved to {output_path}")
        
    except FileNotFoundError:
        print(f"\nBaseline solution not found: {solution_path}")
        print("This is expected if baseline hasn't been run yet.")
    except Exception as e:
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    test_baseline_reliability()
