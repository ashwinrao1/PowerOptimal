"""
Demo script for sensitivity tornado chart visualization using real data.

This script loads actual sensitivity analysis results and creates tornado charts.
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.sensitivity_viz import plot_sensitivity_tornado


def load_sensitivity_results(filepath: str = "results/test_sensitivity_analysis.json"):
    """Load sensitivity analysis results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('parameters', {})


def main():
    """Create tornado chart from real sensitivity analysis data."""
    print("=" * 60)
    print("Sensitivity Tornado Chart Demo")
    print("=" * 60)
    
    # Load real sensitivity results
    print("\nLoading sensitivity analysis results...")
    try:
        sensitivity_results = load_sensitivity_results()
        
        if not sensitivity_results:
            print("No sensitivity results found. Run sensitivity analysis first.")
            return 1
        
        print(f"Loaded {len(sensitivity_results)} parameter analyses")
        
        # Create tornado chart
        print("\nGenerating tornado chart...")
        fig = plot_sensitivity_tornado(
            sensitivity_results,
            metric='total_npv',
            title="Data Center Energy Optimization: Parameter Sensitivity Analysis"
        )
        
        # Save figure
        output_path = "results/figures/sensitivity_tornado_demo.html"
        fig.write_html(output_path)
        print(f"Saved tornado chart to {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary of Parameter Impacts:")
        print("=" * 60)
        
        # Sort by impact score
        sorted_params = sorted(
            sensitivity_results.items(),
            key=lambda x: abs(x[1].get('impact_score', 0)),
            reverse=True
        )
        
        for param_name, analysis in sorted_params:
            impact = analysis.get('impact_score', 0)
            elasticity = analysis.get('elasticity', 0)
            print(f"\n{param_name}:")
            print(f"  Impact Score: {impact:.2f}%")
            print(f"  Elasticity: {elasticity:.2f}")
            print(f"  R-squared: {analysis.get('regression', {}).get('r_squared', 0):.3f}")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except FileNotFoundError:
        print("Error: Sensitivity analysis results file not found.")
        print("Please run sensitivity analysis first to generate the data.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
