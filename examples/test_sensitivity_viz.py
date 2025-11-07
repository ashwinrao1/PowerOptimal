"""
Test script for sensitivity tornado chart visualization.

This script demonstrates the plot_sensitivity_tornado function with mock data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.sensitivity_viz import plot_sensitivity_tornado, plot_sensitivity_comparison


def create_mock_sensitivity_results():
    """Create mock sensitivity analysis results for testing."""
    
    # Mock sensitivity results for multiple parameters
    sensitivity_results = {
        'gas_price_multiplier': {
            'parameter_name': 'gas_price_multiplier',
            'metric': 'total_npv',
            'base_parameter_value': 1.0,
            'base_metric_value': 2_500_000_000,
            'elasticity': 0.15,
            'impact_score': 12.5,
            'parameter_values': [0.5, 0.75, 1.0, 1.25, 1.5],
            'metric_values': [2_350_000_000, 2_425_000_000, 2_500_000_000, 2_575_000_000, 2_650_000_000],
            'percentage_changes': {
                'parameter_pct_change': [-50, -25, 0, 25, 50],
                'metric_pct_change': [-6.0, -3.0, 0, 3.0, 6.0]
            },
            'num_scenarios': 5
        },
        'lmp_multiplier': {
            'parameter_name': 'lmp_multiplier',
            'metric': 'total_npv',
            'base_parameter_value': 1.0,
            'base_metric_value': 2_500_000_000,
            'elasticity': 0.45,
            'impact_score': 28.0,
            'parameter_values': [0.7, 0.85, 1.0, 1.15, 1.3],
            'metric_values': [2_150_000_000, 2_325_000_000, 2_500_000_000, 2_675_000_000, 2_850_000_000],
            'percentage_changes': {
                'parameter_pct_change': [-30, -15, 0, 15, 30],
                'metric_pct_change': [-14.0, -7.0, 0, 7.0, 14.0]
            },
            'num_scenarios': 5
        },
        'battery_cost_per_kwh': {
            'parameter_name': 'battery_cost_per_kwh',
            'metric': 'total_npv',
            'base_parameter_value': 350,
            'base_metric_value': 2_500_000_000,
            'elasticity': 0.08,
            'impact_score': 8.0,
            'parameter_values': [200, 275, 350, 425, 500],
            'metric_values': [2_400_000_000, 2_450_000_000, 2_500_000_000, 2_550_000_000, 2_600_000_000],
            'percentage_changes': {
                'parameter_pct_change': [-42.9, -21.4, 0, 21.4, 42.9],
                'metric_pct_change': [-4.0, -2.0, 0, 2.0, 4.0]
            },
            'num_scenarios': 5
        },
        'reliability_target': {
            'parameter_name': 'reliability_target',
            'metric': 'total_npv',
            'base_parameter_value': 0.9999,
            'base_metric_value': 2_500_000_000,
            'elasticity': 0.25,
            'impact_score': 18.0,
            'parameter_values': [0.999, 0.9995, 0.9999, 0.99995, 0.99999],
            'metric_values': [2_275_000_000, 2_387_500_000, 2_500_000_000, 2_612_500_000, 2_725_000_000],
            'percentage_changes': {
                'parameter_pct_change': [-0.09, -0.04, 0, 0.005, 0.01],
                'metric_pct_change': [-9.0, -4.5, 0, 4.5, 9.0]
            },
            'num_scenarios': 5
        },
        'discount_rate': {
            'parameter_name': 'discount_rate',
            'metric': 'total_npv',
            'base_parameter_value': 0.07,
            'base_metric_value': 2_500_000_000,
            'elasticity': -0.35,
            'impact_score': 22.0,
            'parameter_values': [0.05, 0.06, 0.07, 0.08, 0.09],
            'metric_values': [2_775_000_000, 2_637_500_000, 2_500_000_000, 2_362_500_000, 2_225_000_000],
            'percentage_changes': {
                'parameter_pct_change': [-28.6, -14.3, 0, 14.3, 28.6],
                'metric_pct_change': [11.0, 5.5, 0, -5.5, -11.0]
            },
            'num_scenarios': 5
        }
    }
    
    return sensitivity_results


def create_mock_multi_metric_results():
    """Create mock sensitivity results for multiple metrics."""
    
    base_results = create_mock_sensitivity_results()
    
    # Add carbon emissions metric
    param_names = list(base_results.keys())
    for param_name in param_names:
        base_results[param_name + '_carbon'] = {
            'parameter_name': param_name,
            'metric': 'carbon_tons_annual',
            'base_parameter_value': base_results[param_name]['base_parameter_value'],
            'base_metric_value': 150_000,
            'elasticity': 0.2,
            'impact_score': 15.0,
            'parameter_values': base_results[param_name]['parameter_values'],
            'metric_values': [140_000, 145_000, 150_000, 155_000, 160_000],
            'percentage_changes': {
                'parameter_pct_change': base_results[param_name]['percentage_changes']['parameter_pct_change'],
                'metric_pct_change': [-6.7, -3.3, 0, 3.3, 6.7]
            },
            'num_scenarios': 5
        }
    
    return base_results


def test_basic_tornado():
    """Test basic tornado chart."""
    print("Testing basic tornado chart...")
    
    sensitivity_results = create_mock_sensitivity_results()
    
    fig = plot_sensitivity_tornado(
        sensitivity_results,
        metric='total_npv',
        title="Sensitivity Analysis: Parameter Impact on Total NPV"
    )
    
    output_path = "results/figures/sensitivity_tornado_basic.html"
    fig.write_html(output_path)
    print(f"Saved basic tornado chart to {output_path}")
    
    return fig


def test_top_n_tornado():
    """Test tornado chart with top N parameters."""
    print("\nTesting top N tornado chart...")
    
    sensitivity_results = create_mock_sensitivity_results()
    
    fig = plot_sensitivity_tornado(
        sensitivity_results,
        metric='total_npv',
        title="Top 3 Most Impactful Parameters",
        top_n=3
    )
    
    output_path = "results/figures/sensitivity_tornado_top3.html"
    fig.write_html(output_path)
    print(f"Saved top 3 tornado chart to {output_path}")
    
    return fig


def test_no_values_tornado():
    """Test tornado chart without value labels."""
    print("\nTesting tornado chart without values...")
    
    sensitivity_results = create_mock_sensitivity_results()
    
    fig = plot_sensitivity_tornado(
        sensitivity_results,
        metric='total_npv',
        title="Sensitivity Analysis (Clean View)",
        show_values=False
    )
    
    output_path = "results/figures/sensitivity_tornado_clean.html"
    fig.write_html(output_path)
    print(f"Saved clean tornado chart to {output_path}")
    
    return fig


def test_comparison_chart():
    """Test multi-metric comparison chart."""
    print("\nTesting multi-metric comparison chart...")
    
    sensitivity_results = create_mock_multi_metric_results()
    
    fig = plot_sensitivity_comparison(
        sensitivity_results,
        metrics=['total_npv', 'carbon_tons_annual'],
        title="Parameter Impact Comparison: NPV vs Carbon"
    )
    
    output_path = "results/figures/sensitivity_comparison.html"
    fig.write_html(output_path)
    print(f"Saved comparison chart to {output_path}")
    
    return fig


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Sensitivity Tornado Chart Visualization")
    print("=" * 60)
    
    try:
        # Test basic tornado chart
        test_basic_tornado()
        
        # Test top N parameters
        test_top_n_tornado()
        
        # Test without value labels
        test_no_values_tornado()
        
        # Test comparison chart
        test_comparison_chart()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        print("\nGenerated visualizations:")
        print("  - results/figures/sensitivity_tornado_basic.html")
        print("  - results/figures/sensitivity_tornado_top3.html")
        print("  - results/figures/sensitivity_tornado_clean.html")
        print("  - results/figures/sensitivity_comparison.html")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
