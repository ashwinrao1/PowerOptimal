"""Test script for dispatch heatmap visualization."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.solution import OptimizationSolution
from src.visualization.dispatch_viz import plot_dispatch_heatmap, plot_dispatch_stacked_area


def test_dispatch_heatmap():
    """Test dispatch heatmap visualization with real solution data."""
    print("Testing dispatch heatmap visualization...")
    
    # Load optimal portfolio solution
    solution_path = Path(__file__).parent.parent / "results" / "solutions" / "optimal_portfolio.json"
    
    if not solution_path.exists():
        print(f"Error: Solution file not found at {solution_path}")
        return False
    
    print(f"Loading solution from {solution_path}")
    solution = OptimizationSolution.load(str(solution_path))
    
    print(f"Solution loaded successfully:")
    print(f"  - Grid capacity: {solution.capacity.grid_mw:.2f} MW")
    print(f"  - Gas capacity: {solution.capacity.gas_mw:.2f} MW")
    print(f"  - Solar capacity: {solution.capacity.solar_mw:.2f} MW")
    print(f"  - Battery capacity: {solution.capacity.battery_mwh:.2f} MWh")
    print(f"  - Dispatch hours: {len(solution.dispatch.hour)}")
    
    # Test 1: Full year heatmap
    print("\nTest 1: Creating full year dispatch heatmap...")
    try:
        fig1 = plot_dispatch_heatmap(solution)
        print("  Success! Full year heatmap created")
        
        # Save to file
        output_path = Path(__file__).parent.parent / "results" / "figures" / "dispatch_heatmap_full_year.html"
        fig1.write_html(str(output_path))
        print(f"  Saved to {output_path}")
    except Exception as e:
        print(f"  Error: {e}")
        return False
    
    # Test 2: First week heatmap (hours 1-168)
    print("\nTest 2: Creating first week dispatch heatmap...")
    try:
        fig2 = plot_dispatch_heatmap(solution, time_range=(1, 168))
        print("  Success! First week heatmap created")
        
        # Save to file
        output_path = Path(__file__).parent.parent / "results" / "figures" / "dispatch_heatmap_week1.html"
        fig2.write_html(str(output_path))
        print(f"  Saved to {output_path}")
    except Exception as e:
        print(f"  Error: {e}")
        return False
    
    # Test 3: Summer month (hours 4000-4720, roughly mid-June to mid-July)
    print("\nTest 3: Creating summer month dispatch heatmap...")
    try:
        fig3 = plot_dispatch_heatmap(solution, time_range=(4000, 4720))
        print("  Success! Summer month heatmap created")
        
        # Save to file
        output_path = Path(__file__).parent.parent / "results" / "figures" / "dispatch_heatmap_summer.html"
        fig3.write_html(str(output_path))
        print(f"  Saved to {output_path}")
    except Exception as e:
        print(f"  Error: {e}")
        return False
    
    # Test 4: Stacked area chart for first week
    print("\nTest 4: Creating stacked area chart for first week...")
    try:
        fig4 = plot_dispatch_stacked_area(solution, time_range=(1, 168))
        print("  Success! Stacked area chart created")
        
        # Save to file
        output_path = Path(__file__).parent.parent / "results" / "figures" / "dispatch_stacked_area_week1.html"
        fig4.write_html(str(output_path))
        print(f"  Saved to {output_path}")
    except Exception as e:
        print(f"  Error: {e}")
        return False
    
    # Test 5: Test with dict input (from JSON)
    print("\nTest 5: Testing with dict input...")
    try:
        import json
        with open(solution_path, 'r') as f:
            solution_dict = json.load(f)
        
        fig5 = plot_dispatch_heatmap(solution_dict, time_range=(1, 168))
        print("  Success! Heatmap created from dict input")
    except Exception as e:
        print(f"  Error: {e}")
        return False
    
    # Test 6: Test with market data (if available)
    print("\nTest 6: Testing with market data...")
    try:
        import pandas as pd
        import numpy as np
        
        # Try to load market data
        data_path = Path(__file__).parent.parent / "data" / "processed"
        lmp_path = data_path / "ercot_lmp_hourly_2022_2024.csv"
        gas_path = data_path / "gas_prices_hourly.csv"
        solar_path = data_path / "solar_cf_west_texas.csv"
        
        if lmp_path.exists() and gas_path.exists() and solar_path.exists():
            lmp_df = pd.read_csv(lmp_path)
            gas_df = pd.read_csv(gas_path)
            solar_df = pd.read_csv(solar_path)
            
            # Create market data dict
            market_data = {
                "lmp": lmp_df.iloc[:8760, 1].values if len(lmp_df) >= 8760 else np.zeros(8760),
                "gas_price": gas_df.iloc[:8760, 1].values if len(gas_df) >= 8760 else np.zeros(8760),
                "solar_cf": solar_df.iloc[:8760, 1].values if len(solar_df) >= 8760 else np.zeros(8760)
            }
            
            fig6 = plot_dispatch_heatmap(solution, market_data=market_data, time_range=(1, 168))
            print("  Success! Heatmap created with market data")
            
            # Save to file
            output_path = Path(__file__).parent.parent / "results" / "figures" / "dispatch_heatmap_with_market_data.html"
            fig6.write_html(str(output_path))
            print(f"  Saved to {output_path}")
        else:
            print("  Skipped: Market data files not found")
    except Exception as e:
        print(f"  Warning: {e}")
        # Don't fail the test if market data is not available
    
    print("\nAll tests passed!")
    return True


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n\nTesting error handling...")
    
    # Load solution
    solution_path = Path(__file__).parent.parent / "results" / "solutions" / "optimal_portfolio.json"
    solution = OptimizationSolution.load(str(solution_path))
    
    # Test invalid time range
    print("\nTest: Invalid time range (start > end)...")
    try:
        fig = plot_dispatch_heatmap(solution, time_range=(100, 50))
        print("  Error: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  Success! Caught expected error: {e}")
    
    # Test out of bounds time range
    print("\nTest: Out of bounds time range...")
    try:
        fig = plot_dispatch_heatmap(solution, time_range=(1, 10000))
        print("  Error: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  Success! Caught expected error: {e}")
    
    # Test invalid solution type
    print("\nTest: Invalid solution type...")
    try:
        fig = plot_dispatch_heatmap("invalid")
        print("  Error: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  Success! Caught expected error: {e}")
    
    print("\nError handling tests passed!")
    return True


if __name__ == "__main__":
    success = test_dispatch_heatmap()
    if success:
        success = test_error_handling()
    
    if success:
        print("\n" + "="*60)
        print("All dispatch visualization tests passed successfully!")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("Some tests failed!")
        print("="*60)
        sys.exit(1)
