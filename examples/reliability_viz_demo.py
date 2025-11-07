"""Demo script showing all reliability visualization features with mock data."""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.solution import (
    OptimizationSolution,
    CapacitySolution,
    DispatchSolution,
    SolutionMetrics
)
from src.visualization.reliability_viz import (
    plot_reliability_analysis,
    plot_curtailment_histogram,
    plot_reserve_margin_timeseries,
    plot_worst_reliability_events
)


def create_mock_solution_with_curtailment():
    """Create a mock solution with some curtailment events for demonstration."""
    
    print("Creating mock solution with curtailment events...")
    
    # Create capacity solution
    capacity = CapacitySolution(
        grid_mw=200.0,
        gas_mw=100.0,
        battery_mwh=400.0,
        solar_mw=150.0
    )
    
    # Create dispatch solution with 8760 hours
    hours = 8760
    hour = np.arange(1, hours + 1)
    
    # Base dispatch pattern
    load_mw = 285.0
    
    # Grid provides base load
    grid_power = np.full(hours, 150.0)
    
    # Gas provides peaking
    gas_power = np.full(hours, 50.0)
    # Add some variation
    gas_power += np.random.uniform(-10, 20, hours)
    gas_power = np.clip(gas_power, 0, 100)
    
    # Solar follows daily pattern
    solar_power = np.zeros(hours)
    for h in range(hours):
        hour_of_day = h % 24
        if 6 <= hour_of_day <= 18:
            # Daytime solar generation
            solar_cf = np.sin((hour_of_day - 6) * np.pi / 12) * 0.7
            solar_power[h] = 150.0 * solar_cf
    
    # Battery charges during day, discharges at night
    battery_power = np.zeros(hours)
    battery_soc = np.full(hours, 200.0)  # Start at 50% SOC
    
    for h in range(hours):
        hour_of_day = h % 24
        if 10 <= hour_of_day <= 16:
            # Charge during peak solar
            battery_power[h] = 30.0
        elif 18 <= hour_of_day <= 22:
            # Discharge during evening peak
            battery_power[h] = -40.0
        
        # Update SOC
        if h > 0:
            battery_soc[h] = battery_soc[h-1] + battery_power[h] * 0.85
            battery_soc[h] = np.clip(battery_soc[h], 40, 360)  # 10-90% limits
    
    # Add curtailment events during stress periods
    curtailment = np.zeros(hours)
    
    # Simulate 5 major curtailment events (extreme weather, grid outages)
    stress_hours = [1234, 2456, 3678, 5432, 7890]
    for stress_hour in stress_hours:
        if stress_hour < hours:
            # Major event with 2-3 hour duration
            for offset in range(3):
                h = stress_hour + offset
                if h < hours:
                    curtailment[h] = np.random.uniform(5, 25)
    
    # Add 10 minor curtailment events
    minor_events = np.random.choice(hours, 10, replace=False)
    for h in minor_events:
        curtailment[h] = np.random.uniform(1, 8)
    
    # Create dispatch solution
    dispatch = DispatchSolution(
        hour=hour,
        grid_power=grid_power,
        gas_power=gas_power,
        solar_power=solar_power,
        battery_power=battery_power,
        curtailment=curtailment,
        battery_soc=battery_soc
    )
    
    # Calculate metrics
    total_curtailment_mwh = curtailment.sum()
    num_curtailment_hours = np.sum(curtailment > 0.01)
    reliability_pct = (1 - total_curtailment_mwh / (load_mw * hours)) * 100
    
    metrics = SolutionMetrics(
        total_npv=2_500_000_000,
        capex=450_000_000,
        opex_annual=85_000_000,
        lcoe=45.5,
        reliability_pct=reliability_pct,
        total_curtailment_mwh=total_curtailment_mwh,
        num_curtailment_hours=num_curtailment_hours,
        carbon_tons_annual=125_000,
        carbon_intensity_g_per_kwh=450,
        carbon_reduction_pct=35.0,
        grid_dependence_pct=55.0,
        gas_capacity_factor=0.45,
        battery_cycles_per_year=250,
        solar_capacity_factor=0.28,
        solve_time_seconds=450.0,
        optimality_gap_pct=0.3
    )
    
    # Create complete solution
    solution = OptimizationSolution(
        capacity=capacity,
        dispatch=dispatch,
        metrics=metrics,
        scenario_params={
            "load_mw": load_mw,
            "reliability_target": 0.9999,
            "location": "West Texas"
        }
    )
    
    print(f"Mock solution created:")
    print(f"  Reliability: {reliability_pct:.4f}%")
    print(f"  Total curtailment: {total_curtailment_mwh:.2f} MWh")
    print(f"  Curtailment hours: {num_curtailment_hours}")
    
    return solution


def main():
    """Run reliability visualization demo."""
    
    print("="*70)
    print("Reliability Visualization Demo")
    print("="*70)
    
    # Create mock solution with curtailment
    solution = create_mock_solution_with_curtailment()
    
    print("\n" + "="*70)
    print("Generating Visualizations...")
    print("="*70)
    
    # 1. Comprehensive reliability analysis
    print("\n1. Comprehensive Reliability Analysis (4-panel view)...")
    fig = plot_reliability_analysis(
        solution,
        title="Reliability Analysis - Mock Scenario with Curtailment Events"
    )
    output_path = "results/figures/reliability_demo_comprehensive.html"
    fig.write_html(output_path)
    print(f"   Saved to: {output_path}")
    
    # 2. Curtailment histogram
    print("\n2. Curtailment Event Histogram...")
    fig = plot_curtailment_histogram(
        solution,
        title="Distribution of Curtailment Events - Mock Scenario",
        bins=20
    )
    output_path = "results/figures/reliability_demo_histogram.html"
    fig.write_html(output_path)
    print(f"   Saved to: {output_path}")
    
    # 3. Reserve margin - full year
    print("\n3. Reserve Margin Time Series - Full Year...")
    fig = plot_reserve_margin_timeseries(
        solution,
        title="Reserve Margin Over Full Year - Mock Scenario"
    )
    output_path = "results/figures/reliability_demo_reserve_full.html"
    fig.write_html(output_path)
    print(f"   Saved to: {output_path}")
    
    # 4. Reserve margin - summer period
    print("\n4. Reserve Margin Time Series - Summer Period...")
    fig = plot_reserve_margin_timeseries(
        solution,
        time_range=(4000, 5000),
        title="Reserve Margin - Summer Period (Hours 4000-5000)"
    )
    output_path = "results/figures/reliability_demo_reserve_summer.html"
    fig.write_html(output_path)
    print(f"   Saved to: {output_path}")
    
    # 5. Worst reliability events
    print("\n5. Top 10 Worst Reliability Events...")
    fig = plot_worst_reliability_events(
        solution,
        n_events=10,
        title="Top 10 Worst Reliability Events - Mock Scenario"
    )
    output_path = "results/figures/reliability_demo_worst_events.html"
    fig.write_html(output_path)
    print(f"   Saved to: {output_path}")
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print("\nGenerated visualizations demonstrate:")
    print("  - Multi-panel comprehensive reliability analysis")
    print("  - Curtailment event distribution histogram")
    print("  - Reserve margin time series (full year and zoomed)")
    print("  - Identification of worst-case reliability events")
    print("\nAll visualizations are interactive HTML files that can be opened")
    print("in a web browser for detailed exploration.")
    print("\nFiles saved to results/figures/")


if __name__ == "__main__":
    main()
