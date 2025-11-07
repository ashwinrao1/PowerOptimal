"""
Example usage of the solver interface for data center energy optimization.

This script demonstrates how to:
1. Build an optimization model
2. Solve it using the Gurobi solver
3. Handle various solver outcomes
4. Validate the solution
"""

import pandas as pd
from src.models.market_data import MarketData
from src.models.technology import TechnologyCosts, FacilityParams
from src.optimization import (
    build_optimization_model,
    solve_model,
    get_solver_info,
    validate_solution,
    InfeasibleError,
    UnboundedError,
    NumericalError,
    TimeoutError,
    SolverError
)
import pyomo.environ as pyo


def main():
    """Example workflow for solving the optimization model."""
    
    # Step 1: Check if Gurobi is available
    print("Checking Gurobi availability...")
    solver_info = get_solver_info()
    
    if not solver_info['available']:
        print("ERROR: Gurobi solver is not available.")
        print("Please install Gurobi and obtain a license.")
        print("Visit: https://www.gurobi.com/downloads/")
        return
    
    print(f"Gurobi is available (version: {solver_info.get('version', 'unknown')})")
    
    # Step 2: Load market data
    print("\nLoading market data...")
    lmp_data = pd.read_csv('data/processed/ercot_lmp_hourly_2022_2024.csv')
    solar_data = pd.read_csv('data/processed/solar_cf_west_texas.csv')
    gas_data = pd.read_csv('data/processed/gas_prices_hourly.csv')
    carbon_data = pd.read_csv('data/processed/grid_carbon_intensity.csv')
    
    # Create MarketData object (assuming proper data structure)
    market_data = MarketData(
        timestamp=pd.to_datetime(lmp_data['timestamp']),
        lmp=lmp_data['lmp'].values,
        gas_price=gas_data['price'].values,
        solar_cf=solar_data['capacity_factor'].values,
        grid_carbon_intensity=carbon_data['carbon_intensity'].values
    )
    
    # Step 3: Set up technology costs and facility parameters
    print("Setting up optimization parameters...")
    tech_costs = TechnologyCosts()
    facility_params = FacilityParams(
        it_load_mw=300,
        reliability_target=0.9999,
        carbon_budget=None  # No carbon constraint for this example
    )
    
    # Step 4: Build optimization model
    print("Building optimization model...")
    model = build_optimization_model(
        market_data=market_data,
        tech_costs=tech_costs,
        facility_params=facility_params,
        allow_gas=True,
        allow_battery=True,
        allow_solar=True
    )
    
    print(f"Model built with {len(model.hours)} hours")
    
    # Step 5: Solve the model
    print("\nSolving optimization model...")
    print("This may take several minutes...")
    
    try:
        results, solve_time = solve_model(
            model=model,
            time_limit=1800,  # 30 minutes
            mip_gap=0.005,    # 0.5% optimality gap
            verbose=True
        )
        
        print(f"\nOptimization completed successfully!")
        print(f"Solve time: {solve_time:.2f} seconds ({solve_time/60:.2f} minutes)")
        
        # Step 6: Extract solution
        print("\nOptimal Solution:")
        print(f"  Total Cost (NPV): ${pyo.value(model.total_cost):,.0f}")
        print(f"  Grid Capacity: {pyo.value(model.C_grid):.2f} MW")
        print(f"  Gas Capacity: {pyo.value(model.C_gas):.2f} MW")
        print(f"  Battery Capacity: {pyo.value(model.C_battery):.2f} MWh")
        print(f"  Solar Capacity: {pyo.value(model.C_solar):.2f} MW")
        
        # Step 7: Validate solution
        print("\nValidating solution...")
        is_valid, violations = validate_solution(model, tolerance=1e-6)
        
        if is_valid:
            print("Solution is valid - all constraints satisfied!")
        else:
            print(f"WARNING: Solution has {len(violations)} constraint violations:")
            for violation in violations[:10]:  # Show first 10
                print(f"  - {violation}")
        
    except InfeasibleError as e:
        print(f"\nERROR: Problem is infeasible")
        print(e)
        print("\nSuggestions:")
        print("  - Relax reliability target")
        print("  - Increase capacity bounds")
        print("  - Check carbon budget constraints")
        
    except UnboundedError as e:
        print(f"\nERROR: Problem is unbounded")
        print(e)
        print("\nThis indicates a model formulation error.")
        
    except NumericalError as e:
        print(f"\nERROR: Numerical difficulties")
        print(e)
        
    except TimeoutError as e:
        print(f"\nWARNING: Solver timeout")
        print(e)
        print("\nA feasible solution may still be available.")
        
    except SolverError as e:
        print(f"\nERROR: Solver error")
        print(e)


if __name__ == "__main__":
    main()
