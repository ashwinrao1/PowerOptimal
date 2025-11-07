# Implementation Plan

- [x] 1. Set up project structure and configuration
  - Create directory structure for data, src, tests, notebooks, dashboard, docs, and results
  - Create requirements.txt with all Python dependencies (Pyomo, Gurobi, Pandas, NumPy, Plotly, Streamlit, pytest)
  - Create .gitignore file to exclude data files, results, and virtual environment
  - Create README.md with project overview and setup instructions
  - Create configuration management module for storing constants and parameters
  - _Requirements: 12.1, 12.2_

- [x] 2. Implement data pipeline for ERCOT LMP collection
  - Create ercot_collector.py module with function to fetch hourly LMP data from ERCOT API
  - Implement retry logic with exponential backoff for API failures
  - Parse XML/CSV responses and extract Day-Ahead Market (DAM) and Real-Time Market (RTM) prices
  - Handle missing data using forward-fill for gaps less than 1%
  - Save processed data to data/processed/ercot_lmp_hourly_2022_2024.csv
  - _Requirements: 6.1_

- [x] 3. Implement solar profile generation
  - Create solar_collector.py module with function to call NREL PVWatts API
  - Configure parameters: West Texas coordinates (31.9973°N, 102.0779°W), fixed tilt at 32°, south-facing azimuth
  - Generate 8760 hourly capacity factors for typical meteorological year
  - Validate capacity factors are between 0 and 1
  - Save to data/processed/solar_cf_west_texas.csv
  - _Requirements: 6.2_

- [x] 4. Implement natural gas price collection
  - Create gas_collector.py module to fetch daily Waha Hub prices from EIA API
  - Interpolate daily prices to hourly using 10% peak/off-peak differential pattern
  - Validate prices are non-negative and within reasonable range (0-50 $/MMBtu)
  - Save to data/processed/gas_prices_hourly.csv
  - _Requirements: 6.3_

- [x] 5. Implement grid carbon intensity collection
  - Create carbon_collector.py module to fetch hourly ERCOT West carbon intensity from EIA Electric Grid Monitor
  - Convert units to kg CO2/MWh if needed
  - Validate data completeness and reasonable ranges
  - Save to data/processed/grid_carbon_intensity.csv
  - _Requirements: 6.4_

- [x] 6. Create technology cost database
  - Manually compile technology costs from NREL ATB 2024 documentation
  - Create JSON structure with CAPEX and OPEX for grid, gas, battery, and solar
  - Include parameters: heat rate, efficiency, degradation costs, O&M costs
  - Save to data/tech_costs.json
  - _Requirements: 6.5_

- [x] 7. Implement data validation module
  - Create validator.py with functions to check data completeness (no more than 1% missing)
  - Validate timestamps are continuous and cover full year (8760 hours)
  - Validate values are within expected ranges for each data type
  - Implement forward-fill for small gaps and raise exceptions for large gaps
  - _Requirements: 6.5_

- [ ]* 7.1 Write unit tests for data pipeline
  - Create test_data_pipeline.py with mock API responses
  - Test each collector independently with synthetic data
  - Test validation logic catches invalid values and missing data
  - Test interpolation and gap-filling functions
  - _Requirements: 6.5_

- [x] 8. Create data model classes
  - Create models/market_data.py with MarketData dataclass containing timestamp, lmp, gas_price, solar_cf, grid_carbon_intensity arrays
  - Implement validate() method to ensure data completeness and value ranges
  - Create models/technology.py with TechnologyCosts and FacilityParams dataclasses
  - Create models/solution.py with CapacitySolution, DispatchSolution, SolutionMetrics, and OptimizationSolution dataclasses
  - Implement serialization methods (to_dict, to_dataframe, save, load) for solution classes
  - _Requirements: 1.2, 1.3_

- [x] 9. Build core optimization model with Pyomo
  - Create optimization/model_builder.py with build_optimization_model() function
  - Define decision variables: capacity variables (C_grid, C_gas, C_battery, C_solar) and hourly dispatch variables (p_grid, p_gas, p_battery, p_curtail, p_solar, SOC)
  - Implement objective function: minimize CAPEX + NPV(OPEX) + NPV(Curtailment_Penalty) with 20-year horizon and 7% discount rate
  - Calculate CAPEX using technology costs: grid ($3000/kW), gas ($1000/kW), battery ($350/kWh), solar ($1200/kW)
  - Calculate annual OPEX including grid electricity costs, demand charges, gas fuel costs, gas O&M, battery degradation, and solar O&M
  - _Requirements: 1.1, 1.4, 1.5_

- [x] 10. Implement optimization model constraints
  - Implement energy balance constraint for all 8760 hours: grid + gas + solar - battery + curtailment = load
  - Implement capacity limit constraints: grid power ≤ grid capacity, gas power ≤ gas capacity, battery power within ±25% of capacity
  - Implement solar generation constraint: solar power = solar capacity × capacity factor
  - Implement battery dynamics: SOC[h] = SOC[h-1] + battery_power × efficiency × timestep
  - Implement battery SOC limits: 10% to 90% of capacity for battery health
  - Implement battery periodicity: SOC at hour 8760 equals SOC at hour 1
  - Implement gas ramping constraint: change in gas power ≤ 50% of capacity per hour
  - Implement reliability constraint: total annual curtailment ≤ 2.85 MWh (1 hour for 285 MW load)
  - Implement optional carbon constraint: total emissions ≤ carbon budget when specified
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 5.1, 5.2, 5.3, 5.4, 7.4_

- [x] 11. Implement solver interface
  - Create optimization/solver.py with solve_model() function
  - Configure Gurobi solver with MIPGap=0.005, TimeLimit=1800 seconds, automatic thread selection
  - Set solver method to barrier for LP problems
  - Implement error handling for infeasibility, unbounded, numerical issues, and timeout
  - Return solver results and solve time
  - _Requirements: 12.1, 12.2, 12.3, 12.5_

- [x] 12. Implement solution extraction and validation
  - Create optimization/solution_extractor.py with extract_solution() function
  - Extract capacity decisions from solved model variables
  - Extract hourly dispatch decisions for all 8760 hours
  - Calculate solution metrics: total NPV, CAPEX, annual OPEX, LCOE, reliability percentage, curtailment statistics
  - Calculate carbon metrics: annual emissions, carbon intensity, reduction vs grid-only baseline
  - Calculate operational metrics: grid dependence, gas capacity factor, battery cycles, solar capacity factor
  - Create optimization/validator.py to verify all constraints are satisfied in solution
  - Check energy balance at every hour with numerical tolerance
  - _Requirements: 1.3, 9.1, 9.2, 9.3, 9.4, 9.5, 12.4_

- [ ]* 12.1 Write unit tests for optimization model
  - Create test_optimization_model.py with small test datasets (24 hours)
  - Test model construction creates correct number of variables and constraints
  - Test objective function calculation matches manual computation
  - Test constraint generation for energy balance, capacity limits, battery dynamics
  - Test solution extraction produces correct data structures
  - _Requirements: 12.1_

- [x] 13. Implement baseline grid-only scenario
  - Create script to run optimization with zero capacity for gas, battery, and solar
  - Only allow grid connection to meet full load
  - Calculate total cost, reliability, and carbon emissions for baseline
  - Save baseline solution to results/solutions/baseline_grid_only.json
  - Document baseline metrics for comparison with optimal portfolio
  - _Requirements: 9.5, 11.2_

- [x] 14. Implement optimal portfolio scenario
  - Run optimization with all technologies available (grid, gas, battery, solar)
  - Allow optimizer to determine optimal capacity mix
  - Extract and save optimal solution to results/solutions/optimal_portfolio.json
  - Calculate cost savings, reliability improvement, and carbon reduction vs baseline
  - _Requirements: 9.1, 9.2, 9.5_

- [ ]* 14.1 Create integration test for end-to-end optimization
  - Create test_end_to_end.py with 1-week test dataset
  - Test complete pipeline from data loading to solution extraction
  - Verify solution satisfies all constraints (energy balance, capacity limits, reliability)
  - Verify solution quality (objective value in expected range, no curtailment violations)
  - _Requirements: 12.1, 12.4_

- [x] 15. Implement scenario generator for sensitivity analysis
  - Create analysis/scenario_generator.py with generate_scenarios() function
  - Implement parameter variation logic for gas prices (±50%), grid LMPs (±30%), battery costs ($200-500/kWh)
  - Implement reliability target variations (99.9%, 99.99%, 99.999%)
  - Implement carbon constraint variations (no constraint, 50% reduction, 80% reduction, 100% carbon-free)
  - Generate list of parameter dictionaries for all scenario combinations
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 16. Implement batch solver for parallel scenario execution
  - Create analysis/batch_solver.py with solve_scenarios() function
  - Use Python multiprocessing to solve independent scenarios concurrently
  - Configure number of worker processes based on available CPU cores
  - Collect results from all scenarios into list of solution dictionaries
  - Handle solver failures gracefully and log errors
  - Save scenario results to results/scenario_results.csv
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 17. Implement Pareto frontier calculator
  - Create analysis/pareto_calculator.py with calculate_pareto_frontier() function
  - Implement algorithm to identify non-dominated solutions for two objectives
  - Support objective pairs: cost vs reliability, cost vs carbon, grid dependence vs reliability
  - Return DataFrame of Pareto-optimal solutions
  - Save Pareto frontier data to results/pareto_frontiers.json
  - _Requirements: 9.4_

- [x] 18. Implement sensitivity analyzer
  - Create analysis/sensitivity_analyzer.py with analyze_sensitivity() function
  - Calculate elasticity: percentage change in cost per percentage change in parameter
  - Identify breakeven points where optimal decisions change
  - Rank parameters by impact on total NPV
  - Generate sensitivity metrics for all varied parameters
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 19. Create capacity mix visualizations
  - Create visualization/capacity_viz.py with plot_capacity_mix() function
  - Implement stacked bar chart showing MW capacity for each technology
  - Implement pie chart showing percentage breakdown of capacity
  - Use Plotly for interactive visualizations with hover tooltips
  - Support multiple format options (bar, pie, waterfall)
  - _Requirements: 10.2_

- [x] 20. Create dispatch heatmap visualization
  - Create visualization/dispatch_viz.py with plot_dispatch_heatmap() function
  - Implement 2D heatmap with hours (1-8760) on x-axis and power sources on y-axis
  - Color code by MW contribution from each source
  - Add hover tooltips showing LMP, gas price, and solar capacity factor for each hour
  - Support time range selection for zooming into specific periods
  - _Requirements: 10.3_

- [x] 21. Create cost breakdown visualizations
  - Create visualization/cost_viz.py with plot_cost_breakdown() function
  - Implement waterfall chart showing CAPEX and OPEX components
  - Break down costs by technology: grid CAPEX/OPEX, gas CAPEX/OPEX, battery CAPEX/OPEX, solar CAPEX/OPEX
  - Show total NPV and annual costs
  - _Requirements: 10.4_

- [x] 22. Create Pareto frontier plots
  - Create visualization/pareto_viz.py with plot_pareto_frontier() function
  - Implement scatter plot with two objectives on x and y axes
  - Highlight Pareto-optimal solutions with different color/marker
  - Add annotations for key solutions (baseline, optimal, extreme points)
  - Support multiple objective pairs
  - _Requirements: 9.4_

- [x] 23. Create reliability analysis visualizations
  - Create visualization/reliability_viz.py with plot_reliability_analysis() function
  - Implement histogram of hourly curtailment events
  - Create time series plot showing reserve margin over the year
  - Identify and visualize top 10 worst-case reliability events
  - Show statistics: total curtailment hours, maximum single-hour curtailment
  - _Requirements: 4.4, 4.5, 10.5_

- [x] 24. Create sensitivity tornado chart
  - Add plot_sensitivity_tornado() function to visualization module
  - Implement horizontal bar chart showing parameter impacts on NPV
  - Sort parameters by magnitude of impact (largest at top)
  - Show both positive and negative variations
  - _Requirements: 8.1, 8.2, 8.3_

- [-] 25. Build Streamlit dashboard structure
  - Create dashboard/app.py as main entry point with page navigation
  - Implement sidebar with page selection: Optimization Setup, Optimal Portfolio, Hourly Dispatch, Scenario Comparison, Case Study
  - Set up session state management to persist optimization results and user inputs
  - Configure page layout and theme settings
  - _Requirements: 10.1_

- [ ] 26. Implement optimization setup page
  - Create dashboard/pages/setup.py with input widgets
  - Add sliders for facility size (100-500 MW), reliability target (99.9-99.999%), carbon reduction (0-100%)
  - Add dropdown for location selection and year scenario
  - Add checkboxes for available technologies (grid, gas, battery, solar)
  - Add "Run Optimization" button that triggers model solving
  - Show progress bar and status during optimization
  - Store results in session state when complete
  - _Requirements: 10.1_

- [ ] 27. Implement optimal portfolio page
  - Create dashboard/pages/portfolio.py to display optimization results
  - Show capacity mix visualization (bar chart) using plot_capacity_mix()
  - Show cost breakdown (waterfall chart) using plot_cost_breakdown()
  - Display key metrics cards: total NPV, LCOE, reliability %, carbon intensity, grid independence %
  - Add download button to export results to CSV/JSON
  - _Requirements: 10.2, 10.4_

- [ ] 28. Implement hourly dispatch page
  - Create dashboard/pages/dispatch.py to visualize hourly operations
  - Display dispatch heatmap for full year using plot_dispatch_heatmap()
  - Add time range selector for zooming into specific weeks or days
  - Show statistics panel: gas utilization hours, battery cycles, solar capacity factor, peak grid draw
  - _Requirements: 10.3_

- [ ] 29. Implement scenario comparison page
  - Create dashboard/pages/scenarios.py for multi-scenario analysis
  - Add controls to configure and run multiple scenarios
  - Display Pareto frontier plots for cost vs reliability and cost vs carbon
  - Show scenario comparison table with key metrics
  - Display sensitivity tornado chart
  - _Requirements: 10.5_

- [ ] 30. Implement case study page
  - Create dashboard/pages/case_study.py with pre-computed 300MW West Texas results
  - Display baseline (grid-only) and optimal portfolio solutions side-by-side
  - Show narrative explanation of findings and recommendations
  - Compare to alternative strategies (e.g., Microsoft Three Mile Island approach)
  - Add download button for case study report
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [ ] 31. Implement dashboard caching and performance optimization
  - Add @st.cache_data decorator to data loading functions
  - Add @st.cache_resource decorator for solver initialization
  - Implement lazy loading for visualizations (only render when page is viewed)
  - Pre-compute common scenarios and cache results
  - _Requirements: 12.2_

- [ ]* 31.1 Test dashboard functionality
  - Create test_dashboard.py to verify all pages render without errors
  - Test input widgets trigger correct behavior
  - Test session state persistence across page changes
  - Test download functionality for results export
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 32. Run 300MW West Texas case study analysis
  - Load ERCOT West Texas data for 2022-2024
  - Run baseline optimization with grid-only configuration
  - Run optimal portfolio optimization with all technologies available
  - Calculate upfront investment, annual savings, payback period
  - Generate all visualizations for case study
  - Save results to results/solutions/case_study_results.json
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [ ] 33. Create case study report document
  - Write docs/case_study.md with problem statement for 300MW West Texas facility
  - Document baseline scenario: costs, reliability, carbon emissions
  - Document optimal portfolio: recommended capacity mix and rationale
  - Calculate and present financial analysis: CAPEX, OPEX savings, NPV, payback period
  - Compare to alternative strategies and provide strategic insights
  - Include key visualizations: capacity mix, dispatch heatmap, cost breakdown
  - _Requirements: 11.5_

- [ ] 34. Create comprehensive README documentation
  - Write project overview explaining the business problem and solution approach
  - Document setup instructions: clone repo, create virtual environment, install dependencies
  - Provide quick start guide: download data, run baseline optimization, launch dashboard
  - Document project structure and key files
  - Add links to detailed documentation (design, case study, API reference)
  - Include example usage and screenshots
  - _Requirements: 12.1_

- [ ]* 35. Create mathematical model formulation document
  - Write docs/model_formulation.tex in LaTeX format
  - Document decision variables, objective function, and all constraints with mathematical notation
  - Explain problem formulation and modeling choices
  - Include problem size analysis (number of variables and constraints)
  - Compile to docs/model_formulation.pdf
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 5.1, 5.2, 5.3, 5.4, 5.5, 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 36. Create utility scripts for common operations
  - Create scripts/download_all_data.py to automate data collection from all sources
  - Create scripts/run_baseline.py to run baseline grid-only optimization
  - Create scripts/run_scenarios.py to execute all scenario analyses
  - Add error handling and progress reporting to all scripts
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 37. Deploy dashboard to Streamlit Cloud
  - Push code to GitHub repository with clean commit history
  - Connect Streamlit Cloud account to GitHub repository
  - Configure Python version (3.10+) and dependencies
  - Test deployed dashboard functionality
  - Document deployment URL in README
  - _Requirements: 10.1_
