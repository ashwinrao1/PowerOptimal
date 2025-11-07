# Requirements Document

## Introduction

This document specifies requirements for an optimization model that determines the optimal energy portfolio and hourly dispatch strategy for large-scale AI training data centers. The system addresses the trilemma of cost minimization, reliability maximization, and carbon emission reduction while considering behind-the-meter power generation options including grid connection, natural gas peakers, battery storage, and solar PV.

## Glossary

- **Optimization Model**: The mathematical programming system that determines optimal capacity investments and hourly dispatch decisions
- **Energy Portfolio**: The combination of power generation and storage assets (grid connection, natural gas peakers, battery storage, solar PV)
- **Dispatch Strategy**: The hourly operational decisions for how much power to draw from each energy source
- **Behind-the-Meter (BTM)**: Power generation assets located at the data center site, not purchased from the grid
- **Data Center Load**: The constant 285-315 MW power requirement for AI training operations
- **LMP (Locational Marginal Price)**: The hourly wholesale electricity price at a specific grid location
- **Curtailment**: Emergency reduction of data center load when power supply is insufficient
- **ERCOT**: Electric Reliability Council of Texas, the independent system operator for the Texas power grid
- **PUE (Power Usage Effectiveness)**: Ratio of total facility power to IT equipment power (1.05 = 5% overhead)
- **CAPEX**: Capital expenditure for building power generation and storage assets
- **OPEX**: Operating expenditure for fuel, maintenance, and grid electricity purchases
- **NPV (Net Present Value)**: Total 20-year cost discounted to present value
- **Capacity Factor**: The ratio of actual solar generation to maximum possible generation
- **State of Charge (SOC)**: The current energy stored in the battery system (MWh)

## Requirements

### Requirement 1: Optimization Model Core

**User Story:** As a data center CFO, I want an optimization model that minimizes total 20-year costs, so that I can make informed investment decisions about energy infrastructure.

#### Acceptance Criteria

1. THE Optimization Model SHALL minimize the sum of capital expenditures, net present value of operating expenditures, and net present value of curtailment penalties over a 20-year planning horizon
2. THE Optimization Model SHALL determine optimal capacity investments for grid interconnection (MW), natural gas peaker capacity (MW), battery storage capacity (MWh), and solar PV capacity (MW)
3. THE Optimization Model SHALL determine hourly dispatch decisions for 8760 hours per year including grid power draw, gas generation, battery charge/discharge, and load curtailment
4. THE Optimization Model SHALL incorporate capital costs of $3000/kW for grid interconnection, $1000/kW for gas peakers, $350/kWh for battery storage, and $1200/kW for solar PV
5. THE Optimization Model SHALL incorporate operating costs including hourly grid electricity prices, monthly demand charges, natural gas fuel costs, operations and maintenance costs, and battery degradation costs

### Requirement 2: Energy Balance and Dispatch

**User Story:** As a data center operator, I want the system to maintain continuous power supply through optimal dispatch, so that AI training workloads run without interruption.

#### Acceptance Criteria

1. WHEN dispatching power for each hour, THE Optimization Model SHALL ensure that the sum of grid power, gas generation, solar generation, and battery discharge minus battery charge plus curtailment equals the data center load of 285 MW
2. THE Optimization Model SHALL enforce that grid power draw does not exceed the installed grid interconnection capacity for any hour
3. THE Optimization Model SHALL enforce that natural gas generation does not exceed the installed gas peaker capacity for any hour
4. THE Optimization Model SHALL enforce that battery charge and discharge rates do not exceed 25% of battery capacity per hour (4-hour battery constraint)
5. THE Optimization Model SHALL calculate solar generation as the product of installed solar capacity and hourly capacity factor from weather data

### Requirement 3: Battery Storage Operations

**User Story:** As an energy systems engineer, I want realistic battery operation constraints, so that the model produces implementable dispatch strategies.

#### Acceptance Criteria

1. THE Optimization Model SHALL track battery state of charge across all hours using the equation: SOC(h) = SOC(h-1) + battery_power(h) × efficiency × time_step
2. THE Optimization Model SHALL enforce that battery state of charge remains between 10% and 90% of battery capacity at all hours to protect battery health
3. THE Optimization Model SHALL apply round-trip efficiency of 85% for battery charge and discharge operations
4. THE Optimization Model SHALL incorporate battery degradation costs of $5/MWh for all charge and discharge operations
5. THE Optimization Model SHALL ensure battery state of charge at hour 8760 equals the state of charge at hour 1 for annual periodicity

### Requirement 4: Reliability Requirements

**User Story:** As a data center CTO, I want guaranteed reliability of 99.99% uptime, so that expensive AI training runs are not interrupted.

#### Acceptance Criteria

1. THE Optimization Model SHALL enforce that total annual curtailment does not exceed 2.85 MWh per year (equivalent to 1 hour of downtime for 285 MW load)
2. THE Optimization Model SHALL apply a curtailment penalty of $10,000/MWh to represent the cost of lost compute time and wasted GPU resources
3. THE Optimization Model SHALL allow curtailment as an emergency variable to maintain model feasibility during extreme grid events
4. THE Optimization Model SHALL report the number of hours with non-zero curtailment and the maximum single-hour curtailment in the solution summary
5. THE Optimization Model SHALL identify the worst-case reliability events by reporting the top 10 hours with highest risk of curtailment

### Requirement 5: Natural Gas Operations

**User Story:** As an environmental compliance officer, I want realistic natural gas peaker constraints, so that we can accurately assess emissions and operational feasibility.

#### Acceptance Criteria

1. THE Optimization Model SHALL enforce ramping limits such that gas generation change between consecutive hours does not exceed 50% of installed gas capacity
2. THE Optimization Model SHALL calculate gas fuel costs using hourly natural gas prices, a heat rate of 10 MMBtu/MWh, and thermal efficiency of 35%
3. THE Optimization Model SHALL apply operations and maintenance costs of $15/MWh for all gas generation
4. THE Optimization Model SHALL calculate carbon emissions from gas generation at 0.4 metric tons CO2 per MWh generated
5. THE Optimization Model SHALL report total annual gas generation hours and capacity factor in the solution summary

### Requirement 6: Data Integration

**User Story:** As a data analyst, I want the system to ingest real-world market data, so that optimization results reflect actual market conditions.

#### Acceptance Criteria

1. THE Optimization Model SHALL accept hourly ERCOT locational marginal prices for West Texas hub from 2022-2024 as input data
2. THE Optimization Model SHALL accept hourly solar capacity factors from NREL PVWatts for West Texas location (31.9973°N, 102.0779°W) as input data
3. THE Optimization Model SHALL accept hourly natural gas prices from EIA Waha Hub interpolated from daily data as input data
4. THE Optimization Model SHALL accept hourly grid carbon intensity from EIA Electric Grid Monitor for ERCOT West region as input data
5. THE Optimization Model SHALL validate all input data for completeness and handle missing values through forward-fill interpolation for gaps less than 1% of total data points

### Requirement 7: Carbon Emissions Tracking

**User Story:** As a sustainability manager, I want to track and optionally constrain carbon emissions, so that we can meet corporate climate commitments.

#### Acceptance Criteria

1. THE Optimization Model SHALL calculate hourly carbon emissions as the sum of grid emissions (grid power × grid carbon intensity) and gas emissions (gas generation × 0.4 tons CO2/MWh)
2. THE Optimization Model SHALL report total annual carbon emissions in metric tons CO2 in the solution summary
3. THE Optimization Model SHALL report average carbon intensity in grams CO2 per kWh in the solution summary
4. WHERE a carbon budget is specified, THE Optimization Model SHALL enforce that total annual emissions do not exceed the specified carbon budget
5. THE Optimization Model SHALL compare total emissions to a grid-only baseline and report the percentage carbon reduction achieved

### Requirement 8: Scenario Analysis

**User Story:** As a strategic planner, I want to analyze multiple scenarios with different assumptions, so that I can understand sensitivity to key parameters and market conditions.

#### Acceptance Criteria

1. THE Optimization Model SHALL support scenario analysis with varied natural gas prices ranging from -50% to +50% of baseline prices
2. THE Optimization Model SHALL support scenario analysis with varied grid electricity prices ranging from -30% to +30% of baseline prices
3. THE Optimization Model SHALL support scenario analysis with varied battery costs ranging from $200/kWh to $500/kWh
4. THE Optimization Model SHALL support scenario analysis with varied reliability requirements of 99.9%, 99.99%, and 99.999% uptime
5. THE Optimization Model SHALL support scenario analysis with varied carbon constraints including no constraint, 50% reduction, 80% reduction, and 100% carbon-free operation

### Requirement 9: Solution Output and Reporting

**User Story:** As a decision maker, I want clear visualization of optimal portfolios and trade-offs, so that I can communicate recommendations to executives and stakeholders.

#### Acceptance Criteria

1. THE Optimization Model SHALL output the optimal capacity mix showing MW or MWh for each technology (grid, gas, battery, solar)
2. THE Optimization Model SHALL output hourly dispatch decisions for all 8760 hours showing power contribution from each source
3. THE Optimization Model SHALL calculate and report levelized cost of energy in $/MWh for the optimal solution
4. THE Optimization Model SHALL generate Pareto frontier plots showing trade-offs between cost and reliability, cost and carbon emissions, and grid dependence and reliability
5. THE Optimization Model SHALL report financial metrics including 20-year NPV, payback period for behind-the-meter investments, and annual cost savings compared to grid-only baseline

### Requirement 10: Interactive Dashboard

**User Story:** As an analyst, I want an interactive dashboard to explore different scenarios, so that I can quickly answer "what-if" questions without re-running the optimization model.

#### Acceptance Criteria

1. THE Interactive Dashboard SHALL provide input controls for facility size (100-500 MW), reliability target (99.9-99.999%), and carbon budget (0-100% reduction)
2. THE Interactive Dashboard SHALL display the optimal capacity mix as a bar chart showing MW capacity for each technology
3. THE Interactive Dashboard SHALL display hourly dispatch as a heatmap with hours on x-axis, power sources on y-axis, and color representing MW contribution
4. THE Interactive Dashboard SHALL display cost breakdown showing CAPEX versus OPEX components for the optimal solution
5. THE Interactive Dashboard SHALL display reliability analysis including histogram of hourly unserved energy and identification of worst-case events

### Requirement 11: Case Study Analysis

**User Story:** As a business development manager, I want a specific case study for a 300MW AI data center in West Texas, so that I can present concrete recommendations to potential clients.

#### Acceptance Criteria

1. THE Optimization Model SHALL analyze a baseline scenario of 300MW data center with grid-only power supply and report total cost and reliability metrics
2. THE Optimization Model SHALL analyze an optimal portfolio scenario for the same 300MW data center and report the recommended capacity mix
3. THE Optimization Model SHALL calculate the upfront investment required for the optimal behind-the-meter portfolio
4. THE Optimization Model SHALL calculate annual operational cost savings of the optimal portfolio compared to grid-only baseline
5. THE Optimization Model SHALL generate a written case study report comparing the optimal strategy to alternative approaches including the Microsoft Three Mile Island nuclear approach

### Requirement 12: Model Performance and Scalability

**User Story:** As a software engineer, I want the optimization model to solve efficiently, so that analysts can iterate quickly on different scenarios.

#### Acceptance Criteria

1. THE Optimization Model SHALL formulate the problem as a linear program or mixed-integer linear program suitable for commercial solvers
2. THE Optimization Model SHALL solve the full 8760-hour annual optimization problem in less than 30 minutes on a standard laptop computer
3. THE Optimization Model SHALL configure the solver with a MIP gap tolerance of 0.5% to balance solution quality and solve time
4. THE Optimization Model SHALL validate that all constraints are satisfied in the optimal solution and report any constraint violations
5. THE Optimization Model SHALL provide solver status information including objective value, optimality gap, and solve time in the solution output

### Requirement 13: EMOJIS

**User Story:** As a Software enginner, I don't want emojis anywhere in my code. Should not look AI generated

#### Acceptance Criteria

1. THE code WILL NOT have any emojis
2. The Debugging statements WILL NOT have emojis
3. The comments WILL NOT have emojis
4. The code WILL NOT have any emojis

