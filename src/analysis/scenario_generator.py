"""
Scenario generator for sensitivity analysis.

This module generates parameter combinations for exploring sensitivity
to key inputs including gas prices, grid LMPs, battery costs, reliability
targets, and carbon constraints.
"""

from typing import List, Dict, Optional, Any
from itertools import product
import copy


def generate_scenarios(
    base_params: Optional[Dict[str, Any]] = None,
    gas_price_variations: Optional[List[float]] = None,
    lmp_variations: Optional[List[float]] = None,
    battery_cost_variations: Optional[List[float]] = None,
    reliability_variations: Optional[List[float]] = None,
    carbon_variations: Optional[List[Optional[float]]] = None,
    include_base: bool = True
) -> List[Dict[str, Any]]:
    """
    Generate scenario parameter sets for sensitivity analysis.
    
    Creates all combinations of parameter variations for systematic exploration
    of the solution space. Each scenario is a dictionary of parameters that can
    be used to configure the optimization model.
    
    Args:
        base_params: Base parameter dictionary. If None, uses default values.
        gas_price_variations: List of gas price multipliers (e.g., [0.5, 1.0, 1.5])
        lmp_variations: List of LMP multipliers (e.g., [0.7, 1.0, 1.3])
        battery_cost_variations: List of battery costs in $/kWh (e.g., [200, 350, 500])
        reliability_variations: List of reliability targets (e.g., [0.999, 0.9999, 0.99999])
        carbon_variations: List of carbon reduction percentages or None for no constraint
                          (e.g., [None, 50, 80, 100])
        include_base: If True, include base scenario with no variations
    
    Returns:
        List of parameter dictionaries, one for each scenario combination
        
    Example:
        >>> scenarios = generate_scenarios(
        ...     gas_price_variations=[0.5, 1.0, 1.5],
        ...     reliability_variations=[0.9999, 0.99999]
        ... )
        >>> len(scenarios)
        6  # 3 gas prices × 2 reliability levels
    """
    # Set default base parameters
    if base_params is None:
        base_params = {
            'gas_price_multiplier': 1.0,
            'lmp_multiplier': 1.0,
            'battery_cost_per_kwh': 350.0,
            'reliability_target': 0.9999,
            'carbon_reduction_pct': None,
            'scenario_name': 'base'
        }
    
    # Set default variations if not provided
    if gas_price_variations is None:
        gas_price_variations = [1.0]
    if lmp_variations is None:
        lmp_variations = [1.0]
    if battery_cost_variations is None:
        battery_cost_variations = [350.0]
    if reliability_variations is None:
        reliability_variations = [0.9999]
    if carbon_variations is None:
        carbon_variations = [None]
    
    scenarios = []
    
    # Generate all combinations
    for gas_mult, lmp_mult, batt_cost, reliability, carbon_pct in product(
        gas_price_variations,
        lmp_variations,
        battery_cost_variations,
        reliability_variations,
        carbon_variations
    ):
        # Skip base scenario if it will be added separately
        is_base = (gas_mult == 1.0 and lmp_mult == 1.0 and 
                   batt_cost == 350.0 and reliability == 0.9999 and 
                   carbon_pct is None)
        
        if is_base and not include_base:
            continue
        
        # Create scenario parameters
        scenario = copy.deepcopy(base_params)
        scenario['gas_price_multiplier'] = gas_mult
        scenario['lmp_multiplier'] = lmp_mult
        scenario['battery_cost_per_kwh'] = batt_cost
        scenario['reliability_target'] = reliability
        scenario['carbon_reduction_pct'] = carbon_pct
        
        # Generate descriptive scenario name
        scenario['scenario_name'] = _generate_scenario_name(
            gas_mult, lmp_mult, batt_cost, reliability, carbon_pct
        )
        
        scenarios.append(scenario)
    
    return scenarios


def generate_gas_price_scenarios(
    base_params: Optional[Dict[str, Any]] = None,
    variations: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Generate scenarios varying only gas prices.
    
    Args:
        base_params: Base parameter dictionary
        variations: List of gas price multipliers (default: [0.5, 0.75, 1.0, 1.25, 1.5])
    
    Returns:
        List of scenario dictionaries with varied gas prices
    """
    if variations is None:
        variations = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    return generate_scenarios(
        base_params=base_params,
        gas_price_variations=variations
    )


def generate_lmp_scenarios(
    base_params: Optional[Dict[str, Any]] = None,
    variations: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Generate scenarios varying only grid LMP prices.
    
    Args:
        base_params: Base parameter dictionary
        variations: List of LMP multipliers (default: [0.7, 0.85, 1.0, 1.15, 1.3])
    
    Returns:
        List of scenario dictionaries with varied LMP prices
    """
    if variations is None:
        variations = [0.7, 0.85, 1.0, 1.15, 1.3]
    
    return generate_scenarios(
        base_params=base_params,
        lmp_variations=variations
    )


def generate_battery_cost_scenarios(
    base_params: Optional[Dict[str, Any]] = None,
    variations: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Generate scenarios varying only battery costs.
    
    Args:
        base_params: Base parameter dictionary
        variations: List of battery costs in $/kWh (default: [200, 275, 350, 425, 500])
    
    Returns:
        List of scenario dictionaries with varied battery costs
    """
    if variations is None:
        variations = [200.0, 275.0, 350.0, 425.0, 500.0]
    
    return generate_scenarios(
        base_params=base_params,
        battery_cost_variations=variations
    )


def generate_reliability_scenarios(
    base_params: Optional[Dict[str, Any]] = None,
    variations: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Generate scenarios varying only reliability targets.
    
    Args:
        base_params: Base parameter dictionary
        variations: List of reliability targets (default: [0.999, 0.9999, 0.99999])
    
    Returns:
        List of scenario dictionaries with varied reliability targets
    """
    if variations is None:
        variations = [0.999, 0.9999, 0.99999]
    
    return generate_scenarios(
        base_params=base_params,
        reliability_variations=variations
    )


def generate_carbon_scenarios(
    base_params: Optional[Dict[str, Any]] = None,
    variations: Optional[List[Optional[float]]] = None
) -> List[Dict[str, Any]]:
    """
    Generate scenarios varying only carbon constraints.
    
    Args:
        base_params: Base parameter dictionary
        variations: List of carbon reduction percentages or None for no constraint
                   (default: [None, 50, 80, 100])
    
    Returns:
        List of scenario dictionaries with varied carbon constraints
    """
    if variations is None:
        variations = [None, 50.0, 80.0, 100.0]
    
    return generate_scenarios(
        base_params=base_params,
        carbon_variations=variations
    )


def generate_full_sensitivity_scenarios(
    base_params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate comprehensive set of scenarios for full sensitivity analysis.
    
    Creates scenarios covering all parameter variations:
    - Gas prices: ±50% (0.5, 1.0, 1.5)
    - Grid LMPs: ±30% (0.7, 1.0, 1.3)
    - Battery costs: $200-500/kWh (200, 350, 500)
    - Reliability: 99.9%, 99.99%, 99.999%
    - Carbon: no constraint, 50%, 80%, 100% reduction
    
    Args:
        base_params: Base parameter dictionary
    
    Returns:
        List of all scenario combinations (3×3×3×3×4 = 324 scenarios)
    """
    return generate_scenarios(
        base_params=base_params,
        gas_price_variations=[0.5, 1.0, 1.5],
        lmp_variations=[0.7, 1.0, 1.3],
        battery_cost_variations=[200.0, 350.0, 500.0],
        reliability_variations=[0.999, 0.9999, 0.99999],
        carbon_variations=[None, 50.0, 80.0, 100.0]
    )


def generate_pareto_scenarios(
    base_params: Optional[Dict[str, Any]] = None,
    objective_pair: str = 'cost_reliability'
) -> List[Dict[str, Any]]:
    """
    Generate scenarios for Pareto frontier analysis.
    
    Creates scenarios that explore trade-offs between two objectives
    by varying relevant parameters.
    
    Args:
        base_params: Base parameter dictionary
        objective_pair: Which objectives to explore
                       - 'cost_reliability': Cost vs. reliability trade-off
                       - 'cost_carbon': Cost vs. carbon emissions trade-off
                       - 'grid_reliability': Grid dependence vs. reliability
    
    Returns:
        List of scenario dictionaries optimized for Pareto analysis
    """
    if objective_pair == 'cost_reliability':
        # Vary reliability targets to explore cost-reliability trade-off
        return generate_scenarios(
            base_params=base_params,
            reliability_variations=[0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999]
        )
    
    elif objective_pair == 'cost_carbon':
        # Vary carbon constraints to explore cost-carbon trade-off
        return generate_scenarios(
            base_params=base_params,
            carbon_variations=[None, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0]
        )
    
    elif objective_pair == 'grid_reliability':
        # Vary reliability with different grid/BTM technology mixes
        scenarios = []
        for reliability in [0.999, 0.9999, 0.99999]:
            for gas_mult in [0.5, 1.0, 1.5]:
                scenario = copy.deepcopy(base_params) if base_params else {}
                scenario['reliability_target'] = reliability
                scenario['gas_price_multiplier'] = gas_mult
                scenario['lmp_multiplier'] = 1.0
                scenario['battery_cost_per_kwh'] = 350.0
                scenario['carbon_reduction_pct'] = None
                scenario['scenario_name'] = f'grid_rel_r{reliability}_g{gas_mult}'
                scenarios.append(scenario)
        return scenarios
    
    else:
        raise ValueError(f"Unknown objective pair: {objective_pair}")


def _generate_scenario_name(
    gas_mult: float,
    lmp_mult: float,
    batt_cost: float,
    reliability: float,
    carbon_pct: Optional[float]
) -> str:
    """
    Generate descriptive scenario name from parameters.
    
    Args:
        gas_mult: Gas price multiplier
        lmp_mult: LMP multiplier
        batt_cost: Battery cost in $/kWh
        reliability: Reliability target
        carbon_pct: Carbon reduction percentage or None
    
    Returns:
        Descriptive scenario name string
    """
    parts = []
    
    # Gas price variation
    if gas_mult != 1.0:
        parts.append(f'gas{int(gas_mult*100)}')
    
    # LMP variation
    if lmp_mult != 1.0:
        parts.append(f'lmp{int(lmp_mult*100)}')
    
    # Battery cost variation
    if batt_cost != 350.0:
        parts.append(f'batt{int(batt_cost)}')
    
    # Reliability variation
    if reliability == 0.999:
        parts.append('rel999')
    elif reliability == 0.9999:
        parts.append('rel9999')
    elif reliability == 0.99999:
        parts.append('rel99999')
    elif reliability != 0.9999:
        # For non-standard reliability values
        rel_str = str(reliability).replace('.', '')
        parts.append(f'rel{rel_str}')
    
    # Carbon constraint
    if carbon_pct is not None:
        if carbon_pct == 0.0:
            parts.append('carbon_free')
        else:
            parts.append(f'carbon{int(carbon_pct)}')
    
    # If no variations, it's the base scenario
    if not parts:
        return 'base'
    
    return '_'.join(parts)


def filter_scenarios(
    scenarios: List[Dict[str, Any]],
    filter_func: callable
) -> List[Dict[str, Any]]:
    """
    Filter scenarios based on custom criteria.
    
    Args:
        scenarios: List of scenario dictionaries
        filter_func: Function that takes a scenario dict and returns True to keep it
    
    Returns:
        Filtered list of scenarios
        
    Example:
        >>> # Keep only high reliability scenarios
        >>> high_rel = filter_scenarios(
        ...     scenarios,
        ...     lambda s: s['reliability_target'] >= 0.9999
        ... )
    """
    return [s for s in scenarios if filter_func(s)]


def get_scenario_summary(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get summary statistics about a set of scenarios.
    
    Args:
        scenarios: List of scenario dictionaries
    
    Returns:
        Dictionary with summary information
    """
    if not scenarios:
        return {
            'total_scenarios': 0,
            'parameter_ranges': {}
        }
    
    # Extract parameter ranges
    gas_mults = [s.get('gas_price_multiplier', 1.0) for s in scenarios]
    lmp_mults = [s.get('lmp_multiplier', 1.0) for s in scenarios]
    batt_costs = [s.get('battery_cost_per_kwh', 350.0) for s in scenarios]
    reliabilities = [s.get('reliability_target', 0.9999) for s in scenarios]
    carbon_pcts = [s.get('carbon_reduction_pct') for s in scenarios if s.get('carbon_reduction_pct') is not None]
    
    return {
        'total_scenarios': len(scenarios),
        'parameter_ranges': {
            'gas_price_multiplier': {
                'min': min(gas_mults),
                'max': max(gas_mults),
                'unique_values': len(set(gas_mults))
            },
            'lmp_multiplier': {
                'min': min(lmp_mults),
                'max': max(lmp_mults),
                'unique_values': len(set(lmp_mults))
            },
            'battery_cost_per_kwh': {
                'min': min(batt_costs),
                'max': max(batt_costs),
                'unique_values': len(set(batt_costs))
            },
            'reliability_target': {
                'min': min(reliabilities),
                'max': max(reliabilities),
                'unique_values': len(set(reliabilities))
            },
            'carbon_reduction_pct': {
                'min': min(carbon_pcts) if carbon_pcts else None,
                'max': max(carbon_pcts) if carbon_pcts else None,
                'unique_values': len(set(carbon_pcts)),
                'scenarios_with_constraint': len(carbon_pcts)
            }
        }
    }
