"""
Dashboard Caching Configuration

This module provides caching utilities and configuration for the Streamlit dashboard
to improve performance and reduce redundant computations.

Caching Strategy:
1. Data Loading: Cache market data files (TTL: 1 hour)
2. Visualizations: Cache generated plots (TTL: 30 minutes)
3. Computations: Cache expensive calculations (TTL: 30 minutes)
4. Solver Instance: Persist solver across reruns (cache_resource)
5. Pre-computed Scenarios: Cache pre-computed results (TTL: 24 hours)
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any, List
import json


# Cache TTL (Time To Live) settings in seconds
CACHE_TTL_DATA = 3600          # 1 hour for data files
CACHE_TTL_VISUALIZATION = 1800  # 30 minutes for visualizations
CACHE_TTL_COMPUTATION = 1800    # 30 minutes for computations
CACHE_TTL_PRECOMPUTED = 86400   # 24 hours for pre-computed results


def clear_all_caches():
    """
    Clear all Streamlit caches.
    Useful for forcing fresh data loads or after data updates.
    """
    st.cache_data.clear()
    st.cache_resource.clear()


def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about cached data.
    
    Returns:
        Dictionary with cache statistics
    """
    # Note: Streamlit doesn't provide direct cache introspection
    # This is a placeholder for potential future functionality
    return {
        "data_cache_ttl": CACHE_TTL_DATA,
        "visualization_cache_ttl": CACHE_TTL_VISUALIZATION,
        "computation_cache_ttl": CACHE_TTL_COMPUTATION,
        "precomputed_cache_ttl": CACHE_TTL_PRECOMPUTED
    }


@st.cache_data(ttl=CACHE_TTL_PRECOMPUTED)
def load_precomputed_baseline():
    """
    Load pre-computed baseline (grid-only) solution.
    
    Returns:
        Dictionary with baseline solution or None if not available
    """
    results_dir = Path(__file__).parent.parent / "results" / "solutions"
    baseline_file = results_dir / "baseline_grid_only.json"
    
    try:
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    
    return None


@st.cache_data(ttl=CACHE_TTL_PRECOMPUTED)
def load_precomputed_optimal():
    """
    Load pre-computed optimal portfolio solution.
    
    Returns:
        Dictionary with optimal solution or None if not available
    """
    results_dir = Path(__file__).parent.parent / "results" / "solutions"
    optimal_file = results_dir / "optimal_portfolio.json"
    
    try:
        if optimal_file.exists():
            with open(optimal_file, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    
    return None


@st.cache_data(ttl=CACHE_TTL_PRECOMPUTED)
def load_precomputed_pareto_frontiers():
    """
    Load pre-computed Pareto frontier data.
    
    Returns:
        Dictionary with Pareto frontiers or None if not available
    """
    results_dir = Path(__file__).parent.parent / "results"
    pareto_file = results_dir / "example_pareto_frontiers.json"
    
    try:
        if pareto_file.exists():
            with open(pareto_file, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    
    return None


def precompute_common_scenarios():
    """
    Pre-compute common scenarios for faster dashboard loading.
    This function should be run offline to generate cached results.
    
    Common scenarios to pre-compute:
    1. Baseline (grid-only)
    2. Optimal portfolio (all technologies)
    3. Gas price sensitivity (±50%)
    4. Reliability sensitivity (99.9%, 99.99%, 99.999%)
    5. Carbon constraint scenarios (0%, 50%, 80%, 100% reduction)
    
    Note: This is a placeholder for the actual pre-computation logic.
    In production, this would be run as a separate script.
    """
    # This would be implemented as a separate script that:
    # 1. Loads market data
    # 2. Runs optimization for common scenarios
    # 3. Saves results to results/scenarios/ directory
    # 4. Results are then loaded by the dashboard with caching
    
    pass


def enable_lazy_loading():
    """
    Configure lazy loading for visualizations.
    
    Lazy loading ensures that visualizations are only rendered when
    the user navigates to the page, rather than pre-rendering all
    visualizations on app startup.
    
    This is automatically handled by Streamlit's page navigation,
    but this function documents the strategy.
    """
    # Streamlit's multi-page app structure inherently provides lazy loading
    # Each page module is only imported and executed when navigated to
    
    # Additional optimization: Use st.expander for optional visualizations
    # Use conditional rendering based on user selections
    
    pass


def optimize_dataframe_display(df, max_rows: int = 1000):
    """
    Optimize DataFrame display for large datasets.
    
    Args:
        df: DataFrame to display
        max_rows: Maximum number of rows to display
        
    Returns:
        Optimized DataFrame for display
    """
    if len(df) > max_rows:
        # Show warning and provide download option
        st.warning(f"Large dataset ({len(df)} rows). Showing first {max_rows} rows.")
        
        # Provide download button for full dataset
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset",
            data=csv,
            file_name="full_data.csv",
            mime="text/csv"
        )
        
        return df.head(max_rows)
    
    return df


# Cache configuration documentation
CACHE_DOCUMENTATION = """
# Dashboard Caching Strategy

## Overview
The dashboard implements multi-level caching to improve performance and user experience.

## Cache Levels

### 1. Data Loading (@st.cache_data, TTL: 1 hour)
- Market data files (LMP, solar CF, gas prices, carbon intensity)
- Technology cost parameters
- Pre-computed scenario results

**Why**: Data files are large (100+ MB) and rarely change during a session.

### 2. Visualizations (@st.cache_data, TTL: 30 minutes)
- Capacity mix charts (bar, pie, waterfall)
- Cost breakdown charts
- Dispatch heatmaps and stacked area charts
- Pareto frontier plots
- Sensitivity tornado charts

**Why**: Plotly figure generation is computationally expensive.

### 3. Computations (@st.cache_data, TTL: 30 minutes)
- Operational statistics calculations
- Pareto frontier identification
- Sensitivity analysis
- Scenario comparisons

**Why**: These calculations involve processing 8760-hour datasets.

### 4. Solver Instance (@st.cache_resource)
- Gurobi solver instance
- Persists across reruns

**Why**: Solver initialization has overhead; reuse the same instance.

### 5. Pre-computed Results (@st.cache_data, TTL: 24 hours)
- Baseline grid-only solution
- Optimal portfolio solution
- Common scenario results
- Pareto frontier data

**Why**: These are expensive to compute but rarely change.

## Cache Invalidation

Caches automatically expire based on TTL settings:
- Data: 1 hour (3600 seconds)
- Visualizations: 30 minutes (1800 seconds)
- Computations: 30 minutes (1800 seconds)
- Pre-computed: 24 hours (86400 seconds)

Manual cache clearing:
```python
from dashboard.cache_config import clear_all_caches
clear_all_caches()
```

## Lazy Loading

Visualizations are only rendered when:
1. User navigates to the page
2. User selects visualization options
3. User expands optional sections

This is achieved through:
- Streamlit's multi-page architecture
- Conditional rendering based on user input
- st.expander for optional content

## Performance Impact

Expected performance improvements:
- Initial page load: 2-5 seconds (vs 10-30 seconds without caching)
- Subsequent page loads: <1 second (cached data)
- Visualization rendering: 1-2 seconds (vs 5-10 seconds without caching)
- Scenario comparison: 2-3 seconds (vs 10-20 seconds without caching)

## Memory Considerations

Cached data memory usage:
- Market data: ~100 MB
- Visualizations: ~10-20 MB per chart
- Computations: ~5-10 MB per result
- Total: ~200-300 MB for typical session

Streamlit automatically manages cache memory and evicts old entries when needed.

## Best Practices

1. Use underscore prefix for unhashable parameters:
   ```python
   @st.cache_data
   def my_function(_solution: OptimizationSolution):
       # _solution won't be hashed, preventing errors
       pass
   ```

2. Set appropriate TTL based on data volatility:
   - Static data: Long TTL (hours/days)
   - Dynamic data: Short TTL (minutes)

3. Use cache_data for data/computations:
   - Serializable objects (DataFrames, dicts, lists)
   - Can be pickled

4. Use cache_resource for connections/instances:
   - Database connections
   - Solver instances
   - ML models

5. Clear caches when data updates:
   - After running new optimization
   - After updating data files
   - When debugging

## Monitoring

Monitor cache effectiveness:
- Check page load times
- Monitor memory usage
- Review cache hit rates (if available)
- User feedback on responsiveness

## Future Enhancements

1. Redis cache for multi-user deployments
2. Persistent cache across app restarts
3. Cache warming on startup
4. Intelligent cache invalidation based on data changes
5. Cache analytics and monitoring dashboard
"""


def print_cache_documentation():
    """Print cache documentation to console."""
    print(CACHE_DOCUMENTATION)


if __name__ == "__main__":
    # Print documentation when run as script
    print_cache_documentation()
