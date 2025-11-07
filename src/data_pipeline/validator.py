"""
Data validation module for energy market data.

This module provides functions to validate data completeness, continuity,
and value ranges for all data types used in the optimization model.
"""

import logging
from typing import List, Tuple, Optional
from datetime import timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


def validate_data_completeness(
    df: pd.DataFrame,
    max_missing_pct: float = 1.0,
    exclude_columns: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Check that data has no more than max_missing_pct missing values.
    
    Args:
        df: DataFrame to validate
        max_missing_pct: Maximum allowed percentage of missing data (default: 1.0%)
        exclude_columns: List of column names to exclude from validation
        
    Returns:
        Tuple of (is_valid, list_of_issues)
        
    Example:
        >>> df = pd.DataFrame({'a': [1, 2, None], 'b': [1, 2, 3]})
        >>> is_valid, issues = validate_data_completeness(df, max_missing_pct=1.0)
        >>> print(is_valid, issues)
    """
    issues = []
    exclude_columns = exclude_columns or []
    
    if df.empty:
        issues.append("DataFrame is empty")
        return False, issues
    
    total_rows = len(df)
    
    for col in df.columns:
        if col in exclude_columns:
            continue
            
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_rows) * 100
        
        if missing_pct > max_missing_pct:
            issues.append(
                f"Column '{col}' has {missing_pct:.2f}% missing data "
                f"(exceeds {max_missing_pct}% threshold)"
            )
    
    is_valid = len(issues) == 0
    return is_valid, issues


def validate_timestamps_continuous(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    expected_freq: str = 'h'
) -> Tuple[bool, List[str]]:
    """
    Validate that timestamps are continuous with no gaps.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        expected_freq: Expected frequency ('h' for hourly, 'D' for daily)
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if timestamp_col not in df.columns:
        issues.append(f"Timestamp column '{timestamp_col}' not found")
        return False, issues
    
    if df.empty:
        issues.append("DataFrame is empty")
        return False, issues
    
    timestamps = pd.to_datetime(df[timestamp_col])
    
    if timestamps.isna().any():
        missing_count = timestamps.isna().sum()
        issues.append(f"Timestamp column has {missing_count} missing values")
        return False, issues
    
    timestamps = timestamps.sort_values().reset_index(drop=True)
    
    if expected_freq == 'h':
        expected_delta = timedelta(hours=1)
    elif expected_freq == 'D':
        expected_delta = timedelta(days=1)
    else:
        issues.append(f"Unsupported frequency: {expected_freq}")
        return False, issues
    
    gaps = []
    for i in range(1, len(timestamps)):
        actual_delta = timestamps.iloc[i] - timestamps.iloc[i-1]
        if actual_delta != expected_delta:
            gaps.append({
                'index': i,
                'from': timestamps.iloc[i-1],
                'to': timestamps.iloc[i],
                'gap': actual_delta
            })
    
    if gaps:
        issues.append(f"Found {len(gaps)} gaps in timestamp continuity")
        for gap in gaps[:5]:
            issues.append(
                f"  Gap at index {gap['index']}: {gap['from']} to {gap['to']} "
                f"(delta: {gap['gap']})"
            )
        if len(gaps) > 5:
            issues.append(f"  ... and {len(gaps) - 5} more gaps")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def validate_full_year_coverage(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    expected_hours: int = 8760
) -> Tuple[bool, List[str]]:
    """
    Validate that data covers a full year (8760 hours).
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        expected_hours: Expected number of hours (default: 8760 for non-leap year)
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if timestamp_col not in df.columns:
        issues.append(f"Timestamp column '{timestamp_col}' not found")
        return False, issues
    
    actual_hours = len(df)
    
    if actual_hours < expected_hours:
        issues.append(
            f"Data has {actual_hours} hours, expected {expected_hours} "
            f"(missing {expected_hours - actual_hours} hours)"
        )
    elif actual_hours > expected_hours:
        issues.append(
            f"Data has {actual_hours} hours, expected {expected_hours} "
            f"(extra {actual_hours - expected_hours} hours)"
        )
    
    timestamps = pd.to_datetime(df[timestamp_col])
    if not timestamps.empty:
        date_range = timestamps.max() - timestamps.min()
        expected_range = timedelta(hours=expected_hours - 1)
        
        if abs((date_range - expected_range).total_seconds()) > 3600:
            issues.append(
                f"Date range is {date_range}, expected approximately {expected_range}"
            )
    
    is_valid = len(issues) == 0
    return is_valid, issues


def validate_value_ranges(
    df: pd.DataFrame,
    column: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_negative: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate that values are within expected ranges.
    
    Args:
        df: DataFrame to validate
        column: Column name to check
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)
        allow_negative: Whether negative values are allowed
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if column not in df.columns:
        issues.append(f"Column '{column}' not found")
        return False, issues
    
    values = df[column].dropna()
    
    if values.empty:
        issues.append(f"Column '{column}' has no valid values")
        return False, issues
    
    actual_min = values.min()
    actual_max = values.max()
    
    if not allow_negative and actual_min < 0:
        negative_count = (values < 0).sum()
        issues.append(
            f"Column '{column}' has {negative_count} negative values "
            f"(min: {actual_min:.2f})"
        )
    
    if min_value is not None and actual_min < min_value:
        below_count = (values < min_value).sum()
        issues.append(
            f"Column '{column}' has {below_count} values below {min_value} "
            f"(min: {actual_min:.2f})"
        )
    
    if max_value is not None and actual_max > max_value:
        above_count = (values > max_value).sum()
        issues.append(
            f"Column '{column}' has {above_count} values above {max_value} "
            f"(max: {actual_max:.2f})"
        )
    
    is_valid = len(issues) == 0
    return is_valid, issues


def forward_fill_small_gaps(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_missing_pct: float = 1.0
) -> pd.DataFrame:
    """
    Forward-fill small gaps in data. Raises exception if gaps are too large.
    
    Args:
        df: DataFrame to process
        columns: List of columns to fill (if None, fills all numeric columns)
        max_missing_pct: Maximum allowed percentage of missing data
        
    Returns:
        DataFrame with small gaps filled
        
    Raises:
        DataValidationError: If missing data exceeds threshold
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    total_rows = len(df)
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue
        
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_rows) * 100
        
        if missing_pct > max_missing_pct:
            raise DataValidationError(
                f"Column '{col}' has {missing_pct:.2f}% missing data, "
                f"exceeds threshold of {max_missing_pct}%"
            )
        
        if missing_count > 0:
            logger.info(
                f"Forward-filling {missing_count} missing values "
                f"({missing_pct:.2f}%) in column '{col}'"
            )
            df[col] = df[col].ffill()
            
            remaining_missing = df[col].isna().sum()
            if remaining_missing > 0:
                logger.info(
                    f"Back-filling {remaining_missing} values at start of series "
                    f"in column '{col}'"
                )
                df[col] = df[col].bfill()
    
    return df


def validate_lmp_data(
    df: pd.DataFrame,
    max_missing_pct: float = 1.0,
    min_lmp: float = -100.0,
    max_lmp: float = 5000.0
) -> Tuple[bool, List[str]]:
    """
    Validate ERCOT LMP data.
    
    Args:
        df: DataFrame with LMP data
        max_missing_pct: Maximum allowed percentage of missing data
        min_lmp: Minimum reasonable LMP value ($/MWh)
        max_lmp: Maximum reasonable LMP value ($/MWh)
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    all_issues = []
    
    required_columns = ['timestamp', 'lmp_dam', 'lmp_rtm']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        all_issues.append(f"Missing required columns: {missing_columns}")
        return False, all_issues
    
    is_valid, issues = validate_data_completeness(
        df, max_missing_pct, exclude_columns=['timestamp']
    )
    all_issues.extend(issues)
    
    is_valid, issues = validate_timestamps_continuous(df, 'timestamp', 'h')
    all_issues.extend(issues)
    
    for col in ['lmp_dam', 'lmp_rtm']:
        is_valid, issues = validate_value_ranges(
            df, col, min_value=min_lmp, max_value=max_lmp, allow_negative=True
        )
        all_issues.extend(issues)
    
    is_valid = len(all_issues) == 0
    return is_valid, all_issues


def validate_solar_data(
    df: pd.DataFrame,
    expected_hours: int = 8760
) -> Tuple[bool, List[str]]:
    """
    Validate solar capacity factor data.
    
    Args:
        df: DataFrame with solar capacity factor data
        expected_hours: Expected number of hours
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    all_issues = []
    
    required_columns = ['hour_of_year', 'capacity_factor']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        all_issues.append(f"Missing required columns: {missing_columns}")
        return False, all_issues
    
    is_valid, issues = validate_data_completeness(df, max_missing_pct=0.0)
    all_issues.extend(issues)
    
    if len(df) != expected_hours:
        all_issues.append(
            f"Expected {expected_hours} hours, got {len(df)} hours"
        )
    
    is_valid, issues = validate_value_ranges(
        df, 'capacity_factor', min_value=0.0, max_value=1.0, allow_negative=False
    )
    all_issues.extend(issues)
    
    hours = df['hour_of_year'].values
    expected_hours_array = np.arange(1, expected_hours + 1)
    if not np.array_equal(hours, expected_hours_array):
        all_issues.append("Hour of year values are not sequential from 1 to 8760")
    
    is_valid = len(all_issues) == 0
    return is_valid, all_issues


def validate_gas_price_data(
    df: pd.DataFrame,
    max_missing_pct: float = 1.0,
    min_price: float = 0.0,
    max_price: float = 50.0
) -> Tuple[bool, List[str]]:
    """
    Validate natural gas price data.
    
    Args:
        df: DataFrame with gas price data
        max_missing_pct: Maximum allowed percentage of missing data
        min_price: Minimum reasonable gas price ($/MMBtu)
        max_price: Maximum reasonable gas price ($/MMBtu)
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    all_issues = []
    
    required_columns = ['timestamp', 'price_mmbtu']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        all_issues.append(f"Missing required columns: {missing_columns}")
        return False, all_issues
    
    is_valid, issues = validate_data_completeness(
        df, max_missing_pct, exclude_columns=['timestamp']
    )
    all_issues.extend(issues)
    
    is_valid, issues = validate_timestamps_continuous(df, 'timestamp', 'h')
    all_issues.extend(issues)
    
    is_valid, issues = validate_value_ranges(
        df, 'price_mmbtu', min_value=min_price, max_value=max_price, 
        allow_negative=False
    )
    all_issues.extend(issues)
    
    is_valid = len(all_issues) == 0
    return is_valid, all_issues


def validate_carbon_intensity_data(
    df: pd.DataFrame,
    max_missing_pct: float = 1.0,
    min_intensity: float = 0.0,
    max_intensity: float = 2000.0
) -> Tuple[bool, List[str]]:
    """
    Validate grid carbon intensity data.
    
    Args:
        df: DataFrame with carbon intensity data
        max_missing_pct: Maximum allowed percentage of missing data
        min_intensity: Minimum reasonable carbon intensity (kg CO2/MWh)
        max_intensity: Maximum reasonable carbon intensity (kg CO2/MWh)
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    all_issues = []
    
    required_columns = ['timestamp', 'carbon_intensity_kg_per_mwh']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        all_issues.append(f"Missing required columns: {missing_columns}")
        return False, all_issues
    
    is_valid, issues = validate_data_completeness(
        df, max_missing_pct, exclude_columns=['timestamp']
    )
    all_issues.extend(issues)
    
    is_valid, issues = validate_timestamps_continuous(df, 'timestamp', 'h')
    all_issues.extend(issues)
    
    is_valid, issues = validate_value_ranges(
        df, 'carbon_intensity_kg_per_mwh', 
        min_value=min_intensity, max_value=max_intensity, allow_negative=False
    )
    all_issues.extend(issues)
    
    is_valid = len(all_issues) == 0
    return is_valid, all_issues


def validate_all_data(
    lmp_df: pd.DataFrame,
    solar_df: pd.DataFrame,
    gas_df: pd.DataFrame,
    carbon_df: pd.DataFrame,
    max_missing_pct: float = 1.0,
    raise_on_error: bool = True
) -> Tuple[bool, dict]:
    """
    Validate all data sources for optimization model.
    
    Args:
        lmp_df: ERCOT LMP data
        solar_df: Solar capacity factor data
        gas_df: Natural gas price data
        carbon_df: Grid carbon intensity data
        max_missing_pct: Maximum allowed percentage of missing data
        raise_on_error: Whether to raise exception on validation failure
        
    Returns:
        Tuple of (all_valid, validation_results_dict)
        
    Raises:
        DataValidationError: If validation fails and raise_on_error is True
    """
    results = {}
    
    logger.info("Validating LMP data...")
    is_valid, issues = validate_lmp_data(lmp_df, max_missing_pct)
    results['lmp'] = {'valid': is_valid, 'issues': issues}
    if issues:
        for issue in issues:
            logger.warning(f"LMP: {issue}")
    
    logger.info("Validating solar data...")
    is_valid, issues = validate_solar_data(solar_df)
    results['solar'] = {'valid': is_valid, 'issues': issues}
    if issues:
        for issue in issues:
            logger.warning(f"Solar: {issue}")
    
    logger.info("Validating gas price data...")
    is_valid, issues = validate_gas_price_data(gas_df, max_missing_pct)
    results['gas'] = {'valid': is_valid, 'issues': issues}
    if issues:
        for issue in issues:
            logger.warning(f"Gas: {issue}")
    
    logger.info("Validating carbon intensity data...")
    is_valid, issues = validate_carbon_intensity_data(carbon_df, max_missing_pct)
    results['carbon'] = {'valid': is_valid, 'issues': issues}
    if issues:
        for issue in issues:
            logger.warning(f"Carbon: {issue}")
    
    all_valid = all(r['valid'] for r in results.values())
    
    if not all_valid and raise_on_error:
        error_msg = "Data validation failed:\n"
        for data_type, result in results.items():
            if not result['valid']:
                error_msg += f"\n{data_type.upper()}:\n"
                for issue in result['issues']:
                    error_msg += f"  - {issue}\n"
        raise DataValidationError(error_msg)
    
    if all_valid:
        logger.info("All data validation checks passed")
    
    return all_valid, results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Loading data files for validation...")
    
    lmp_df = pd.read_csv("data/processed/ercot_lmp_hourly_2022_2024.csv")
    solar_df = pd.read_csv("data/processed/solar_cf_west_texas.csv")
    gas_df = pd.read_csv("data/processed/gas_prices_hourly.csv")
    carbon_df = pd.read_csv("data/processed/grid_carbon_intensity.csv")
    
    logger.info(f"Loaded {len(lmp_df)} LMP records")
    logger.info(f"Loaded {len(solar_df)} solar records")
    logger.info(f"Loaded {len(gas_df)} gas price records")
    logger.info(f"Loaded {len(carbon_df)} carbon intensity records")
    
    all_valid, results = validate_all_data(
        lmp_df, solar_df, gas_df, carbon_df,
        max_missing_pct=1.0,
        raise_on_error=False
    )
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for data_type, result in results.items():
        status = "PASS" if result['valid'] else "FAIL"
        print(f"\n{data_type.upper()}: {status}")
        if result['issues']:
            for issue in result['issues']:
                print(f"  - {issue}")
    
    print("\n" + "="*60)
    if all_valid:
        print("Overall: ALL CHECKS PASSED")
    else:
        print("Overall: VALIDATION FAILED")
    print("="*60)
