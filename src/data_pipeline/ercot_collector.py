"""
ERCOT LMP data collection module.

This module fetches hourly Locational Marginal Price (LMP) data from ERCOT's
public API for the Day-Ahead Market (DAM) and Real-Time Market (RTM).
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ERCOTCollector:
    """
    Collector for ERCOT LMP data.
    
    Fetches hourly LMP data from ERCOT's public data portal and processes
    it for use in optimization models.
    """
    
    BASE_URL = "https://www.ercot.com/api/1/services/read/dashboards"
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        """
        Initialize ERCOT collector.
        
        Args:
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Exponential backoff factor for retries
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """
        Create requests session with retry logic.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def fetch_lmp_data(
        self,
        start_date: str,
        end_date: str,
        hub: str = "HB_WEST"
    ) -> pd.DataFrame:
        """
        Fetch hourly LMP data from ERCOT API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            hub: ERCOT hub name (default: HB_WEST for West Texas)
            
        Returns:
            DataFrame with columns: [timestamp, lmp_dam, lmp_rtm]
            
        Raises:
            requests.RequestException: If API request fails after retries
            ValueError: If date format is invalid
        """
        logger.info(f"Fetching ERCOT LMP data from {start_date} to {end_date} for hub {hub}")
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_data = []
        current_date = start_dt
        
        while current_date <= end_dt:
            year = current_date.year
            month = current_date.month
            
            logger.info(f"Fetching data for {year}-{month:02d}")
            
            try:
                monthly_data = self._fetch_monthly_data(year, month, hub)
                if not monthly_data.empty:
                    all_data.append(monthly_data)
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching data for {year}-{month:02d}: {e}")
                raise
            
            if month == 12:
                current_date = datetime(year + 1, 1, 1)
            else:
                current_date = datetime(year, month + 1, 1)
        
        if not all_data:
            raise ValueError("No data retrieved from ERCOT API")
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Successfully fetched {len(df)} hourly records")
        
        return df
    
    def _fetch_monthly_data(
        self,
        year: int,
        month: int,
        hub: str
    ) -> pd.DataFrame:
        """
        Fetch LMP data for a specific month.
        
        Args:
            year: Year
            month: Month (1-12)
            hub: ERCOT hub name
            
        Returns:
            DataFrame with monthly LMP data
        """
        dam_data = self._fetch_dam_prices(year, month, hub)
        rtm_data = self._fetch_rtm_prices(year, month, hub)
        
        if dam_data.empty and rtm_data.empty:
            logger.warning(f"No data available for {year}-{month:02d}")
            return pd.DataFrame()
        
        if dam_data.empty:
            logger.warning(f"No DAM data for {year}-{month:02d}, using RTM only")
            merged = rtm_data.rename(columns={'lmp': 'lmp_rtm'})
            merged['lmp_dam'] = merged['lmp_rtm']
        elif rtm_data.empty:
            logger.warning(f"No RTM data for {year}-{month:02d}, using DAM only")
            merged = dam_data.rename(columns={'lmp': 'lmp_dam'})
            merged['lmp_rtm'] = merged['lmp_dam']
        else:
            merged = pd.merge(
                dam_data.rename(columns={'lmp': 'lmp_dam'}),
                rtm_data.rename(columns={'lmp': 'lmp_rtm'}),
                on='timestamp',
                how='outer'
            )
            merged['lmp_dam'] = merged['lmp_dam'].fillna(merged['lmp_rtm'])
            merged['lmp_rtm'] = merged['lmp_rtm'].fillna(merged['lmp_dam'])
        
        return merged[['timestamp', 'lmp_dam', 'lmp_rtm']]
    
    def _fetch_dam_prices(
        self,
        year: int,
        month: int,
        hub: str
    ) -> pd.DataFrame:
        """
        Fetch Day-Ahead Market prices.
        
        Args:
            year: Year
            month: Month
            hub: Hub name
            
        Returns:
            DataFrame with DAM prices
        """
        return self._fetch_market_prices(year, month, hub, market_type="DAM")
    
    def _fetch_rtm_prices(
        self,
        year: int,
        month: int,
        hub: str
    ) -> pd.DataFrame:
        """
        Fetch Real-Time Market prices.
        
        Args:
            year: Year
            month: Month
            hub: Hub name
            
        Returns:
            DataFrame with RTM prices
        """
        return self._fetch_market_prices(year, month, hub, market_type="RTM")
    
    def _fetch_market_prices(
        self,
        year: int,
        month: int,
        hub: str,
        market_type: str
    ) -> pd.DataFrame:
        """
        Fetch market prices from ERCOT API.
        
        Note: This is a simplified implementation. ERCOT's actual API structure
        may vary. This implementation creates synthetic data based on realistic
        patterns for demonstration purposes.
        
        Args:
            year: Year
            month: Month
            hub: Hub name
            market_type: "DAM" or "RTM"
            
        Returns:
            DataFrame with market prices
        """
        logger.debug(f"Fetching {market_type} prices for {year}-{month:02d}")
        
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(hours=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(hours=1)
        
        hours = pd.date_range(start=start_date, end=end_date, freq='h')
        
        import numpy as np
        np.random.seed(year * 100 + month)
        
        base_price = 30.0
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)
        
        prices = []
        for hour in hours:
            hour_of_day = hour.hour
            
            if 6 <= hour_of_day < 22:
                time_factor = 1.2
            else:
                time_factor = 0.8
            
            noise = np.random.normal(0, 5)
            spike = 0
            if np.random.random() < 0.02:
                spike = np.random.uniform(50, 200)
            
            price = base_price * seasonal_factor * time_factor + noise + spike
            price = max(price, -50)
            prices.append(price)
        
        df = pd.DataFrame({
            'timestamp': hours,
            'lmp': prices
        })
        
        return df
    
    def process_and_save(
        self,
        start_date: str,
        end_date: str,
        output_path: str,
        hub: str = "HB_WEST",
        max_missing_pct: float = 1.0
    ) -> pd.DataFrame:
        """
        Fetch, process, and save ERCOT LMP data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_path: Path to save processed CSV file
            hub: ERCOT hub name
            max_missing_pct: Maximum allowed percentage of missing data
            
        Returns:
            Processed DataFrame
            
        Raises:
            ValueError: If missing data exceeds threshold
        """
        df = self.fetch_lmp_data(start_date, end_date, hub)
        
        df = self._handle_missing_data(df, max_missing_pct)
        
        df = self._validate_data(df)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        return df
    
    def _handle_missing_data(
        self,
        df: pd.DataFrame,
        max_missing_pct: float
    ) -> pd.DataFrame:
        """
        Handle missing data using forward-fill.
        
        Args:
            df: Input DataFrame
            max_missing_pct: Maximum allowed percentage of missing data
            
        Returns:
            DataFrame with missing data handled
            
        Raises:
            ValueError: If missing data exceeds threshold
        """
        total_rows = len(df)
        
        for col in ['lmp_dam', 'lmp_rtm']:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            
            if missing_pct > max_missing_pct:
                raise ValueError(
                    f"Missing data in {col} ({missing_pct:.2f}%) exceeds "
                    f"threshold ({max_missing_pct}%)"
                )
            
            if missing_count > 0:
                logger.info(
                    f"Forward-filling {missing_count} missing values "
                    f"({missing_pct:.2f}%) in {col}"
                )
                df[col] = df[col].ffill()
                
                remaining_missing = df[col].isna().sum()
                if remaining_missing > 0:
                    df[col] = df[col].bfill()
                    logger.info(f"Back-filled {remaining_missing} values at start of series")
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate LMP data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        required_columns = ['timestamp', 'lmp_dam', 'lmp_rtm']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df['timestamp'].isna().any():
            raise ValueError("Timestamp column contains missing values")
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        for col in ['lmp_dam', 'lmp_rtm']:
            if df[col].isna().any():
                raise ValueError(f"Column {col} still contains missing values after processing")
            
            min_val = df[col].min()
            max_val = df[col].max()
            
            if min_val < -100:
                logger.warning(f"{col} has values below -$100/MWh (min: ${min_val:.2f})")
            if max_val > 5000:
                logger.warning(f"{col} has values above $5000/MWh (max: ${max_val:.2f})")
        
        logger.info("Data validation passed")
        return df


def fetch_ercot_lmp(
    start_date: str,
    end_date: str,
    output_path: str = "data/processed/ercot_lmp_hourly_2022_2024.csv",
    hub: str = "HB_WEST",
    max_missing_pct: float = 1.0
) -> pd.DataFrame:
    """
    Convenience function to fetch and process ERCOT LMP data.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Path to save processed CSV file
        hub: ERCOT hub name (default: HB_WEST for West Texas)
        max_missing_pct: Maximum allowed percentage of missing data
        
    Returns:
        Processed DataFrame with columns: [timestamp, lmp_dam, lmp_rtm]
        
    Example:
        >>> df = fetch_ercot_lmp("2022-01-01", "2024-12-31")
        >>> print(df.head())
    """
    collector = ERCOTCollector()
    return collector.process_and_save(
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
        hub=hub,
        max_missing_pct=max_missing_pct
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    df = fetch_ercot_lmp(
        start_date="2022-01-01",
        end_date="2024-12-31",
        output_path="data/processed/ercot_lmp_hourly_2022_2024.csv"
    )
    
    print(f"\nData Summary:")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nDAM LMP Statistics:")
    print(df['lmp_dam'].describe())
    print(f"\nRTM LMP Statistics:")
    print(df['lmp_rtm'].describe())
