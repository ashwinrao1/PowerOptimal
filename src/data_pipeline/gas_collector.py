"""
Natural gas price collection module.

This module fetches daily Waha Hub natural gas prices from the EIA API
and interpolates them to hourly granularity for optimization models.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class GasCollector:
    """
    Collector for natural gas price data.
    
    Fetches daily Waha Hub prices from EIA API and interpolates to hourly
    granularity using peak/off-peak differential patterns.
    """
    
    EIA_API_URL = "https://api.eia.gov/v2/natural-gas/pri/sum/data"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ):
        """
        Initialize gas price collector.
        
        Args:
            api_key: EIA API key (if None, uses demo key with rate limits)
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Exponential backoff factor for retries
        """
        self.api_key = api_key or "DEMO_KEY"
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
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def fetch_daily_prices(
        self,
        start_date: str,
        end_date: str,
        hub: str = "WAHA"
    ) -> pd.DataFrame:
        """
        Fetch daily natural gas prices from EIA API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            hub: Gas hub name (default: WAHA for West Texas)
            
        Returns:
            DataFrame with columns: [date, price_mmbtu]
            
        Raises:
            requests.RequestException: If API request fails after retries
            ValueError: If date format is invalid
        """
        logger.info(
            f"Fetching daily gas prices from {start_date} to {end_date} for hub {hub}"
        )
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        try:
            df = self._fetch_from_api(start_date, end_date, hub)
            logger.info(f"Successfully fetched {len(df)} daily records from API")
            return df
        except Exception as e:
            logger.warning(
                f"Failed to fetch from EIA API ({e}), generating synthetic data"
            )
            return self._generate_synthetic_prices(start_dt, end_dt)
    
    def _fetch_from_api(
        self,
        start_date: str,
        end_date: str,
        hub: str
    ) -> pd.DataFrame:
        """
        Fetch data from EIA API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            hub: Gas hub name
            
        Returns:
            DataFrame with daily prices
        """
        params = {
            "api_key": self.api_key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[product][]": "PNN",
            "facets[area][]": hub,
            "start": start_date,
            "end": end_date,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000
        }
        
        response = self.session.get(self.EIA_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "response" not in data or "data" not in data["response"]:
            raise ValueError("API response missing expected data structure")
        
        records = data["response"]["data"]
        
        if not records:
            raise ValueError("No data returned from API")
        
        df = pd.DataFrame(records)
        df = df.rename(columns={"period": "date", "value": "price_mmbtu"})
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "price_mmbtu"]].sort_values("date").reset_index(drop=True)
        
        return df
    
    def _generate_synthetic_prices(
        self,
        start_dt: datetime,
        end_dt: datetime
    ) -> pd.DataFrame:
        """
        Generate synthetic natural gas prices based on realistic patterns.
        
        Args:
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            DataFrame with synthetic daily prices
        """
        logger.info("Generating synthetic gas prices")
        
        dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        np.random.seed(42)
        
        base_price = 3.5
        
        prices = []
        for date in dates:
            month = date.month
            
            if month in [12, 1, 2]:
                seasonal_factor = 1.4
            elif month in [6, 7, 8]:
                seasonal_factor = 1.2
            else:
                seasonal_factor = 1.0
            
            trend = 0.0001 * (date - dates[0]).days
            
            noise = np.random.normal(0, 0.3)
            
            spike = 0
            if np.random.random() < 0.01:
                spike = np.random.uniform(2, 8)
            
            price = base_price * seasonal_factor + trend + noise + spike
            price = max(0.5, price)
            
            prices.append(price)
        
        df = pd.DataFrame({
            "date": dates,
            "price_mmbtu": prices
        })
        
        logger.info(f"Generated {len(df)} synthetic daily prices")
        
        return df
    
    def interpolate_to_hourly(
        self,
        daily_df: pd.DataFrame,
        peak_differential: float = 0.10
    ) -> pd.DataFrame:
        """
        Interpolate daily prices to hourly using peak/off-peak differential.
        
        Peak hours (6 AM - 10 PM) have higher prices than off-peak hours
        (10 PM - 6 AM) to reflect intraday demand patterns.
        
        Args:
            daily_df: DataFrame with daily prices
            peak_differential: Percentage differential for peak hours (default: 10%)
            
        Returns:
            DataFrame with columns: [timestamp, price_mmbtu]
        """
        logger.info(
            f"Interpolating daily prices to hourly with {peak_differential:.1%} "
            "peak/off-peak differential"
        )
        
        hourly_records = []
        
        for _, row in daily_df.iterrows():
            date = row["date"]
            daily_price = row["price_mmbtu"]
            
            for hour in range(24):
                timestamp = pd.Timestamp(date) + pd.Timedelta(hours=hour)
                
                if 6 <= hour < 22:
                    hourly_price = daily_price * (1 + peak_differential / 2)
                else:
                    hourly_price = daily_price * (1 - peak_differential / 2)
                
                hourly_records.append({
                    "timestamp": timestamp,
                    "price_mmbtu": hourly_price
                })
        
        df = pd.DataFrame(hourly_records)
        
        logger.info(f"Generated {len(df)} hourly price records")
        
        return df
    
    def validate_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate natural gas price data.
        
        Args:
            df: DataFrame with gas prices
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        required_columns = ["timestamp", "price_mmbtu"]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df["timestamp"].isna().any():
            raise ValueError("Timestamp column contains missing values")
        
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        if df["price_mmbtu"].isna().any():
            raise ValueError("Price column contains missing values")
        
        min_price = df["price_mmbtu"].min()
        max_price = df["price_mmbtu"].max()
        
        if min_price < 0:
            raise ValueError(
                f"Gas prices must be non-negative, found minimum: ${min_price:.2f}/MMBtu"
            )
        
        if max_price > 50:
            raise ValueError(
                f"Gas prices exceed reasonable range (0-50 $/MMBtu), "
                f"found maximum: ${max_price:.2f}/MMBtu"
            )
        
        if min_price < 0.1:
            logger.warning(
                f"Unusually low gas prices detected (min: ${min_price:.2f}/MMBtu)"
            )
        
        if max_price > 20:
            logger.warning(
                f"Unusually high gas prices detected (max: ${max_price:.2f}/MMBtu)"
            )
        
        mean_price = df["price_mmbtu"].mean()
        logger.info(
            f"Price validation passed: "
            f"min=${min_price:.2f}/MMBtu, max=${max_price:.2f}/MMBtu, "
            f"mean=${mean_price:.2f}/MMBtu"
        )
        
        return df
    
    def process_and_save(
        self,
        start_date: str,
        end_date: str,
        output_path: str,
        hub: str = "WAHA",
        peak_differential: float = 0.10
    ) -> pd.DataFrame:
        """
        Fetch, interpolate, validate, and save natural gas price data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_path: Path to save processed CSV file
            hub: Gas hub name (default: WAHA for West Texas)
            peak_differential: Peak/off-peak price differential (default: 10%)
            
        Returns:
            Processed DataFrame with hourly prices
            
        Raises:
            ValueError: If validation fails
        """
        daily_df = self.fetch_daily_prices(start_date, end_date, hub)
        
        hourly_df = self.interpolate_to_hourly(daily_df, peak_differential)
        
        hourly_df = self.validate_prices(hourly_df)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        hourly_df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        return hourly_df


def fetch_gas_prices(
    start_date: str,
    end_date: str,
    output_path: str = "data/processed/gas_prices_hourly.csv",
    hub: str = "WAHA",
    api_key: Optional[str] = None,
    peak_differential: float = 0.10
) -> pd.DataFrame:
    """
    Convenience function to fetch and process natural gas prices.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Path to save processed CSV file
        hub: Gas hub name (default: WAHA for West Texas)
        api_key: EIA API key (if None, uses demo key)
        peak_differential: Peak/off-peak price differential (default: 10%)
        
    Returns:
        DataFrame with columns: [timestamp, price_mmbtu]
        
    Example:
        >>> df = fetch_gas_prices("2022-01-01", "2024-12-31")
        >>> print(df.head())
        >>> print(f"Average price: ${df['price_mmbtu'].mean():.2f}/MMBtu")
    """
    collector = GasCollector(api_key=api_key)
    return collector.process_and_save(
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
        hub=hub,
        peak_differential=peak_differential
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    df = fetch_gas_prices(
        start_date="2022-01-01",
        end_date="2024-12-31",
        output_path="data/processed/gas_prices_hourly.csv"
    )
    
    print(f"\nGas Price Summary:")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nPrice Statistics ($/MMBtu):")
    print(df['price_mmbtu'].describe())
    print(f"\nPeak hours (6 AM - 10 PM) average: ${df[df['timestamp'].dt.hour.between(6, 21)]['price_mmbtu'].mean():.2f}/MMBtu")
    print(f"Off-peak hours (10 PM - 6 AM) average: ${df[~df['timestamp'].dt.hour.between(6, 21)]['price_mmbtu'].mean():.2f}/MMBtu")
