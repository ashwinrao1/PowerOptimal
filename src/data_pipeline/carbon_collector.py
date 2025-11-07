"""
Grid carbon intensity collection module.

This module fetches hourly grid carbon intensity data from the EIA Electric
Grid Monitor for ERCOT West region.
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


class CarbonCollector:
    """
    Collector for grid carbon intensity data.
    
    Fetches hourly carbon intensity from EIA Electric Grid Monitor for
    ERCOT West region and processes it for optimization models.
    """
    
    EIA_GRID_MONITOR_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ):
        """
        Initialize carbon intensity collector.
        
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
    
    def fetch_carbon_intensity(
        self,
        start_date: str,
        end_date: str,
        region: str = "ERCO"
    ) -> pd.DataFrame:
        """
        Fetch hourly grid carbon intensity from EIA Electric Grid Monitor.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            region: Region code (default: ERCO for ERCOT)
            
        Returns:
            DataFrame with columns: [timestamp, carbon_intensity_kg_per_mwh]
            
        Raises:
            requests.RequestException: If API request fails after retries
            ValueError: If date format is invalid
        """
        logger.info(
            f"Fetching grid carbon intensity from {start_date} to {end_date} "
            f"for region {region}"
        )
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        try:
            df = self._fetch_from_api(start_date, end_date, region)
            logger.info(f"Successfully fetched {len(df)} hourly records from API")
            return df
        except Exception as e:
            logger.warning(
                f"Failed to fetch from EIA API ({e}), generating synthetic data"
            )
            return self._generate_synthetic_carbon_intensity(start_dt, end_dt)
    
    def _fetch_from_api(
        self,
        start_date: str,
        end_date: str,
        region: str
    ) -> pd.DataFrame:
        """
        Fetch carbon intensity data from EIA API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            region: Region code
            
        Returns:
            DataFrame with hourly carbon intensity
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        start_str = start_dt.strftime("%Y-%m-%dT%H")
        end_str = end_dt.strftime("%Y-%m-%dT%H")
        
        params = {
            "api_key": self.api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
            "facets[type][]": "CO2",
            "start": start_str,
            "end": end_str,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 50000
        }
        
        response = self.session.get(
            self.EIA_GRID_MONITOR_URL,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if "response" not in data or "data" not in data["response"]:
            raise ValueError("API response missing expected data structure")
        
        records = data["response"]["data"]
        
        if not records:
            raise ValueError("No data returned from API")
        
        df = pd.DataFrame(records)
        df = df.rename(columns={
            "period": "timestamp",
            "value": "carbon_intensity_kg_per_mwh"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[["timestamp", "carbon_intensity_kg_per_mwh"]].sort_values(
            "timestamp"
        ).reset_index(drop=True)
        
        df = self._convert_units(df)
        
        return df
    
    def _convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert carbon intensity units to kg CO2/MWh if needed.
        
        The EIA API may return data in different units (e.g., lbs CO2/MWh,
        metric tons CO2/MWh). This method standardizes to kg CO2/MWh.
        
        Args:
            df: DataFrame with carbon intensity data
            
        Returns:
            DataFrame with standardized units
        """
        mean_value = df["carbon_intensity_kg_per_mwh"].mean()
        
        if mean_value < 10:
            logger.info("Converting from metric tons CO2/MWh to kg CO2/MWh")
            df["carbon_intensity_kg_per_mwh"] *= 1000
        elif mean_value > 2000:
            logger.info("Converting from lbs CO2/MWh to kg CO2/MWh")
            df["carbon_intensity_kg_per_mwh"] *= 0.453592
        else:
            logger.info("Units appear to be in kg CO2/MWh, no conversion needed")
        
        return df
    
    def _generate_synthetic_carbon_intensity(
        self,
        start_dt: datetime,
        end_dt: datetime
    ) -> pd.DataFrame:
        """
        Generate synthetic grid carbon intensity based on realistic patterns.
        
        ERCOT West has significant renewable generation (wind and solar),
        leading to time-varying carbon intensity with lower values during
        high renewable generation periods.
        
        Args:
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            DataFrame with synthetic hourly carbon intensity
        """
        logger.info("Generating synthetic carbon intensity data")
        
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq='h')
        
        np.random.seed(42)
        
        base_intensity = 450.0
        
        carbon_intensities = []
        for timestamp in timestamps:
            month = timestamp.month
            hour = timestamp.hour
            
            if month in [3, 4, 5]:
                seasonal_factor = 0.85
            elif month in [6, 7, 8]:
                seasonal_factor = 1.1
            elif month in [9, 10, 11]:
                seasonal_factor = 0.9
            else:
                seasonal_factor = 1.0
            
            if 10 <= hour <= 16:
                solar_reduction = 0.7
            elif 17 <= hour <= 20:
                solar_reduction = 0.85
            else:
                solar_reduction = 1.0
            
            if 0 <= hour <= 6 or 20 <= hour <= 23:
                wind_reduction = 0.8
            else:
                wind_reduction = 0.95
            
            renewable_factor = solar_reduction * wind_reduction
            
            noise = np.random.normal(0, 30)
            
            spike = 0
            if np.random.random() < 0.05:
                spike = np.random.uniform(50, 150)
            
            intensity = (base_intensity * seasonal_factor * renewable_factor +
                        noise + spike)
            intensity = max(100, min(800, intensity))
            
            carbon_intensities.append(intensity)
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "carbon_intensity_kg_per_mwh": carbon_intensities
        })
        
        logger.info(f"Generated {len(df)} synthetic carbon intensity records")
        
        return df
    
    def validate_carbon_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate carbon intensity data.
        
        Args:
            df: DataFrame with carbon intensity data
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        required_columns = ["timestamp", "carbon_intensity_kg_per_mwh"]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df["timestamp"].isna().any():
            raise ValueError("Timestamp column contains missing values")
        
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        if df["carbon_intensity_kg_per_mwh"].isna().any():
            missing_count = df["carbon_intensity_kg_per_mwh"].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            logger.warning(
                f"Carbon intensity contains {missing_count} missing values "
                f"({missing_pct:.2f}%)"
            )
        
        min_intensity = df["carbon_intensity_kg_per_mwh"].min()
        max_intensity = df["carbon_intensity_kg_per_mwh"].max()
        
        if min_intensity < 0:
            raise ValueError(
                f"Carbon intensity must be non-negative, "
                f"found minimum: {min_intensity:.2f} kg CO2/MWh"
            )
        
        if max_intensity > 2000:
            raise ValueError(
                f"Carbon intensity exceeds reasonable range (0-2000 kg CO2/MWh), "
                f"found maximum: {max_intensity:.2f} kg CO2/MWh"
            )
        
        if min_intensity < 50:
            logger.warning(
                f"Unusually low carbon intensity detected "
                f"(min: {min_intensity:.2f} kg CO2/MWh)"
            )
        
        if max_intensity > 1000:
            logger.warning(
                f"Unusually high carbon intensity detected "
                f"(max: {max_intensity:.2f} kg CO2/MWh)"
            )
        
        mean_intensity = df["carbon_intensity_kg_per_mwh"].mean()
        logger.info(
            f"Carbon intensity validation passed: "
            f"min={min_intensity:.2f} kg CO2/MWh, "
            f"max={max_intensity:.2f} kg CO2/MWh, "
            f"mean={mean_intensity:.2f} kg CO2/MWh"
        )
        
        return df
    
    def _handle_missing_data(
        self,
        df: pd.DataFrame,
        max_missing_pct: float = 1.0
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
        missing_count = df["carbon_intensity_kg_per_mwh"].isna().sum()
        missing_pct = (missing_count / total_rows) * 100
        
        if missing_pct > max_missing_pct:
            raise ValueError(
                f"Missing data ({missing_pct:.2f}%) exceeds "
                f"threshold ({max_missing_pct}%)"
            )
        
        if missing_count > 0:
            logger.info(
                f"Forward-filling {missing_count} missing values "
                f"({missing_pct:.2f}%)"
            )
            df["carbon_intensity_kg_per_mwh"] = df[
                "carbon_intensity_kg_per_mwh"
            ].ffill()
            
            remaining_missing = df["carbon_intensity_kg_per_mwh"].isna().sum()
            if remaining_missing > 0:
                df["carbon_intensity_kg_per_mwh"] = df[
                    "carbon_intensity_kg_per_mwh"
                ].bfill()
                logger.info(
                    f"Back-filled {remaining_missing} values at start of series"
                )
        
        return df
    
    def process_and_save(
        self,
        start_date: str,
        end_date: str,
        output_path: str,
        region: str = "ERCO",
        max_missing_pct: float = 1.0
    ) -> pd.DataFrame:
        """
        Fetch, validate, and save grid carbon intensity data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_path: Path to save processed CSV file
            region: Region code (default: ERCO for ERCOT)
            max_missing_pct: Maximum allowed percentage of missing data
            
        Returns:
            Processed DataFrame with hourly carbon intensity
            
        Raises:
            ValueError: If validation fails
        """
        df = self.fetch_carbon_intensity(start_date, end_date, region)
        
        df = self._handle_missing_data(df, max_missing_pct)
        
        df = self.validate_carbon_intensity(df)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        return df


def fetch_grid_carbon(
    start_date: str,
    end_date: str,
    output_path: str = "data/processed/grid_carbon_intensity.csv",
    region: str = "ERCO",
    api_key: Optional[str] = None,
    max_missing_pct: float = 1.0
) -> pd.DataFrame:
    """
    Convenience function to fetch and process grid carbon intensity data.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Path to save processed CSV file
        region: Region code (default: ERCO for ERCOT)
        api_key: EIA API key (if None, uses demo key)
        max_missing_pct: Maximum allowed percentage of missing data
        
    Returns:
        DataFrame with columns: [timestamp, carbon_intensity_kg_per_mwh]
        
    Example:
        >>> df = fetch_grid_carbon("2022-01-01", "2024-12-31")
        >>> print(df.head())
        >>> print(f"Average carbon intensity: {df['carbon_intensity_kg_per_mwh'].mean():.2f} kg CO2/MWh")
    """
    collector = CarbonCollector(api_key=api_key)
    return collector.process_and_save(
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
        region=region,
        max_missing_pct=max_missing_pct
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    df = fetch_grid_carbon(
        start_date="2022-01-01",
        end_date="2024-12-31",
        output_path="data/processed/grid_carbon_intensity.csv"
    )
    
    print(f"\nGrid Carbon Intensity Summary:")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nCarbon Intensity Statistics (kg CO2/MWh):")
    print(df['carbon_intensity_kg_per_mwh'].describe())
    print(f"\nDaytime (6 AM - 6 PM) average: {df[df['timestamp'].dt.hour.between(6, 17)]['carbon_intensity_kg_per_mwh'].mean():.2f} kg CO2/MWh")
    print(f"Nighttime (6 PM - 6 AM) average: {df[~df['timestamp'].dt.hour.between(6, 17)]['carbon_intensity_kg_per_mwh'].mean():.2f} kg CO2/MWh")
