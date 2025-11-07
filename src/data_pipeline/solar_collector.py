"""
Solar profile generation module.

This module generates hourly solar capacity factors using the NREL PVWatts API
for a typical meteorological year.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class SolarCollector:
    """
    Collector for solar capacity factor data.
    
    Generates 8760 hourly capacity factors using NREL PVWatts API for
    a specified location and system configuration.
    """
    
    PVWATTS_API_URL = "https://developer.nrel.gov/api/pvwatts/v8.json"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ):
        """
        Initialize solar collector.
        
        Args:
            api_key: NREL API key (if None, uses demo key with rate limits)
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
    
    def generate_solar_profile(
        self,
        lat: float,
        lon: float,
        tilt: float,
        azimuth: float,
        system_capacity: float = 1.0,
        module_type: int = 0,
        array_type: int = 1,
        losses: float = 14.0
    ) -> pd.DataFrame:
        """
        Generate hourly solar capacity factors using NREL PVWatts API.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            tilt: Tilt angle in degrees (0 = horizontal, 90 = vertical)
            azimuth: Azimuth angle in degrees (180 = south-facing)
            system_capacity: System capacity in kW (default: 1.0 for normalized output)
            module_type: Module type (0=Standard, 1=Premium, 2=Thin film)
            array_type: Array type (0=Fixed open rack, 1=Fixed roof mount, 2=1-axis, 3=1-axis backtrack, 4=2-axis)
            losses: System losses in percent (default: 14%)
            
        Returns:
            DataFrame with columns: [hour_of_year, capacity_factor]
            
        Raises:
            requests.RequestException: If API request fails after retries
            ValueError: If API returns invalid data
        """
        logger.info(
            f"Generating solar profile for location ({lat:.4f}, {lon:.4f}), "
            f"tilt={tilt}°, azimuth={azimuth}°"
        )
        
        params = {
            "api_key": self.api_key,
            "lat": lat,
            "lon": lon,
            "system_capacity": system_capacity,
            "azimuth": azimuth,
            "tilt": tilt,
            "module_type": module_type,
            "array_type": array_type,
            "losses": losses,
            "timeframe": "hourly"
        }
        
        try:
            response = self.session.get(self.PVWATTS_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "errors" in data:
                raise ValueError(f"API returned errors: {data['errors']}")
            
            if "outputs" not in data or "ac" not in data["outputs"]:
                raise ValueError("API response missing expected output data")
            
            ac_output = data["outputs"]["ac"]
            
            if len(ac_output) != 8760:
                raise ValueError(
                    f"Expected 8760 hourly values, got {len(ac_output)}"
                )
            
            capacity_factors = [ac / (system_capacity * 1000) for ac in ac_output]
            
            df = pd.DataFrame({
                "hour_of_year": range(1, 8761),
                "capacity_factor": capacity_factors
            })
            
            logger.info(
                f"Successfully generated solar profile with {len(df)} hourly values"
            )
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch data from NREL PVWatts API: {e}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing API response: {e}")
            raise
    
    def generate_synthetic_profile(
        self,
        lat: float,
        lon: float,
        tilt: float,
        azimuth: float
    ) -> pd.DataFrame:
        """
        Generate synthetic solar profile based on location and configuration.
        
        This is a fallback method that creates realistic solar profiles using
        mathematical models when the API is unavailable or rate-limited.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            tilt: Tilt angle in degrees
            azimuth: Azimuth angle in degrees
            
        Returns:
            DataFrame with columns: [hour_of_year, capacity_factor]
        """
        logger.info("Generating synthetic solar profile")
        
        capacity_factors = []
        
        for day in range(1, 366):
            declination = 23.45 * np.sin(np.radians(360 * (284 + day) / 365))
            
            for hour in range(24):
                hour_angle = 15 * (hour - 12)
                
                solar_altitude = np.degrees(np.arcsin(
                    np.sin(np.radians(lat)) * np.sin(np.radians(declination)) +
                    np.cos(np.radians(lat)) * np.cos(np.radians(declination)) *
                    np.cos(np.radians(hour_angle))
                ))
                
                if solar_altitude <= 0:
                    cf = 0.0
                else:
                    solar_azimuth = np.degrees(np.arcsin(
                        np.cos(np.radians(declination)) * np.sin(np.radians(hour_angle)) /
                        np.cos(np.radians(solar_altitude))
                    ))
                    
                    incident_angle = np.degrees(np.arccos(
                        np.sin(np.radians(solar_altitude)) * np.cos(np.radians(tilt)) +
                        np.cos(np.radians(solar_altitude)) * np.sin(np.radians(tilt)) *
                        np.cos(np.radians(solar_azimuth - azimuth))
                    ))
                    
                    if incident_angle > 90:
                        cf = 0.0
                    else:
                        air_mass = 1 / (np.cos(np.radians(90 - solar_altitude)) + 0.50572 *
                                       (96.07995 - (90 - solar_altitude)) ** -1.6364)
                        
                        dni = 1000 * 0.7 ** (air_mass ** 0.678)
                        
                        cf = (dni * np.cos(np.radians(incident_angle)) / 1000) * 0.20
                        
                        cf = max(0.0, min(1.0, cf))
                        
                        seasonal_factor = 1.0 - 0.1 * np.cos(2 * np.pi * day / 365)
                        cf *= seasonal_factor
                
                capacity_factors.append(cf)
        
        df = pd.DataFrame({
            "hour_of_year": range(1, 8761),
            "capacity_factor": capacity_factors
        })
        
        logger.info(f"Generated synthetic profile with {len(df)} hourly values")
        
        return df
    
    def validate_capacity_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate capacity factor data.
        
        Args:
            df: DataFrame with capacity factors
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if len(df) != 8760:
            raise ValueError(f"Expected 8760 hours, got {len(df)}")
        
        required_columns = ["hour_of_year", "capacity_factor"]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df["capacity_factor"].isna().any():
            raise ValueError("Capacity factor column contains missing values")
        
        min_cf = df["capacity_factor"].min()
        max_cf = df["capacity_factor"].max()
        
        if min_cf < 0:
            raise ValueError(f"Capacity factors must be >= 0, found minimum: {min_cf}")
        
        if max_cf > 1:
            raise ValueError(f"Capacity factors must be <= 1, found maximum: {max_cf}")
        
        mean_cf = df["capacity_factor"].mean()
        logger.info(
            f"Capacity factor validation passed: "
            f"min={min_cf:.4f}, max={max_cf:.4f}, mean={mean_cf:.4f}"
        )
        
        return df
    
    def process_and_save(
        self,
        lat: float,
        lon: float,
        tilt: float,
        azimuth: float,
        output_path: str,
        use_api: bool = True
    ) -> pd.DataFrame:
        """
        Generate, validate, and save solar capacity factor data.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            tilt: Tilt angle in degrees
            azimuth: Azimuth angle in degrees
            output_path: Path to save processed CSV file
            use_api: If True, use NREL API; if False, use synthetic generation
            
        Returns:
            Processed DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        if use_api:
            try:
                df = self.generate_solar_profile(lat, lon, tilt, azimuth)
            except Exception as e:
                logger.warning(
                    f"Failed to use NREL API ({e}), falling back to synthetic generation"
                )
                df = self.generate_synthetic_profile(lat, lon, tilt, azimuth)
        else:
            df = self.generate_synthetic_profile(lat, lon, tilt, azimuth)
        
        df = self.validate_capacity_factors(df)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved solar profile to {output_path}")
        
        return df


def generate_solar_profile(
    lat: float = 31.9973,
    lon: float = -102.0779,
    tilt: float = 32.0,
    azimuth: float = 180.0,
    output_path: str = "data/processed/solar_cf_west_texas.csv",
    api_key: Optional[str] = None,
    use_api: bool = True
) -> pd.DataFrame:
    """
    Convenience function to generate solar capacity factor profile.
    
    Default parameters are configured for West Texas location with optimal
    fixed-tilt south-facing configuration.
    
    Args:
        lat: Latitude in decimal degrees (default: 31.9973 for West Texas)
        lon: Longitude in decimal degrees (default: -102.0779 for West Texas)
        tilt: Tilt angle in degrees (default: 32, approximately equal to latitude)
        azimuth: Azimuth angle in degrees (default: 180 for south-facing)
        output_path: Path to save processed CSV file
        api_key: NREL API key (if None, uses demo key)
        use_api: If True, use NREL API; if False, use synthetic generation
        
    Returns:
        DataFrame with columns: [hour_of_year, capacity_factor]
        
    Example:
        >>> df = generate_solar_profile()
        >>> print(df.head())
        >>> print(f"Annual average capacity factor: {df['capacity_factor'].mean():.2%}")
    """
    collector = SolarCollector(api_key=api_key)
    return collector.process_and_save(
        lat=lat,
        lon=lon,
        tilt=tilt,
        azimuth=azimuth,
        output_path=output_path,
        use_api=use_api
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    df = generate_solar_profile(
        lat=31.9973,
        lon=-102.0779,
        tilt=32.0,
        azimuth=180.0,
        output_path="data/processed/solar_cf_west_texas.csv",
        use_api=False
    )
    
    print(f"\nSolar Profile Summary:")
    print(f"Total hours: {len(df)}")
    print(f"Capacity Factor Statistics:")
    print(df['capacity_factor'].describe())
    print(f"\nAnnual average capacity factor: {df['capacity_factor'].mean():.2%}")
    print(f"Peak capacity factor: {df['capacity_factor'].max():.2%}")
    print(f"Hours with generation (CF > 0): {(df['capacity_factor'] > 0).sum()}")
