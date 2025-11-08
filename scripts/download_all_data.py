"""
Download all data from external sources.

This script automates the collection of all required data for the optimization
model including ERCOT LMP data, solar profiles, natural gas prices, and grid
carbon intensity.
"""

import sys
from pathlib import Path
import logging
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.ercot_collector import fetch_ercot_lmp
from src.data_pipeline.solar_collector import generate_solar_profile
from src.data_pipeline.gas_collector import fetch_gas_prices
from src.data_pipeline.carbon_collector import fetch_grid_carbon

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_ercot_lmp(start_date: str, end_date: str, output_path: str) -> bool:
    """
    Download ERCOT LMP data.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Path to save processed CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("="*80)
        logger.info("DOWNLOADING ERCOT LMP DATA")
        logger.info("="*80)
        
        df = fetch_ercot_lmp(
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            hub="HB_WEST"
        )
        
        logger.info(f"Successfully downloaded {len(df)} hourly LMP records")
        logger.info(f"Saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download ERCOT LMP data: {e}", exc_info=True)
        return False


def download_solar_profile(output_path: str, use_api: bool = False) -> bool:
    """
    Generate solar capacity factor profile.
    
    Args:
        output_path: Path to save processed CSV file
        use_api: If True, use NREL API; if False, use synthetic generation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("="*80)
        logger.info("GENERATING SOLAR CAPACITY FACTOR PROFILE")
        logger.info("="*80)
        
        df = generate_solar_profile(
            lat=31.9973,
            lon=-102.0779,
            tilt=32.0,
            azimuth=180.0,
            output_path=output_path,
            use_api=use_api
        )
        
        logger.info(f"Successfully generated {len(df)} hourly capacity factors")
        logger.info(f"Average capacity factor: {df['capacity_factor'].mean():.2%}")
        logger.info(f"Saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate solar profile: {e}", exc_info=True)
        return False


def download_gas_prices(start_date: str, end_date: str, output_path: str) -> bool:
    """
    Download natural gas prices.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Path to save processed CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("="*80)
        logger.info("DOWNLOADING NATURAL GAS PRICES")
        logger.info("="*80)
        
        df = fetch_gas_prices(
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            hub="WAHA"
        )
        
        logger.info(f"Successfully downloaded {len(df)} hourly gas price records")
        logger.info(f"Average price: ${df['price_mmbtu'].mean():.2f}/MMBtu")
        logger.info(f"Saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download gas prices: {e}", exc_info=True)
        return False


def download_grid_carbon(start_date: str, end_date: str, output_path: str) -> bool:
    """
    Download grid carbon intensity data.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Path to save processed CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("="*80)
        logger.info("DOWNLOADING GRID CARBON INTENSITY DATA")
        logger.info("="*80)
        
        df = fetch_grid_carbon(
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            region="ERCO"
        )
        
        logger.info(f"Successfully downloaded {len(df)} hourly carbon intensity records")
        logger.info(f"Average intensity: {df['carbon_intensity_kg_per_mwh'].mean():.2f} kg CO2/MWh")
        logger.info(f"Saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download grid carbon intensity: {e}", exc_info=True)
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download all data required for datacenter energy optimization"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2022-01-01",
        help="Start date in YYYY-MM-DD format (default: 2022-01-01)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date in YYYY-MM-DD format (default: 2024-12-31)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data (default: data/processed)"
    )
    parser.add_argument(
        "--use-nrel-api",
        action="store_true",
        help="Use NREL PVWatts API for solar data (requires API key)"
    )
    parser.add_argument(
        "--skip-ercot",
        action="store_true",
        help="Skip ERCOT LMP data download"
    )
    parser.add_argument(
        "--skip-solar",
        action="store_true",
        help="Skip solar profile generation"
    )
    parser.add_argument(
        "--skip-gas",
        action="store_true",
        help="Skip gas price download"
    )
    parser.add_argument(
        "--skip-carbon",
        action="store_true",
        help="Skip carbon intensity download"
    )
    
    args = parser.parse_args()
    
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format. Use YYYY-MM-DD")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("DATA DOWNLOAD AUTOMATION SCRIPT")
    logger.info("="*80)
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"End date: {args.end_date}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*80)
    
    results = {}
    
    if not args.skip_ercot:
        ercot_path = output_dir / "ercot_lmp_hourly_2022_2024.csv"
        results['ercot'] = download_ercot_lmp(
            args.start_date,
            args.end_date,
            str(ercot_path)
        )
    else:
        logger.info("Skipping ERCOT LMP data download")
        results['ercot'] = None
    
    if not args.skip_solar:
        solar_path = output_dir / "solar_cf_west_texas.csv"
        results['solar'] = download_solar_profile(
            str(solar_path),
            use_api=args.use_nrel_api
        )
    else:
        logger.info("Skipping solar profile generation")
        results['solar'] = None
    
    if not args.skip_gas:
        gas_path = output_dir / "gas_prices_hourly.csv"
        results['gas'] = download_gas_prices(
            args.start_date,
            args.end_date,
            str(gas_path)
        )
    else:
        logger.info("Skipping gas price download")
        results['gas'] = None
    
    if not args.skip_carbon:
        carbon_path = output_dir / "grid_carbon_intensity.csv"
        results['carbon'] = download_grid_carbon(
            args.start_date,
            args.end_date,
            str(carbon_path)
        )
    else:
        logger.info("Skipping carbon intensity download")
        results['carbon'] = None
    
    logger.info("\n" + "="*80)
    logger.info("DATA DOWNLOAD SUMMARY")
    logger.info("="*80)
    
    success_count = sum(1 for v in results.values() if v is True)
    skip_count = sum(1 for v in results.values() if v is None)
    fail_count = sum(1 for v in results.values() if v is False)
    
    for dataset, status in results.items():
        if status is True:
            status_str = "SUCCESS"
        elif status is False:
            status_str = "FAILED"
        else:
            status_str = "SKIPPED"
        logger.info(f"  {dataset.upper()}: {status_str}")
    
    logger.info("="*80)
    logger.info(f"Total: {success_count} successful, {fail_count} failed, {skip_count} skipped")
    
    if fail_count > 0:
        logger.error(f"\n{fail_count} dataset(s) failed to download")
        logger.error("Check the logs above for error details")
        return 1
    
    if success_count > 0:
        logger.info("\nAll requested data downloaded successfully!")
        logger.info(f"Data saved to: {args.output_dir}")
        logger.info("\nNext steps:")
        logger.info("  1. Review downloaded data files")
        logger.info("  2. Run baseline optimization: python scripts/run_baseline.py")
        logger.info("  3. Run optimal portfolio: python scripts/run_optimal_portfolio.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
