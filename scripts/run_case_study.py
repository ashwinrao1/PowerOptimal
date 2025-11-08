"""
Run 300MW West Texas case study analysis.

This script runs a comprehensive case study analysis including:
1. Baseline grid-only optimization
2. Optimal portfolio optimization with all technologies
3. Calculation of financial metrics (upfront investment, annual savings, payback)
4. Generation of all visualizations
5. Saving comprehensive results to case_study_results.json
"""

import sys
from pathlib import Path
import logging
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.models.market_data import MarketData
from src.models.technology import TechnologyCosts, FacilityParams
from src.optimization.model_builder import build_optimization_model
from src.optimization.solver import solve_model
from src.optimization.solution_extractor import extract_solution

# Import visualization modules
from src.visualization.capacity_viz import plot_capacity_mix
from src.visualization.dispatch_viz import plot_dispatch_heatmap
from src.visualization.cost_viz import plot_cost_breakdown
from src.visualization.reliability_viz import plot_reliability_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_market_data(year: int = 2023) -> MarketData:
    """Load market data from processed CSV files."""
    logger.info(f"Loading market data for year {year}...")
    
    # Load ERCOT LMP data
    lmp_df = pd.read_csv('data/processed/ercot_lmp_hourly_2022_2024.csv')
    lmp_df['timestamp'] = pd.to_datetime(lmp_df['timestamp'])
    
    # Filter to specific year
    lmp_df = lmp_df[lmp_df['timestamp'].dt.year == year].copy()
    
    # Handle leap years by taking first 8760 hours
    if len(lmp_df) > 8760:
        logger.warning(f"Year {year} has {len(lmp_df)} hours (leap year). Using first 8760 hours.")
        lmp_df = lmp_df.iloc[:8760].copy()
    
    if len(lmp_df) != 8760:
        raise ValueError(f"Expected 8760 hours for year {year}, got {len(lmp_df)}")
    
    # Load other data files
    solar_df = pd.read_csv('data/processed/solar_cf_west_texas.csv')
    gas_df = pd.read_csv('data/processed/gas_prices_hourly.csv')
    gas_df['timestamp'] = pd.to_datetime(gas_df['timestamp'])
    gas_df = gas_df[gas_df['timestamp'].dt.year == year].copy()
    
    if len(gas_df) > 8760:
        gas_df = gas_df.iloc[:8760].copy()
    
    carbon_df = pd.read_csv('data/processed/grid_carbon_intensity.csv')
    carbon_df['timestamp'] = pd.to_datetime(carbon_df['timestamp'])
    carbon_df = carbon_df[carbon_df['timestamp'].dt.year == year].copy()
    
    if len(carbon_df) > 8760:
        carbon_df = carbon_df.iloc[:8760].copy()
    
    # Create MarketData object
    market_data = MarketData(
        timestamp=lmp_df['timestamp'],
        lmp=lmp_df['lmp_dam'].values,
        gas_price=gas_df['price_mmbtu'].values,
        solar_cf=solar_df['capacity_factor'].values,
        grid_carbon_intensity=carbon_df['carbon_intensity_kg_per_mwh'].values
    )
    
    logger.info("Market data loaded successfully")
    return market_data


def load_technology_costs() -> TechnologyCosts:
    """Load technology costs from JSON file."""
    logger.info("Loading technology costs...")
    
    with open('data/tech_costs.json', 'r') as f:
        costs_data = json.load(f)
