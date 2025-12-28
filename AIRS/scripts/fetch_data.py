#!/usr/bin/env python3
"""
Fetch market and economic data for AIRS.

This script fetches data from multiple sources (Yahoo Finance, FRED, Alpha Vantage)
and prepares it for feature engineering and model training.

Usage:
    python scripts/fetch_data.py --all
    python scripts/fetch_data.py --incremental
    python scripts/fetch_data.py --start 2020-01-01 --end 2024-01-01
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from airs.config import get_settings
from airs.data import DataAggregator
from airs.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch market and economic data for AIRS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch all historical data (default: last 10 years)",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Fetch only new data since last fetch",
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Default: 10 years ago",
    )

    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: today",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Default: ./data/processed",
    )

    parser.add_argument(
        "--skip-macro",
        action="store_true",
        help="Skip macroeconomic indicators (faster)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh, ignore cache",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def get_date_range(args: argparse.Namespace) -> tuple[str, str]:
    """Determine start and end dates based on arguments."""
    settings = get_settings()

    # Default end date is today
    if args.end:
        end_date = args.end
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # Default start date
    if args.start:
        start_date = args.start
    elif args.incremental:
        # For incremental, fetch last 30 days
        start_dt = datetime.now() - timedelta(days=30)
        start_date = start_dt.strftime("%Y-%m-%d")
    else:
        # Default: 10 years of data
        start_dt = datetime.now() - timedelta(days=365 * 10)
        start_date = start_dt.strftime("%Y-%m-%d")

    return start_date, end_date


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    logger.info("=" * 60)
    logger.info("AIRS Data Fetch")
    logger.info("=" * 60)

    # Get settings
    settings = get_settings()

    # Validate API keys
    if not settings.fred_api_key:
        logger.error("FRED_API_KEY not set in environment or .env file")
        logger.error("Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return 1

    # Get date range
    start_date, end_date = get_date_range(args)
    logger.info(f"Date range: {start_date} to {end_date}")

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = settings.data_dir / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    try:
        # Initialize aggregator
        logger.info("Initializing data aggregator...")
        aggregator = DataAggregator(
            fred_api_key=settings.fred_api_key,
            alpha_vantage_api_key=settings.alpha_vantage_api_key or None,
        )

        # Fetch all data
        logger.info("Fetching data from all sources...")
        logger.info("  - Yahoo Finance: Prices, sectors, volatility")
        logger.info("  - FRED: Yields, spreads, macro indicators")

        include_macro = not args.skip_macro

        # Use the prepare_features_dataset for complete pipeline
        df = aggregator.prepare_features_dataset(
            start_date=start_date,
            end_date=end_date,
            forward_fill=True,
            drop_na_threshold=0.5,
        )

        logger.info(f"Fetched data shape: {df.shape}")
        logger.info(f"Date range in data: {df.index.min()} to {df.index.max()}")
        logger.info(f"Columns: {len(df.columns)}")

        # Save dataset
        output_path = aggregator.save_dataset(
            df,
            name="market_data",
            format="parquet",
        )
        logger.info(f"Saved to: {output_path}")

        # Also save as CSV for inspection
        csv_path = output_dir / "market_data.csv"
        df.to_csv(csv_path)
        logger.info(f"CSV copy saved to: {csv_path}")

        # Print summary statistics
        logger.info("")
        logger.info("=" * 60)
        logger.info("Data Summary")
        logger.info("=" * 60)
        logger.info(f"Total observations: {len(df)}")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # Check for missing data
        missing_pct = df.isnull().sum() / len(df) * 100
        high_missing = missing_pct[missing_pct > 5]
        if len(high_missing) > 0:
            logger.warning(f"Columns with >5% missing: {len(high_missing)}")
            for col, pct in high_missing.items():
                logger.warning(f"  {col}: {pct:.1f}%")
        else:
            logger.info("No columns with >5% missing data")

        logger.info("")
        logger.info("Data fetch completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
