"""
Data aggregator for combining data from multiple sources.

Handles data alignment, quality checks, and point-in-time adjustments.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from airs.data.fred import FREDFetcher
from airs.data.yahoo import YahooFetcher
from airs.data.alpha_vantage import AlphaVantageFetcher
from airs.data.quality import DataQualityChecker
from airs.utils.logging import get_logger

logger = get_logger(__name__)


class DataAggregator:
    """
    Aggregates and aligns data from multiple sources.

    Ensures consistent timestamps, handles missing data,
    and applies point-in-time adjustments.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        fred_api_key: str | None = None,
        alpha_vantage_api_key: str | None = None,
    ):
        """
        Initialize data aggregator with all data sources.

        Args:
            cache_dir: Directory for caching data
            fred_api_key: FRED API key
            alpha_vantage_api_key: Alpha Vantage API key
        """
        self.cache_dir = cache_dir or Path("./data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize fetchers
        self.fred = FREDFetcher(
            api_key=fred_api_key,
            cache_dir=self.cache_dir / "fred",
        )
        self.yahoo = YahooFetcher(
            cache_dir=self.cache_dir / "yahoo",
        )
        self.alpha_vantage = AlphaVantageFetcher(
            api_key=alpha_vantage_api_key,
            cache_dir=self.cache_dir / "alpha_vantage",
        )

        # Initialize quality checker
        self.quality_checker = DataQualityChecker()

    def fetch_all(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        include_macro: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch all data from all sources.

        Args:
            start_date: Start date
            end_date: End date
            include_macro: Whether to include macro indicators

        Returns:
            Dictionary of DataFrames by category
        """
        data = {}

        logger.info("Fetching market data from Yahoo Finance...")
        data["prices"] = self._fetch_prices(start_date, end_date)
        data["sectors"] = self._fetch_sectors(start_date, end_date)

        logger.info("Fetching economic data from FRED...")
        data["yields"] = self._fetch_yields(start_date, end_date)
        data["spreads"] = self._fetch_spreads(start_date, end_date)
        data["credit"] = self._fetch_credit(start_date, end_date)
        data["financial_conditions"] = self._fetch_financial_conditions(
            start_date, end_date
        )

        if include_macro:
            data["macro"] = self._fetch_macro(start_date, end_date)

        return data

    def _fetch_prices(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """Fetch core asset prices."""
        symbols = ["SPY", "VEU", "AGG", "DJP", "VNQ", "^VIX", "GLD"]
        try:
            return self.yahoo.fetch_prices(symbols, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to fetch prices: {e}")
            return pd.DataFrame()

    def _fetch_sectors(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """Fetch sector ETF prices."""
        try:
            return self.yahoo.fetch_sector_etfs(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to fetch sectors: {e}")
            return pd.DataFrame()

    def _fetch_yields(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """Fetch Treasury yield data."""
        try:
            return self.fred.fetch_yields(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to fetch yields: {e}")
            return pd.DataFrame()

    def _fetch_spreads(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """Fetch yield spread data."""
        try:
            return self.fred.fetch_spreads(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to fetch spreads: {e}")
            return pd.DataFrame()

    def _fetch_credit(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """Fetch credit spread data."""
        try:
            return self.fred.fetch_credit(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to fetch credit: {e}")
            return pd.DataFrame()

    def _fetch_financial_conditions(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """Fetch financial conditions indices."""
        try:
            return self.fred.fetch_financial_conditions(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to fetch financial conditions: {e}")
            return pd.DataFrame()

    def _fetch_macro(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """Fetch macro indicators with point-in-time adjustment."""
        try:
            df = self.fred.fetch_macro(start_date, end_date)

            # Apply point-in-time adjustments for release delays
            adjusted_dfs = []
            for col in df.columns:
                series_df = df[[col]].copy()
                series_df = self.fred.apply_point_in_time(series_df, col)
                adjusted_dfs.append(series_df)

            if adjusted_dfs:
                return pd.concat(adjusted_dfs, axis=1)
            return df

        except Exception as e:
            logger.error(f"Failed to fetch macro: {e}")
            return pd.DataFrame()

    def combine_data(
        self,
        data: dict[str, pd.DataFrame],
        resample_freq: str = "D",
    ) -> pd.DataFrame:
        """
        Combine all data into a single DataFrame.

        Args:
            data: Dictionary of DataFrames
            resample_freq: Resampling frequency (D=daily, W=weekly)

        Returns:
            Combined DataFrame with aligned timestamps
        """
        dfs = []

        for name, df in data.items():
            if df.empty:
                continue

            # Flatten multi-level columns if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(col).strip() for col in df.columns.values]

            # Prefix columns with category
            df.columns = [f"{name}_{col}" for col in df.columns]
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        # Combine all DataFrames
        combined = pd.concat(dfs, axis=1)
        combined = combined.sort_index()

        # Resample if needed (for aligning different frequencies)
        if resample_freq != "D":
            combined = combined.resample(resample_freq).last()

        return combined

    def prepare_features_dataset(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        forward_fill: bool = True,
        drop_na_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Prepare a complete dataset for feature engineering.

        Args:
            start_date: Start date
            end_date: End date
            forward_fill: Whether to forward-fill missing values
            drop_na_threshold: Drop columns with more than this fraction missing

        Returns:
            Cleaned and aligned DataFrame
        """
        # Fetch all data
        data = self.fetch_all(start_date, end_date)

        # Combine into single DataFrame
        df = self.combine_data(data)

        if df.empty:
            logger.warning("No data fetched")
            return df

        # Check data quality
        quality_report = self.quality_checker.check_all(df)
        logger.info(f"Data quality report: {quality_report['summary']}")

        # Drop columns with too many missing values
        missing_pct = df.isna().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > drop_na_threshold].index
        if len(cols_to_drop) > 0:
            logger.warning(f"Dropping columns with >{drop_na_threshold*100}% missing: {list(cols_to_drop)}")
            df = df.drop(columns=cols_to_drop)

        # Forward fill missing values (with limit)
        if forward_fill:
            df = df.ffill(limit=5)

        # Drop remaining rows with NaN
        df = df.dropna()

        logger.info(
            f"Prepared dataset: {len(df)} rows, {len(df.columns)} columns, "
            f"from {df.index.min()} to {df.index.max()}"
        )

        return df

    def save_dataset(
        self,
        df: pd.DataFrame,
        name: str = "features_dataset",
        format: str = "parquet",
    ) -> Path:
        """
        Save dataset to file.

        Args:
            df: DataFrame to save
            name: Dataset name
            format: File format (parquet, csv)

        Returns:
            Path to saved file
        """
        output_dir = self.cache_dir.parent / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            path = output_dir / f"{name}.parquet"
            df.to_parquet(path)
        elif format == "csv":
            path = output_dir / f"{name}.csv"
            df.to_csv(path)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved dataset to {path}")
        return path

    def load_dataset(
        self,
        name: str = "features_dataset",
        format: str = "parquet",
    ) -> pd.DataFrame:
        """
        Load dataset from file.

        Args:
            name: Dataset name
            format: File format

        Returns:
            Loaded DataFrame
        """
        output_dir = self.cache_dir.parent / "processed"

        if format == "parquet":
            path = output_dir / f"{name}.parquet"
            return pd.read_parquet(path)
        elif format == "csv":
            path = output_dir / f"{name}.csv"
            return pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unknown format: {format}")
