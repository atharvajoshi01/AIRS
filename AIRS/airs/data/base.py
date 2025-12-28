"""
Base class for data fetchers.

Defines the interface and common functionality for all data sources.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from airs.utils.logging import get_logger

logger = get_logger(__name__)


class DataFetcher(ABC):
    """Abstract base class for data fetchers."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        rate_limit: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize data fetcher.

        Args:
            cache_dir: Directory for caching data
            rate_limit: Maximum requests per minute
            max_retries: Maximum retry attempts for failed requests
        """
        self.cache_dir = cache_dir or Path("./data/raw")
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self._last_request_time: datetime | None = None

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of the data source."""
        pass

    @abstractmethod
    def fetch(
        self,
        symbols: list[str],
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch data for the specified symbols and date range.

        Args:
            symbols: List of symbols to fetch
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with fetched data
        """
        pass

    @abstractmethod
    def fetch_single(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch data for a single symbol.

        Args:
            symbol: Symbol to fetch
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with fetched data
        """
        pass

    def get_cache_path(self, symbol: str) -> Path:
        """
        Get the cache file path for a symbol.

        Args:
            symbol: Symbol name

        Returns:
            Path to cache file
        """
        # Sanitize symbol for filename
        safe_symbol = symbol.replace("^", "").replace("/", "_")
        return self.cache_dir / f"{self.source_name}_{safe_symbol}.parquet"

    def load_from_cache(self, symbol: str) -> pd.DataFrame | None:
        """
        Load data from cache if available.

        Args:
            symbol: Symbol to load

        Returns:
            DataFrame if cached, None otherwise
        """
        cache_path = self.get_cache_path(symbol)
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                logger.debug(f"Loaded {symbol} from cache")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")
        return None

    def save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Save data to cache.

        Args:
            symbol: Symbol name
            df: DataFrame to cache
        """
        cache_path = self.get_cache_path(symbol)
        try:
            df.to_parquet(cache_path)
            logger.debug(f"Cached {symbol} to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache {symbol}: {e}")

    def fetch_with_cache(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch data with caching support.

        Args:
            symbol: Symbol to fetch
            start_date: Start date
            end_date: End date
            force_refresh: If True, bypass cache

        Returns:
            DataFrame with data
        """
        if not force_refresh:
            cached = self.load_from_cache(symbol)
            if cached is not None:
                # Check if cache covers requested range
                if len(cached) > 0:
                    cache_start = cached.index.min()
                    cache_end = cached.index.max()
                    req_start = pd.Timestamp(start_date)
                    req_end = pd.Timestamp(end_date)

                    if cache_start <= req_start and cache_end >= req_end:
                        # Filter to requested range
                        return cached.loc[req_start:req_end]

        # Fetch fresh data
        df = self.fetch_single(symbol, start_date, end_date)

        # Update cache
        if len(df) > 0:
            self.save_to_cache(symbol, df)

        return df

    def validate_data(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Validate fetched data quality.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        results = {
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": None,
            "missing_pct": {},
            "has_nulls": False,
        }

        if len(df) > 0:
            results["date_range"] = (
                df.index.min().isoformat(),
                df.index.max().isoformat(),
            )

            for col in df.columns:
                missing = df[col].isna().sum() / len(df) * 100
                results["missing_pct"][col] = round(missing, 2)
                if missing > 0:
                    results["has_nulls"] = True

        return results
