"""
FRED (Federal Reserve Economic Data) fetcher.

Fetches economic indicators, interest rates, and credit spreads from FRED.
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from fredapi import Fred

from airs.config.settings import get_settings
from airs.data.base import DataFetcher
from airs.utils.logging import get_logger

logger = get_logger(__name__)


# FRED series definitions with metadata
FRED_SERIES = {
    # Treasury Yields
    "DGS2": {"name": "2-Year Treasury Rate", "frequency": "daily", "units": "percent"},
    "DGS5": {"name": "5-Year Treasury Rate", "frequency": "daily", "units": "percent"},
    "DGS10": {"name": "10-Year Treasury Rate", "frequency": "daily", "units": "percent"},
    "DGS30": {"name": "30-Year Treasury Rate", "frequency": "daily", "units": "percent"},
    "DTB3": {"name": "3-Month T-Bill Rate", "frequency": "daily", "units": "percent"},
    # Yield Spreads
    "T10Y2Y": {"name": "10Y-2Y Treasury Spread", "frequency": "daily", "units": "percent"},
    "T10Y3M": {"name": "10Y-3M Treasury Spread", "frequency": "daily", "units": "percent"},
    # Credit Spreads
    "BAMLC0A0CM": {
        "name": "ICE BofA US Corporate IG OAS",
        "frequency": "daily",
        "units": "percent",
    },
    "BAMLH0A0HYM2": {
        "name": "ICE BofA US High Yield OAS",
        "frequency": "daily",
        "units": "percent",
    },
    # Interest Rates
    "SOFR": {"name": "SOFR Rate", "frequency": "daily", "units": "percent"},
    "FEDFUNDS": {"name": "Fed Funds Rate", "frequency": "daily", "units": "percent"},
    "TEDRATE": {"name": "TED Spread", "frequency": "daily", "units": "percent"},
    # Volatility
    "VIXCLS": {"name": "VIX Index", "frequency": "daily", "units": "index"},
    # Financial Conditions
    "NFCI": {
        "name": "Chicago Fed NFCI",
        "frequency": "weekly",
        "units": "index",
        "release_lag": 7,
    },
    "ANFCI": {
        "name": "Chicago Fed Adjusted NFCI",
        "frequency": "weekly",
        "units": "index",
        "release_lag": 7,
    },
    # Macro Indicators
    "USSLIND": {
        "name": "Leading Index",
        "frequency": "monthly",
        "units": "index",
        "release_lag": 21,
    },
    "ICSA": {
        "name": "Initial Claims",
        "frequency": "weekly",
        "units": "thousands",
        "release_lag": 5,
    },
    "UMCSENT": {
        "name": "Consumer Sentiment",
        "frequency": "monthly",
        "units": "index",
        "release_lag": 14,
    },
    "PERMIT": {
        "name": "Building Permits",
        "frequency": "monthly",
        "units": "thousands",
        "release_lag": 18,
    },
    # Money Supply
    "M2SL": {
        "name": "M2 Money Stock",
        "frequency": "monthly",
        "units": "billions",
        "release_lag": 14,
    },
}


class FREDFetcher(DataFetcher):
    """Fetcher for FRED economic data."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        rate_limit: int = 120,  # FRED allows 120 requests per minute
    ):
        """
        Initialize FRED fetcher.

        Args:
            api_key: FRED API key (defaults to settings)
            cache_dir: Cache directory
            rate_limit: Requests per minute limit
        """
        super().__init__(cache_dir=cache_dir, rate_limit=rate_limit)

        settings = get_settings()
        self.api_key = api_key or settings.fred_api_key

        if not self.api_key:
            logger.warning("FRED API key not set. Some functionality may be limited.")
            self._client = None
        else:
            self._client = Fred(api_key=self.api_key)

        self._request_count = 0
        self._window_start = time.time()

    @property
    def source_name(self) -> str:
        return "fred"

    def _rate_limit_wait(self) -> None:
        """Enforce rate limiting."""
        current_time = time.time()

        # Reset counter every minute
        if current_time - self._window_start > 60:
            self._request_count = 0
            self._window_start = current_time

        # Wait if at limit
        if self._request_count >= self.rate_limit:
            wait_time = 60 - (current_time - self._window_start)
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self._request_count = 0
                self._window_start = time.time()

        self._request_count += 1

    def fetch_single(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch a single FRED series.

        Args:
            symbol: FRED series ID
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with the series data
        """
        if self._client is None:
            raise ValueError("FRED API key not configured")

        self._rate_limit_wait()

        try:
            # Fetch series
            series = self._client.get_series(
                symbol,
                observation_start=pd.Timestamp(start_date),
                observation_end=pd.Timestamp(end_date),
            )

            # Convert to DataFrame
            df = pd.DataFrame({symbol: series})
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"

            # Get metadata
            if symbol in FRED_SERIES:
                meta = FRED_SERIES[symbol]
                df.attrs["name"] = meta["name"]
                df.attrs["frequency"] = meta["frequency"]
                df.attrs["units"] = meta["units"]

            logger.info(f"Fetched {symbol}: {len(df)} observations")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            raise

    def fetch(
        self,
        symbols: list[str],
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch multiple FRED series.

        Args:
            symbols: List of FRED series IDs
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with all series
        """
        dfs = []

        for symbol in symbols:
            try:
                df = self.fetch_with_cache(symbol, start_date, end_date)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue

        if not dfs:
            return pd.DataFrame()

        # Combine all series
        result = pd.concat(dfs, axis=1)
        result = result.sort_index()

        return result

    def fetch_yields(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch all Treasury yield series.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with yield curve data
        """
        symbols = ["DTB3", "DGS2", "DGS5", "DGS10", "DGS30"]
        return self.fetch(symbols, start_date, end_date)

    def fetch_spreads(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch yield spread series.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with spread data
        """
        symbols = ["T10Y2Y", "T10Y3M", "TEDRATE"]
        return self.fetch(symbols, start_date, end_date)

    def fetch_credit(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch credit spread series.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with credit spread data
        """
        symbols = ["BAMLC0A0CM", "BAMLH0A0HYM2"]
        return self.fetch(symbols, start_date, end_date)

    def fetch_macro(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch macro indicator series.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with macro data
        """
        symbols = ["USSLIND", "ICSA", "UMCSENT", "PERMIT", "M2SL"]
        return self.fetch(symbols, start_date, end_date)

    def fetch_financial_conditions(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch financial conditions indices.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with financial conditions data
        """
        symbols = ["NFCI", "ANFCI"]
        return self.fetch(symbols, start_date, end_date)

    def get_release_calendar(self, symbol: str) -> dict:
        """
        Get release calendar information for a series.

        Args:
            symbol: FRED series ID

        Returns:
            Dictionary with release information
        """
        if symbol not in FRED_SERIES:
            return {}

        meta = FRED_SERIES[symbol]
        return {
            "series": symbol,
            "name": meta["name"],
            "frequency": meta["frequency"],
            "release_lag_days": meta.get("release_lag", 0),
        }

    def apply_point_in_time(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Apply point-in-time adjustment for release delays.

        Shifts data forward by the release lag to ensure
        no lookahead bias.

        Args:
            df: DataFrame with data
            symbol: Series symbol for release lag lookup

        Returns:
            Point-in-time adjusted DataFrame
        """
        if symbol not in FRED_SERIES:
            return df

        release_lag = FRED_SERIES[symbol].get("release_lag", 0)

        if release_lag > 0:
            # Shift data forward by release lag
            df = df.copy()
            df.index = df.index + pd.Timedelta(days=release_lag)
            logger.debug(f"Applied {release_lag} day release lag to {symbol}")

        return df
