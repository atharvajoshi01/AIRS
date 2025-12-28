"""
Alpha Vantage data fetcher.

Fetches additional market data not available from Yahoo/FRED.
Note: Free tier has 25 requests/day, 5 requests/minute limit.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from airs.config.settings import get_settings
from airs.data.base import DataFetcher
from airs.utils.logging import get_logger

logger = get_logger(__name__)


class AlphaVantageFetcher(DataFetcher):
    """Fetcher for Alpha Vantage data."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        rate_limit: int = 5,  # 5 requests per minute for free tier
    ):
        """
        Initialize Alpha Vantage fetcher.

        Args:
            api_key: Alpha Vantage API key (defaults to settings)
            cache_dir: Cache directory
            rate_limit: Requests per minute limit
        """
        super().__init__(cache_dir=cache_dir, rate_limit=rate_limit)

        settings = get_settings()
        self.api_key = api_key or settings.alpha_vantage_api_key

        if not self.api_key:
            logger.warning(
                "Alpha Vantage API key not set. Functionality will be limited."
            )

        self._request_count = 0
        self._window_start = time.time()
        self._daily_count = 0
        self._daily_limit = 25  # Free tier daily limit

    @property
    def source_name(self) -> str:
        return "alpha_vantage"

    def _rate_limit_wait(self) -> None:
        """Enforce rate limiting."""
        current_time = time.time()

        # Reset minute counter every minute
        if current_time - self._window_start > 60:
            self._request_count = 0
            self._window_start = current_time

        # Check minute limit
        if self._request_count >= self.rate_limit:
            wait_time = 60 - (current_time - self._window_start)
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time + 1)  # Add 1 second buffer
                self._request_count = 0
                self._window_start = time.time()

        # Check daily limit
        if self._daily_count >= self._daily_limit:
            logger.warning(
                "Daily API limit reached (25 requests). Use cached data or wait."
            )
            raise ValueError("Alpha Vantage daily limit exceeded")

        self._request_count += 1
        self._daily_count += 1

    def _make_request(self, params: dict[str, Any]) -> dict:
        """
        Make API request with rate limiting.

        Args:
            params: Request parameters

        Returns:
            JSON response
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not configured")

        self._rate_limit_wait()

        params["apikey"] = self.api_key

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for API error messages
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def fetch_single(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch daily time series for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
        }

        data = self._make_request(params)

        if "Time Series (Daily)" not in data:
            logger.warning(f"No time series data for {symbol}")
            return pd.DataFrame()

        # Parse the time series
        ts_data = data["Time Series (Daily)"]
        records = []

        for date_str, values in ts_data.items():
            records.append(
                {
                    "date": pd.Timestamp(date_str),
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "adjusted_close": float(values["5. adjusted close"]),
                    "volume": int(values["6. volume"]),
                    "dividend": float(values["7. dividend amount"]),
                    "split_coef": float(values["8. split coefficient"]),
                }
            )

        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        # Filter to date range
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        df = df[(df.index >= start) & (df.index <= end)]

        logger.info(f"Fetched {symbol}: {len(df)} observations")
        return df

    def fetch(
        self,
        symbols: list[str],
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch data for multiple symbols.

        Note: Due to rate limits, this can be slow. Consider using cache.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with data for all symbols
        """
        dfs = []

        for symbol in symbols:
            try:
                df = self.fetch_with_cache(symbol, start_date, end_date)
                df = df[["adjusted_close"]].rename(columns={"adjusted_close": symbol})
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, axis=1)
        return result.sort_index()

    def fetch_treasury_yield(
        self,
        maturity: str = "10year",
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch Treasury yield curve data.

        Args:
            maturity: Treasury maturity (3month, 2year, 5year, 7year, 10year, 30year)
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            DataFrame with Treasury yield data
        """
        params = {
            "function": "TREASURY_YIELD",
            "interval": "daily",
            "maturity": maturity,
        }

        data = self._make_request(params)

        if "data" not in data:
            logger.warning(f"No Treasury yield data for {maturity}")
            return pd.DataFrame()

        records = []
        for item in data["data"]:
            if item["value"] != ".":
                records.append(
                    {
                        "date": pd.Timestamp(item["date"]),
                        "yield": float(item["value"]),
                    }
                )

        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        # Filter to date range if provided
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        df.columns = [f"treasury_{maturity}"]
        return df

    def fetch_economic_indicator(
        self,
        indicator: str,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch economic indicator data.

        Args:
            indicator: Indicator name (GDP, REAL_GDP, CPI, INFLATION,
                      RETAIL_SALES, DURABLES, UNEMPLOYMENT, NONFARM_PAYROLL)
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            DataFrame with indicator data
        """
        valid_indicators = [
            "GDP",
            "REAL_GDP",
            "CPI",
            "INFLATION",
            "RETAIL_SALES",
            "DURABLES",
            "UNEMPLOYMENT",
            "NONFARM_PAYROLL",
        ]

        if indicator not in valid_indicators:
            raise ValueError(f"Invalid indicator. Must be one of: {valid_indicators}")

        params = {"function": indicator}

        data = self._make_request(params)

        if "data" not in data:
            logger.warning(f"No data for indicator {indicator}")
            return pd.DataFrame()

        records = []
        for item in data["data"]:
            if item["value"] != ".":
                records.append(
                    {
                        "date": pd.Timestamp(item["date"]),
                        "value": float(item["value"]),
                    }
                )

        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        # Filter to date range if provided
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        df.columns = [indicator.lower()]
        return df

    def fetch_forex(
        self,
        from_currency: str = "EUR",
        to_currency: str = "USD",
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch forex exchange rate data.

        Args:
            from_currency: Source currency
            to_currency: Target currency
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with exchange rate data
        """
        params = {
            "function": "FX_DAILY",
            "from_symbol": from_currency,
            "to_symbol": to_currency,
            "outputsize": "full",
        }

        data = self._make_request(params)

        key = "Time Series FX (Daily)"
        if key not in data:
            logger.warning(f"No forex data for {from_currency}/{to_currency}")
            return pd.DataFrame()

        records = []
        for date_str, values in data[key].items():
            records.append(
                {
                    "date": pd.Timestamp(date_str),
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                }
            )

        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        # Filter to date range
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        # Add currency pair name to columns
        pair = f"{from_currency}{to_currency}"
        df.columns = [f"{pair}_{col}" for col in df.columns]

        return df
