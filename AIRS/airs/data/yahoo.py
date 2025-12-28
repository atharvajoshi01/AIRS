"""
Yahoo Finance data fetcher.

Fetches price data for ETFs, stocks, indices, and commodities.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

from airs.data.base import DataFetcher
from airs.utils.logging import get_logger

logger = get_logger(__name__)


# Asset definitions with metadata
YAHOO_ASSETS = {
    # Core Portfolio ETFs
    "SPY": {"name": "SPDR S&P 500 ETF", "asset_class": "us_equity"},
    "VEU": {"name": "Vanguard FTSE All-World ex-US ETF", "asset_class": "intl_equity"},
    "AGG": {"name": "iShares Core US Aggregate Bond ETF", "asset_class": "bonds"},
    "DJP": {"name": "iPath Bloomberg Commodity Index", "asset_class": "commodities"},
    "VNQ": {"name": "Vanguard Real Estate ETF", "asset_class": "reits"},
    # Indices
    "^GSPC": {"name": "S&P 500 Index", "asset_class": "index"},
    "^DJI": {"name": "Dow Jones Industrial Average", "asset_class": "index"},
    "^IXIC": {"name": "NASDAQ Composite", "asset_class": "index"},
    "^RUT": {"name": "Russell 2000", "asset_class": "index"},
    "^VIX": {"name": "CBOE Volatility Index", "asset_class": "volatility"},
    # Sector ETFs
    "XLF": {"name": "Financial Select Sector SPDR", "asset_class": "sector"},
    "XLE": {"name": "Energy Select Sector SPDR", "asset_class": "sector"},
    "XLK": {"name": "Technology Select Sector SPDR", "asset_class": "sector"},
    "XLV": {"name": "Health Care Select Sector SPDR", "asset_class": "sector"},
    "XLI": {"name": "Industrial Select Sector SPDR", "asset_class": "sector"},
    "XLP": {"name": "Consumer Staples Select Sector SPDR", "asset_class": "sector"},
    "XLY": {"name": "Consumer Discretionary Select Sector SPDR", "asset_class": "sector"},
    "XLU": {"name": "Utilities Select Sector SPDR", "asset_class": "sector"},
    "XLB": {"name": "Materials Select Sector SPDR", "asset_class": "sector"},
    # Commodities
    "GLD": {"name": "SPDR Gold Shares", "asset_class": "commodities"},
    "USO": {"name": "United States Oil Fund", "asset_class": "commodities"},
    "DBA": {"name": "Invesco DB Agriculture Fund", "asset_class": "commodities"},
    # International
    "EEM": {"name": "iShares MSCI Emerging Markets ETF", "asset_class": "intl_equity"},
    "EFA": {"name": "iShares MSCI EAFE ETF", "asset_class": "intl_equity"},
    # Fixed Income
    "TLT": {"name": "iShares 20+ Year Treasury Bond ETF", "asset_class": "bonds"},
    "IEF": {"name": "iShares 7-10 Year Treasury Bond ETF", "asset_class": "bonds"},
    "SHY": {"name": "iShares 1-3 Year Treasury Bond ETF", "asset_class": "bonds"},
    "LQD": {"name": "iShares iBoxx Investment Grade Corporate Bond", "asset_class": "bonds"},
    "HYG": {"name": "iShares iBoxx High Yield Corporate Bond", "asset_class": "bonds"},
    # Volatility Products
    "VIXY": {"name": "ProShares VIX Short-Term Futures", "asset_class": "volatility"},
}


class YahooFetcher(DataFetcher):
    """Fetcher for Yahoo Finance data."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        rate_limit: int = 2000,  # Yahoo is generous with rate limits
    ):
        """
        Initialize Yahoo Finance fetcher.

        Args:
            cache_dir: Cache directory
            rate_limit: Requests per minute limit
        """
        super().__init__(cache_dir=cache_dir, rate_limit=rate_limit)

    @property
    def source_name(self) -> str:
        return "yahoo"

    def fetch_single(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch data for a single symbol.

        Args:
            symbol: Yahoo Finance symbol
            start_date: Start date
            end_date: End date
            interval: Data interval (1d, 1wk, 1mo)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=pd.Timestamp(start_date),
                end=pd.Timestamp(end_date) + pd.Timedelta(days=1),  # End is exclusive
                interval=interval,
                auto_adjust=True,  # Adjust for splits and dividends
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Standardize column names
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            # Ensure datetime index
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)  # Remove timezone
            df.index.name = "date"

            # Add metadata
            if symbol in YAHOO_ASSETS:
                df.attrs["name"] = YAHOO_ASSETS[symbol]["name"]
                df.attrs["asset_class"] = YAHOO_ASSETS[symbol]["asset_class"]

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
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of Yahoo Finance symbols
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            DataFrame with data for all symbols
        """
        try:
            # Use yfinance download for multiple symbols (more efficient)
            df = yf.download(
                symbols,
                start=pd.Timestamp(start_date),
                end=pd.Timestamp(end_date) + pd.Timedelta(days=1),
                interval=interval,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )

            if df.empty:
                logger.warning(f"No data returned for {symbols}")
                return pd.DataFrame()

            # Ensure datetime index
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)
            df.index.name = "date"

            logger.info(f"Fetched {len(symbols)} symbols: {len(df)} observations")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            raise

    def fetch_prices(
        self,
        symbols: list[str],
        start_date: str | datetime,
        end_date: str | datetime,
        price_col: Literal["close", "open", "high", "low"] = "close",
    ) -> pd.DataFrame:
        """
        Fetch just price data for multiple symbols.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            price_col: Which price to return

        Returns:
            DataFrame with prices only
        """
        df = self.fetch(symbols, start_date, end_date)

        if df.empty:
            return df

        # Handle capitalization - yfinance returns 'Close', 'Open', etc.
        price_col_cap = price_col.capitalize()

        # Extract just the price column
        if isinstance(df.columns, pd.MultiIndex):
            # Multi-level columns from yfinance
            # Structure can be (Ticker, PriceType) or (PriceType, Ticker) depending on group_by
            try:
                # Try level 1 first (group_by='ticker' gives (Ticker, PriceType))
                prices = df.xs(price_col_cap, axis=1, level=1)
            except KeyError:
                try:
                    # Try level 0 (group_by='column' gives (PriceType, Ticker))
                    prices = df.xs(price_col_cap, axis=1, level=0)
                except KeyError:
                    # Fallback: try lowercase
                    try:
                        prices = df.xs(price_col, axis=1, level=1)
                    except KeyError:
                        prices = df.xs(price_col, axis=1, level=0)
            return prices
        else:
            # Single-level columns
            if price_col_cap in df.columns:
                return df[[price_col_cap]]
            elif price_col in df.columns:
                return df[[price_col]]
            else:
                logger.error(f"Column {price_col} not found in {df.columns.tolist()}")
                return pd.DataFrame()

    def fetch_returns(
        self,
        symbols: list[str],
        start_date: str | datetime,
        end_date: str | datetime,
        method: Literal["simple", "log"] = "simple",
    ) -> pd.DataFrame:
        """
        Fetch returns for multiple symbols.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            method: Return calculation method

        Returns:
            DataFrame with returns
        """
        prices = self.fetch_prices(symbols, start_date, end_date)

        if prices.empty:
            return prices

        if method == "simple":
            returns = prices.pct_change()
        else:  # log returns
            import numpy as np

            returns = np.log(prices / prices.shift(1))

        return returns.dropna()

    def fetch_portfolio_assets(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch data for core portfolio assets.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with portfolio asset data
        """
        portfolio_symbols = ["SPY", "VEU", "AGG", "DJP", "VNQ"]
        return self.fetch_prices(portfolio_symbols, start_date, end_date)

    def fetch_sector_etfs(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch data for sector ETFs.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with sector ETF data
        """
        sector_symbols = ["XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]
        return self.fetch_prices(sector_symbols, start_date, end_date)

    def fetch_vix(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Fetch VIX data.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with VIX data
        """
        return self.fetch_single("^VIX", start_date, end_date)

    def calculate_portfolio_value(
        self,
        prices: pd.DataFrame,
        weights: dict[str, float],
        initial_value: float = 100.0,
    ) -> pd.Series:
        """
        Calculate portfolio value over time given weights.

        Args:
            prices: DataFrame of prices
            weights: Dictionary of symbol -> weight
            initial_value: Initial portfolio value

        Returns:
            Series of portfolio values
        """
        # Calculate returns
        returns = prices.pct_change()

        # Calculate portfolio returns
        portfolio_returns = pd.Series(0, index=returns.index)
        for symbol, weight in weights.items():
            if symbol in returns.columns:
                portfolio_returns += weight * returns[symbol]

        # Calculate cumulative portfolio value
        portfolio_value = (1 + portfolio_returns).cumprod() * initial_value

        return portfolio_value
