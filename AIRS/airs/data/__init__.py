"""Data collection layer for AIRS."""

from airs.data.base import DataFetcher
from airs.data.fred import FREDFetcher
from airs.data.yahoo import YahooFetcher
from airs.data.alpha_vantage import AlphaVantageFetcher
from airs.data.aggregator import DataAggregator

__all__ = [
    "DataFetcher",
    "FREDFetcher",
    "YahooFetcher",
    "AlphaVantageFetcher",
    "DataAggregator",
]
