"""
AIRS: AI-Driven Early-Warning System for Portfolio Drawdown Risk

A production-ready system for:
- Detecting multi-asset portfolio stress events
- Predicting large drawdowns (10%+) with 5-20 day lead time
- Recommending de-risking allocation shifts
- Backtesting with realistic transaction costs
"""

__version__ = "0.1.0"
__author__ = "AIRS Team"

from airs.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
