"""
Drawdown calculation module.

Calculates historical and forward-looking drawdowns for target variable generation.
"""

from typing import Literal

import numpy as np
import pandas as pd

from airs.utils.logging import get_logger

logger = get_logger(__name__)


class DrawdownCalculator:
    """
    Calculator for portfolio drawdowns.

    Computes:
    - Historical drawdowns from peak
    - Forward-looking drawdowns for prediction targets
    - Drawdown statistics (depth, duration, recovery)
    """

    # Default multi-asset portfolio weights
    DEFAULT_WEIGHTS = {
        "SPY": 0.40,   # US Equity
        "VEU": 0.20,   # International Equity
        "AGG": 0.25,   # Bonds
        "DJP": 0.10,   # Commodities
        "VNQ": 0.05,   # REITs
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        initial_value: float = 100.0,
    ):
        """
        Initialize drawdown calculator.

        Args:
            weights: Portfolio weights (defaults to multi-asset diversified)
            initial_value: Initial portfolio value
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.initial_value = initial_value

        # Validate weights sum to 1
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0, rtol=0.01):
            logger.warning(f"Portfolio weights sum to {weight_sum:.2f}, not 1.0")

    def calculate_portfolio_value(
        self,
        prices: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate portfolio value over time.

        Args:
            prices: DataFrame with asset prices (columns match weight keys)

        Returns:
            Series of portfolio values
        """
        # Calculate returns
        returns = prices.pct_change()

        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns.index)

        for asset, weight in self.weights.items():
            if asset in returns.columns:
                portfolio_returns += weight * returns[asset].fillna(0)
            else:
                logger.warning(f"Asset {asset} not found in prices")

        # Calculate cumulative portfolio value
        portfolio_value = (1 + portfolio_returns).cumprod() * self.initial_value
        portfolio_value.name = "portfolio_value"

        return portfolio_value

    def calculate_drawdown(
        self,
        prices: pd.Series | pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate drawdown from peak.

        Args:
            prices: Price series or portfolio value

        Returns:
            Drawdown series (negative values)
        """
        if isinstance(prices, pd.DataFrame):
            prices = self.calculate_portfolio_value(prices)

        # Running maximum
        running_max = prices.expanding().max()

        # Drawdown
        drawdown = (prices - running_max) / running_max
        drawdown.name = "drawdown"

        return drawdown

    def calculate_forward_drawdown(
        self,
        prices: pd.Series | pd.DataFrame,
        horizon: int = 15,
    ) -> pd.Series:
        """
        Calculate forward-looking maximum drawdown.

        For each day, calculates the maximum drawdown that will occur
        in the next `horizon` days.

        Args:
            prices: Price series or DataFrame
            horizon: Number of days to look ahead

        Returns:
            Forward drawdown series
        """
        if isinstance(prices, pd.DataFrame):
            prices = self.calculate_portfolio_value(prices)

        forward_dd = pd.Series(index=prices.index, dtype=float, name="forward_drawdown")

        for i in range(len(prices) - horizon):
            start_price = prices.iloc[i]
            future_prices = prices.iloc[i : i + horizon + 1]

            # Calculate drawdown from starting point
            min_future = future_prices.min()
            drawdown = (min_future - start_price) / start_price

            forward_dd.iloc[i] = drawdown

        return forward_dd

    def calculate_forward_max_drawdown(
        self,
        prices: pd.Series | pd.DataFrame,
        horizon: int = 15,
    ) -> pd.Series:
        """
        Calculate forward-looking maximum drawdown (more conservative).

        For each day, calculates the worst drawdown from any peak
        in the next `horizon` days.

        Args:
            prices: Price series or DataFrame
            horizon: Number of days to look ahead

        Returns:
            Forward max drawdown series
        """
        if isinstance(prices, pd.DataFrame):
            prices = self.calculate_portfolio_value(prices)

        forward_dd = pd.Series(index=prices.index, dtype=float, name="forward_max_drawdown")

        for i in range(len(prices) - horizon):
            future_prices = prices.iloc[i : i + horizon + 1]

            # Calculate drawdown within the future window
            running_max = future_prices.expanding().max()
            window_drawdowns = (future_prices - running_max) / running_max

            forward_dd.iloc[i] = window_drawdowns.min()

        return forward_dd

    def get_drawdown_periods(
        self,
        prices: pd.Series | pd.DataFrame,
        threshold: float = -0.05,
    ) -> pd.DataFrame:
        """
        Identify drawdown periods exceeding threshold.

        Args:
            prices: Price series
            threshold: Drawdown threshold (negative, e.g., -0.05 for 5%)

        Returns:
            DataFrame with drawdown period details
        """
        if isinstance(prices, pd.DataFrame):
            prices = self.calculate_portfolio_value(prices)

        drawdown = self.calculate_drawdown(prices)

        periods = []
        in_drawdown = False
        start_date = None
        peak_value = None

        for date, dd in drawdown.items():
            if dd < threshold and not in_drawdown:
                # Start of drawdown period
                in_drawdown = True
                start_date = date
                peak_value = prices.loc[:date].max()

            elif dd >= threshold * 0.5 and in_drawdown:
                # End of drawdown period (recovered to half the threshold)
                trough_date = drawdown.loc[start_date:date].idxmin()
                trough_dd = drawdown.loc[trough_date]

                periods.append({
                    "start_date": start_date,
                    "trough_date": trough_date,
                    "end_date": date,
                    "max_drawdown": trough_dd,
                    "duration_to_trough": (trough_date - start_date).days,
                    "duration_to_recovery": (date - start_date).days,
                    "peak_value": peak_value,
                    "trough_value": prices.loc[trough_date],
                })

                in_drawdown = False

        return pd.DataFrame(periods)

    def calculate_drawdown_statistics(
        self,
        prices: pd.Series | pd.DataFrame,
    ) -> dict:
        """
        Calculate comprehensive drawdown statistics.

        Args:
            prices: Price series

        Returns:
            Dictionary with drawdown statistics
        """
        if isinstance(prices, pd.DataFrame):
            prices = self.calculate_portfolio_value(prices)

        drawdown = self.calculate_drawdown(prices)

        # Maximum drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()

        # Average drawdown
        avg_dd = drawdown[drawdown < 0].mean()

        # Drawdown periods
        periods = self.get_drawdown_periods(prices, threshold=-0.05)

        # Time underwater
        underwater_pct = (drawdown < 0).mean() * 100

        return {
            "max_drawdown": max_dd,
            "max_drawdown_date": max_dd_date,
            "average_drawdown": avg_dd,
            "num_5pct_drawdowns": len(periods),
            "avg_recovery_days": periods["duration_to_recovery"].mean() if len(periods) > 0 else 0,
            "time_underwater_pct": underwater_pct,
        }


class DrawdownEventDetector:
    """
    Detects significant drawdown events for labeling.

    Identifies:
    - Start of drawdown periods
    - Severity classification
    - Lead time for prediction
    """

    def __init__(
        self,
        thresholds: list[float] = [-0.05, -0.075, -0.10],
        horizons: list[int] = [10, 15, 20],
    ):
        """
        Initialize event detector.

        Args:
            thresholds: Drawdown thresholds (negative values)
            horizons: Forward-looking horizons in days
        """
        self.thresholds = thresholds
        self.horizons = horizons

    def detect_events(
        self,
        prices: pd.Series,
        threshold: float = -0.05,
        min_gap: int = 5,
    ) -> pd.DataFrame:
        """
        Detect drawdown events.

        Args:
            prices: Price series
            threshold: Drawdown threshold
            min_gap: Minimum gap between events

        Returns:
            DataFrame with event details
        """
        calculator = DrawdownCalculator()
        drawdown = calculator.calculate_drawdown(prices)

        events = []
        last_event_idx = -min_gap

        for i, (date, dd) in enumerate(drawdown.items()):
            if dd < threshold and i - last_event_idx >= min_gap:
                # Check if this is a new drawdown (not continuation)
                if i == 0 or drawdown.iloc[i - 1] >= threshold * 0.8:
                    # Find the peak before this drawdown
                    peak_idx = prices.iloc[:i + 1].idxmax()

                    events.append({
                        "date": date,
                        "peak_date": peak_idx,
                        "drawdown": dd,
                        "severity": self._classify_severity(dd),
                    })

                    last_event_idx = i

        return pd.DataFrame(events)

    def _classify_severity(self, drawdown: float) -> str:
        """Classify drawdown severity."""
        if drawdown <= -0.10:
            return "severe"
        elif drawdown <= -0.075:
            return "moderate"
        elif drawdown <= -0.05:
            return "mild"
        else:
            return "none"
