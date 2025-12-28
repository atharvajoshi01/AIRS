"""
Backtest engine for AIRS.

Simulates portfolio performance with signal-driven de-risking.
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from airs.backtest.portfolio import Portfolio
from airs.backtest.costs import TransactionCostModel
from airs.backtest.metrics import PerformanceMetrics
from airs.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest."""

    # Portfolio settings
    initial_value: float = 100_000.0
    target_weights: dict[str, float] | None = None

    # De-risking settings
    derisk_equity_reduction: float = 0.5  # Reduce equity by 50%
    derisk_cash_increase: float = 0.3  # Increase cash allocation
    rerisk_days: int = 5  # Days to gradually re-risk
    min_signal_days: int = 3  # Minimum days of alert before action

    # Signal settings
    signal_lag: int = 1  # Days between signal and trade
    alert_threshold: float = 0.5  # Probability threshold for alert

    # Cost settings
    trading_cost_bps: float = 10.0  # 10 bps per trade
    slippage_bps: float = 5.0  # 5 bps slippage

    # Rebalancing
    rebalance_frequency: str = "monthly"  # monthly, quarterly, never
    rebalance_threshold: float = 0.05  # Min deviation to trigger rebalance


class BacktestEngine:
    """
    Engine for running portfolio backtests with signal-driven de-risking.
    """

    # Default multi-asset allocation
    DEFAULT_WEIGHTS = {
        "SPY": 0.40,
        "VEU": 0.20,
        "AGG": 0.25,
        "DJP": 0.10,
        "VNQ": 0.05,
    }

    def __init__(self, config: BacktestConfig | None = None):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.config.target_weights = self.config.target_weights or self.DEFAULT_WEIGHTS

        self.cost_model = TransactionCostModel(
            trading_cost_bps=self.config.trading_cost_bps,
            slippage_bps=self.config.slippage_bps,
        )

        # Results storage
        self.portfolio_values: pd.Series | None = None
        self.positions_history: pd.DataFrame | None = None
        self.trades: list[dict] = []
        self.alerts_triggered: list[dict] = []

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Run backtest.

        Args:
            prices: Asset prices (columns match weight keys)
            signals: Signal series (0-1 probability or binary)
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dictionary with backtest results
        """
        # Align data
        prices, signals = self._align_data(prices, signals, start_date, end_date)

        if len(prices) == 0:
            raise ValueError("No data available for backtest")

        logger.info(
            f"Running backtest from {prices.index[0]} to {prices.index[-1]} "
            f"({len(prices)} days)"
        )

        # Initialize portfolio
        portfolio = Portfolio(
            initial_value=self.config.initial_value,
            target_weights=self.config.target_weights,
        )

        # Initialize on first day
        portfolio.initialize(prices.iloc[0])

        # Storage for daily values
        portfolio_values = []
        positions_history = []

        # State tracking
        in_derisk_mode = False
        derisk_start_date = None
        rerisk_days_remaining = 0
        consecutive_alert_days = 0

        # Run simulation day by day
        for i, (date, row) in enumerate(prices.iterrows()):
            current_prices = row

            # Get lagged signal (1-day lag)
            signal_idx = max(0, i - self.config.signal_lag)
            signal_value = signals.iloc[signal_idx] if signal_idx < len(signals) else 0

            # Check for alert
            is_alert = signal_value >= self.config.alert_threshold

            if is_alert:
                consecutive_alert_days += 1
            else:
                consecutive_alert_days = 0

            # De-risking logic
            if not in_derisk_mode and consecutive_alert_days >= self.config.min_signal_days:
                # Enter de-risk mode
                in_derisk_mode = True
                derisk_start_date = date

                # Calculate de-risked weights
                derisk_weights = self._calculate_derisk_weights()

                # Execute de-risking trades
                trades = portfolio.rebalance_to_weights(
                    derisk_weights,
                    current_prices,
                    self.cost_model,
                )
                self.trades.extend(trades)

                self.alerts_triggered.append({
                    "date": date,
                    "signal": signal_value,
                    "action": "derisk",
                })

                logger.info(f"De-risking triggered on {date}, signal={signal_value:.2f}")

            elif in_derisk_mode and not is_alert:
                # Start re-risking process
                if rerisk_days_remaining == 0:
                    rerisk_days_remaining = self.config.rerisk_days

                rerisk_days_remaining -= 1

                if rerisk_days_remaining == 0:
                    # Return to normal allocation
                    in_derisk_mode = False

                    trades = portfolio.rebalance_to_weights(
                        self.config.target_weights,
                        current_prices,
                        self.cost_model,
                    )
                    self.trades.extend(trades)

                    self.alerts_triggered.append({
                        "date": date,
                        "signal": signal_value,
                        "action": "rerisk",
                    })

                    logger.info(f"Re-risking completed on {date}")

            # Update portfolio with current prices
            portfolio.update_values(current_prices)

            # Periodic rebalancing (if not in de-risk mode)
            if not in_derisk_mode and self._should_rebalance(date, i):
                trades = portfolio.rebalance_to_weights(
                    self.config.target_weights,
                    current_prices,
                    self.cost_model,
                )
                self.trades.extend(trades)

            # Record daily state
            portfolio_values.append({
                "date": date,
                "value": portfolio.total_value,
                "in_derisk_mode": in_derisk_mode,
                "signal": signal_value,
            })

            positions_history.append({
                "date": date,
                **{f"{k}_weight": v for k, v in portfolio.current_weights.items()},
                **{f"{k}_value": v for k, v in portfolio.position_values.items()},
            })

        # Convert to DataFrames
        self.portfolio_values = pd.DataFrame(portfolio_values).set_index("date")
        self.positions_history = pd.DataFrame(positions_history).set_index("date")

        # Calculate metrics
        metrics = PerformanceMetrics()
        results = metrics.calculate_all(
            self.portfolio_values["value"],
            prices,
            self.config.target_weights,
        )

        results["trades"] = self.trades
        results["alerts"] = self.alerts_triggered
        results["n_trades"] = len(self.trades)
        results["n_alerts"] = len(self.alerts_triggered)

        return results

    def _align_data(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Align prices and signals to common date range."""
        # Filter to date range
        if start_date:
            prices = prices[prices.index >= start_date]
            signals = signals[signals.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]
            signals = signals[signals.index <= end_date]

        # Align indices
        common_dates = prices.index.intersection(signals.index)
        prices = prices.loc[common_dates]
        signals = signals.loc[common_dates]

        return prices, signals

    def _calculate_derisk_weights(self) -> dict[str, float]:
        """Calculate de-risked portfolio weights."""
        derisk_weights = {}

        for asset, weight in self.config.target_weights.items():
            if asset in ["SPY", "VEU", "VNQ"]:  # Equity-like
                derisk_weights[asset] = weight * (1 - self.config.derisk_equity_reduction)
            elif asset in ["AGG"]:  # Bonds
                derisk_weights[asset] = weight * 1.2  # Increase bonds
            else:
                derisk_weights[asset] = weight * 0.8  # Reduce commodities

        # Add cash allocation
        total_allocated = sum(derisk_weights.values())
        derisk_weights["CASH"] = 1.0 - total_allocated

        return derisk_weights

    def _should_rebalance(self, date: pd.Timestamp, day_index: int) -> bool:
        """Check if periodic rebalancing should occur."""
        if self.config.rebalance_frequency == "never":
            return False
        elif self.config.rebalance_frequency == "monthly":
            return date.day == 1 or day_index == 0
        elif self.config.rebalance_frequency == "quarterly":
            return date.month in [1, 4, 7, 10] and date.day == 1
        return False

    def run_benchmark(
        self,
        prices: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """
        Run benchmark (buy-and-hold) portfolio.

        Args:
            prices: Asset prices
            start_date: Start date
            end_date: End date

        Returns:
            Benchmark portfolio values
        """
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]

        # Calculate weighted returns
        returns = prices.pct_change()
        portfolio_returns = pd.Series(0.0, index=returns.index)

        for asset, weight in self.config.target_weights.items():
            if asset in returns.columns:
                portfolio_returns += weight * returns[asset].fillna(0)

        # Calculate cumulative value
        benchmark = (1 + portfolio_returns).cumprod() * self.config.initial_value
        benchmark.name = "benchmark"

        return benchmark

    def compare_to_benchmark(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
    ) -> dict[str, Any]:
        """
        Compare strategy to benchmark.

        Args:
            prices: Asset prices
            signals: Trading signals

        Returns:
            Comparison results
        """
        # Run strategy
        strategy_results = self.run(prices, signals)

        # Run benchmark
        benchmark = self.run_benchmark(prices)

        # Calculate comparative metrics
        metrics = PerformanceMetrics()

        strategy_metrics = strategy_results
        benchmark_metrics = metrics.calculate_all(
            benchmark, prices, self.config.target_weights
        )

        return {
            "strategy": strategy_metrics,
            "benchmark": benchmark_metrics,
            "excess_return": strategy_metrics["total_return"] - benchmark_metrics["total_return"],
            "excess_sharpe": strategy_metrics["sharpe_ratio"] - benchmark_metrics["sharpe_ratio"],
            "drawdown_improvement": benchmark_metrics["max_drawdown"] - strategy_metrics["max_drawdown"],
        }
