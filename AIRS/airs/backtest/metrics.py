"""
Performance metrics for backtesting.

Comprehensive risk and return metrics.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from airs.utils.logging import get_logger
from airs.utils.stats import (
    calculate_returns,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_rolling_drawdown,
)

logger = get_logger(__name__)


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,  # 2% annual
        trading_days: int = 252,
    ):
        """
        Initialize metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate
            trading_days: Trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def calculate_all(
        self,
        portfolio_values: pd.Series,
        prices: pd.DataFrame | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate all performance metrics.

        Args:
            portfolio_values: Portfolio value time series
            prices: Asset prices (for benchmark)
            weights: Asset weights (for benchmark)

        Returns:
            Dictionary of metrics
        """
        returns = portfolio_values.pct_change().dropna()

        # Return metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        n_years = len(returns) / self.trading_days
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Risk metrics
        volatility = returns.std() * np.sqrt(self.trading_days)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(self.trading_days) if len(downside_returns) > 0 else 0

        # Drawdown
        max_dd, peak_date, trough_date = calculate_max_drawdown(portfolio_values)
        drawdowns = calculate_rolling_drawdown(portfolio_values)

        # Risk-adjusted metrics
        daily_rf = self.risk_free_rate / self.trading_days
        excess_returns = returns - daily_rf

        sharpe = (
            np.sqrt(self.trading_days) * excess_returns.mean() / returns.std()
            if returns.std() > 0
            else 0
        )

        sortino = (
            np.sqrt(self.trading_days) * excess_returns.mean() / downside_vol
            if downside_vol > 0
            else 0
        )

        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

        # Value at Risk
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        # Tail metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Win rate
        win_rate = (returns > 0).mean()

        # Underwater analysis
        underwater_days = (drawdowns < 0).sum()
        avg_drawdown = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).sum() > 0 else 0

        return {
            # Returns
            "total_return": total_return,
            "annualized_return": annualized_return,
            "avg_daily_return": returns.mean(),
            # Risk
            "volatility": volatility,
            "downside_volatility": downside_vol,
            "max_drawdown": max_dd,
            "max_drawdown_date": trough_date,
            "avg_drawdown": avg_drawdown,
            # Risk-adjusted
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            # VaR
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            # Distribution
            "skewness": skewness,
            "kurtosis": kurtosis,
            "win_rate": win_rate,
            # Time analysis
            "underwater_days": underwater_days,
            "n_days": len(returns),
            "n_years": n_years,
        }

    def calculate_rolling_metrics(
        self,
        portfolio_values: pd.Series,
        window: int = 63,  # ~3 months
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Args:
            portfolio_values: Portfolio values
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        returns = portfolio_values.pct_change()

        rolling_return = returns.rolling(window).mean() * self.trading_days
        rolling_vol = returns.rolling(window).std() * np.sqrt(self.trading_days)
        rolling_sharpe = rolling_return / rolling_vol

        return pd.DataFrame({
            "return": rolling_return,
            "volatility": rolling_vol,
            "sharpe": rolling_sharpe,
        })

    def calculate_drawdown_stats(
        self,
        portfolio_values: pd.Series,
        threshold: float = -0.05,
    ) -> dict[str, Any]:
        """
        Calculate detailed drawdown statistics.

        Args:
            portfolio_values: Portfolio values
            threshold: Threshold for significant drawdown

        Returns:
            Drawdown statistics
        """
        drawdowns = calculate_rolling_drawdown(portfolio_values)

        # Find drawdown periods
        in_drawdown = drawdowns < threshold
        drawdown_periods = []

        start = None
        for i, (date, dd) in enumerate(drawdowns.items()):
            if dd < threshold and start is None:
                start = date
            elif dd >= 0 and start is not None:
                drawdown_periods.append({
                    "start": start,
                    "end": date,
                    "trough_date": drawdowns.loc[start:date].idxmin(),
                    "depth": drawdowns.loc[start:date].min(),
                    "duration": (date - start).days,
                })
                start = None

        return {
            "n_drawdowns": len(drawdown_periods),
            "avg_depth": np.mean([d["depth"] for d in drawdown_periods]) if drawdown_periods else 0,
            "avg_duration": np.mean([d["duration"] for d in drawdown_periods]) if drawdown_periods else 0,
            "max_duration": max([d["duration"] for d in drawdown_periods]) if drawdown_periods else 0,
            "periods": drawdown_periods,
            "time_underwater_pct": in_drawdown.mean() * 100,
        }

    def compare_strategies(
        self,
        strategy1: pd.Series,
        strategy2: pd.Series,
        strategy1_name: str = "Strategy",
        strategy2_name: str = "Benchmark",
    ) -> dict[str, Any]:
        """
        Compare two strategies.

        Args:
            strategy1: First strategy values
            strategy2: Second strategy values
            strategy1_name: Name for first strategy
            strategy2_name: Name for second strategy

        Returns:
            Comparison metrics
        """
        metrics1 = self.calculate_all(strategy1)
        metrics2 = self.calculate_all(strategy2)

        # Calculate information ratio
        returns1 = strategy1.pct_change()
        returns2 = strategy2.pct_change()
        active_returns = returns1 - returns2
        tracking_error = active_returns.std() * np.sqrt(self.trading_days)
        information_ratio = (
            active_returns.mean() * self.trading_days / tracking_error
            if tracking_error > 0
            else 0
        )

        # Statistical test
        t_stat, p_value = stats.ttest_ind(returns1.dropna(), returns2.dropna())

        return {
            strategy1_name: metrics1,
            strategy2_name: metrics2,
            "excess_return": metrics1["total_return"] - metrics2["total_return"],
            "excess_sharpe": metrics1["sharpe_ratio"] - metrics2["sharpe_ratio"],
            "drawdown_improvement": metrics2["max_drawdown"] - metrics1["max_drawdown"],
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_outperformance": p_value < 0.05 and metrics1["total_return"] > metrics2["total_return"],
        }

    def generate_report(
        self,
        metrics: dict[str, Any],
        name: str = "Portfolio",
    ) -> str:
        """Generate human-readable performance report."""
        report = [
            f"\n{'='*60}",
            f"Performance Report: {name}",
            f"{'='*60}",
            "",
            "Return Metrics:",
            f"  Total Return:      {metrics['total_return']*100:>10.2f}%",
            f"  Annualized Return: {metrics['annualized_return']*100:>10.2f}%",
            "",
            "Risk Metrics:",
            f"  Volatility:        {metrics['volatility']*100:>10.2f}%",
            f"  Max Drawdown:      {metrics['max_drawdown']*100:>10.2f}%",
            f"  VaR (95%):         {metrics['var_95']*100:>10.2f}%",
            "",
            "Risk-Adjusted Metrics:",
            f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>10.2f}",
            f"  Sortino Ratio:     {metrics['sortino_ratio']:>10.2f}",
            f"  Calmar Ratio:      {metrics['calmar_ratio']:>10.2f}",
            "",
            "Other Metrics:",
            f"  Win Rate:          {metrics['win_rate']*100:>10.2f}%",
            f"  Skewness:          {metrics['skewness']:>10.2f}",
            f"  Kurtosis:          {metrics['kurtosis']:>10.2f}",
            "",
            f"Period: {metrics['n_days']} days ({metrics['n_years']:.1f} years)",
            "=" * 60,
        ]

        return "\n".join(report)
