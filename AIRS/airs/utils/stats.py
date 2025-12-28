"""
Statistical utility functions for AIRS.

Common calculations for financial time series analysis.
"""

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


def calculate_returns(
    prices: pd.Series | pd.DataFrame,
    method: Literal["simple", "log"] = "simple",
) -> pd.Series | pd.DataFrame:
    """
    Calculate returns from price series.

    Args:
        prices: Price series or DataFrame
        method: Return calculation method ('simple' or 'log')

    Returns:
        Returns series/DataFrame
    """
    if method == "simple":
        return prices.pct_change()
    elif method == "log":
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """
    Calculate rolling volatility.

    Args:
        returns: Returns series
        window: Rolling window size
        annualize: Whether to annualize the volatility
        trading_days: Number of trading days per year

    Returns:
        Volatility series
    """
    vol = returns.rolling(window=window).std()

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).

    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()

    if downside_std == 0:
        return np.inf if excess_returns.mean() > 0 else -np.inf

    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def calculate_max_drawdown(prices: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and the period.

    Args:
        prices: Price series

    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
    """
    # Calculate running maximum
    running_max = prices.expanding().max()

    # Calculate drawdowns
    drawdowns = (prices - running_max) / running_max

    # Find maximum drawdown
    max_dd = drawdowns.min()

    # Find peak and trough dates
    trough_date = drawdowns.idxmin()
    peak_date = prices[:trough_date].idxmax()

    return max_dd, peak_date, trough_date


def calculate_rolling_drawdown(prices: pd.Series) -> pd.Series:
    """
    Calculate rolling drawdown from peak.

    Args:
        prices: Price series

    Returns:
        Drawdown series (negative values)
    """
    running_max = prices.expanding().max()
    return (prices - running_max) / running_max


def calculate_forward_drawdown(
    prices: pd.Series,
    horizon: int = 10,
) -> pd.Series:
    """
    Calculate forward-looking maximum drawdown.

    For each day, calculate the max drawdown that occurs
    in the next `horizon` days.

    Args:
        prices: Price series
        horizon: Number of days to look ahead

    Returns:
        Forward drawdown series
    """
    forward_dd = pd.Series(index=prices.index, dtype=float)

    for i in range(len(prices) - horizon):
        start_price = prices.iloc[i]
        future_prices = prices.iloc[i : i + horizon + 1]
        running_max = future_prices.expanding().max()
        drawdown = ((future_prices - running_max) / running_max).min()
        forward_dd.iloc[i] = drawdown

    return forward_dd


def calculate_zscore(
    series: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Calculate rolling z-score.

    Args:
        series: Input series
        window: Rolling window size

    Returns:
        Z-score series
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    return (series - rolling_mean) / rolling_std


def calculate_percentile_rank(
    series: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Calculate rolling percentile rank.

    Args:
        series: Input series
        window: Rolling window size

    Returns:
        Percentile rank series (0-100)
    """

    def percentile_rank(x: np.ndarray) -> float:
        if len(x) < 2:
            return 50.0
        return stats.percentileofscore(x[:-1], x[-1])

    return series.rolling(window=window).apply(percentile_rank, raw=True)


def calculate_rolling_correlation(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 21,
) -> pd.Series:
    """
    Calculate rolling correlation between two series.

    Args:
        series1: First series
        series2: Second series
        window: Rolling window size

    Returns:
        Correlation series
    """
    return series1.rolling(window=window).corr(series2)


def winsorize(
    series: pd.Series,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.Series:
    """
    Winsorize series at specified percentiles.

    Args:
        series: Input series
        lower_pct: Lower percentile (0-1)
        upper_pct: Upper percentile (0-1)

    Returns:
        Winsorized series
    """
    lower = series.quantile(lower_pct)
    upper = series.quantile(upper_pct)

    return series.clip(lower=lower, upper=upper)


def check_stationarity(
    series: pd.Series,
    significance_level: float = 0.05,
) -> dict:
    """
    Test for stationarity using Augmented Dickey-Fuller test.

    Args:
        series: Input series
        significance_level: Significance level for the test

    Returns:
        Dictionary with test results
    """
    from statsmodels.tsa.stattools import adfuller

    # Drop NaN values
    clean_series = series.dropna()

    if len(clean_series) < 20:
        return {
            "is_stationary": None,
            "p_value": None,
            "error": "Not enough data points",
        }

    try:
        result = adfuller(clean_series, autolag="AIC")
        return {
            "is_stationary": result[1] < significance_level,
            "p_value": result[1],
            "test_statistic": result[0],
            "critical_values": result[4],
        }
    except Exception as e:
        return {
            "is_stationary": None,
            "p_value": None,
            "error": str(e),
        }
