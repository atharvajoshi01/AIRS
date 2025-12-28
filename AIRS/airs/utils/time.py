"""
Time and date utilities for AIRS.

Handles timezone conversions, trading calendars, and business day calculations.
"""

from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

# US market holidays (simplified - use pandas_market_calendars for production)
US_HOLIDAYS_2024 = [
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # MLK Day
    "2024-02-19",  # Presidents Day
    "2024-03-29",  # Good Friday
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas
]


def ensure_utc(dt: datetime | pd.Timestamp) -> pd.Timestamp:
    """
    Ensure a datetime is in UTC timezone.

    Args:
        dt: Datetime to convert

    Returns:
        Timezone-aware timestamp in UTC
    """
    if isinstance(dt, datetime):
        dt = pd.Timestamp(dt)

    if dt.tzinfo is None:
        return dt.tz_localize("UTC")
    return dt.tz_convert("UTC")


def get_trading_calendar(
    start_date: str | datetime,
    end_date: str | datetime,
    market: Literal["NYSE", "NASDAQ"] = "NYSE",
) -> pd.DatetimeIndex:
    """
    Get trading days between start and end date.

    Args:
        start_date: Start date
        end_date: End date
        market: Market calendar to use

    Returns:
        DatetimeIndex of trading days
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Generate all business days
    all_days = pd.bdate_range(start=start, end=end)

    # Remove holidays (simplified)
    holidays = pd.to_datetime(US_HOLIDAYS_2024)
    trading_days = all_days[~all_days.isin(holidays)]

    return trading_days


def is_trading_day(
    date: str | datetime,
    market: Literal["NYSE", "NASDAQ"] = "NYSE",
) -> bool:
    """
    Check if a date is a trading day.

    Args:
        date: Date to check
        market: Market calendar to use

    Returns:
        True if trading day, False otherwise
    """
    dt = pd.Timestamp(date)

    # Check if weekend
    if dt.dayofweek >= 5:
        return False

    # Check if holiday
    holidays = pd.to_datetime(US_HOLIDAYS_2024)
    if dt in holidays:
        return False

    return True


def to_business_days(calendar_days: int) -> int:
    """
    Convert calendar days to approximate business days.

    Args:
        calendar_days: Number of calendar days

    Returns:
        Approximate number of business days
    """
    # Rough approximation: 5 business days per 7 calendar days
    return int(calendar_days * 5 / 7)


def get_previous_trading_day(
    date: str | datetime,
    market: Literal["NYSE", "NASDAQ"] = "NYSE",
) -> pd.Timestamp:
    """
    Get the previous trading day.

    Args:
        date: Reference date
        market: Market calendar to use

    Returns:
        Previous trading day
    """
    dt = pd.Timestamp(date)

    # Go back one day
    prev_day = dt - timedelta(days=1)

    # Keep going back until we find a trading day
    while not is_trading_day(prev_day, market):
        prev_day -= timedelta(days=1)

    return prev_day


def get_next_trading_day(
    date: str | datetime,
    market: Literal["NYSE", "NASDAQ"] = "NYSE",
) -> pd.Timestamp:
    """
    Get the next trading day.

    Args:
        date: Reference date
        market: Market calendar to use

    Returns:
        Next trading day
    """
    dt = pd.Timestamp(date)

    # Go forward one day
    next_day = dt + timedelta(days=1)

    # Keep going forward until we find a trading day
    while not is_trading_day(next_day, market):
        next_day += timedelta(days=1)

    return next_day


def calculate_rolling_window_dates(
    end_date: str | datetime,
    window_days: int,
    calendar_days: bool = False,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Calculate start and end dates for a rolling window.

    Args:
        end_date: End date of the window
        window_days: Number of days in window
        calendar_days: If True, use calendar days; if False, use trading days

    Returns:
        Tuple of (start_date, end_date)
    """
    end = pd.Timestamp(end_date)

    if calendar_days:
        start = end - timedelta(days=window_days)
    else:
        # Go back trading days
        start = end
        days_back = 0
        while days_back < window_days:
            start -= timedelta(days=1)
            if is_trading_day(start):
                days_back += 1

    return start, end


def resample_to_frequency(
    df: pd.DataFrame,
    frequency: Literal["D", "W", "M", "Q"] = "D",
    method: Literal["last", "mean", "first"] = "last",
) -> pd.DataFrame:
    """
    Resample time series data to a different frequency.

    Args:
        df: DataFrame with DatetimeIndex
        frequency: Target frequency (D=daily, W=weekly, M=monthly, Q=quarterly)
        method: Resampling method

    Returns:
        Resampled DataFrame
    """
    if method == "last":
        return df.resample(frequency).last()
    elif method == "mean":
        return df.resample(frequency).mean()
    elif method == "first":
        return df.resample(frequency).first()
    else:
        raise ValueError(f"Unknown resampling method: {method}")
