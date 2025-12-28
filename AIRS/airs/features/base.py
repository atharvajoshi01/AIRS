"""
Base class for feature generators.

Defines the interface and common functionality for all feature generators.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from airs.utils.logging import get_logger
from airs.utils.stats import calculate_zscore, calculate_percentile_rank

logger = get_logger(__name__)


class FeatureGenerator(ABC):
    """Abstract base class for feature generators."""

    def __init__(
        self,
        lookback_window: int = 252,
        min_periods: int = 21,
    ):
        """
        Initialize feature generator.

        Args:
            lookback_window: Default lookback window for rolling calculations
            min_periods: Minimum periods for rolling calculations
        """
        self.lookback_window = lookback_window
        self.min_periods = min_periods

    @property
    @abstractmethod
    def feature_group(self) -> str:
        """Return the feature group name."""
        pass

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return list of feature names this generator produces."""
        pass

    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from input data.

        Args:
            data: Input DataFrame with required columns

        Returns:
            DataFrame with generated features
        """
        pass

    def validate_input(self, data: pd.DataFrame, required_columns: list[str]) -> bool:
        """
        Validate that required columns are present.

        Args:
            data: Input DataFrame
            required_columns: List of required column names

        Returns:
            True if valid, raises ValueError otherwise
        """
        missing = set(required_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True

    def add_zscore(
        self,
        series: pd.Series,
        window: int | None = None,
        suffix: str = "_zscore",
    ) -> pd.Series:
        """
        Add z-score version of a series.

        Args:
            series: Input series
            window: Rolling window (defaults to lookback_window)
            suffix: Suffix for the new column name

        Returns:
            Z-score series
        """
        window = window or self.lookback_window
        zscore = calculate_zscore(series, window=window)
        zscore.name = f"{series.name}{suffix}"
        return zscore

    def add_percentile(
        self,
        series: pd.Series,
        window: int | None = None,
        suffix: str = "_pctl",
    ) -> pd.Series:
        """
        Add percentile rank version of a series.

        Args:
            series: Input series
            window: Rolling window (defaults to lookback_window)
            suffix: Suffix for the new column name

        Returns:
            Percentile rank series (0-100)
        """
        window = window or self.lookback_window
        pctl = calculate_percentile_rank(series, window=window)
        pctl.name = f"{series.name}{suffix}"
        return pctl

    def add_momentum(
        self,
        series: pd.Series,
        periods: list[int] = [5, 10, 21],
        suffix: str = "_mom",
    ) -> pd.DataFrame:
        """
        Add momentum (change) features for multiple periods.

        Args:
            series: Input series
            periods: List of periods to calculate momentum
            suffix: Suffix for new column names

        Returns:
            DataFrame with momentum columns
        """
        momentum_df = pd.DataFrame(index=series.index)

        for period in periods:
            col_name = f"{series.name}{suffix}_{period}d"
            momentum_df[col_name] = series.diff(period)

        return momentum_df

    def add_rolling_stats(
        self,
        series: pd.Series,
        windows: list[int] = [21, 63],
    ) -> pd.DataFrame:
        """
        Add rolling statistics (mean, std, min, max).

        Args:
            series: Input series
            windows: List of rolling window sizes

        Returns:
            DataFrame with rolling statistics
        """
        stats_df = pd.DataFrame(index=series.index)

        for window in windows:
            prefix = f"{series.name}_{window}d"
            stats_df[f"{prefix}_mean"] = series.rolling(
                window, min_periods=self.min_periods
            ).mean()
            stats_df[f"{prefix}_std"] = series.rolling(
                window, min_periods=self.min_periods
            ).std()
            stats_df[f"{prefix}_min"] = series.rolling(
                window, min_periods=self.min_periods
            ).min()
            stats_df[f"{prefix}_max"] = series.rolling(
                window, min_periods=self.min_periods
            ).max()

        return stats_df

    def prefix_columns(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add prefix to all column names."""
        df = df.copy()
        df.columns = [f"{prefix}_{col}" for col in df.columns]
        return df

    def log_features_generated(self, df: pd.DataFrame) -> None:
        """Log information about generated features."""
        non_null = df.notna().sum()
        logger.info(
            f"{self.feature_group}: Generated {len(df.columns)} features, "
            f"{len(df)} rows, {non_null.mean():.0f} avg non-null values"
        )
