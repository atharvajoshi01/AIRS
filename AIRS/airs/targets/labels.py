"""
Label generation for drawdown prediction.

Creates target variables for supervised learning.
"""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from airs.targets.drawdown import DrawdownCalculator
from airs.utils.logging import get_logger

logger = get_logger(__name__)


class LabelGenerator:
    """
    Target label generator for drawdown prediction.

    Generates:
    - Binary classification labels
    - Multi-class labels (severity)
    - Regression targets (drawdown magnitude)
    """

    def __init__(
        self,
        threshold: float = -0.05,
        horizon: int = 15,
        portfolio_weights: dict[str, float] | None = None,
    ):
        """
        Initialize label generator.

        Args:
            threshold: Drawdown threshold for binary classification (negative)
            horizon: Forward-looking horizon in trading days
            portfolio_weights: Portfolio weights for drawdown calculation
        """
        self.threshold = threshold
        self.horizon = horizon
        self.calculator = DrawdownCalculator(weights=portfolio_weights)

    def generate_binary_labels(
        self,
        prices: pd.DataFrame | pd.Series,
        threshold: float | None = None,
        horizon: int | None = None,
    ) -> pd.Series:
        """
        Generate binary labels for drawdown prediction.

        Y = 1 if forward drawdown exceeds threshold
        Y = 0 otherwise

        Args:
            prices: Asset prices or portfolio value
            threshold: Override default threshold
            horizon: Override default horizon

        Returns:
            Binary label series
        """
        threshold = threshold or self.threshold
        horizon = horizon or self.horizon

        # Calculate forward drawdown
        if isinstance(prices, pd.DataFrame):
            portfolio_value = self.calculator.calculate_portfolio_value(prices)
        else:
            portfolio_value = prices

        forward_dd = self.calculator.calculate_forward_drawdown(
            portfolio_value, horizon=horizon
        )

        # Create binary labels
        labels = (forward_dd <= threshold).astype(int)
        labels.name = f"drawdown_{abs(int(threshold*100))}pct_{horizon}d"

        # Log class balance
        positive_rate = labels.mean()
        logger.info(
            f"Generated labels: {len(labels)} samples, "
            f"{positive_rate*100:.1f}% positive (threshold={threshold*100}%, horizon={horizon}d)"
        )

        return labels

    def generate_multiclass_labels(
        self,
        prices: pd.DataFrame | pd.Series,
        thresholds: list[float] = [-0.05, -0.075, -0.10],
        horizon: int | None = None,
    ) -> pd.Series:
        """
        Generate multi-class labels based on drawdown severity.

        Classes:
        0 = No significant drawdown
        1 = Mild drawdown (5-7.5%)
        2 = Moderate drawdown (7.5-10%)
        3 = Severe drawdown (>10%)

        Args:
            prices: Asset prices or portfolio value
            thresholds: List of thresholds for classification
            horizon: Forward-looking horizon

        Returns:
            Multi-class label series
        """
        horizon = horizon or self.horizon

        if isinstance(prices, pd.DataFrame):
            portfolio_value = self.calculator.calculate_portfolio_value(prices)
        else:
            portfolio_value = prices

        forward_dd = self.calculator.calculate_forward_drawdown(
            portfolio_value, horizon=horizon
        )

        # Create multi-class labels
        labels = pd.Series(0, index=forward_dd.index, name="drawdown_severity")

        # Sort thresholds from least severe to most severe
        sorted_thresholds = sorted(thresholds, reverse=True)

        for i, thresh in enumerate(sorted_thresholds, start=1):
            labels[forward_dd <= thresh] = i

        # Log class distribution
        class_dist = labels.value_counts().sort_index()
        logger.info(f"Multi-class distribution:\n{class_dist}")

        return labels

    def generate_regression_target(
        self,
        prices: pd.DataFrame | pd.Series,
        horizon: int | None = None,
        clip_percentile: float = 0.01,
    ) -> pd.Series:
        """
        Generate regression target (actual drawdown magnitude).

        Args:
            prices: Asset prices or portfolio value
            horizon: Forward-looking horizon
            clip_percentile: Percentile for winsorizing

        Returns:
            Regression target series (drawdown magnitude)
        """
        horizon = horizon or self.horizon

        if isinstance(prices, pd.DataFrame):
            portfolio_value = self.calculator.calculate_portfolio_value(prices)
        else:
            portfolio_value = prices

        forward_dd = self.calculator.calculate_forward_drawdown(
            portfolio_value, horizon=horizon
        )

        # Convert to positive magnitude for easier interpretation
        target = forward_dd.abs()
        target.name = f"drawdown_magnitude_{horizon}d"

        # Winsorize extremes
        lower = target.quantile(clip_percentile)
        upper = target.quantile(1 - clip_percentile)
        target = target.clip(lower=lower, upper=upper)

        logger.info(
            f"Regression target: mean={target.mean()*100:.2f}%, "
            f"std={target.std()*100:.2f}%, max={target.max()*100:.2f}%"
        )

        return target

    def generate_multiple_horizons(
        self,
        prices: pd.DataFrame | pd.Series,
        horizons: list[int] = [10, 15, 20],
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """
        Generate labels for multiple forecast horizons.

        Args:
            prices: Asset prices
            horizons: List of horizons
            threshold: Drawdown threshold

        Returns:
            DataFrame with labels for each horizon
        """
        threshold = threshold or self.threshold

        labels_df = pd.DataFrame(index=prices.index)

        for horizon in horizons:
            labels = self.generate_binary_labels(prices, threshold, horizon)
            labels_df[f"y_{horizon}d"] = labels

        return labels_df

    def generate_multiple_thresholds(
        self,
        prices: pd.DataFrame | pd.Series,
        thresholds: list[float] = [-0.05, -0.075, -0.10],
        horizon: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate labels for multiple drawdown thresholds.

        Args:
            prices: Asset prices
            thresholds: List of thresholds
            horizon: Forecast horizon

        Returns:
            DataFrame with labels for each threshold
        """
        horizon = horizon or self.horizon

        labels_df = pd.DataFrame(index=prices.index)

        for threshold in thresholds:
            labels = self.generate_binary_labels(prices, threshold, horizon)
            pct = abs(int(threshold * 100))
            labels_df[f"y_{pct}pct"] = labels

        return labels_df

    def get_class_weights(
        self,
        labels: pd.Series,
        method: Literal["balanced", "custom"] = "balanced",
        custom_weights: dict[int, float] | None = None,
    ) -> dict[int, float]:
        """
        Calculate class weights for imbalanced classification.

        Args:
            labels: Label series
            method: Weight calculation method
            custom_weights: Custom weights if method='custom'

        Returns:
            Dictionary of class weights
        """
        if method == "custom" and custom_weights:
            return custom_weights

        classes = np.unique(labels.dropna())
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=labels.dropna(),
        )

        return dict(zip(classes, weights))

    def analyze_class_balance(
        self,
        labels: pd.Series,
    ) -> dict:
        """
        Analyze class balance and suggest strategies.

        Args:
            labels: Label series

        Returns:
            Analysis dictionary
        """
        clean_labels = labels.dropna()
        value_counts = clean_labels.value_counts().sort_index()
        class_pcts = value_counts / len(clean_labels) * 100

        # Calculate imbalance ratio
        if len(value_counts) == 2:
            imbalance_ratio = value_counts.max() / value_counts.min()
        else:
            imbalance_ratio = value_counts.max() / value_counts.min()

        # Suggest strategies
        strategies = []
        if imbalance_ratio > 10:
            strategies.extend([
                "Use class weights in training",
                "Consider SMOTE or other oversampling",
                "Use precision-recall curve instead of ROC",
                "Consider anomaly detection framing",
            ])
        elif imbalance_ratio > 5:
            strategies.extend([
                "Use class weights in training",
                "Monitor both precision and recall",
            ])

        return {
            "class_counts": value_counts.to_dict(),
            "class_percentages": class_pcts.to_dict(),
            "imbalance_ratio": imbalance_ratio,
            "total_samples": len(clean_labels),
            "missing_samples": labels.isna().sum(),
            "recommended_strategies": strategies,
        }


class TimeSeriesLabelSplitter:
    """
    Handles train/validation/test splits for time series labels.

    Ensures no data leakage and proper temporal ordering.
    """

    def __init__(
        self,
        train_end: str = "2017-12-31",
        val_end: str = "2018-12-31",
        embargo_days: int = 5,
    ):
        """
        Initialize splitter.

        Args:
            train_end: End date for training period
            val_end: End date for validation period
            embargo_days: Gap between train/val and val/test
        """
        self.train_end = pd.Timestamp(train_end)
        self.val_end = pd.Timestamp(val_end)
        self.embargo_days = embargo_days

    def split(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> tuple[
        tuple[pd.DataFrame, pd.Series],
        tuple[pd.DataFrame, pd.Series],
        tuple[pd.DataFrame, pd.Series],
    ]:
        """
        Split features and labels into train/val/test.

        Args:
            features: Feature DataFrame
            labels: Label Series

        Returns:
            Tuple of (train, val, test) where each is (X, y) tuple
        """
        # Align features and labels
        common_idx = features.index.intersection(labels.index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]

        # Calculate embargo dates
        train_embargo = self.train_end + pd.Timedelta(days=self.embargo_days)
        val_embargo = self.val_end + pd.Timedelta(days=self.embargo_days)

        # Split
        train_mask = features.index <= self.train_end
        val_mask = (features.index > train_embargo) & (features.index <= self.val_end)
        test_mask = features.index > val_embargo

        X_train = features[train_mask]
        y_train = labels[train_mask]

        X_val = features[val_mask]
        y_val = labels[val_mask]

        X_test = features[test_mask]
        y_test = labels[test_mask]

        logger.info(
            f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def walk_forward_split(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        initial_train_years: int = 5,
        test_months: int = 12,
        step_months: int = 3,
    ) -> list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """
        Create walk-forward validation splits.

        Args:
            features: Feature DataFrame
            labels: Label Series
            initial_train_years: Initial training period in years
            test_months: Test period in months
            step_months: Step size in months

        Returns:
            List of (X_train, y_train, X_test, y_test) tuples
        """
        # Align
        common_idx = features.index.intersection(labels.index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]

        splits = []
        start_date = features.index.min()
        end_date = features.index.max()

        train_end = start_date + pd.DateOffset(years=initial_train_years)

        while train_end < end_date:
            test_end = train_end + pd.DateOffset(months=test_months)
            test_start = train_end + pd.Timedelta(days=self.embargo_days)

            if test_end > end_date:
                test_end = end_date

            # Get splits
            train_mask = features.index <= train_end
            test_mask = (features.index > test_start) & (features.index <= test_end)

            X_train = features[train_mask]
            y_train = labels[train_mask]
            X_test = features[test_mask]
            y_test = labels[test_mask]

            if len(X_test) > 0:
                splits.append((X_train, y_train, X_test, y_test))

            # Step forward
            train_end = train_end + pd.DateOffset(months=step_months)

        logger.info(f"Created {len(splits)} walk-forward splits")

        return splits
