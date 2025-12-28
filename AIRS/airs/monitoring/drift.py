"""
Drift detection for AIRS.

Monitors data and model drift to ensure prediction quality.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from airs.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection."""

    feature_name: str
    metric_name: str
    value: float
    threshold: float
    is_drifted: bool
    severity: str  # "none", "low", "medium", "high"
    computed_at: datetime


class FeatureDriftDetector:
    """
    Detect drift in feature distributions.

    Uses Population Stability Index (PSI) and statistical tests
    to detect distribution shifts.
    """

    # PSI thresholds
    PSI_THRESHOLDS = {
        "low": 0.1,
        "medium": 0.2,
        "high": 0.25,
    }

    def __init__(
        self,
        reference_window: int = 252,  # 1 year of trading days
        current_window: int = 21,  # 1 month
        n_bins: int = 10,
    ):
        """
        Initialize detector.

        Args:
            reference_window: Number of days for reference distribution
            current_window: Number of days for current distribution
            n_bins: Number of bins for PSI calculation
        """
        self.reference_window = reference_window
        self.current_window = current_window
        self.n_bins = n_bins

    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """
        Calculate Population Stability Index.

        Args:
            reference: Reference distribution
            current: Current distribution

        Returns:
            PSI value
        """
        # Create bins from reference distribution
        _, bin_edges = np.histogram(reference, bins=self.n_bins)

        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        # Add small epsilon to avoid division by zero
        eps = 1e-6
        ref_props = (ref_counts + eps) / (len(reference) + eps * self.n_bins)
        curr_props = (curr_counts + eps) / (len(current) + eps * self.n_bins)

        # Calculate PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))

        return float(psi)

    def detect_drift(
        self,
        data: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> list[DriftResult]:
        """
        Detect drift for all features.

        Args:
            data: DataFrame with feature columns (indexed by date)
            feature_names: Features to check (default: all columns)

        Returns:
            List of drift results
        """
        if feature_names is None:
            feature_names = list(data.columns)

        # Split into reference and current windows
        reference_data = data.iloc[-self.reference_window - self.current_window:-self.current_window]
        current_data = data.iloc[-self.current_window:]

        results = []
        computed_at = datetime.utcnow()

        for feature in feature_names:
            if feature not in data.columns:
                continue

            reference = reference_data[feature].dropna().values
            current = current_data[feature].dropna().values

            if len(reference) < 10 or len(current) < 5:
                continue

            # Calculate PSI
            psi = self.calculate_psi(reference, current)

            # Determine severity
            if psi < self.PSI_THRESHOLDS["low"]:
                severity = "none"
                is_drifted = False
            elif psi < self.PSI_THRESHOLDS["medium"]:
                severity = "low"
                is_drifted = True
            elif psi < self.PSI_THRESHOLDS["high"]:
                severity = "medium"
                is_drifted = True
            else:
                severity = "high"
                is_drifted = True

            results.append(
                DriftResult(
                    feature_name=feature,
                    metric_name="psi",
                    value=psi,
                    threshold=self.PSI_THRESHOLDS["low"],
                    is_drifted=is_drifted,
                    severity=severity,
                    computed_at=computed_at,
                )
            )

        return results

    def get_summary(self, results: list[DriftResult]) -> dict[str, Any]:
        """Summarize drift detection results."""
        drifted = [r for r in results if r.is_drifted]
        high_severity = [r for r in results if r.severity == "high"]

        return {
            "total_features": len(results),
            "drifted_features": len(drifted),
            "high_severity": len(high_severity),
            "drift_rate": len(drifted) / len(results) if results else 0,
            "avg_psi": np.mean([r.value for r in results]) if results else 0,
            "max_psi": max([r.value for r in results]) if results else 0,
            "drifted_names": [r.feature_name for r in drifted],
        }


class PredictionDriftDetector:
    """
    Detect drift in model predictions.

    Monitors prediction distribution and calibration.
    """

    def __init__(
        self,
        reference_window: int = 252,
        current_window: int = 21,
    ):
        """
        Initialize detector.

        Args:
            reference_window: Days for reference distribution
            current_window: Days for current distribution
        """
        self.reference_window = reference_window
        self.current_window = current_window

    def detect_drift(
        self,
        predictions: pd.Series,
    ) -> dict[str, Any]:
        """
        Detect drift in predictions.

        Args:
            predictions: Series of predictions indexed by date

        Returns:
            Drift analysis results
        """
        # Split into windows
        reference = predictions.iloc[-self.reference_window - self.current_window:-self.current_window]
        current = predictions.iloc[-self.current_window:]

        if len(reference) < 10 or len(current) < 5:
            return {"error": "Insufficient data for drift detection"}

        # Calculate statistics
        ref_mean = reference.mean()
        ref_std = reference.std()
        curr_mean = current.mean()
        curr_std = current.std()

        # Mean shift
        mean_shift = curr_mean - ref_mean
        mean_shift_zscore = mean_shift / ref_std if ref_std > 0 else 0

        # Variance change
        std_change = (curr_std - ref_std) / ref_std if ref_std > 0 else 0

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(reference.dropna(), current.dropna())

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(reference.dropna(), current.dropna())

        # Determine if drifted
        is_drifted = (
            abs(mean_shift_zscore) > 2.0 or
            abs(std_change) > 0.3 or
            ks_pvalue < 0.01
        )

        return {
            "is_drifted": is_drifted,
            "reference_mean": float(ref_mean),
            "reference_std": float(ref_std),
            "current_mean": float(curr_mean),
            "current_std": float(curr_std),
            "mean_shift": float(mean_shift),
            "mean_shift_zscore": float(mean_shift_zscore),
            "std_change_pct": float(std_change * 100),
            "t_statistic": float(t_stat),
            "t_pvalue": float(p_value),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "computed_at": datetime.utcnow().isoformat(),
        }


class ModelPerformanceMonitor:
    """
    Monitor model performance over time.

    Tracks precision, recall, and other metrics.
    """

    def __init__(
        self,
        baseline_precision: float = 0.70,
        baseline_recall: float = 0.60,
        degradation_threshold: float = 0.10,
    ):
        """
        Initialize monitor.

        Args:
            baseline_precision: Expected baseline precision
            baseline_recall: Expected baseline recall
            degradation_threshold: Alert threshold for degradation
        """
        self.baseline_precision = baseline_precision
        self.baseline_recall = baseline_recall
        self.degradation_threshold = degradation_threshold

    def calculate_metrics(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            predictions: Predicted probabilities
            actuals: Actual binary outcomes
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        # Align indices
        common_idx = predictions.index.intersection(actuals.index)
        preds = predictions.loc[common_idx]
        acts = actuals.loc[common_idx]

        # Binary predictions
        pred_binary = (preds >= threshold).astype(int)

        # Calculate metrics
        tp = ((pred_binary == 1) & (acts == 1)).sum()
        fp = ((pred_binary == 1) & (acts == 0)).sum()
        fn = ((pred_binary == 0) & (acts == 1)).sum()
        tn = ((pred_binary == 0) & (acts == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(acts) if len(acts) > 0 else 0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(accuracy),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "n_samples": len(acts),
        }

    def check_performance(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
        lookback_days: int = 90,
    ) -> dict[str, Any]:
        """
        Check if model performance has degraded.

        Args:
            predictions: Predicted probabilities
            actuals: Actual outcomes
            lookback_days: Number of days to evaluate

        Returns:
            Performance analysis
        """
        # Filter to recent data
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_preds = predictions[predictions.index >= cutoff]
        recent_acts = actuals[actuals.index >= cutoff]

        if len(recent_preds) < 10:
            return {"error": "Insufficient recent data for performance check"}

        # Calculate current metrics
        metrics = self.calculate_metrics(recent_preds, recent_acts)

        # Check for degradation
        precision_degradation = (
            (self.baseline_precision - metrics["precision"]) / self.baseline_precision
        )
        recall_degradation = (
            (self.baseline_recall - metrics["recall"]) / self.baseline_recall
        )

        is_degraded = (
            precision_degradation > self.degradation_threshold or
            recall_degradation > self.degradation_threshold
        )

        return {
            "is_degraded": is_degraded,
            "current_metrics": metrics,
            "baseline_precision": self.baseline_precision,
            "baseline_recall": self.baseline_recall,
            "precision_degradation_pct": float(precision_degradation * 100),
            "recall_degradation_pct": float(recall_degradation * 100),
            "lookback_days": lookback_days,
            "computed_at": datetime.utcnow().isoformat(),
        }


class DriftMonitor:
    """
    Combined drift monitoring for features, predictions, and performance.
    """

    def __init__(self):
        """Initialize all drift detectors."""
        self.feature_detector = FeatureDriftDetector()
        self.prediction_detector = PredictionDriftDetector()
        self.performance_monitor = ModelPerformanceMonitor()

    def run_full_check(
        self,
        features: pd.DataFrame,
        predictions: pd.Series,
        actuals: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Run comprehensive drift check.

        Args:
            features: Feature DataFrame
            predictions: Prediction Series
            actuals: Actual outcomes (optional)

        Returns:
            Complete drift analysis
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Feature drift
        feature_results = self.feature_detector.detect_drift(features)
        results["feature_drift"] = self.feature_detector.get_summary(feature_results)

        # Prediction drift
        results["prediction_drift"] = self.prediction_detector.detect_drift(predictions)

        # Performance check (if actuals available)
        if actuals is not None:
            results["performance"] = self.performance_monitor.check_performance(
                predictions, actuals
            )

        # Overall status
        has_feature_drift = results["feature_drift"]["high_severity"] > 0
        has_prediction_drift = results["prediction_drift"].get("is_drifted", False)
        has_performance_issue = results.get("performance", {}).get("is_degraded", False)

        if has_feature_drift or has_prediction_drift or has_performance_issue:
            results["status"] = "alert"
            results["issues"] = []
            if has_feature_drift:
                results["issues"].append("High severity feature drift detected")
            if has_prediction_drift:
                results["issues"].append("Prediction distribution drift detected")
            if has_performance_issue:
                results["issues"].append("Model performance degradation detected")
        else:
            results["status"] = "healthy"
            results["issues"] = []

        return results
