"""
Model validation and evaluation for AIRS.

Walk-forward validation and comprehensive metrics.
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    classification_report,
)

from airs.models.base import BaseModel
from airs.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: np.ndarray
    optimal_threshold: float
    precision_at_thresholds: dict[float, float]
    lead_time_analysis: dict | None = None


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.

    Implements proper temporal cross-validation without lookahead bias.
    """

    def __init__(
        self,
        initial_train_size: int = 1260,  # ~5 years of trading days
        test_size: int = 63,  # ~3 months
        step_size: int = 21,  # ~1 month
        embargo_size: int = 5,  # 5 day gap
        expanding_window: bool = True,
    ):
        """
        Initialize walk-forward validator.

        Args:
            initial_train_size: Initial training set size in days
            test_size: Test set size in days
            step_size: Step size for rolling forward
            embargo_size: Gap between train and test
            expanding_window: If True, expand training window; else slide
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.embargo_size = embargo_size
        self.expanding_window = expanding_window

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for walk-forward validation.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        splits = []

        train_end = self.initial_train_size

        while train_end + self.embargo_size + self.test_size <= n_samples:
            if self.expanding_window:
                train_start = 0
            else:
                train_start = max(0, train_end - self.initial_train_size)

            test_start = train_end + self.embargo_size
            test_end = min(test_start + self.test_size, n_samples)

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            splits.append((train_indices, test_indices))

            train_end += self.step_size

        logger.info(f"Created {len(splits)} walk-forward splits")

        return splits

    def validate(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        metrics_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """
        Run walk-forward validation.

        Args:
            model: Model to validate
            X: Feature matrix
            y: Labels
            metrics_fn: Custom metrics function

        Returns:
            Dictionary with validation results
        """
        splits = self.split(X, y)

        all_predictions = []
        all_probabilities = []
        all_actuals = []
        all_dates = []
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # Fit model on training fold
            params = model.get_params()
            # Update random_state to vary by fold
            params["random_state"] = model.random_state + fold_idx
            model_clone = model.__class__(**params)
            model_clone.fit(X_train, y_train)

            # Get predictions
            predictions = model_clone.predict(X_test)
            probabilities = model_clone.predict_proba(X_test)[:, 1]

            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_actuals.extend(y_test.values)
            all_dates.extend(X_test.index.tolist())

            # Compute fold metrics
            if metrics_fn:
                fold_metric = metrics_fn(y_test, predictions, probabilities)
            else:
                fold_metric = {
                    "precision": precision_score(y_test, predictions, zero_division=0),
                    "recall": recall_score(y_test, predictions, zero_division=0),
                    "f1": f1_score(y_test, predictions, zero_division=0),
                }

            fold_metrics.append(fold_metric)

        # Aggregate results
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_actuals = np.array(all_actuals)

        # Overall metrics
        evaluator = ModelEvaluator()
        overall_metrics = evaluator.evaluate(
            all_actuals, all_predictions, all_probabilities
        )

        return {
            "predictions": pd.Series(all_predictions, index=all_dates),
            "probabilities": pd.Series(all_probabilities, index=all_dates),
            "actuals": pd.Series(all_actuals, index=all_dates),
            "fold_metrics": fold_metrics,
            "overall_metrics": overall_metrics,
            "n_folds": len(splits),
        }


class ModelEvaluator:
    """
    Comprehensive model evaluation with focus on drawdown prediction metrics.
    """

    def __init__(
        self,
        target_precision: float = 0.7,
        target_recall: float = 0.5,
    ):
        """
        Initialize evaluator.

        Args:
            target_precision: Target precision for threshold optimization
            target_recall: Target recall for threshold optimization
        """
        self.target_precision = target_precision
        self.target_recall = target_recall

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> EvaluationMetrics:
        """
        Evaluate model predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities

        Returns:
            EvaluationMetrics object
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # AUC metrics
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            roc_auc = 0.5

        try:
            pr_auc = average_precision_score(y_true, y_proba)
        except ValueError:
            pr_auc = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Find optimal threshold
        optimal_threshold = self._find_optimal_threshold(y_true, y_proba)

        # Precision at various thresholds
        precision_at_thresholds = self._precision_at_thresholds(
            y_true, y_proba, [0.3, 0.5, 0.7, 0.9]
        )

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            confusion_matrix=cm,
            optimal_threshold=optimal_threshold,
            precision_at_thresholds=precision_at_thresholds,
        )

    def _find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = "f1",
    ) -> float:
        """Find optimal probability threshold."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        if metric == "f1":
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
        elif metric == "precision":
            # Find highest threshold that meets recall target
            valid_idx = np.where(recall >= self.target_recall)[0]
            if len(valid_idx) > 0:
                best_idx = valid_idx[np.argmax(precision[valid_idx])]
            else:
                best_idx = 0
        else:
            best_idx = len(thresholds) // 2

        if best_idx < len(thresholds):
            return thresholds[best_idx]
        return 0.5

    def _precision_at_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: list[float],
    ) -> dict[float, float]:
        """Calculate precision at specific thresholds."""
        results = {}

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            results[thresh] = precision_score(y_true, y_pred, zero_division=0)

        return results

    def calculate_lead_time(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
        horizon: int = 15,
    ) -> dict:
        """
        Calculate lead time statistics for true positives.

        Args:
            predictions: Predicted labels with datetime index
            actuals: Actual labels with datetime index
            horizon: Prediction horizon in days

        Returns:
            Lead time analysis dictionary
        """
        # Find true positive predictions
        tp_mask = (predictions == 1) & (actuals == 1)
        tp_dates = predictions[tp_mask].index

        lead_times = []

        for date in tp_dates:
            # Find when the drawdown actually started
            future_window = actuals.loc[date:][:horizon]
            if future_window.sum() > 0:
                # First positive label in window
                drawdown_start = future_window[future_window == 1].index[0]
                lead_time = (drawdown_start - date).days
                lead_times.append(lead_time)

        if not lead_times:
            return {
                "avg_lead_time": None,
                "median_lead_time": None,
                "min_lead_time": None,
                "max_lead_time": None,
            }

        return {
            "avg_lead_time": np.mean(lead_times),
            "median_lead_time": np.median(lead_times),
            "min_lead_time": np.min(lead_times),
            "max_lead_time": np.max(lead_times),
            "lead_times": lead_times,
        }

    def generate_report(
        self,
        metrics: EvaluationMetrics,
        model_name: str = "Model",
    ) -> str:
        """Generate human-readable evaluation report."""
        report = [
            f"\n{'='*60}",
            f"Model Evaluation Report: {model_name}",
            f"{'='*60}",
            "",
            "Classification Metrics:",
            f"  Accuracy:  {metrics.accuracy:.4f}",
            f"  Precision: {metrics.precision:.4f}",
            f"  Recall:    {metrics.recall:.4f}",
            f"  F1 Score:  {metrics.f1:.4f}",
            "",
            "AUC Metrics:",
            f"  ROC-AUC: {metrics.roc_auc:.4f}",
            f"  PR-AUC:  {metrics.pr_auc:.4f}",
            "",
            f"Optimal Threshold: {metrics.optimal_threshold:.3f}",
            "",
            "Precision at Thresholds:",
        ]

        for thresh, prec in metrics.precision_at_thresholds.items():
            report.append(f"  {thresh:.1f}: {prec:.4f}")

        report.extend([
            "",
            "Confusion Matrix:",
            f"  TN: {metrics.confusion_matrix[0,0]:5d}  FP: {metrics.confusion_matrix[0,1]:5d}",
            f"  FN: {metrics.confusion_matrix[1,0]:5d}  TP: {metrics.confusion_matrix[1,1]:5d}",
            "",
        ])

        if metrics.lead_time_analysis:
            report.extend([
                "Lead Time Analysis:",
                f"  Average: {metrics.lead_time_analysis['avg_lead_time']:.1f} days",
                f"  Median:  {metrics.lead_time_analysis['median_lead_time']:.1f} days",
                "",
            ])

        report.append("=" * 60)

        return "\n".join(report)
