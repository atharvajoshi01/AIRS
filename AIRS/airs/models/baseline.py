"""
Baseline models for AIRS.

Simple models for establishing performance baselines.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from airs.models.base import BaseModel
from airs.utils.logging import get_logger

logger = get_logger(__name__)


class ThresholdModel(BaseModel):
    """
    Simple threshold-based model.

    Uses simple rules based on VIX, credit spreads, etc.
    """

    def __init__(
        self,
        vix_threshold: float = 25.0,
        hy_spread_threshold: float = 5.0,  # 500 bps
        curve_inversion_threshold: int = 5,  # days inverted
        name: str = "threshold_model",
        version: str = "v1",
    ):
        """
        Initialize threshold model.

        Args:
            vix_threshold: VIX level threshold
            hy_spread_threshold: HY spread threshold (percentage points)
            curve_inversion_threshold: Days of curve inversion
            name: Model name
            version: Model version
        """
        super().__init__(name=name, version=version)
        self.vix_threshold = vix_threshold
        self.hy_spread_threshold = hy_spread_threshold
        self.curve_inversion_threshold = curve_inversion_threshold

        # Feature names to look for
        self.vix_cols = ["vix_level", "volatility_vix_level", "VIXCLS"]
        self.hy_cols = ["hy_spread", "credit_hy_spread", "BAMLH0A0HYM2"]
        self.curve_cols = ["yc_inversion_duration", "rates_yc_inversion_duration"]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> "ThresholdModel":
        """
        Fit is a no-op for threshold model (rules are predefined).
        """
        self.feature_names = list(X.columns)
        self.is_fitted = True

        # Find actual column names
        self._vix_col = next((c for c in self.vix_cols if c in X.columns), None)
        self._hy_col = next((c for c in self.hy_cols if c in X.columns), None)
        self._curve_col = next((c for c in self.curve_cols if c in X.columns), None)

        logger.info(
            f"Threshold model using: VIX={self._vix_col}, "
            f"HY={self._hy_col}, Curve={self._curve_col}"
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions based on thresholds."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate probability based on threshold rules.

        Returns probability proportional to how many rules are triggered
        and by how much.
        """
        X = self._validate_input(X)
        n_samples = len(X)

        # Initialize scores
        scores = np.zeros(n_samples)
        max_score = 0

        # VIX rule
        if self._vix_col and self._vix_col in X.columns:
            vix = X[self._vix_col].values
            # Score based on how much VIX exceeds threshold
            vix_score = np.clip((vix - self.vix_threshold) / 10, 0, 1)
            scores += vix_score * 0.4
            max_score += 0.4

        # HY spread rule
        if self._hy_col and self._hy_col in X.columns:
            hy = X[self._hy_col].values
            hy_score = np.clip((hy - self.hy_spread_threshold) / 3, 0, 1)
            scores += hy_score * 0.35
            max_score += 0.35

        # Curve inversion rule
        if self._curve_col and self._curve_col in X.columns:
            curve = X[self._curve_col].values
            curve_score = np.clip(curve / 20, 0, 1)  # Max at 20 days
            scores += curve_score * 0.25
            max_score += 0.25

        # Normalize scores to probabilities
        if max_score > 0:
            probabilities = scores / max_score
        else:
            probabilities = np.full(n_samples, 0.5)

        # Return as 2D array [prob_class_0, prob_class_1]
        return np.column_stack([1 - probabilities, probabilities])


class LogisticModel(BaseModel):
    """
    Logistic regression model with regularization.

    Provides interpretable baseline with feature coefficients.
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        class_weight: str | dict = "balanced",
        max_iter: int = 1000,
        name: str = "logistic_model",
        version: str = "v1",
        random_state: int = 42,
    ):
        """
        Initialize logistic regression model.

        Args:
            C: Regularization strength (smaller = stronger)
            penalty: Regularization type ('l1', 'l2', 'elasticnet')
            class_weight: Class weighting strategy
            max_iter: Maximum iterations
            name: Model name
            version: Model version
            random_state: Random seed
        """
        super().__init__(name=name, version=version, random_state=random_state)
        self.C = C
        self.penalty = penalty
        self.class_weight = class_weight
        self.max_iter = max_iter

        # Solver selection based on penalty
        if penalty == "l1":
            solver = "saga"
        elif penalty == "elasticnet":
            solver = "saga"
        else:
            solver = "lbfgs"

        self._model = LogisticRegression(
            C=C,
            penalty=penalty,
            class_weight=class_weight,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
        )
        self._scaler = StandardScaler()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> "LogisticModel":
        """
        Fit logistic regression model.

        Args:
            X: Feature matrix
            y: Target labels
        """
        self.feature_names = list(X.columns)

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Fit model
        self._model.fit(X_scaled, y)
        self.is_fitted = True

        logger.info(
            f"Fitted logistic model with {len(self.feature_names)} features, "
            f"C={self.C}, penalty={self.penalty}"
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X = self._validate_input(X)
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        X = self._validate_input(X)
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)

    def feature_importance(self) -> pd.Series:
        """
        Get feature coefficients as importance.

        Returns:
            Series of absolute coefficient values
        """
        if not self.is_fitted:
            return None

        coefficients = self._model.coef_[0]
        importance = pd.Series(
            np.abs(coefficients),
            index=self.feature_names,
            name="importance",
        )

        return importance.sort_values(ascending=False)

    def get_coefficients(self) -> pd.Series:
        """
        Get actual coefficients (with sign).

        Returns:
            Series of coefficients
        """
        if not self.is_fitted:
            return None

        return pd.Series(
            self._model.coef_[0],
            index=self.feature_names,
            name="coefficient",
        ).sort_values(ascending=False)


class MovingAverageModel(BaseModel):
    """
    Simple moving average model for stress score.

    Uses rolling averages of stress indicators.
    """

    def __init__(
        self,
        window: int = 21,
        threshold: float = 0.6,
        name: str = "ma_model",
        version: str = "v1",
    ):
        """
        Initialize moving average model.

        Args:
            window: Rolling window size
            threshold: Threshold for positive prediction
            name: Model name
            version: Model version
        """
        super().__init__(name=name, version=version)
        self.window = window
        self.threshold = threshold

        # Features to use for stress score
        self.stress_features = [
            "composite_stress_index",
            "composite_early_warning_score",
        ]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> "MovingAverageModel":
        """Fit is a no-op for MA model."""
        self.feature_names = list(X.columns)
        self.is_fitted = True

        # Find available stress features
        self._stress_cols = [
            c for c in self.stress_features if c in X.columns
        ]

        if not self._stress_cols:
            # Fallback to any feature with 'stress' in name
            self._stress_cols = [c for c in X.columns if "stress" in c.lower()]

        logger.info(f"MA model using features: {self._stress_cols}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate probability from rolling stress average."""
        X = self._validate_input(X)

        if self._stress_cols:
            # Average of stress features
            stress_avg = X[self._stress_cols].mean(axis=1)
            # Normalize to 0-1
            probabilities = (stress_avg / 100).clip(0, 1).values
        else:
            probabilities = np.full(len(X), 0.5)

        return np.column_stack([1 - probabilities, probabilities])
