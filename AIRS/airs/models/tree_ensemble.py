"""
Tree ensemble models for AIRS.

XGBoost, LightGBM, and Random Forest implementations.
"""

from typing import Any

import numpy as np
import pandas as pd

from airs.models.base import BaseModel
from airs.utils.logging import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost classifier for drawdown prediction.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: float | None = None,
        early_stopping_rounds: int = 20,
        name: str = "xgboost_model",
        version: str = "v1",
        random_state: int = 42,
    ):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Feature subsample ratio
            min_child_weight: Minimum child weight
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            scale_pos_weight: Class weight for positive class
            early_stopping_rounds: Early stopping patience
            name: Model name
            version: Model version
            random_state: Random seed
        """
        super().__init__(name=name, version=version, random_state=random_state)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds

        self._model = None
        self._best_iteration = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs,
    ) -> "XGBoostModel":
        """
        Fit XGBoost model.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels
        """
        import xgboost as xgb

        self.feature_names = list(X.columns)

        # Calculate scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            self.scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        # Create model (XGBoost 2.0+ API)
        model_params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_state,
            "eval_metric": "logloss",
        }

        # Add early_stopping_rounds to constructor if we have validation data
        if X_val is not None and y_val is not None:
            model_params["early_stopping_rounds"] = self.early_stopping_rounds

        self._model = xgb.XGBClassifier(**model_params)

        # Fit with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            self._model.fit(
                X,
                y,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self._best_iteration = self._model.best_iteration
        else:
            self._model.fit(X, y)
            self._best_iteration = self.n_estimators

        self.is_fitted = True
        logger.info(
            f"Fitted XGBoost with {self._best_iteration} trees, "
            f"scale_pos_weight={self.scale_pos_weight:.2f}"
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X = self._validate_input(X)
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        X = self._validate_input(X)
        return self._model.predict_proba(X)

    def feature_importance(self) -> pd.Series:
        """Get feature importance (gain-based)."""
        if not self.is_fitted:
            return None

        importance = pd.Series(
            self._model.feature_importances_,
            index=self.feature_names,
            name="importance",
        )

        return importance.sort_values(ascending=False)


class LightGBMModel(BaseModel):
    """
    LightGBM classifier for drawdown prediction.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        class_weight: str | dict = "balanced",
        early_stopping_rounds: int = 20,
        name: str = "lightgbm_model",
        version: str = "v1",
        random_state: int = 42,
    ):
        """
        Initialize LightGBM model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            num_leaves: Number of leaves per tree
            subsample: Subsample ratio
            colsample_bytree: Feature subsample ratio
            min_child_samples: Minimum samples per leaf
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            class_weight: Class weights
            early_stopping_rounds: Early stopping patience
            name: Model name
            version: Model version
            random_state: Random seed
        """
        super().__init__(name=name, version=version, random_state=random_state)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.class_weight = class_weight
        self.early_stopping_rounds = early_stopping_rounds

        self._model = None
        self._best_iteration = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs,
    ) -> "LightGBMModel":
        """
        Fit LightGBM model.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels
        """
        import lightgbm as lgb

        self.feature_names = list(X.columns)

        # Create model
        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbose=-1,
        )

        # Fit with early stopping if validation set provided
        callbacks = []
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
            callbacks.append(lgb.log_evaluation(period=0))

            self._model.fit(
                X,
                y,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
            self._best_iteration = self._model.best_iteration_
        else:
            self._model.fit(X, y)
            self._best_iteration = self.n_estimators

        self.is_fitted = True
        logger.info(f"Fitted LightGBM with {self._best_iteration} trees")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X = self._validate_input(X)
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        X = self._validate_input(X)
        return self._model.predict_proba(X)

    def feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if not self.is_fitted:
            return None

        importance = pd.Series(
            self._model.feature_importances_,
            index=self.feature_names,
            name="importance",
        )

        return importance.sort_values(ascending=False)


class RandomForestModel(BaseModel):
    """
    Random Forest classifier for drawdown prediction.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        class_weight: str | dict = "balanced",
        name: str = "random_forest_model",
        version: str = "v1",
        random_state: int = 42,
    ):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples for split
            min_samples_leaf: Minimum samples per leaf
            max_features: Features to consider for split
            class_weight: Class weights
            name: Model name
            version: Model version
            random_state: Random seed
        """
        super().__init__(name=name, version=version, random_state=random_state)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight

        self._model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> "RandomForestModel":
        """
        Fit Random Forest model.

        Args:
            X: Training features
            y: Training labels
        """
        from sklearn.ensemble import RandomForestClassifier

        self.feature_names = list(X.columns)

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self._model.fit(X, y)
        self.is_fitted = True

        # Get OOB score if available
        if hasattr(self._model, "oob_score_"):
            logger.info(
                f"Fitted Random Forest with {self.n_estimators} trees, "
                f"OOB score: {self._model.oob_score_:.4f}"
            )
        else:
            logger.info(f"Fitted Random Forest with {self.n_estimators} trees")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X = self._validate_input(X)
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        X = self._validate_input(X)
        return self._model.predict_proba(X)

    def feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if not self.is_fitted:
            return None

        importance = pd.Series(
            self._model.feature_importances_,
            index=self.feature_names,
            name="importance",
        )

        return importance.sort_values(ascending=False)
