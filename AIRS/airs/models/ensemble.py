"""
Ensemble models for AIRS.

Stacking ensemble and regime-aware ensemble implementations.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from airs.models.base import BaseModel
from airs.models.baseline import LogisticModel
from airs.models.tree_ensemble import XGBoostModel, LightGBMModel, RandomForestModel
from airs.utils.logging import get_logger

logger = get_logger(__name__)


class StackingEnsemble(BaseModel):
    """
    Stacking ensemble that combines multiple base models.

    Architecture:
    - Layer 1: Diverse base models (XGBoost, LightGBM, RF, LogReg)
    - Layer 2: Meta-learner (Logistic Regression)
    """

    def __init__(
        self,
        base_models: list[BaseModel] | None = None,
        meta_learner: BaseModel | None = None,
        use_probas: bool = True,
        cv_folds: int = 5,
        name: str = "stacking_ensemble",
        version: str = "v1",
        random_state: int = 42,
    ):
        """
        Initialize stacking ensemble.

        Args:
            base_models: List of base models (default creates standard set)
            meta_learner: Meta-learner model (default is LogisticRegression)
            use_probas: Use probabilities vs predictions for stacking
            cv_folds: Cross-validation folds for stacking
            name: Model name
            version: Model version
            random_state: Random seed
        """
        super().__init__(name=name, version=version, random_state=random_state)

        self.use_probas = use_probas
        self.cv_folds = cv_folds

        # Default base models
        if base_models is None:
            self.base_models = [
                XGBoostModel(name="xgb_base", random_state=random_state),
                LightGBMModel(name="lgbm_base", random_state=random_state),
                RandomForestModel(name="rf_base", random_state=random_state),
                LogisticModel(name="logreg_base", random_state=random_state),
            ]
        else:
            self.base_models = base_models

        # Default meta-learner
        if meta_learner is None:
            self._meta_learner = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                random_state=random_state,
            )
        else:
            self._meta_learner = meta_learner

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs,
    ) -> "StackingEnsemble":
        """
        Fit stacking ensemble.

        Uses out-of-fold predictions to avoid leakage.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        from sklearn.model_selection import StratifiedKFold

        self.feature_names = list(X.columns)
        n_models = len(self.base_models)

        # Prepare meta-features matrix
        meta_features = np.zeros((len(X), n_models))

        # Cross-validation for stacking
        kfold = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        logger.info(f"Fitting {n_models} base models with {self.cv_folds}-fold CV")

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]

            for model_idx, model in enumerate(self.base_models):
                # Clone model for this fold
                params = model.get_params()
                # Update random_state to vary by fold
                params["random_state"] = self.random_state + fold_idx
                model_clone = model.__class__(**params)

                # Fit on fold
                model_clone.fit(X_fold_train, y_fold_train)

                # Get predictions for held-out samples
                if self.use_probas:
                    preds = model_clone.predict_proba(X_fold_val)[:, 1]
                else:
                    preds = model_clone.predict(X_fold_val)

                meta_features[val_idx, model_idx] = preds

        # Fit base models on full training data
        for model in self.base_models:
            if X_val is not None and y_val is not None:
                model.fit(X, y, X_val=X_val, y_val=y_val)
            else:
                model.fit(X, y)

        # Fit meta-learner
        self._meta_learner.fit(meta_features, y)

        self.is_fitted = True
        logger.info("Fitted stacking ensemble")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using ensemble."""
        X = self._validate_input(X)

        # Get base model predictions
        meta_features = self._get_meta_features(X)

        # Get meta-learner predictions
        return self._meta_learner.predict_proba(meta_features)

    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Get meta-features from base models."""
        meta_features = np.zeros((len(X), len(self.base_models)))

        for i, model in enumerate(self.base_models):
            if self.use_probas:
                meta_features[:, i] = model.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = model.predict(X)

        return meta_features

    def feature_importance(self) -> pd.Series:
        """Get average feature importance across base models."""
        if not self.is_fitted:
            return None

        importance_sum = None

        for model in self.base_models:
            imp = model.feature_importance()
            if imp is not None:
                if importance_sum is None:
                    importance_sum = imp.copy()
                else:
                    importance_sum += imp.reindex(importance_sum.index, fill_value=0)

        if importance_sum is not None:
            importance_avg = importance_sum / len(self.base_models)
            return importance_avg.sort_values(ascending=False)

        return None

    def get_base_model_weights(self) -> pd.Series:
        """Get meta-learner weights for base models."""
        if not self.is_fitted:
            return None

        weights = pd.Series(
            self._meta_learner.coef_[0],
            index=[m.name for m in self.base_models],
            name="weight",
        )

        return weights.sort_values(ascending=False)

    def save(self, path: str | Path) -> Path:
        """
        Save stacking ensemble to disk.

        Properly saves all base models and meta-learner.
        """
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save base models with their state
        base_models_data = []
        for model in self.base_models:
            model_data = {
                "class": model.__class__.__name__,
                "model": model._model,
                "name": model.name,
                "version": model.version,
                "random_state": model.random_state,
                "is_fitted": model.is_fitted,
                "feature_names": model.feature_names,
                "params": model.get_params(),
            }
            # Save any additional model-specific attributes
            if hasattr(model, "_best_iteration"):
                model_data["_best_iteration"] = model._best_iteration
            if hasattr(model, "scale_pos_weight"):
                model_data["scale_pos_weight"] = model.scale_pos_weight
            if hasattr(model, "_scaler"):
                model_data["_scaler"] = model._scaler
            base_models_data.append(model_data)

        ensemble_data = {
            "base_models_data": base_models_data,
            "meta_learner": self._meta_learner,
            "use_probas": self.use_probas,
            "cv_folds": self.cv_folds,
            "name": self.name,
            "version": self.version,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
        }

        joblib.dump(ensemble_data, path)
        logger.info(f"Saved stacking ensemble to {path}")

        return path

    def load(self, path: str | Path) -> "StackingEnsemble":
        """
        Load stacking ensemble from disk.

        Properly restores all base models and meta-learner.
        """
        import joblib

        path = Path(path)
        ensemble_data = joblib.load(path)

        # Restore ensemble attributes
        self._meta_learner = ensemble_data["meta_learner"]
        self.use_probas = ensemble_data["use_probas"]
        self.cv_folds = ensemble_data["cv_folds"]
        self.name = ensemble_data["name"]
        self.version = ensemble_data["version"]
        self.random_state = ensemble_data["random_state"]
        self.is_fitted = ensemble_data["is_fitted"]
        self.feature_names = ensemble_data["feature_names"]

        # Restore base models
        from airs.models.tree_ensemble import XGBoostModel, LightGBMModel, RandomForestModel
        from airs.models.baseline import LogisticModel

        class_map = {
            "XGBoostModel": XGBoostModel,
            "LightGBMModel": LightGBMModel,
            "RandomForestModel": RandomForestModel,
            "LogisticModel": LogisticModel,
        }

        self.base_models = []
        for model_data in ensemble_data["base_models_data"]:
            model_class = class_map.get(model_data["class"])
            if model_class is None:
                raise ValueError(f"Unknown model class: {model_data['class']}")

            # Create model instance with saved params
            params = model_data["params"]
            model = model_class(**params)

            # Restore fitted state
            model._model = model_data["model"]
            model.is_fitted = model_data["is_fitted"]
            model.feature_names = model_data["feature_names"]

            # Restore model-specific attributes
            if "_best_iteration" in model_data:
                model._best_iteration = model_data["_best_iteration"]
            if "scale_pos_weight" in model_data:
                model.scale_pos_weight = model_data["scale_pos_weight"]
            if "_scaler" in model_data:
                model._scaler = model_data["_scaler"]

            self.base_models.append(model)

        logger.info(f"Loaded stacking ensemble from {path}")

        return self


class RegimeAwareEnsemble(BaseModel):
    """
    Regime-aware ensemble that adapts to market conditions.

    Uses different model weights or separate models for different regimes.
    """

    def __init__(
        self,
        base_ensemble: StackingEnsemble | None = None,
        regime_feature: str = "regime_regime",
        n_regimes: int = 3,
        regime_weights: dict[int, dict[str, float]] | None = None,
        name: str = "regime_aware_ensemble",
        version: str = "v1",
        random_state: int = 42,
    ):
        """
        Initialize regime-aware ensemble.

        Args:
            base_ensemble: Base stacking ensemble
            regime_feature: Name of regime feature
            n_regimes: Number of regimes
            regime_weights: Predefined weights per regime
            name: Model name
            version: Model version
            random_state: Random seed
        """
        super().__init__(name=name, version=version, random_state=random_state)

        self.regime_feature = regime_feature
        self.n_regimes = n_regimes
        self.regime_weights = regime_weights

        # Base ensemble
        if base_ensemble is None:
            self.base_ensemble = StackingEnsemble(random_state=random_state)
        else:
            self.base_ensemble = base_ensemble

        # Regime-specific adjustments
        self._regime_calibrators: dict[int, Any] = {}
        self._regime_thresholds: dict[int, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs,
    ) -> "RegimeAwareEnsemble":
        """
        Fit regime-aware ensemble.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        self.feature_names = list(X.columns)

        # Check if regime feature exists
        if self.regime_feature not in X.columns:
            logger.warning(
                f"Regime feature '{self.regime_feature}' not found. "
                "Falling back to standard ensemble."
            )
            self.base_ensemble.fit(X, y, X_val=X_val, y_val=y_val)
            self.is_fitted = True
            return self

        # Fit base ensemble
        X_no_regime = X.drop(columns=[self.regime_feature])
        if X_val is not None:
            X_val_no_regime = X_val.drop(columns=[self.regime_feature], errors="ignore")
        else:
            X_val_no_regime = None

        self.base_ensemble.fit(
            X_no_regime, y, X_val=X_val_no_regime, y_val=y_val
        )

        # Calibrate per regime
        for regime in range(self.n_regimes):
            regime_mask = X[self.regime_feature] == regime

            if regime_mask.sum() < 50:
                logger.warning(f"Insufficient samples for regime {regime}")
                continue

            X_regime = X_no_regime[regime_mask]
            y_regime = y[regime_mask]

            # Get base predictions for this regime
            base_proba = self.base_ensemble.predict_proba(X_regime)[:, 1]

            # Calculate optimal threshold for this regime
            from sklearn.metrics import precision_recall_curve

            precision, recall, thresholds = precision_recall_curve(y_regime, base_proba)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            self._regime_thresholds[regime] = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

            logger.info(
                f"Regime {regime}: {regime_mask.sum()} samples, "
                f"optimal threshold: {self._regime_thresholds[regime]:.3f}"
            )

        self.is_fitted = True
        logger.info("Fitted regime-aware ensemble")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with regime-specific thresholds."""
        X = self._validate_input(X)

        proba = self.predict_proba(X)[:, 1]
        predictions = np.zeros(len(X), dtype=int)

        if self.regime_feature in X.columns:
            regimes = X[self.regime_feature].values

            for regime in range(self.n_regimes):
                regime_mask = regimes == regime
                threshold = self._regime_thresholds.get(regime, 0.5)
                predictions[regime_mask] = (proba[regime_mask] >= threshold).astype(int)
        else:
            predictions = (proba >= 0.5).astype(int)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        X = self._validate_input(X)

        # Remove regime feature for base ensemble
        if self.regime_feature in X.columns:
            X_no_regime = X.drop(columns=[self.regime_feature])
        else:
            X_no_regime = X

        # Get base predictions
        proba = self.base_ensemble.predict_proba(X_no_regime)

        # Apply regime-specific adjustments if defined
        if self.regime_weights and self.regime_feature in X.columns:
            regimes = X[self.regime_feature].values

            for regime, weights in self.regime_weights.items():
                regime_mask = regimes == regime

                # Adjust probability based on regime weights
                if "bias" in weights:
                    proba[regime_mask, 1] += weights["bias"]
                if "scale" in weights:
                    proba[regime_mask, 1] *= weights["scale"]

            # Clip to valid range
            proba = np.clip(proba, 0, 1)
            proba[:, 0] = 1 - proba[:, 1]

        return proba

    def feature_importance(self) -> pd.Series:
        """Get feature importance from base ensemble."""
        return self.base_ensemble.feature_importance()

    def get_regime_thresholds(self) -> dict[int, float]:
        """Get optimal thresholds per regime."""
        return self._regime_thresholds.copy()
