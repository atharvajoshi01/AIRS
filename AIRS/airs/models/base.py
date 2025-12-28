"""
Base model interface for AIRS.

Defines the interface that all models must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from airs.utils.logging import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all AIRS models.

    Defines the interface for training, prediction, and persistence.
    """

    def __init__(
        self,
        name: str = "base_model",
        version: str = "v1",
        random_state: int = 42,
    ):
        """
        Initialize base model.

        Args:
            name: Model name
            version: Model version
            random_state: Random seed for reproducibility
        """
        self.name = name
        self.version = version
        self.random_state = random_state
        self.is_fitted = False
        self.feature_names: list[str] = []
        self._model: Any = None

    @property
    def model_id(self) -> str:
        """Return unique model identifier."""
        return f"{self.name}_{self.version}"

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> "BaseModel":
        """
        Fit the model to training data.

        Args:
            X: Feature matrix
            y: Target labels
            **kwargs: Additional training arguments

        Returns:
            Self
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability array (n_samples, n_classes)
        """
        pass

    def get_probability(self, X: pd.DataFrame) -> pd.Series:
        """
        Get probability of positive class.

        Args:
            X: Feature matrix

        Returns:
            Series of probabilities
        """
        proba = self.predict_proba(X)
        if proba.ndim == 2:
            proba = proba[:, 1]  # Probability of class 1

        return pd.Series(proba, index=X.index, name="probability")

    def feature_importance(self) -> pd.Series | None:
        """
        Get feature importance scores.

        Returns:
            Series of importance scores or None if not available
        """
        return None

    def save(self, path: str | Path) -> Path:
        """
        Save model to disk.

        Args:
            path: Save path

        Returns:
            Path to saved model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self._model,
            "name": self.name,
            "version": self.version,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
        }

        joblib.dump(model_data, path)
        logger.info(f"Saved model to {path}")

        return path

    def load(self, path: str | Path) -> "BaseModel":
        """
        Load model from disk.

        Args:
            path: Load path

        Returns:
            Self
        """
        path = Path(path)
        model_data = joblib.load(path)

        self._model = model_data["model"]
        self.name = model_data["name"]
        self.version = model_data["version"]
        self.random_state = model_data["random_state"]
        self.is_fitted = model_data["is_fitted"]
        self.feature_names = model_data["feature_names"]

        logger.info(f"Loaded model from {path}")

        return self

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare input features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Check for missing features
        if self.feature_names:
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Missing features: {missing}")

            # Reorder columns to match training
            X = X[self.feature_names]

        return X

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            "name": self.name,
            "version": self.version,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "BaseModel":
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"
