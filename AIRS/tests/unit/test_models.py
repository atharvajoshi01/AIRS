"""
Unit tests for ML models.
"""

import numpy as np
import pandas as pd
import pytest


class TestBaselineModels:
    """Tests for baseline models."""

    def test_threshold_model(self, sample_features: pd.DataFrame):
        """Test threshold-based model."""
        from airs.models.baseline import ThresholdModel

        model = ThresholdModel()

        # Fit the model first
        y = pd.Series(np.random.randint(0, 2, len(sample_features)), index=sample_features.index)
        model.fit(sample_features, y)

        # Create predictions
        predictions = model.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert all(p in [0, 1] for p in predictions)

    def test_logistic_model_fit(
        self,
        sample_features: pd.DataFrame,
        sample_labels: pd.Series,
    ):
        """Test logistic regression model training."""
        from airs.models.baseline import LogisticModel

        # Align data - keep as DataFrame
        common_idx = sample_features.index.intersection(sample_labels.index)
        X = sample_features.loc[common_idx].fillna(0)
        y = sample_labels.loc[common_idx]

        model = LogisticModel()
        model.fit(X, y)

        # Check predictions
        predictions = model.predict_proba(X)
        assert len(predictions) == len(X)
        # predict_proba returns 2D array
        assert predictions.shape[1] == 2

    def test_logistic_model_feature_importance(
        self,
        sample_features: pd.DataFrame,
        sample_labels: pd.Series,
    ):
        """Test feature importance extraction."""
        from airs.models.baseline import LogisticModel

        common_idx = sample_features.index.intersection(sample_labels.index)
        X = sample_features.loc[common_idx].fillna(0)
        y = sample_labels.loc[common_idx]

        model = LogisticModel()
        model.fit(X, y)

        # Method is feature_importance(), not get_feature_importance()
        importance = model.feature_importance()
        assert len(importance) == X.shape[1]


class TestTreeEnsembles:
    """Tests for tree ensemble models."""

    def test_xgboost_fit(
        self,
        sample_features: pd.DataFrame,
        sample_labels: pd.Series,
    ):
        """Test XGBoost model training."""
        pytest.importorskip("xgboost")
        from airs.models.tree_ensemble import XGBoostModel

        common_idx = sample_features.index.intersection(sample_labels.index)
        X = sample_features.loc[common_idx].fillna(0)
        y = sample_labels.loc[common_idx]

        model = XGBoostModel(n_estimators=10)
        model.fit(X, y)

        predictions = model.predict_proba(X)
        assert len(predictions) == len(X)
        # predict_proba returns 2D array
        assert predictions.shape[1] == 2

    def test_lightgbm_fit(
        self,
        sample_features: pd.DataFrame,
        sample_labels: pd.Series,
    ):
        """Test LightGBM model training."""
        pytest.importorskip("lightgbm")
        from airs.models.tree_ensemble import LightGBMModel

        common_idx = sample_features.index.intersection(sample_labels.index)
        X = sample_features.loc[common_idx].fillna(0)
        y = sample_labels.loc[common_idx]

        model = LightGBMModel(n_estimators=10)
        model.fit(X, y)

        predictions = model.predict_proba(X)
        assert len(predictions) == len(X)

    def test_random_forest_fit(
        self,
        sample_features: pd.DataFrame,
        sample_labels: pd.Series,
    ):
        """Test Random Forest model training."""
        from airs.models.tree_ensemble import RandomForestModel

        common_idx = sample_features.index.intersection(sample_labels.index)
        X = sample_features.loc[common_idx].fillna(0)
        y = sample_labels.loc[common_idx]

        model = RandomForestModel(n_estimators=10)
        model.fit(X, y)

        predictions = model.predict_proba(X)
        assert len(predictions) == len(X)

    def test_feature_importance(
        self,
        sample_features: pd.DataFrame,
        sample_labels: pd.Series,
    ):
        """Test feature importance from tree models."""
        from airs.models.tree_ensemble import RandomForestModel

        common_idx = sample_features.index.intersection(sample_labels.index)
        X = sample_features.loc[common_idx].fillna(0)
        y = sample_labels.loc[common_idx]

        model = RandomForestModel(n_estimators=10)
        model.fit(X, y)

        importance = model.feature_importance()
        assert len(importance) == X.shape[1]
        assert importance.sum() > 0  # Should have some importance


class TestEnsembleModels:
    """Tests for ensemble models."""

    def test_stacking_ensemble(
        self,
        sample_features: pd.DataFrame,
        sample_labels: pd.Series,
    ):
        """Test stacking ensemble."""
        from airs.models.ensemble import StackingEnsemble
        from airs.models.baseline import LogisticModel

        common_idx = sample_features.index.intersection(sample_labels.index)
        X = sample_features.loc[common_idx].fillna(0)
        y = sample_labels.loc[common_idx]

        # Use simple base models for testing
        ensemble = StackingEnsemble(
            base_models=[LogisticModel(name="logreg_test")],
            cv_folds=2,
        )
        ensemble.fit(X, y)

        predictions = ensemble.predict_proba(X)
        assert len(predictions) == len(X)
        # Check probabilities are valid
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)


class TestModelValidation:
    """Tests for model validation."""

    def test_walk_forward_splits(self, sample_features: pd.DataFrame):
        """Test walk-forward validation splits."""
        from airs.models.validation import WalkForwardValidator

        # Use actual parameter names from implementation
        validator = WalkForwardValidator(
            initial_train_size=252,
            test_size=63,
            step_size=21,
            embargo_size=5,
        )

        splits = validator.split(sample_features, pd.Series(0, index=sample_features.index))

        assert len(splits) > 0

        for train_idx, test_idx in splits:
            # Test indices should come after train indices
            assert train_idx.max() < test_idx.min()
            # Embargo should be respected
            assert test_idx.min() - train_idx.max() >= 5

    def test_no_lookahead_in_cv(self, sample_features: pd.DataFrame):
        """Test that cross-validation doesn't leak future data."""
        from airs.models.validation import WalkForwardValidator

        validator = WalkForwardValidator(
            initial_train_size=252,
            test_size=63,
            step_size=21,
            embargo_size=5,
        )

        y = pd.Series(0, index=sample_features.index)
        for train_idx, test_idx in validator.split(sample_features, y):
            train_dates = sample_features.index[train_idx]
            test_dates = sample_features.index[test_idx]

            # All train dates should be before all test dates
            assert train_dates.max() < test_dates.min()


class TestModelMetrics:
    """Tests for model evaluation metrics."""

    def test_precision_recall(
        self,
        sample_predictions: pd.Series,
        sample_labels: pd.Series,
    ):
        """Test precision and recall calculation."""
        from sklearn.metrics import precision_score, recall_score

        common_idx = sample_predictions.index.intersection(sample_labels.index)
        preds = (sample_predictions.loc[common_idx] >= 0.5).astype(int)
        labels = sample_labels.loc[common_idx]

        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1

    def test_auc_score(
        self,
        sample_predictions: pd.Series,
        sample_labels: pd.Series,
    ):
        """Test AUC score calculation."""
        from sklearn.metrics import roc_auc_score

        common_idx = sample_predictions.index.intersection(sample_labels.index)
        preds = sample_predictions.loc[common_idx]
        labels = sample_labels.loc[common_idx]

        # Only calculate if we have both classes
        if len(labels.unique()) > 1:
            auc = roc_auc_score(labels, preds)
            assert 0 <= auc <= 1
