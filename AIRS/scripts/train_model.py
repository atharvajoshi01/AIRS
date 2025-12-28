#!/usr/bin/env python3
"""
Train AIRS drawdown prediction models.

This script trains the ensemble model for predicting portfolio drawdowns.
It includes feature engineering, model training, validation, and saving.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --track  # With MLflow tracking
    python scripts/train_model.py --model xgboost  # Single model
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from airs.config import get_settings
from airs.data import DataAggregator
from airs.features import FeaturePipeline
from airs.targets import DrawdownCalculator, LabelGenerator
from airs.models import (
    XGBoostModel,
    LightGBMModel,
    RandomForestModel,
    LogisticModel,
    StackingEnsemble,
    RegimeAwareEnsemble,
    WalkForwardValidator,
    ModelEvaluator,
)
from airs.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AIRS drawdown prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to market data. Default: ./data/processed/market_data.parquet",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["xgboost", "lightgbm", "random_forest", "logistic", "ensemble", "regime_aware"],
        help="Model type to train. Default: ensemble",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for models. Default: ./data/models",
    )

    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable MLflow experiment tracking",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run walk-forward validation (default: True)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=-0.05,
        help="Drawdown threshold for labels. Default: -0.05 (-5%%)",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=15,
        help="Prediction horizon in days. Default: 15",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def load_data(data_path: Path) -> pd.DataFrame:
    """Load market data from file."""
    if data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        return pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")


def prepare_features_and_labels(
    data: pd.DataFrame,
    threshold: float,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate features and labels from raw data."""
    logger.info("Generating features...")

    # Initialize feature pipeline
    pipeline = FeaturePipeline(
        lookback_window=252,
        include_regime=True,
        include_composite=True,
    )

    # Generate features
    features = pipeline.generate_features(data, add_prefix=True)
    logger.info(f"Generated {len(features.columns)} features")

    # Clean features
    features = pipeline.clean_features(
        features,
        max_missing_pct=0.3,
        drop_constant=True,
        fill_method="ffill",
        fill_limit=5,
    )
    logger.info(f"After cleaning: {features.shape}")

    # Generate labels
    logger.info("Generating labels...")

    # Get portfolio prices for drawdown calculation
    price_cols = [c for c in data.columns if c in ["SPY", "prices_SPY", "prices_close"]]
    if price_cols:
        prices = data[price_cols[0]]
    else:
        # Try to find any price column
        price_cols = [c for c in data.columns if "SPY" in c or "close" in c.lower()]
        if price_cols:
            prices = data[price_cols[0]]
        else:
            raise ValueError("Could not find price data for label generation")

    label_gen = LabelGenerator(
        threshold=threshold,
        horizon=horizon,
    )

    labels = label_gen.generate_binary_labels(prices)
    logger.info(f"Label distribution: {labels.value_counts().to_dict()}")

    # Align features and labels
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    # Drop any remaining NaN
    valid_mask = ~(features.isnull().any(axis=1) | labels.isnull())
    features = features[valid_mask]
    labels = labels[valid_mask]

    logger.info(f"Final dataset: {features.shape[0]} samples, {features.shape[1]} features")

    return features, labels


def create_model(model_type: str):
    """Create model instance based on type."""
    if model_type == "xgboost":
        return XGBoostModel(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            early_stopping_rounds=20,
        )
    elif model_type == "lightgbm":
        return LightGBMModel(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            early_stopping_rounds=20,
        )
    elif model_type == "random_forest":
        return RandomForestModel(
            n_estimators=200,
            max_depth=10,
        )
    elif model_type == "logistic":
        return LogisticModel(
            C=1.0,
            penalty="l2",
        )
    elif model_type == "ensemble":
        return StackingEnsemble(
            cv_folds=5,
        )
    elif model_type == "regime_aware":
        return RegimeAwareEnsemble(
            n_regimes=3,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(
    model,
    features: pd.DataFrame,
    labels: pd.Series,
    validate: bool = True,
) -> dict:
    """Train model and run evaluation."""
    results = {}

    # Split data temporally
    n = len(features)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train = features.iloc[:train_end]
    y_train = labels.iloc[:train_end]
    X_val = features.iloc[train_end:val_end]
    y_val = labels.iloc[train_end:val_end]
    X_test = features.iloc[val_end:]
    y_test = labels.iloc[val_end:]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train model
    logger.info(f"Training {model.name}...")
    if hasattr(model, "fit") and "X_val" in model.fit.__code__.co_varnames:
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    evaluator = ModelEvaluator()

    test_preds = model.predict(X_test)
    test_proba = model.predict_proba(X_test)

    # Handle probability format
    if len(test_proba.shape) > 1:
        test_proba = test_proba[:, 1]

    metrics = evaluator.evaluate(y_test.values, test_preds, test_proba)

    results["test_metrics"] = metrics
    results["test_report"] = evaluator.generate_report(metrics, model.name)

    # Walk-forward validation
    if validate:
        logger.info("Running walk-forward validation...")
        validator = WalkForwardValidator(
            initial_train_size=1260,  # ~5 years
            test_size=63,  # ~3 months
            step_size=21,  # ~1 month
            embargo_size=5,
        )

        # Map model names to create_model types
        model_name_to_type = {
            "xgboost_model": "xgboost",
            "lightgbm_model": "lightgbm",
            "random_forest_model": "random_forest",
            "logistic_model": "logistic",
            "stacking_ensemble": "ensemble",
            "regime_aware_ensemble": "regime_aware",
        }
        model_type = model_name_to_type.get(model.name, model.name)
        fresh_model = create_model(model_type)
        wf_results = validator.validate(fresh_model, features, labels)

        results["walk_forward"] = wf_results
        logger.info(f"Walk-forward ROC-AUC: {wf_results['overall_metrics'].roc_auc:.4f}")

    # Feature importance
    importance = model.feature_importance()
    if importance is not None:
        results["feature_importance"] = importance.head(20)
        logger.info("Top 10 features:")
        for feat, imp in importance.head(10).items():
            logger.info(f"  {feat}: {imp:.4f}")

    return results


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    logger.info("=" * 60)
    logger.info("AIRS Model Training")
    logger.info("=" * 60)

    # Get settings
    settings = get_settings()

    # Determine data path
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = settings.data_dir / "processed" / "market_data.parquet"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Run 'make fetch-data' or 'python scripts/fetch_data.py' first")
        return 1

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = settings.data_dir / "models"

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        logger.info(f"Loading data from {data_path}...")
        data = load_data(data_path)
        logger.info(f"Loaded data: {data.shape}")

        # Prepare features and labels
        features, labels = prepare_features_and_labels(
            data,
            threshold=args.threshold,
            horizon=args.horizon,
        )

        # Setup MLflow tracking if requested
        if args.track:
            try:
                import mlflow

                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                mlflow.set_experiment(settings.mlflow_experiment_name)
                mlflow.start_run(run_name=f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

                # Log parameters
                mlflow.log_param("model_type", args.model)
                mlflow.log_param("threshold", args.threshold)
                mlflow.log_param("horizon", args.horizon)
                mlflow.log_param("n_features", features.shape[1])
                mlflow.log_param("n_samples", features.shape[0])

                logger.info("MLflow tracking enabled")
            except ImportError:
                logger.warning("MLflow not available, skipping tracking")
                args.track = False

        # Create and train model
        model = create_model(args.model)
        results = train_and_evaluate(
            model,
            features,
            labels,
            validate=args.validate,
        )

        # Print results
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training Results")
        logger.info("=" * 60)
        print(results["test_report"])

        # Log to MLflow
        if args.track:
            metrics = results["test_metrics"]
            mlflow.log_metric("test_accuracy", metrics.accuracy)
            mlflow.log_metric("test_precision", metrics.precision)
            mlflow.log_metric("test_recall", metrics.recall)
            mlflow.log_metric("test_f1", metrics.f1)
            mlflow.log_metric("test_roc_auc", metrics.roc_auc)

            if "walk_forward" in results:
                wf = results["walk_forward"]["overall_metrics"]
                mlflow.log_metric("wf_roc_auc", wf.roc_auc)
                mlflow.log_metric("wf_precision", wf.precision)
                mlflow.log_metric("wf_recall", wf.recall)

            mlflow.end_run()

        # Save model
        model_path = output_dir / f"{args.model}_model.pkl"
        model.save(model_path)
        logger.info(f"Model saved to: {model_path}")

        # Save feature importance
        if "feature_importance" in results:
            importance_path = output_dir / f"{args.model}_feature_importance.csv"
            results["feature_importance"].to_csv(importance_path)
            logger.info(f"Feature importance saved to: {importance_path}")

        logger.info("")
        logger.info("Training completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception(e)
        if args.track:
            try:
                import mlflow
                mlflow.end_run(status="FAILED")
            except:
                pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
