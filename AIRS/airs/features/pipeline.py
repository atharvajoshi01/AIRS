"""
Feature pipeline for orchestrating feature generation.

Combines all feature generators into a unified pipeline.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from airs.features.base import FeatureGenerator
from airs.features.rates import RateFeatures, GlobalRateFeatures
from airs.features.credit import CreditFeatures, CreditQualityFeatures
from airs.features.volatility import VolatilityFeatures, RateVolatilityFeatures
from airs.features.macro import MacroFeatures, MoneySupplyFeatures
from airs.features.cross_asset import CrossAssetFeatures
from airs.features.regime import RegimeDetector, VolatilityRegimeDetector
from airs.features.composite import CompositeFeatures
from airs.utils.logging import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """
    Feature generation pipeline.

    Orchestrates all feature generators and produces a unified feature matrix.
    """

    def __init__(
        self,
        lookback_window: int = 252,
        include_regime: bool = True,
        include_composite: bool = True,
    ):
        """
        Initialize feature pipeline.

        Args:
            lookback_window: Default lookback for rolling calculations
            include_regime: Whether to include regime detection
            include_composite: Whether to include composite features
        """
        self.lookback_window = lookback_window
        self.include_regime = include_regime
        self.include_composite = include_composite

        # Initialize generators
        self.generators: list[FeatureGenerator] = [
            RateFeatures(lookback_window=lookback_window),
            CreditFeatures(lookback_window=lookback_window),
            VolatilityFeatures(lookback_window=lookback_window),
            RateVolatilityFeatures(lookback_window=lookback_window),
            MacroFeatures(lookback_window=lookback_window),
            CrossAssetFeatures(lookback_window=lookback_window),
        ]

        if include_regime:
            self.generators.append(
                RegimeDetector(n_regimes=3, method="combined", lookback_window=lookback_window)
            )
            self.generators.append(
                VolatilityRegimeDetector(lookback_window=lookback_window)
            )

        # Composite generator is added last as it depends on other features
        self.composite_generator = CompositeFeatures(lookback_window=lookback_window) if include_composite else None

    def generate_features(
        self,
        data: pd.DataFrame,
        add_prefix: bool = True,
    ) -> pd.DataFrame:
        """
        Generate all features from input data.

        Args:
            data: DataFrame with raw market data
            add_prefix: Whether to add feature group prefix to column names

        Returns:
            DataFrame with all features
        """
        all_features = []

        logger.info(f"Generating features from {len(self.generators)} generators")

        for generator in self.generators:
            try:
                features = generator.generate(data)

                if add_prefix and not features.empty:
                    features = generator.prefix_columns(features, generator.feature_group)

                all_features.append(features)

            except Exception as e:
                logger.error(f"Error in {generator.feature_group}: {e}")
                continue

        if not all_features:
            logger.warning("No features generated")
            return pd.DataFrame(index=data.index)

        # Combine all features
        combined = pd.concat(all_features, axis=1)

        # Generate composite features (needs combined data)
        if self.composite_generator is not None:
            try:
                composite = self.composite_generator.generate(combined)
                if add_prefix:
                    composite = self.composite_generator.prefix_columns(
                        composite, self.composite_generator.feature_group
                    )
                combined = pd.concat([combined, composite], axis=1)
            except Exception as e:
                logger.error(f"Error generating composite features: {e}")

        logger.info(f"Generated {len(combined.columns)} total features")

        return combined

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names."""
        names = []
        for generator in self.generators:
            for name in generator.feature_names:
                names.append(f"{generator.feature_group}_{name}")

        if self.composite_generator:
            for name in self.composite_generator.feature_names:
                names.append(f"{self.composite_generator.feature_group}_{name}")

        return names

    def get_feature_groups(self) -> dict[str, list[str]]:
        """Get feature names organized by group."""
        groups = {}
        for generator in self.generators:
            groups[generator.feature_group] = generator.feature_names

        if self.composite_generator:
            groups[self.composite_generator.feature_group] = (
                self.composite_generator.feature_names
            )

        return groups

    def validate_features(
        self,
        features: pd.DataFrame,
        max_missing_pct: float = 0.3,
    ) -> dict[str, Any]:
        """
        Validate generated features.

        Args:
            features: DataFrame of generated features
            max_missing_pct: Maximum acceptable missing percentage

        Returns:
            Validation results
        """
        results = {
            "total_features": len(features.columns),
            "total_rows": len(features),
            "missing_summary": {},
            "dropped_features": [],
            "warnings": [],
        }

        # Check missing values
        missing_pct = features.isna().sum() / len(features)
        results["missing_summary"] = missing_pct.to_dict()

        # Identify features with too many missing values
        high_missing = missing_pct[missing_pct > max_missing_pct]
        results["dropped_features"] = list(high_missing.index)

        if len(results["dropped_features"]) > 0:
            results["warnings"].append(
                f"{len(results['dropped_features'])} features have >{max_missing_pct*100}% missing"
            )

        # Check for constant features
        constant_features = []
        for col in features.columns:
            if features[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            results["warnings"].append(
                f"{len(constant_features)} constant features detected"
            )
            results["constant_features"] = constant_features

        # Check for infinite values
        inf_counts = (features == np.inf).sum() + (features == -np.inf).sum()
        inf_features = inf_counts[inf_counts > 0]
        if len(inf_features) > 0:
            results["warnings"].append(
                f"{len(inf_features)} features contain infinite values"
            )
            results["inf_features"] = inf_features.to_dict()

        return results

    def clean_features(
        self,
        features: pd.DataFrame,
        max_missing_pct: float = 0.3,
        drop_constant: bool = True,
        fill_method: str = "ffill",
        fill_limit: int = 5,
    ) -> pd.DataFrame:
        """
        Clean features by handling missing values and problematic columns.

        Args:
            features: DataFrame of features
            max_missing_pct: Maximum missing percentage to keep
            drop_constant: Whether to drop constant features
            fill_method: Method for filling missing values
            fill_limit: Limit for forward/backward filling

        Returns:
            Cleaned feature DataFrame
        """
        import numpy as np

        cleaned = features.copy()

        # Replace infinite values with NaN
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

        # Drop columns with too many missing values
        missing_pct = cleaned.isna().sum() / len(cleaned)
        cols_to_drop = missing_pct[missing_pct > max_missing_pct].index
        if len(cols_to_drop) > 0:
            logger.info(f"Dropping {len(cols_to_drop)} features with >{max_missing_pct*100}% missing")
            cleaned = cleaned.drop(columns=cols_to_drop)

        # Drop constant columns
        if drop_constant:
            constant_cols = [col for col in cleaned.columns if cleaned[col].nunique() <= 1]
            if constant_cols:
                logger.info(f"Dropping {len(constant_cols)} constant features")
                cleaned = cleaned.drop(columns=constant_cols)

        # Fill missing values
        if fill_method == "ffill":
            cleaned = cleaned.ffill(limit=fill_limit)
        elif fill_method == "bfill":
            cleaned = cleaned.bfill(limit=fill_limit)
        elif fill_method == "mean":
            cleaned = cleaned.fillna(cleaned.mean())

        # Drop remaining rows with NaN
        initial_rows = len(cleaned)
        cleaned = cleaned.dropna()
        dropped_rows = initial_rows - len(cleaned)

        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with remaining NaN values")

        return cleaned

    def save_features(
        self,
        features: pd.DataFrame,
        path: str | Path,
        format: str = "parquet",
    ) -> Path:
        """Save features to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            features.to_parquet(path)
        elif format == "csv":
            features.to_csv(path)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved {len(features.columns)} features to {path}")
        return path

    def load_features(self, path: str | Path, format: str = "parquet") -> pd.DataFrame:
        """Load features from file."""
        path = Path(path)

        if format == "parquet":
            return pd.read_parquet(path)
        elif format == "csv":
            return pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unknown format: {format}")


# Import numpy for clean_features method
import numpy as np
