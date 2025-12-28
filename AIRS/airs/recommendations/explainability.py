"""
Model explainability for AIRS recommendations.

Provides SHAP-based feature attribution and interpretation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from airs.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureAttribution:
    """Attribution for a single feature."""

    feature_name: str
    value: float
    contribution: float
    direction: str  # "risk_increasing" or "risk_decreasing"
    percentile: float | None = None
    historical_context: str | None = None


class ModelExplainer:
    """
    Explain model predictions using SHAP values.

    Provides feature attribution and interpretation for
    model predictions.
    """

    # Feature descriptions for natural language
    FEATURE_DESCRIPTIONS = {
        # Rate features
        "yield_10y": "10-year Treasury yield",
        "yield_2y": "2-year Treasury yield",
        "yield_curve_slope": "yield curve slope (10Y-2Y)",
        "yield_curve_curvature": "yield curve curvature",
        "curve_inversion_flag": "yield curve inversion",
        "rate_momentum_10d": "10-day rate momentum",
        "rate_vol_21d": "21-day rate volatility",
        # Credit features
        "hy_spread": "high-yield credit spread",
        "ig_spread": "investment-grade spread",
        "hy_ig_diff": "HY-IG spread differential",
        "spread_momentum": "credit spread momentum",
        "credit_stress_flag": "credit stress indicator",
        # Volatility features
        "vix_level": "VIX index level",
        "vix_percentile": "VIX percentile (2Y)",
        "vix_term_structure": "VIX term structure",
        "vix_momentum": "VIX momentum",
        "realized_vol": "realized volatility",
        "vol_risk_premium": "volatility risk premium",
        # Macro features
        "lei_yoy": "Leading Economic Index YoY",
        "pmi_level": "ISM Manufacturing PMI",
        "claims_4wma": "jobless claims (4-week avg)",
        "consumer_sentiment": "consumer sentiment",
        "recession_prob": "recession probability",
        # Cross-asset features
        "equity_bond_corr": "equity-bond correlation",
        "risk_on_off_score": "risk-on/off indicator",
        "cross_asset_momentum": "cross-asset momentum",
        # Regime features
        "regime_high_vol_prob": "high-volatility regime probability",
        "regime_transition_prob": "regime transition probability",
        # Composite features
        "composite_stress_index": "composite stress index",
        "early_warning_score": "early warning score",
    }

    # Thresholds for interpreting feature values
    FEATURE_THRESHOLDS = {
        "vix_level": {"low": 15, "moderate": 20, "high": 25, "extreme": 35},
        "hy_spread": {"low": 300, "moderate": 400, "high": 500, "extreme": 700},
        "yield_curve_slope": {"inverted": 0, "flat": 50, "normal": 100},
        "lei_yoy": {"contraction": 0, "slow": 2, "normal": 4},
    }

    def __init__(self, model: Any = None, feature_names: list[str] | None = None):
        """
        Initialize explainer.

        Args:
            model: Trained model for SHAP
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names or []
        self.explainer = None
        self.background_data = None

    def fit(
        self,
        background_data: pd.DataFrame | np.ndarray,
        model: Any | None = None,
    ) -> "ModelExplainer":
        """
        Fit SHAP explainer on background data.

        Args:
            background_data: Data for SHAP background
            model: Model to explain (optional if set in init)

        Returns:
            Self
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, using fallback attribution")
            return self

        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided for explanation")

        # Store background data
        if isinstance(background_data, pd.DataFrame):
            self.feature_names = list(background_data.columns)
            self.background_data = background_data.values
        else:
            self.background_data = background_data

        # Sample background if too large
        if len(self.background_data) > 100:
            indices = np.random.choice(
                len(self.background_data), 100, replace=False
            )
            background_sample = self.background_data[indices]
        else:
            background_sample = self.background_data

        # Create appropriate explainer based on model type
        model_type = type(self.model).__name__.lower()

        try:
            if "xgb" in model_type or "lgb" in model_type:
                self.explainer = shap.TreeExplainer(self.model)
            elif "randomforest" in model_type or "gradient" in model_type:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba
                    if hasattr(self.model, "predict_proba")
                    else self.model.predict,
                    background_sample,
                )

            logger.info(f"SHAP explainer initialized: {type(self.explainer).__name__}")

        except Exception as e:
            logger.warning(f"Failed to create SHAP explainer: {e}")
            self.explainer = None

        return self

    def explain(
        self,
        features: pd.Series | pd.DataFrame | np.ndarray,
    ) -> list[FeatureAttribution]:
        """
        Explain a single prediction.

        Args:
            features: Feature values for explanation

        Returns:
            List of feature attributions
        """
        # Convert to numpy if needed
        if isinstance(features, pd.Series):
            feature_names = list(features.index)
            feature_values = features.values.reshape(1, -1)
        elif isinstance(features, pd.DataFrame):
            feature_names = list(features.columns)
            feature_values = features.values
        else:
            feature_names = self.feature_names
            feature_values = features.reshape(1, -1) if features.ndim == 1 else features

        # Get SHAP values
        if self.explainer is not None and SHAP_AVAILABLE:
            try:
                shap_values = self.explainer.shap_values(feature_values)

                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # Binary classification - use positive class
                    shap_values = shap_values[1]

                contributions = shap_values[0] if shap_values.ndim > 1 else shap_values

            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
                contributions = self._fallback_attribution(feature_values[0])
        else:
            contributions = self._fallback_attribution(feature_values[0])

        # Create attribution objects
        attributions = []
        for i, (name, contrib) in enumerate(zip(feature_names, contributions)):
            value = feature_values[0, i] if feature_values.ndim > 1 else feature_values[i]

            attribution = FeatureAttribution(
                feature_name=name,
                value=float(value),
                contribution=float(contrib),
                direction="risk_increasing" if contrib > 0 else "risk_decreasing",
                percentile=self._calculate_percentile(name, value),
                historical_context=self._get_historical_context(name, value),
            )
            attributions.append(attribution)

        # Sort by absolute contribution
        attributions.sort(key=lambda x: abs(x.contribution), reverse=True)

        return attributions

    def _fallback_attribution(self, features: np.ndarray) -> np.ndarray:
        """
        Fallback attribution when SHAP is unavailable.

        Uses simple feature importance-weighted z-scores.
        """
        # Simple heuristic: larger absolute values contribute more
        if self.background_data is not None and len(self.background_data) > 0:
            means = np.mean(self.background_data, axis=0)
            stds = np.std(self.background_data, axis=0)
            stds = np.where(stds == 0, 1, stds)
            z_scores = (features - means) / stds
            return z_scores * 0.1  # Scale to reasonable range
        else:
            return features * 0.01

    def _calculate_percentile(self, feature_name: str, value: float) -> float | None:
        """Calculate percentile of feature value vs background."""
        if self.background_data is None or feature_name not in self.feature_names:
            return None

        try:
            idx = self.feature_names.index(feature_name)
            feature_data = self.background_data[:, idx]
            percentile = (feature_data < value).mean() * 100
            return float(percentile)
        except Exception:
            return None

    def _get_historical_context(
        self, feature_name: str, value: float
    ) -> str | None:
        """Get historical context for feature value."""
        if feature_name not in self.FEATURE_THRESHOLDS:
            return None

        thresholds = self.FEATURE_THRESHOLDS[feature_name]

        if feature_name == "vix_level":
            if value < thresholds["low"]:
                return "Low volatility environment"
            elif value < thresholds["moderate"]:
                return "Normal volatility"
            elif value < thresholds["high"]:
                return "Elevated volatility"
            elif value < thresholds["extreme"]:
                return "High volatility - stress regime"
            else:
                return "Extreme volatility - crisis conditions"

        elif feature_name == "hy_spread":
            if value < thresholds["low"]:
                return "Tight spreads - risk-on"
            elif value < thresholds["moderate"]:
                return "Normal credit conditions"
            elif value < thresholds["high"]:
                return "Widening spreads - credit stress"
            else:
                return "Wide spreads - significant credit stress"

        elif feature_name == "yield_curve_slope":
            if value < thresholds["inverted"]:
                return "Inverted yield curve - recession signal"
            elif value < thresholds["flat"]:
                return "Flat yield curve"
            else:
                return "Normal yield curve"

        return None

    def get_top_drivers(
        self,
        attributions: list[FeatureAttribution],
        n: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Get top N drivers with formatted explanations.

        Args:
            attributions: List of feature attributions
            n: Number of top drivers

        Returns:
            List of driver dictionaries
        """
        drivers = []

        for attr in attributions[:n]:
            description = self.FEATURE_DESCRIPTIONS.get(
                attr.feature_name, attr.feature_name.replace("_", " ")
            )

            driver = {
                "feature": attr.feature_name,
                "description": description,
                "value": attr.value,
                "contribution": attr.contribution,
                "direction": attr.direction,
                "impact": "high" if abs(attr.contribution) > 0.1 else "moderate",
            }

            if attr.percentile is not None:
                driver["percentile"] = attr.percentile
                driver["rarity"] = self._get_rarity_label(attr.percentile)

            if attr.historical_context:
                driver["context"] = attr.historical_context

            drivers.append(driver)

        return drivers

    def _get_rarity_label(self, percentile: float) -> str:
        """Get rarity label for percentile."""
        if percentile < 5 or percentile > 95:
            return "rare"
        elif percentile < 10 or percentile > 90:
            return "uncommon"
        elif percentile < 25 or percentile > 75:
            return "somewhat unusual"
        else:
            return "typical"

    def generate_explanation_text(
        self,
        attributions: list[FeatureAttribution],
        prediction: float,
    ) -> str:
        """
        Generate natural language explanation.

        Args:
            attributions: Feature attributions
            prediction: Model prediction probability

        Returns:
            Explanation text
        """
        top_drivers = self.get_top_drivers(attributions, n=3)

        if not top_drivers:
            return "Unable to determine key drivers for this prediction."

        prob_pct = prediction * 100
        text_parts = [
            f"The model predicts a {prob_pct:.0f}% probability of significant drawdown."
        ]

        # Add top driver explanations
        text_parts.append("\nKey factors driving this assessment:")

        for i, driver in enumerate(top_drivers, 1):
            direction = "increasing" if driver["direction"] == "risk_increasing" else "decreasing"
            desc = driver["description"]
            value = driver["value"]

            factor_text = f"{i}. {desc}: {value:.2f}"

            if "context" in driver:
                factor_text += f" ({driver['context']})"

            if "percentile" in driver:
                factor_text += f" - {driver['percentile']:.0f}th percentile"

            factor_text += f" - {direction} risk"

            text_parts.append(factor_text)

        return "\n".join(text_parts)

    def find_similar_historical_periods(
        self,
        current_features: pd.Series,
        historical_features: pd.DataFrame,
        n_similar: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Find historical periods with similar feature profiles.

        Args:
            current_features: Current feature values
            historical_features: Historical feature data
            n_similar: Number of similar periods to return

        Returns:
            List of similar historical periods
        """
        # Normalize features
        means = historical_features.mean()
        stds = historical_features.std()
        stds = stds.replace(0, 1)

        current_normalized = (current_features - means) / stds
        historical_normalized = (historical_features - means) / stds

        # Calculate distances
        distances = np.sqrt(
            ((historical_normalized - current_normalized) ** 2).sum(axis=1)
        )

        # Get top N similar periods
        similar_indices = distances.nsmallest(n_similar).index

        similar_periods = []
        for idx in similar_indices:
            period_info = {
                "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                "distance": float(distances[idx]),
                "similarity_score": float(1 / (1 + distances[idx])),
            }
            similar_periods.append(period_info)

        return similar_periods
