"""
Interest rate features.

Generates features from Treasury yields, yield curve, and rate volatility.
"""

import numpy as np
import pandas as pd

from airs.features.base import FeatureGenerator
from airs.utils.logging import get_logger

logger = get_logger(__name__)


class RateFeatures(FeatureGenerator):
    """
    Interest rate feature generator.

    Generates features including:
    - Yield curve level, slope, curvature
    - Yield changes and momentum
    - Rate volatility
    - Curve inversion indicators
    """

    @property
    def feature_group(self) -> str:
        return "rates"

    @property
    def feature_names(self) -> list[str]:
        return [
            # Yield curve shape
            "yc_level",
            "yc_slope_10y2y",
            "yc_slope_30y5y",
            "yc_slope_10y3m",
            "yc_curvature",
            # Inversion
            "yc_inverted_10y2y",
            "yc_inverted_10y3m",
            "yc_inversion_depth",
            "yc_inversion_duration",
            # Momentum
            "rate_10y_chg_5d",
            "rate_10y_chg_10d",
            "rate_10y_chg_21d",
            "rate_10y_zscore",
            "rate_10y_pctl",
            # Rate change acceleration
            "rate_10y_accel",
            # Volatility
            "rate_vol_21d",
            "rate_vol_63d",
            "rate_vol_zscore",
            # Term premium proxy
            "term_premium_proxy",
        ]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate rate features.

        Args:
            data: DataFrame with columns:
                - DGS2, DGS5, DGS10, DGS30 (Treasury yields)
                - DTB3 (3-month T-bill)
                - T10Y2Y, T10Y3M (spreads, optional)

        Returns:
            DataFrame with rate features
        """
        features = pd.DataFrame(index=data.index)

        # Get available yield columns
        yield_cols = ["DTB3", "DGS2", "DGS5", "DGS10", "DGS30"]
        available_yields = [c for c in yield_cols if c in data.columns]

        if len(available_yields) < 3:
            logger.warning(
                f"Limited yield data available: {available_yields}. "
                "Some features will be NaN."
            )

        # Yield curve level (average of available yields)
        if available_yields:
            features["yc_level"] = data[available_yields].mean(axis=1)

        # Yield curve slope (10Y - 2Y)
        if "DGS10" in data.columns and "DGS2" in data.columns:
            features["yc_slope_10y2y"] = data["DGS10"] - data["DGS2"]
        elif "T10Y2Y" in data.columns:
            features["yc_slope_10y2y"] = data["T10Y2Y"]

        # Alternative slopes
        if "DGS30" in data.columns and "DGS5" in data.columns:
            features["yc_slope_30y5y"] = data["DGS30"] - data["DGS5"]

        if "DGS10" in data.columns and "DTB3" in data.columns:
            features["yc_slope_10y3m"] = data["DGS10"] - data["DTB3"]
        elif "T10Y3M" in data.columns:
            features["yc_slope_10y3m"] = data["T10Y3M"]

        # Yield curve curvature: 2*5Y - (2Y + 10Y)
        if all(c in data.columns for c in ["DGS2", "DGS5", "DGS10"]):
            features["yc_curvature"] = (
                2 * data["DGS5"] - (data["DGS2"] + data["DGS10"])
            )

        # Inversion flags
        if "yc_slope_10y2y" in features.columns:
            features["yc_inverted_10y2y"] = (features["yc_slope_10y2y"] < 0).astype(int)
            features["yc_inversion_depth"] = features["yc_slope_10y2y"].clip(upper=0)

            # Inversion duration (consecutive days inverted)
            inverted = features["yc_inverted_10y2y"]
            features["yc_inversion_duration"] = (
                inverted.groupby((inverted != inverted.shift()).cumsum()).cumcount() + 1
            ) * inverted

        if "yc_slope_10y3m" in features.columns:
            features["yc_inverted_10y3m"] = (features["yc_slope_10y3m"] < 0).astype(int)

        # 10Y yield momentum
        if "DGS10" in data.columns:
            rate_10y = data["DGS10"]

            # Changes
            features["rate_10y_chg_5d"] = rate_10y.diff(5)
            features["rate_10y_chg_10d"] = rate_10y.diff(10)
            features["rate_10y_chg_21d"] = rate_10y.diff(21)

            # Z-score and percentile
            features["rate_10y_zscore"] = self.add_zscore(rate_10y)
            features["rate_10y_pctl"] = self.add_percentile(rate_10y)

            # Acceleration (second derivative)
            features["rate_10y_accel"] = features["rate_10y_chg_5d"].diff(5)

            # Rate volatility
            rate_changes = rate_10y.diff()
            features["rate_vol_21d"] = rate_changes.rolling(21).std()
            features["rate_vol_63d"] = rate_changes.rolling(63).std()
            features["rate_vol_zscore"] = self.add_zscore(
                features["rate_vol_21d"], suffix=""
            )

        # Term premium proxy (10Y - 3M slope minus expected rate changes)
        # Simplified: just use the slope as proxy
        if "yc_slope_10y3m" in features.columns:
            features["term_premium_proxy"] = features["yc_slope_10y3m"]

        self.log_features_generated(features)
        return features


class GlobalRateFeatures(FeatureGenerator):
    """
    Global interest rate features for policy divergence analysis.

    Requires additional data: German Bunds, UK Gilts, JGB yields.
    """

    @property
    def feature_group(self) -> str:
        return "global_rates"

    @property
    def feature_names(self) -> list[str]:
        return [
            "us_de_spread",
            "us_uk_spread",
            "us_jp_spread",
            "global_rate_divergence",
            "global_rate_correlation",
        ]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate global rate divergence features.

        Args:
            data: DataFrame with US and foreign yields

        Returns:
            DataFrame with global rate features
        """
        features = pd.DataFrame(index=data.index)

        # US 10Y is required
        if "DGS10" not in data.columns:
            logger.warning("DGS10 not available, skipping global rate features")
            return features

        us_10y = data["DGS10"]

        # US-Germany spread
        if "DE10Y" in data.columns:
            features["us_de_spread"] = us_10y - data["DE10Y"]

        # US-UK spread
        if "UK10Y" in data.columns:
            features["us_uk_spread"] = us_10y - data["UK10Y"]

        # US-Japan spread
        if "JP10Y" in data.columns:
            features["us_jp_spread"] = us_10y - data["JP10Y"]

        # Global divergence (average spread from US)
        spread_cols = [c for c in features.columns if c.startswith("us_")]
        if spread_cols:
            features["global_rate_divergence"] = features[spread_cols].abs().mean(axis=1)

        # Rolling correlation with foreign rates
        foreign_cols = ["DE10Y", "UK10Y", "JP10Y"]
        available_foreign = [c for c in foreign_cols if c in data.columns]

        if available_foreign:
            correlations = []
            for col in available_foreign:
                corr = us_10y.rolling(63).corr(data[col])
                correlations.append(corr)

            features["global_rate_correlation"] = (
                pd.concat(correlations, axis=1).mean(axis=1)
            )

        self.log_features_generated(features)
        return features
