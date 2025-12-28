"""
Credit spread features.

Generates features from investment grade and high yield credit spreads.
"""

import numpy as np
import pandas as pd

from airs.features.base import FeatureGenerator
from airs.utils.logging import get_logger

logger = get_logger(__name__)


class CreditFeatures(FeatureGenerator):
    """
    Credit spread feature generator.

    Generates features including:
    - Spread levels and percentiles
    - Spread momentum and velocity
    - HY-IG differential
    - Credit stress indicators
    """

    # Stress thresholds (in percentage points)
    HY_STRESS_THRESHOLD = 5.0  # 500 bps
    IG_STRESS_THRESHOLD = 2.0  # 200 bps

    @property
    def feature_group(self) -> str:
        return "credit"

    @property
    def feature_names(self) -> list[str]:
        return [
            # Spread levels
            "ig_spread",
            "hy_spread",
            "ig_spread_pctl",
            "hy_spread_pctl",
            "ig_spread_zscore",
            "hy_spread_zscore",
            # Differential
            "hy_ig_diff",
            "hy_ig_ratio",
            # Momentum
            "ig_spread_chg_5d",
            "ig_spread_chg_10d",
            "ig_spread_chg_21d",
            "hy_spread_chg_5d",
            "hy_spread_chg_10d",
            "hy_spread_chg_21d",
            # Velocity
            "hy_spread_velocity",
            "ig_spread_velocity",
            # Stress indicators
            "hy_stress_flag",
            "ig_stress_flag",
            "credit_widening_flag",
            "credit_2sigma_flag",
            # Volatility
            "hy_spread_vol_21d",
            "credit_vol_zscore",
        ]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate credit features.

        Args:
            data: DataFrame with columns:
                - BAMLC0A0CM (IG spread)
                - BAMLH0A0HYM2 (HY spread)

        Returns:
            DataFrame with credit features
        """
        features = pd.DataFrame(index=data.index)

        # Check for required columns
        has_ig = "BAMLC0A0CM" in data.columns
        has_hy = "BAMLH0A0HYM2" in data.columns

        if not has_ig and not has_hy:
            logger.warning("No credit spread data available")
            return features

        # Investment Grade features
        if has_ig:
            ig_spread = data["BAMLC0A0CM"]
            features["ig_spread"] = ig_spread

            # Percentile and z-score
            features["ig_spread_pctl"] = self.add_percentile(ig_spread)
            features["ig_spread_zscore"] = self.add_zscore(ig_spread)

            # Momentum
            features["ig_spread_chg_5d"] = ig_spread.diff(5)
            features["ig_spread_chg_10d"] = ig_spread.diff(10)
            features["ig_spread_chg_21d"] = ig_spread.diff(21)

            # Velocity (rate of change)
            features["ig_spread_velocity"] = ig_spread.diff(5) / 5

            # Stress flag
            features["ig_stress_flag"] = (ig_spread > self.IG_STRESS_THRESHOLD).astype(
                int
            )

        # High Yield features
        if has_hy:
            hy_spread = data["BAMLH0A0HYM2"]
            features["hy_spread"] = hy_spread

            # Percentile and z-score
            features["hy_spread_pctl"] = self.add_percentile(hy_spread)
            features["hy_spread_zscore"] = self.add_zscore(hy_spread)

            # Momentum
            features["hy_spread_chg_5d"] = hy_spread.diff(5)
            features["hy_spread_chg_10d"] = hy_spread.diff(10)
            features["hy_spread_chg_21d"] = hy_spread.diff(21)

            # Velocity
            features["hy_spread_velocity"] = hy_spread.diff(5) / 5

            # Volatility
            features["hy_spread_vol_21d"] = hy_spread.diff().rolling(21).std()
            features["credit_vol_zscore"] = self.add_zscore(
                features["hy_spread_vol_21d"], suffix=""
            )

            # Stress flag
            features["hy_stress_flag"] = (hy_spread > self.HY_STRESS_THRESHOLD).astype(
                int
            )

        # Combined features
        if has_ig and has_hy:
            ig_spread = data["BAMLC0A0CM"]
            hy_spread = data["BAMLH0A0HYM2"]

            # HY-IG differential
            features["hy_ig_diff"] = hy_spread - ig_spread
            features["hy_ig_ratio"] = hy_spread / ig_spread.replace(0, np.nan)

            # Credit widening flag (both widening)
            ig_widening = features["ig_spread_chg_5d"] > 0
            hy_widening = features["hy_spread_chg_5d"] > 0
            features["credit_widening_flag"] = (ig_widening & hy_widening).astype(int)

            # 2-sigma widening flag
            hy_zscore = features["hy_spread_zscore"]
            features["credit_2sigma_flag"] = (hy_zscore > 2).astype(int)

        self.log_features_generated(features)
        return features


class CreditQualityFeatures(FeatureGenerator):
    """
    Additional credit quality features.

    Includes CDS basis, credit default indicators, and relative value.
    """

    @property
    def feature_group(self) -> str:
        return "credit_quality"

    @property
    def feature_names(self) -> list[str]:
        return [
            "hy_vix_ratio",
            "credit_risk_premium",
            "hy_momentum_score",
        ]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate credit quality features.

        Args:
            data: DataFrame with credit spreads and VIX

        Returns:
            DataFrame with credit quality features
        """
        features = pd.DataFrame(index=data.index)

        # HY spread / VIX ratio (credit vs equity vol)
        if "BAMLH0A0HYM2" in data.columns:
            hy_spread = data["BAMLH0A0HYM2"]

            # Look for VIX in various column names
            vix_cols = ["VIXCLS", "^VIX", "VIX"]
            vix_col = next((c for c in vix_cols if c in data.columns), None)

            if vix_col:
                vix = data[vix_col]
                # Convert HY spread to bps if in percentage, normalize
                features["hy_vix_ratio"] = (hy_spread * 100) / vix.replace(0, np.nan)

            # Credit risk premium (HY spread minus expected defaults proxy)
            # Simplified: use deviation from rolling mean
            hy_mean = hy_spread.rolling(252).mean()
            features["credit_risk_premium"] = hy_spread - hy_mean

            # Momentum score (composite)
            chg_5d = hy_spread.diff(5)
            chg_10d = hy_spread.diff(10)
            chg_21d = hy_spread.diff(21)

            # Normalize changes
            chg_5d_z = (chg_5d - chg_5d.rolling(252).mean()) / chg_5d.rolling(252).std()
            chg_10d_z = (chg_10d - chg_10d.rolling(252).mean()) / chg_10d.rolling(
                252
            ).std()
            chg_21d_z = (chg_21d - chg_21d.rolling(252).mean()) / chg_21d.rolling(
                252
            ).std()

            # Weighted momentum score (more weight on recent)
            features["hy_momentum_score"] = (
                0.5 * chg_5d_z + 0.3 * chg_10d_z + 0.2 * chg_21d_z
            )

        self.log_features_generated(features)
        return features
