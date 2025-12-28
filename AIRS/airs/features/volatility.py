"""
Volatility features.

Generates features from VIX, realized volatility, and volatility term structure.
"""

import numpy as np
import pandas as pd

from airs.features.base import FeatureGenerator
from airs.utils.logging import get_logger
from airs.utils.stats import calculate_volatility

logger = get_logger(__name__)


class VolatilityFeatures(FeatureGenerator):
    """
    Volatility feature generator.

    Generates features including:
    - VIX level and percentile
    - VIX term structure
    - Realized volatility
    - Implied vs realized spread
    - Volatility of volatility
    """

    @property
    def feature_group(self) -> str:
        return "volatility"

    @property
    def feature_names(self) -> list[str]:
        return [
            # VIX level
            "vix_level",
            "vix_pctl",
            "vix_zscore",
            # VIX momentum
            "vix_chg_5d",
            "vix_chg_10d",
            "vix_chg_21d",
            "vix_chg_pct_5d",
            # Term structure
            "vix_term_structure",
            "vix_contango_flag",
            # Realized vol
            "realized_vol_5d",
            "realized_vol_21d",
            "realized_vol_63d",
            # Implied vs realized
            "impl_real_spread",
            "impl_real_ratio",
            "vol_risk_premium",
            # VVIX (vol of vol)
            "vix_vol_21d",
            "vvix_zscore",
            # Regime indicators
            "vix_above_20",
            "vix_above_30",
            "vix_spike",
        ]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volatility features.

        Args:
            data: DataFrame with columns:
                - VIXCLS or ^VIX (VIX index)
                - SPY close prices (for realized vol)
                - VIX3M, VIX6M (optional, for term structure)

        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=data.index)

        # Find VIX column
        vix_cols = ["VIXCLS", "^VIX", "VIX"]
        vix_col = next((c for c in vix_cols if c in data.columns), None)

        if vix_col is None:
            logger.warning("No VIX data available")
            return features

        vix = data[vix_col]

        # VIX level features
        features["vix_level"] = vix
        features["vix_pctl"] = self.add_percentile(vix)
        features["vix_zscore"] = self.add_zscore(vix)

        # VIX momentum
        features["vix_chg_5d"] = vix.diff(5)
        features["vix_chg_10d"] = vix.diff(10)
        features["vix_chg_21d"] = vix.diff(21)
        features["vix_chg_pct_5d"] = vix.pct_change(5) * 100

        # VIX term structure (VIX - VIX3M)
        if "VIX3M" in data.columns:
            vix3m = data["VIX3M"]
            features["vix_term_structure"] = vix - vix3m
            # Contango (positive when VIX < VIX3M)
            features["vix_contango_flag"] = (features["vix_term_structure"] < 0).astype(
                int
            )
        elif "VXV" in data.columns:  # 3-month VIX
            features["vix_term_structure"] = vix - data["VXV"]
            features["vix_contango_flag"] = (features["vix_term_structure"] < 0).astype(
                int
            )

        # Realized volatility from SPY
        spy_cols = ["SPY", "prices_SPY", "close_SPY"]
        spy_col = next((c for c in spy_cols if c in data.columns), None)

        if spy_col:
            spy_returns = data[spy_col].pct_change()

            features["realized_vol_5d"] = calculate_volatility(
                spy_returns, window=5, annualize=True
            )
            features["realized_vol_21d"] = calculate_volatility(
                spy_returns, window=21, annualize=True
            )
            features["realized_vol_63d"] = calculate_volatility(
                spy_returns, window=63, annualize=True
            )

            # Implied vs Realized spread
            realized_21d = features["realized_vol_21d"]
            features["impl_real_spread"] = vix - realized_21d
            features["impl_real_ratio"] = vix / realized_21d.replace(0, np.nan)

            # Volatility risk premium (VIX - future realized vol)
            # Approximated with current realized vol
            features["vol_risk_premium"] = features["impl_real_spread"]

        # VVIX (volatility of volatility)
        vix_returns = vix.pct_change()
        features["vix_vol_21d"] = vix_returns.rolling(21).std() * np.sqrt(252) * 100
        features["vvix_zscore"] = self.add_zscore(features["vix_vol_21d"], suffix="")

        # Regime indicators
        features["vix_above_20"] = (vix > 20).astype(int)
        features["vix_above_30"] = (vix > 30).astype(int)

        # VIX spike (>20% increase in 5 days)
        features["vix_spike"] = (features["vix_chg_pct_5d"] > 20).astype(int)

        self.log_features_generated(features)
        return features


class RateVolatilityFeatures(FeatureGenerator):
    """
    Interest rate volatility features (MOVE index and related).
    """

    @property
    def feature_group(self) -> str:
        return "rate_volatility"

    @property
    def feature_names(self) -> list[str]:
        return [
            "move_level",
            "move_pctl",
            "move_zscore",
            "move_chg_5d",
            "move_vix_ratio",
            "rate_vol_regime",
        ]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate rate volatility features.

        Note: MOVE index requires specialized data source.
        Falls back to realized rate volatility if unavailable.

        Args:
            data: DataFrame with MOVE index or Treasury yields

        Returns:
            DataFrame with rate volatility features
        """
        features = pd.DataFrame(index=data.index)

        # Check for MOVE index
        if "MOVE" in data.columns:
            move = data["MOVE"]
            features["move_level"] = move
            features["move_pctl"] = self.add_percentile(move)
            features["move_zscore"] = self.add_zscore(move)
            features["move_chg_5d"] = move.diff(5)

            # MOVE/VIX ratio
            vix_cols = ["VIXCLS", "^VIX", "VIX"]
            vix_col = next((c for c in vix_cols if c in data.columns), None)
            if vix_col:
                features["move_vix_ratio"] = move / data[vix_col].replace(0, np.nan)

            # Rate vol regime
            move_pctl = features["move_pctl"]
            features["rate_vol_regime"] = pd.cut(
                move_pctl,
                bins=[0, 25, 50, 75, 100],
                labels=[0, 1, 2, 3],  # Low, Normal, Elevated, High
            ).astype(float)

        else:
            # Fallback: compute realized rate volatility from 10Y yield
            if "DGS10" in data.columns:
                rate_10y = data["DGS10"]
                rate_changes = rate_10y.diff()

                # Annualized rate volatility
                rate_vol = rate_changes.rolling(21).std() * np.sqrt(252)
                features["move_level"] = rate_vol * 100  # Scale to MOVE-like range
                features["move_pctl"] = self.add_percentile(features["move_level"])
                features["move_zscore"] = self.add_zscore(features["move_level"])
                features["move_chg_5d"] = features["move_level"].diff(5)

        self.log_features_generated(features)
        return features
