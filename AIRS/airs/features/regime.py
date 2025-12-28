"""
Regime detection module.

Detects market regimes using HMM and threshold-based methods.
"""

import numpy as np
import pandas as pd
from typing import Literal

from airs.features.base import FeatureGenerator
from airs.utils.logging import get_logger

logger = get_logger(__name__)


class RegimeDetector(FeatureGenerator):
    """
    Market regime detector.

    Detects regimes using:
    - Hidden Markov Models (HMM)
    - Threshold-based rules
    - Volatility clustering

    Regimes:
    - 0: Low volatility / Risk-on
    - 1: Normal / Neutral
    - 2: High volatility / Risk-off
    - 3: Crisis / Extreme stress (optional)
    """

    def __init__(
        self,
        n_regimes: int = 3,
        method: Literal["hmm", "threshold", "combined"] = "combined",
        lookback_window: int = 252,
    ):
        """
        Initialize regime detector.

        Args:
            n_regimes: Number of regimes to detect
            method: Detection method
            lookback_window: Lookback for calculations
        """
        super().__init__(lookback_window=lookback_window)
        self.n_regimes = n_regimes
        self.method = method

        # VIX thresholds for threshold-based detection
        self.vix_thresholds = {
            "low": 15,
            "normal": 25,
            "elevated": 35,
        }

        # Credit spread thresholds (HY OAS in percentage points)
        self.credit_thresholds = {
            "low": 3.0,
            "normal": 5.0,
            "elevated": 7.0,
        }

    @property
    def feature_group(self) -> str:
        return "regime"

    @property
    def feature_names(self) -> list[str]:
        return [
            "regime",
            "regime_prob_low",
            "regime_prob_normal",
            "regime_prob_high",
            "regime_transition",
            "regime_duration",
            "regime_vol",
            "regime_credit",
        ]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate regime features.

        Args:
            data: DataFrame with VIX, credit spreads, and returns

        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=data.index)

        if self.method == "hmm":
            features = self._detect_hmm(data, features)
        elif self.method == "threshold":
            features = self._detect_threshold(data, features)
        else:  # combined
            hmm_features = self._detect_hmm(data, pd.DataFrame(index=data.index))
            threshold_features = self._detect_threshold(
                data, pd.DataFrame(index=data.index)
            )

            # Combine: use HMM if available, fallback to threshold
            if "regime" in hmm_features.columns and not hmm_features["regime"].isna().all():
                features = hmm_features
            else:
                features = threshold_features

        # Add derived features
        if "regime" in features.columns:
            features = self._add_derived_features(features)

        self.log_features_generated(features)
        return features

    def _detect_hmm(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Detect regimes using Hidden Markov Model."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed, falling back to threshold method")
            return self._detect_threshold(data, features)

        # Prepare observation data
        obs_cols = []

        # VIX
        vix_cols = ["VIXCLS", "^VIX", "VIX", "vix_level"]
        vix_col = next((c for c in vix_cols if c in data.columns), None)
        if vix_col:
            obs_cols.append(vix_col)

        # Credit spread
        credit_cols = ["BAMLH0A0HYM2", "hy_spread"]
        credit_col = next((c for c in credit_cols if c in data.columns), None)
        if credit_col:
            obs_cols.append(credit_col)

        # Realized volatility
        vol_cols = ["realized_vol_21d"]
        vol_col = next((c for c in vol_cols if c in data.columns), None)
        if vol_col:
            obs_cols.append(vol_col)

        if len(obs_cols) < 1:
            logger.warning("Insufficient data for HMM, using threshold method")
            return self._detect_threshold(data, features)

        # Prepare observations
        observations = data[obs_cols].dropna()

        if len(observations) < 100:
            logger.warning("Insufficient observations for HMM")
            return self._detect_threshold(data, features)

        # Standardize
        obs_mean = observations.mean()
        obs_std = observations.std()
        obs_normalized = (observations - obs_mean) / obs_std

        try:
            # Fit HMM
            model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42,
            )
            model.fit(obs_normalized.values)

            # Predict regimes
            regimes = model.predict(obs_normalized.values)

            # Get regime probabilities
            probs = model.predict_proba(obs_normalized.values)

            # Create features aligned with original index
            regime_series = pd.Series(index=observations.index, data=regimes)
            features["regime"] = regime_series.reindex(data.index)

            # Sort regimes by volatility (0 = low vol, 2 = high vol)
            regime_vols = {}
            for r in range(self.n_regimes):
                mask = regimes == r
                if mask.sum() > 0 and vix_col:
                    regime_vols[r] = observations.loc[observations.index[mask], vix_col].mean()

            if regime_vols:
                sorted_regimes = sorted(regime_vols.keys(), key=lambda x: regime_vols[x])
                regime_map = {old: new for new, old in enumerate(sorted_regimes)}
                features["regime"] = features["regime"].map(regime_map)

            # Regime probabilities
            prob_df = pd.DataFrame(
                probs,
                index=observations.index,
                columns=[f"regime_prob_{i}" for i in range(self.n_regimes)],
            )

            if self.n_regimes >= 3:
                features["regime_prob_low"] = prob_df.iloc[:, 0].reindex(data.index)
                features["regime_prob_normal"] = prob_df.iloc[:, 1].reindex(data.index)
                features["regime_prob_high"] = prob_df.iloc[:, 2].reindex(data.index)

        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return self._detect_threshold(data, features)

        return features

    def _detect_threshold(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Detect regimes using simple thresholds."""

        # Initialize with neutral regime
        features["regime"] = 1  # Normal

        # VIX-based regime
        vix_cols = ["VIXCLS", "^VIX", "VIX", "vix_level"]
        vix_col = next((c for c in vix_cols if c in data.columns), None)

        if vix_col:
            vix = data[vix_col]
            features["regime_vol"] = pd.cut(
                vix,
                bins=[-np.inf, self.vix_thresholds["low"],
                      self.vix_thresholds["normal"],
                      self.vix_thresholds["elevated"], np.inf],
                labels=[0, 1, 2, 3],
            ).astype(float)

            # Update main regime
            features.loc[vix <= self.vix_thresholds["low"], "regime"] = 0
            features.loc[vix > self.vix_thresholds["normal"], "regime"] = 2
            features.loc[vix > self.vix_thresholds["elevated"], "regime"] = 3

        # Credit-based regime adjustment
        credit_cols = ["BAMLH0A0HYM2", "hy_spread"]
        credit_col = next((c for c in credit_cols if c in data.columns), None)

        if credit_col:
            credit = data[credit_col]
            features["regime_credit"] = pd.cut(
                credit,
                bins=[-np.inf, self.credit_thresholds["low"],
                      self.credit_thresholds["normal"],
                      self.credit_thresholds["elevated"], np.inf],
                labels=[0, 1, 2, 3],
            ).astype(float)

            # Credit stress overrides to higher regime
            features.loc[credit > self.credit_thresholds["elevated"], "regime"] = 3
            features.loc[
                (credit > self.credit_thresholds["normal"]) &
                (features["regime"] < 2),
                "regime"
            ] = 2

        # Clip to n_regimes
        features["regime"] = features["regime"].clip(0, self.n_regimes - 1)

        # Simple probability approximation
        if vix_col:
            vix = data[vix_col]
            vix_pctl = vix.rolling(252).apply(
                lambda x: (x.iloc[-1] <= x).mean() * 100 if len(x) > 0 else 50
            )

            features["regime_prob_low"] = (100 - vix_pctl) / 100
            features["regime_prob_high"] = vix_pctl / 100
            features["regime_prob_normal"] = 1 - features["regime_prob_low"].abs() - features["regime_prob_high"].abs()
            features["regime_prob_normal"] = features["regime_prob_normal"].clip(0, 1)

        return features

    def _add_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add derived regime features."""

        if "regime" not in features.columns:
            return features

        regime = features["regime"]

        # Regime transition (change from previous day)
        features["regime_transition"] = (regime != regime.shift(1)).astype(int)

        # Regime duration (consecutive days in current regime)
        regime_changes = regime != regime.shift(1)
        regime_groups = regime_changes.cumsum()
        features["regime_duration"] = regime.groupby(regime_groups).cumcount() + 1

        return features


class VolatilityRegimeDetector(FeatureGenerator):
    """
    Specialized volatility regime detector using GARCH-based approach.
    """

    @property
    def feature_group(self) -> str:
        return "vol_regime"

    @property
    def feature_names(self) -> list[str]:
        return [
            "vol_regime",
            "vol_persistence",
            "vol_clustering",
        ]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility regime features."""
        features = pd.DataFrame(index=data.index)

        # Find returns data
        ret_cols = [c for c in data.columns if "ret" in c.lower() or "spy" in c.lower()]
        if not ret_cols:
            logger.warning("No return data for volatility regime detection")
            return features

        returns = data[ret_cols[0]]
        if returns.abs().mean() > 0.1:
            returns = returns.pct_change()

        # Rolling volatility
        vol_21d = returns.rolling(21).std() * np.sqrt(252)
        vol_63d = returns.rolling(63).std() * np.sqrt(252)

        # Volatility regime based on percentile
        vol_pctl = vol_21d.rolling(252).apply(
            lambda x: (x.iloc[-1] <= x).mean() * 100 if len(x) > 0 else 50
        )

        features["vol_regime"] = pd.cut(
            vol_pctl,
            bins=[0, 25, 75, 100],
            labels=[0, 1, 2],  # Low, Normal, High
        ).astype(float)

        # Volatility persistence (autocorrelation of squared returns)
        sq_returns = returns ** 2
        features["vol_persistence"] = sq_returns.rolling(63).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 10 else np.nan
        )

        # Volatility clustering (ratio of short to long vol)
        features["vol_clustering"] = vol_21d / vol_63d.replace(0, np.nan)

        self.log_features_generated(features)
        return features
