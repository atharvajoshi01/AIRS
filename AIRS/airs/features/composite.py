"""
Composite and aggregate stress features.

Combines multiple indicators into unified stress scores.
"""

import numpy as np
import pandas as pd

from airs.features.base import FeatureGenerator
from airs.utils.logging import get_logger

logger = get_logger(__name__)


class CompositeFeatures(FeatureGenerator):
    """
    Composite stress indicator generator.

    Generates aggregate indicators:
    - Overall stress index (0-100)
    - Early warning composite
    - Multi-factor risk score
    """

    @property
    def feature_group(self) -> str:
        return "composite"

    @property
    def feature_names(self) -> list[str]:
        return [
            # Stress indices
            "stress_index",
            "stress_index_zscore",
            "stress_regime",
            # Early warning
            "early_warning_score",
            "alert_level",
            # Component scores
            "vol_stress_score",
            "credit_stress_score",
            "rate_stress_score",
            "macro_stress_score",
            "cross_asset_stress_score",
            # Trend
            "stress_momentum_5d",
            "stress_momentum_10d",
            # Historical comparison
            "stress_pctl_1y",
            "stress_pctl_3y",
        ]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate composite features.

        Args:
            data: DataFrame with individual feature groups already computed

        Returns:
            DataFrame with composite features
        """
        features = pd.DataFrame(index=data.index)

        # Calculate component stress scores
        features["vol_stress_score"] = self._calculate_vol_stress(data)
        features["credit_stress_score"] = self._calculate_credit_stress(data)
        features["rate_stress_score"] = self._calculate_rate_stress(data)
        features["macro_stress_score"] = self._calculate_macro_stress(data)
        features["cross_asset_stress_score"] = self._calculate_cross_asset_stress(data)

        # Overall stress index (weighted average)
        features["stress_index"] = self._calculate_stress_index(features)

        # Stress index z-score
        features["stress_index_zscore"] = self.add_zscore(
            features["stress_index"], window=252
        )

        # Stress regime (0=low, 1=normal, 2=elevated, 3=high)
        features["stress_regime"] = pd.cut(
            features["stress_index"],
            bins=[0, 25, 50, 75, 100],
            labels=[0, 1, 2, 3],
        ).astype(float)

        # Early warning score (forward-looking composite)
        features["early_warning_score"] = self._calculate_early_warning(features, data)

        # Alert level
        features["alert_level"] = self._determine_alert_level(features)

        # Stress momentum
        stress = features["stress_index"]
        features["stress_momentum_5d"] = stress.diff(5)
        features["stress_momentum_10d"] = stress.diff(10)

        # Historical percentiles
        features["stress_pctl_1y"] = self.add_percentile(stress, window=252)
        features["stress_pctl_3y"] = self.add_percentile(stress, window=756)

        self.log_features_generated(features)
        return features

    def _calculate_vol_stress(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volatility stress component (0-100)."""
        score = pd.Series(50.0, index=data.index)
        contributions = 0

        # VIX percentile
        vix_pctl_cols = ["vix_pctl", "volatility_vix_pctl"]
        vix_pctl_col = next((c for c in vix_pctl_cols if c in data.columns), None)
        if vix_pctl_col:
            score = data[vix_pctl_col].fillna(50)
            contributions += 1

        # VIX z-score adjustment
        vix_z_cols = ["vix_zscore", "volatility_vix_zscore"]
        vix_z_col = next((c for c in vix_z_cols if c in data.columns), None)
        if vix_z_col and contributions > 0:
            vix_z = data[vix_z_col].fillna(0)
            score += vix_z * 10  # Adjust based on z-score
            contributions += 1

        # VIX term structure (backwardation = stress)
        term_cols = ["vix_term_structure", "volatility_vix_term_structure"]
        term_col = next((c for c in term_cols if c in data.columns), None)
        if term_col:
            term = data[term_col].fillna(0)
            # Positive term structure (backwardation) adds to stress
            score += (term > 0).astype(float) * 10
            contributions += 1

        return score.clip(0, 100)

    def _calculate_credit_stress(self, data: pd.DataFrame) -> pd.Series:
        """Calculate credit stress component (0-100)."""
        score = pd.Series(50.0, index=data.index)

        # HY spread percentile
        hy_pctl_cols = ["hy_spread_pctl", "credit_hy_spread_pctl"]
        hy_pctl_col = next((c for c in hy_pctl_cols if c in data.columns), None)
        if hy_pctl_col:
            score = data[hy_pctl_col].fillna(50)

        # Credit widening flag adds stress
        widening_cols = ["credit_widening_flag", "credit_credit_widening_flag"]
        widening_col = next((c for c in widening_cols if c in data.columns), None)
        if widening_col:
            widening = data[widening_col].fillna(0)
            score += widening * 15

        # HY stress flag
        stress_cols = ["hy_stress_flag", "credit_hy_stress_flag"]
        stress_col = next((c for c in stress_cols if c in data.columns), None)
        if stress_col:
            stress_flag = data[stress_col].fillna(0)
            score += stress_flag * 20

        return score.clip(0, 100)

    def _calculate_rate_stress(self, data: pd.DataFrame) -> pd.Series:
        """Calculate interest rate stress component (0-100)."""
        score = pd.Series(50.0, index=data.index)

        # Rate volatility z-score
        rate_vol_cols = ["rate_vol_zscore", "rates_rate_vol_zscore"]
        rate_vol_col = next((c for c in rate_vol_cols if c in data.columns), None)
        if rate_vol_col:
            rate_vol_z = data[rate_vol_col].fillna(0)
            score += rate_vol_z * 15

        # Curve inversion (adds stress, especially if prolonged)
        inversion_cols = ["yc_inverted_10y2y", "rates_yc_inverted_10y2y"]
        inversion_col = next((c for c in inversion_cols if c in data.columns), None)
        if inversion_col:
            inverted = data[inversion_col].fillna(0)
            score += inverted * 15

        # Inversion duration
        duration_cols = ["yc_inversion_duration", "rates_yc_inversion_duration"]
        duration_col = next((c for c in duration_cols if c in data.columns), None)
        if duration_col:
            duration = data[duration_col].fillna(0)
            # Longer inversion = more stress
            score += (duration > 20).astype(float) * 10

        return score.clip(0, 100)

    def _calculate_macro_stress(self, data: pd.DataFrame) -> pd.Series:
        """Calculate macro stress component (0-100)."""
        score = pd.Series(50.0, index=data.index)

        # NFCI
        nfci_cols = ["nfci", "macro_nfci"]
        nfci_col = next((c for c in nfci_cols if c in data.columns), None)
        if nfci_col:
            nfci = data[nfci_col].fillna(0)
            # Positive NFCI = tighter conditions = stress
            score += nfci * 20

        # LEI momentum (negative = stress)
        lei_cols = ["lei_mom_6m", "macro_lei_mom_6m"]
        lei_col = next((c for c in lei_cols if c in data.columns), None)
        if lei_col:
            lei_mom = data[lei_col].fillna(0)
            # Negative momentum increases stress
            score += (lei_mom < 0).astype(float) * 15

        # Recession probability
        prob_cols = ["recession_prob", "macro_recession_prob"]
        prob_col = next((c for c in prob_cols if c in data.columns), None)
        if prob_col:
            rec_prob = data[prob_col].fillna(0)
            score += rec_prob * 30

        return score.clip(0, 100)

    def _calculate_cross_asset_stress(self, data: pd.DataFrame) -> pd.Series:
        """Calculate cross-asset stress component (0-100)."""
        score = pd.Series(50.0, index=data.index)

        # Stock-bond correlation (positive = stress in diversified portfolio)
        corr_cols = ["stock_bond_corr_21d", "cross_asset_stock_bond_corr_21d"]
        corr_col = next((c for c in corr_cols if c in data.columns), None)
        if corr_col:
            corr = data[corr_col].fillna(0)
            # Positive correlation = stress (diversification fails)
            score += corr * 20

        # Average correlation (high = systemic stress)
        avg_corr_cols = ["avg_pairwise_corr", "cross_asset_avg_pairwise_corr"]
        avg_corr_col = next((c for c in avg_corr_cols if c in data.columns), None)
        if avg_corr_col:
            avg_corr = data[avg_corr_col].fillna(0.5)
            score += (avg_corr - 0.3) * 50  # Deviation from normal (~0.3)

        # PC1 variance (high = systemic)
        pc1_cols = ["pc1_variance_ratio", "cross_asset_pc1_variance_ratio"]
        pc1_col = next((c for c in pc1_cols if c in data.columns), None)
        if pc1_col:
            pc1 = data[pc1_col].fillna(0.3)
            score += (pc1 - 0.3) * 40

        return score.clip(0, 100)

    def _calculate_stress_index(self, features: pd.DataFrame) -> pd.Series:
        """Calculate weighted stress index."""

        # Weights for each component
        weights = {
            "vol_stress_score": 0.25,
            "credit_stress_score": 0.25,
            "rate_stress_score": 0.20,
            "macro_stress_score": 0.15,
            "cross_asset_stress_score": 0.15,
        }

        stress_index = pd.Series(0.0, index=features.index)
        total_weight = 0

        for col, weight in weights.items():
            if col in features.columns:
                stress_index += features[col].fillna(50) * weight
                total_weight += weight

        if total_weight > 0:
            stress_index = stress_index / total_weight * sum(weights.values())

        return stress_index.clip(0, 100)

    def _calculate_early_warning(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.Series:
        """Calculate early warning score focused on leading indicators."""

        score = pd.Series(50.0, index=data.index)

        # Credit momentum (leading)
        credit_mom_cols = ["hy_spread_velocity", "credit_hy_spread_velocity"]
        credit_mom_col = next((c for c in credit_mom_cols if c in data.columns), None)
        if credit_mom_col:
            credit_mom = data[credit_mom_col].fillna(0)
            # Positive velocity (widening) = warning
            score += credit_mom * 100

        # VIX term structure (leading)
        term_cols = ["vix_term_structure", "volatility_vix_term_structure"]
        term_col = next((c for c in term_cols if c in data.columns), None)
        if term_col:
            term = data[term_col].fillna(0)
            # Backwardation (positive) = warning
            score += (term > 0).astype(float) * 15

        # Stress momentum
        if "stress_momentum_5d" in features.columns:
            mom = features["stress_momentum_5d"].fillna(0)
            # Rising stress = warning
            score += mom * 2

        # Correlation regime change (leading)
        regime_cols = ["corr_regime_change", "cross_asset_corr_regime_change"]
        regime_col = next((c for c in regime_cols if c in data.columns), None)
        if regime_col:
            regime_change = data[regime_col].fillna(0)
            score += regime_change * 20

        return score.clip(0, 100)

    def _determine_alert_level(self, features: pd.DataFrame) -> pd.Series:
        """
        Determine alert level based on stress and early warning scores.

        Levels:
        0 = None
        1 = Low
        2 = Medium
        3 = High
        4 = Critical
        """
        alert = pd.Series(0, index=features.index)

        stress = features.get("stress_index", pd.Series(50, index=features.index))
        warning = features.get("early_warning_score", pd.Series(50, index=features.index))

        # Combine scores
        combined = stress * 0.6 + warning * 0.4

        alert[combined > 40] = 1  # Low
        alert[combined > 55] = 2  # Medium
        alert[combined > 70] = 3  # High
        alert[combined > 85] = 4  # Critical

        return alert
