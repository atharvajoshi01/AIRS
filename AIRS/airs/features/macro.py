"""
Macroeconomic features.

Generates features from leading indicators, sentiment, and financial conditions.
"""

import numpy as np
import pandas as pd

from airs.features.base import FeatureGenerator
from airs.utils.logging import get_logger

logger = get_logger(__name__)


class MacroFeatures(FeatureGenerator):
    """
    Macroeconomic feature generator.

    Generates features including:
    - Leading Economic Index (LEI) momentum
    - Initial claims trends
    - Consumer sentiment
    - Financial conditions
    - Recession probability indicators
    """

    @property
    def feature_group(self) -> str:
        return "macro"

    @property
    def feature_names(self) -> list[str]:
        return [
            # LEI
            "lei_level",
            "lei_mom_3m",
            "lei_mom_6m",
            "lei_yoy",
            "lei_diffusion",
            # Claims
            "claims_4wk_ma",
            "claims_trend",
            "claims_yoy_pct",
            # Sentiment
            "consumer_sentiment",
            "sentiment_zscore",
            "sentiment_trend",
            # Financial conditions
            "nfci",
            "nfci_zscore",
            "nfci_stress_flag",
            # Recession indicators
            "recession_prob",
            "macro_stress_score",
        ]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate macro features.

        Args:
            data: DataFrame with columns:
                - USSLIND (LEI)
                - ICSA (Initial Claims)
                - UMCSENT (Consumer Sentiment)
                - NFCI (Financial Conditions)

        Returns:
            DataFrame with macro features
        """
        features = pd.DataFrame(index=data.index)

        # Leading Economic Index
        if "USSLIND" in data.columns:
            lei = data["USSLIND"]
            features["lei_level"] = lei

            # Forward fill monthly data to daily
            lei_filled = lei.ffill()

            # Momentum (use month-end aligned data)
            features["lei_mom_3m"] = lei_filled.diff(63)  # ~3 months
            features["lei_mom_6m"] = lei_filled.diff(126)  # ~6 months
            features["lei_yoy"] = lei_filled.pct_change(252) * 100

            # LEI diffusion (simplified: sign of momentum)
            features["lei_diffusion"] = (features["lei_mom_6m"] > 0).astype(int)

        # Initial Claims
        if "ICSA" in data.columns:
            claims = data["ICSA"]

            # 4-week moving average
            features["claims_4wk_ma"] = claims.rolling(20).mean()  # ~4 weeks

            # Trend (positive = rising claims = bad)
            features["claims_trend"] = claims.rolling(10).mean().diff(10)

            # Year-over-year percentage change
            features["claims_yoy_pct"] = claims.pct_change(252) * 100

        # Consumer Sentiment
        if "UMCSENT" in data.columns:
            sentiment = data["UMCSENT"]
            features["consumer_sentiment"] = sentiment

            # Forward fill monthly to daily
            sentiment_filled = sentiment.ffill()

            features["sentiment_zscore"] = self.add_zscore(sentiment_filled)

            # Trend
            features["sentiment_trend"] = sentiment_filled.diff(63)  # 3-month change

        # Financial Conditions (NFCI)
        if "NFCI" in data.columns:
            nfci = data["NFCI"]
            features["nfci"] = nfci

            # Forward fill weekly to daily
            nfci_filled = nfci.ffill()

            features["nfci_zscore"] = self.add_zscore(nfci_filled)

            # Stress flag (NFCI > 0 indicates tighter than average conditions)
            features["nfci_stress_flag"] = (nfci_filled > 0).astype(int)

        # Composite recession probability
        features["recession_prob"] = self._calculate_recession_prob(features, data)

        # Macro stress score (0-100)
        features["macro_stress_score"] = self._calculate_stress_score(features)

        self.log_features_generated(features)
        return features

    def _calculate_recession_prob(
        self,
        features: pd.DataFrame,
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate recession probability using multiple indicators.

        Uses:
        - Yield curve inversion
        - LEI momentum
        - Claims trend
        """
        prob = pd.Series(0.0, index=features.index)
        weights_sum = 0

        # Yield curve inversion contribution
        if "T10Y2Y" in data.columns or "yc_slope_10y2y" in features.columns:
            slope = (
                data.get("T10Y2Y", features.get("yc_slope_10y2y", pd.Series()))
            )
            if len(slope) > 0:
                # Inverted curve increases probability
                curve_prob = (slope < 0).astype(float) * 0.3
                # Deep inversion increases more
                curve_prob += (slope < -0.5).astype(float) * 0.2
                prob += curve_prob.reindex(features.index, fill_value=0)
                weights_sum += 0.5

        # LEI momentum contribution
        if "lei_mom_6m" in features.columns:
            lei_mom = features["lei_mom_6m"]
            # Negative LEI momentum increases probability
            lei_prob = (lei_mom < 0).astype(float) * 0.2
            lei_prob += (lei_mom < -2).astype(float) * 0.1
            prob += lei_prob.fillna(0)
            weights_sum += 0.3

        # Claims trend contribution
        if "claims_trend" in features.columns:
            claims_trend = features["claims_trend"]
            # Rising claims increases probability
            claims_prob = (claims_trend > 0).astype(float) * 0.1
            claims_prob += (claims_trend > 10000).astype(float) * 0.1
            prob += claims_prob.fillna(0)
            weights_sum += 0.2

        # Normalize to 0-1 range
        if weights_sum > 0:
            prob = prob / weights_sum
            prob = prob.clip(0, 1)

        return prob

    def _calculate_stress_score(self, features: pd.DataFrame) -> pd.Series:
        """
        Calculate composite macro stress score (0-100).
        """
        score = pd.Series(50.0, index=features.index)  # Neutral baseline
        contributions = 0

        # NFCI contribution
        if "nfci_zscore" in features.columns:
            nfci_z = features["nfci_zscore"].fillna(0)
            # Higher NFCI z-score = more stress
            score += nfci_z * 10
            contributions += 1

        # LEI momentum contribution
        if "lei_mom_6m" in features.columns:
            lei_mom = features["lei_mom_6m"].fillna(0)
            # Negative momentum = more stress
            lei_contribution = -lei_mom * 2  # Scale factor
            score += lei_contribution.clip(-20, 20)
            contributions += 1

        # Sentiment contribution
        if "sentiment_zscore" in features.columns:
            sent_z = features["sentiment_zscore"].fillna(0)
            # Lower sentiment = more stress
            score -= sent_z * 8
            contributions += 1

        # Normalize and clip
        score = score.clip(0, 100)

        return score


class MoneySupplyFeatures(FeatureGenerator):
    """
    Money supply and liquidity features.
    """

    @property
    def feature_group(self) -> str:
        return "money_supply"

    @property
    def feature_names(self) -> list[str]:
        return [
            "m2_growth_yoy",
            "m2_growth_3m",
            "liquidity_score",
        ]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate money supply features."""
        features = pd.DataFrame(index=data.index)

        if "M2SL" in data.columns:
            m2 = data["M2SL"]

            # Forward fill monthly to daily
            m2_filled = m2.ffill()

            # Year-over-year growth
            features["m2_growth_yoy"] = m2_filled.pct_change(252) * 100

            # 3-month growth (annualized)
            features["m2_growth_3m"] = m2_filled.pct_change(63) * 4 * 100

            # Liquidity score (high M2 growth = more liquidity = lower score)
            m2_growth = features["m2_growth_yoy"].fillna(0)
            # Invert: high growth = low stress
            features["liquidity_score"] = 50 - m2_growth * 2
            features["liquidity_score"] = features["liquidity_score"].clip(0, 100)

        self.log_features_generated(features)
        return features
