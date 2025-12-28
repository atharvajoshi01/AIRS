"""
Cross-asset features.

Generates features from correlations and relationships across asset classes.
"""

import numpy as np
import pandas as pd
from scipy import stats

from airs.features.base import FeatureGenerator
from airs.utils.logging import get_logger
from airs.utils.stats import calculate_rolling_correlation

logger = get_logger(__name__)


class CrossAssetFeatures(FeatureGenerator):
    """
    Cross-asset correlation and relationship feature generator.

    Generates features including:
    - Stock-bond correlation
    - Cross-asset correlations
    - Correlation regime changes
    - Risk-on/risk-off indicators
    """

    @property
    def feature_group(self) -> str:
        return "cross_asset"

    @property
    def feature_names(self) -> list[str]:
        return [
            # Correlations
            "stock_bond_corr_21d",
            "stock_bond_corr_63d",
            "stock_gold_corr_21d",
            "stock_vix_corr_21d",
            # Correlation changes
            "corr_regime_change",
            "avg_pairwise_corr",
            # Risk indicators
            "risk_on_off_score",
            "flight_to_quality",
            "safe_haven_demand",
            # Dispersion
            "sector_dispersion",
            "asset_class_dispersion",
            # PCA-based
            "pc1_variance_ratio",
        ]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cross-asset features.

        Args:
            data: DataFrame with returns/prices for multiple assets

        Returns:
            DataFrame with cross-asset features
        """
        features = pd.DataFrame(index=data.index)

        # Identify asset columns
        equity_cols = self._find_columns(data, ["SPY", "prices_SPY"])
        bond_cols = self._find_columns(data, ["AGG", "prices_AGG", "TLT"])
        gold_cols = self._find_columns(data, ["GLD", "prices_GLD"])
        vix_cols = self._find_columns(data, ["VIXCLS", "^VIX", "VIX"])

        # Calculate returns if we have prices
        returns = self._get_returns(data)

        # Stock-Bond correlation
        if equity_cols and bond_cols:
            spy_ret = returns.get(equity_cols[0])
            agg_ret = returns.get(bond_cols[0])

            if spy_ret is not None and agg_ret is not None:
                features["stock_bond_corr_21d"] = calculate_rolling_correlation(
                    spy_ret, agg_ret, window=21
                )
                features["stock_bond_corr_63d"] = calculate_rolling_correlation(
                    spy_ret, agg_ret, window=63
                )

                # Correlation regime change (sudden shift)
                corr_21d = features["stock_bond_corr_21d"]
                corr_change = corr_21d.diff(5).abs()
                features["corr_regime_change"] = (corr_change > 0.3).astype(int)

        # Stock-Gold correlation
        if equity_cols and gold_cols:
            spy_ret = returns.get(equity_cols[0])
            gld_ret = returns.get(gold_cols[0])

            if spy_ret is not None and gld_ret is not None:
                features["stock_gold_corr_21d"] = calculate_rolling_correlation(
                    spy_ret, gld_ret, window=21
                )

        # Stock-VIX correlation (should be negative)
        if equity_cols and vix_cols:
            spy_ret = returns.get(equity_cols[0])
            vix_col = vix_cols[0]

            if spy_ret is not None and vix_col in data.columns:
                vix_chg = data[vix_col].pct_change()
                features["stock_vix_corr_21d"] = calculate_rolling_correlation(
                    spy_ret, vix_chg, window=21
                )

        # Average pairwise correlation
        features["avg_pairwise_corr"] = self._calculate_avg_correlation(returns)

        # Risk-on/risk-off score
        features["risk_on_off_score"] = self._calculate_risk_on_off(features, data)

        # Flight to quality indicator
        if equity_cols and bond_cols:
            features["flight_to_quality"] = self._calculate_flight_to_quality(
                data, equity_cols[0], bond_cols[0]
            )

        # Safe haven demand (gold and bonds rising when stocks fall)
        features["safe_haven_demand"] = self._calculate_safe_haven_demand(
            returns, equity_cols, bond_cols, gold_cols
        )

        # Sector dispersion
        features["sector_dispersion"] = self._calculate_sector_dispersion(returns)

        # Asset class dispersion
        features["asset_class_dispersion"] = self._calculate_asset_dispersion(returns)

        # PC1 variance ratio (systemic risk indicator)
        features["pc1_variance_ratio"] = self._calculate_pc1_variance(returns)

        self.log_features_generated(features)
        return features

    def _find_columns(self, data: pd.DataFrame, candidates: list[str]) -> list[str]:
        """Find matching columns from candidates."""
        return [c for c in candidates if c in data.columns]

    def _get_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns for price columns."""
        returns = pd.DataFrame(index=data.index)

        price_like_cols = [
            c for c in data.columns if any(
                x in c.lower() for x in ["spy", "agg", "gld", "veu", "vnq", "djp"]
            )
        ]

        for col in price_like_cols:
            # Skip if already looks like returns
            if data[col].abs().mean() < 0.1:
                returns[col] = data[col]
            else:
                returns[col] = data[col].pct_change()

        return returns

    def _calculate_avg_correlation(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate average pairwise correlation."""
        if len(returns.columns) < 2:
            return pd.Series(index=returns.index, dtype=float)

        def rolling_avg_corr(window_data):
            if len(window_data) < 10:
                return np.nan
            corr_matrix = window_data.corr()
            # Get upper triangle (excluding diagonal)
            upper = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
            return np.nanmean(upper)

        avg_corr = returns.rolling(21).apply(
            lambda x: rolling_avg_corr(returns.loc[x.index]),
            raw=False,
        )

        # Take mean across columns
        if isinstance(avg_corr, pd.DataFrame):
            return avg_corr.mean(axis=1)
        return avg_corr

    def _calculate_risk_on_off(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.Series:
        """Calculate risk-on/risk-off score (-1 to 1)."""
        score = pd.Series(0.0, index=features.index)
        weights = 0

        # Stock-bond correlation (negative = risk-off)
        if "stock_bond_corr_21d" in features.columns:
            corr = features["stock_bond_corr_21d"]
            # Positive correlation = risk-on, negative = risk-off
            score += corr.fillna(0) * 0.5
            weights += 0.5

        # VIX level (high = risk-off)
        vix_cols = ["VIXCLS", "^VIX", "VIX"]
        vix_col = next((c for c in vix_cols if c in data.columns), None)
        if vix_col:
            vix = data[vix_col]
            vix_zscore = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
            # High VIX = negative (risk-off)
            score -= vix_zscore.fillna(0) * 0.3
            weights += 0.3

        if weights > 0:
            score = score / weights
            score = score.clip(-1, 1)

        return score

    def _calculate_flight_to_quality(
        self,
        data: pd.DataFrame,
        equity_col: str,
        bond_col: str,
    ) -> pd.Series:
        """Calculate flight-to-quality indicator."""
        equity_ret = data[equity_col].pct_change()
        bond_ret = data[bond_col].pct_change()

        # Flight to quality: stocks down, bonds up
        ftq = pd.Series(0, index=data.index)

        # Stocks down
        equity_down = equity_ret < -0.01  # More than 1% down

        # Bonds up
        bond_up = bond_ret > 0.002  # More than 0.2% up

        ftq[equity_down & bond_up] = 1

        # Rolling sum over 5 days
        return ftq.rolling(5).sum()

    def _calculate_safe_haven_demand(
        self,
        returns: pd.DataFrame,
        equity_cols: list[str],
        bond_cols: list[str],
        gold_cols: list[str],
    ) -> pd.Series:
        """Calculate safe haven demand indicator."""
        demand = pd.Series(0.0, index=returns.index)

        if not equity_cols:
            return demand

        equity_ret = returns.get(equity_cols[0])
        if equity_ret is None:
            return demand

        # When stocks are down
        stocks_down = equity_ret < -0.005

        # Check if bonds and gold are up
        for cols, weight in [(bond_cols, 0.5), (gold_cols, 0.5)]:
            if cols:
                safe_ret = returns.get(cols[0])
                if safe_ret is not None:
                    safe_up = safe_ret > 0
                    demand[stocks_down & safe_up] += weight

        return demand.rolling(5).mean()

    def _calculate_sector_dispersion(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate sector return dispersion."""
        sector_cols = [
            c for c in returns.columns if any(
                x in c.upper() for x in ["XLF", "XLE", "XLK", "XLV", "XLI"]
            )
        ]

        if len(sector_cols) < 2:
            return pd.Series(index=returns.index, dtype=float)

        sector_returns = returns[sector_cols]
        return sector_returns.rolling(21).std().mean(axis=1)

    def _calculate_asset_dispersion(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate cross-asset return dispersion."""
        if len(returns.columns) < 2:
            return pd.Series(index=returns.index, dtype=float)

        return returns.rolling(21).std().mean(axis=1)

    def _calculate_pc1_variance(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate PC1 variance ratio (systemic risk indicator)."""
        if len(returns.columns) < 3:
            return pd.Series(index=returns.index, dtype=float)

        def pca_variance_ratio(window_data):
            if len(window_data) < 20:
                return np.nan

            clean_data = window_data.dropna(axis=1, how="any")
            if clean_data.shape[1] < 2:
                return np.nan

            try:
                # Standardize
                standardized = (clean_data - clean_data.mean()) / clean_data.std()
                # Compute covariance
                cov = standardized.cov()
                # Eigenvalues
                eigenvalues = np.linalg.eigvalsh(cov)
                # Variance explained by largest eigenvalue
                return eigenvalues[-1] / eigenvalues.sum()
            except Exception:
                return np.nan

        # Rolling calculation
        result = pd.Series(index=returns.index, dtype=float)

        for i in range(63, len(returns)):
            window = returns.iloc[i - 63 : i]
            result.iloc[i] = pca_variance_ratio(window)

        return result
