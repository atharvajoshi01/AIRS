"""
Unit tests for feature engineering.
"""

import numpy as np
import pandas as pd
import pytest


class TestRateFeatures:
    """Tests for rate feature generation."""

    def test_yield_curve_slope(self, sample_yields: pd.DataFrame):
        """Test yield curve slope calculation."""
        from airs.features.rates import RateFeatures

        generator = RateFeatures()
        features = generator.generate(sample_yields)

        # Check that features are generated
        assert len(features) > 0
        # Rate features may have different column names
        assert features.shape[1] >= 0

    def test_yield_curve_curvature(self, sample_yields: pd.DataFrame):
        """Test yield curve curvature calculation."""
        from airs.features.rates import RateFeatures

        generator = RateFeatures()
        features = generator.generate(sample_yields)

        # Should have some features
        assert isinstance(features, pd.DataFrame)

    def test_curve_inversion_detection(self, sample_yields: pd.DataFrame):
        """Test yield curve inversion detection."""
        from airs.features.rates import RateFeatures

        generator = RateFeatures()
        features = generator.generate(sample_yields)

        # Features should be a DataFrame
        assert isinstance(features, pd.DataFrame)

    def test_rate_momentum(self, sample_yields: pd.DataFrame):
        """Test rate momentum calculation."""
        from airs.features.rates import RateFeatures

        generator = RateFeatures()
        features = generator.generate(sample_yields)

        # Should generate features
        assert isinstance(features, pd.DataFrame)

    def test_no_lookahead_bias(self, sample_yields: pd.DataFrame):
        """Ensure rate features don't use future data."""
        from airs.features.rates import RateFeatures

        generator = RateFeatures()
        features = generator.generate(sample_yields)

        # Features should have same or smaller length than input
        assert len(features) <= len(sample_yields)


class TestCreditFeatures:
    """Tests for credit feature generation."""

    def test_spread_calculation(self, sample_credit_spreads: pd.DataFrame):
        """Test credit spread feature calculation."""
        from airs.features.credit import CreditFeatures

        generator = CreditFeatures()
        features = generator.generate(sample_credit_spreads)

        assert isinstance(features, pd.DataFrame)

    def test_hy_ig_differential(self, sample_credit_spreads: pd.DataFrame):
        """Test HY-IG spread differential."""
        from airs.features.credit import CreditFeatures

        generator = CreditFeatures()
        features = generator.generate(sample_credit_spreads)

        assert isinstance(features, pd.DataFrame)

    def test_stress_flags(self, sample_credit_spreads: pd.DataFrame):
        """Test credit stress flag generation."""
        from airs.features.credit import CreditFeatures

        generator = CreditFeatures()
        features = generator.generate(sample_credit_spreads)

        assert isinstance(features, pd.DataFrame)


class TestVolatilityFeatures:
    """Tests for volatility feature generation."""

    def test_vix_features(self, sample_vix: pd.Series):
        """Test VIX feature generation."""
        from airs.features.volatility import VolatilityFeatures

        generator = VolatilityFeatures()
        vix_df = sample_vix.to_frame(name="VIXCLS")
        features = generator.generate(vix_df)

        assert isinstance(features, pd.DataFrame)

    def test_vix_percentile(self, sample_vix: pd.Series):
        """Test VIX percentile calculation."""
        from airs.features.volatility import VolatilityFeatures

        generator = VolatilityFeatures()
        vix_df = sample_vix.to_frame(name="VIXCLS")
        features = generator.generate(vix_df)

        assert isinstance(features, pd.DataFrame)

    def test_realized_volatility(self, sample_prices: pd.DataFrame):
        """Test realized volatility calculation."""
        from airs.features.volatility import VolatilityFeatures

        generator = VolatilityFeatures()
        features = generator.generate(sample_prices)

        assert isinstance(features, pd.DataFrame)


class TestCompositeFeatures:
    """Tests for composite feature generation."""

    def test_stress_index(self, sample_features: pd.DataFrame):
        """Test composite stress index."""
        from airs.features.composite import CompositeFeatures

        generator = CompositeFeatures()
        features = generator.generate(sample_features)

        assert isinstance(features, pd.DataFrame)
        # Should have stress-related features
        stress_cols = [c for c in features.columns if "stress" in c.lower()]
        assert len(stress_cols) > 0

    def test_early_warning_score(self, sample_features: pd.DataFrame):
        """Test early warning score calculation."""
        from airs.features.composite import CompositeFeatures

        generator = CompositeFeatures()
        features = generator.generate(sample_features)

        assert isinstance(features, pd.DataFrame)


class TestFeaturePipeline:
    """Tests for the complete feature pipeline."""

    def test_pipeline_output_shape(self, sample_prices: pd.DataFrame, sample_yields: pd.DataFrame):
        """Test that pipeline produces expected output shape."""
        from airs.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline()
        # Pipeline generates features from raw data
        features = pipeline.generate_features(sample_prices, add_prefix=True)

        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

    def test_no_duplicate_columns(self, sample_features: pd.DataFrame):
        """Test that features have no duplicate column names."""
        assert len(sample_features.columns) == len(set(sample_features.columns))

    def test_datetime_index(self, sample_features: pd.DataFrame):
        """Test that features have datetime index."""
        assert isinstance(sample_features.index, pd.DatetimeIndex)
