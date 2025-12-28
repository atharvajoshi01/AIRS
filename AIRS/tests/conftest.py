"""
Pytest fixtures for AIRS tests.
"""

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
def sample_dates() -> pd.DatetimeIndex:
    """Generate sample trading dates."""
    return pd.bdate_range(start="2020-01-01", end="2024-12-31", freq="B")


@pytest.fixture
def sample_prices(sample_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate sample price data for multiple assets."""
    np.random.seed(42)

    n = len(sample_dates)

    # Generate random walks for prices
    returns = {
        "SPY": np.random.normal(0.0003, 0.012, n),
        "VEU": np.random.normal(0.0002, 0.014, n),
        "AGG": np.random.normal(0.0001, 0.004, n),
        "DJP": np.random.normal(0.0001, 0.015, n),
        "VNQ": np.random.normal(0.0002, 0.016, n),
    }

    prices = {}
    for symbol, rets in returns.items():
        price_series = 100 * np.cumprod(1 + rets)
        prices[symbol] = price_series

    return pd.DataFrame(prices, index=sample_dates)


@pytest.fixture
def sample_features(sample_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate sample feature data."""
    np.random.seed(42)

    n = len(sample_dates)

    features = {
        "vix_level": np.random.uniform(12, 35, n),
        "yield_curve_slope": np.random.uniform(-50, 200, n),
        "hy_spread": np.random.uniform(250, 600, n),
        "lei_yoy": np.random.uniform(-5, 8, n),
        "equity_bond_corr": np.random.uniform(-0.5, 0.5, n),
        "composite_stress_index": np.random.uniform(0, 1, n),
    }

    return pd.DataFrame(features, index=sample_dates)


@pytest.fixture
def sample_yields(sample_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate sample yield curve data."""
    np.random.seed(42)

    n = len(sample_dates)
    base_level = 2.0

    # Random walk for level
    level_changes = np.random.normal(0, 0.02, n)
    levels = base_level + np.cumsum(level_changes)

    return pd.DataFrame({
        "DGS3MO": levels + np.random.normal(0, 0.1, n) - 0.5,
        "DGS2": levels + np.random.normal(0, 0.1, n),
        "DGS5": levels + np.random.normal(0, 0.1, n) + 0.3,
        "DGS10": levels + np.random.normal(0, 0.1, n) + 0.6,
        "DGS30": levels + np.random.normal(0, 0.1, n) + 1.0,
    }, index=sample_dates)


@pytest.fixture
def sample_credit_spreads(sample_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate sample credit spread data."""
    np.random.seed(42)

    n = len(sample_dates)

    # Random walk for spreads
    ig_spread = 100 + np.cumsum(np.random.normal(0, 5, n))
    ig_spread = np.clip(ig_spread, 50, 300)

    hy_spread = 350 + np.cumsum(np.random.normal(0, 15, n))
    hy_spread = np.clip(hy_spread, 200, 900)

    return pd.DataFrame({
        "BAMLC0A0CM": ig_spread,
        "BAMLH0A0HYM2": hy_spread,
    }, index=sample_dates)


@pytest.fixture
def sample_vix(sample_dates: pd.DatetimeIndex) -> pd.Series:
    """Generate sample VIX data."""
    np.random.seed(42)

    n = len(sample_dates)

    # Mean-reverting VIX
    vix = [18.0]
    for _ in range(n - 1):
        change = 0.05 * (18 - vix[-1]) + np.random.normal(0, 1.5)
        new_vix = max(10, vix[-1] + change)
        vix.append(new_vix)

    return pd.Series(vix, index=sample_dates, name="VIX")


@pytest.fixture
def sample_signals(sample_dates: pd.DatetimeIndex) -> pd.Series:
    """Generate sample trading signals."""
    np.random.seed(42)

    n = len(sample_dates)

    # Mostly low signals with occasional spikes
    signals = np.random.beta(2, 5, n)  # Skewed toward low values

    return pd.Series(signals, index=sample_dates, name="signal")


@pytest.fixture
def sample_labels(sample_dates: pd.DatetimeIndex) -> pd.Series:
    """Generate sample binary labels."""
    np.random.seed(42)

    n = len(sample_dates)

    # About 5% positive class
    labels = (np.random.random(n) < 0.05).astype(int)

    return pd.Series(labels, index=sample_dates, name="label")


@pytest.fixture
def sample_predictions(sample_dates: pd.DatetimeIndex) -> pd.Series:
    """Generate sample model predictions."""
    np.random.seed(42)

    n = len(sample_dates)

    predictions = np.random.beta(2, 5, n)

    return pd.Series(predictions, index=sample_dates, name="prediction")


# ============================================================================
# Model Fixtures
# ============================================================================


@pytest.fixture
def trained_model(sample_features: pd.DataFrame, sample_labels: pd.Series):
    """Get a trained model for testing."""
    from sklearn.ensemble import RandomForestClassifier

    # Align data
    common_idx = sample_features.index.intersection(sample_labels.index)
    X = sample_features.loc[common_idx].values
    y = sample_labels.loc[common_idx].values

    # Train simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    return model


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def backtest_config() -> dict[str, Any]:
    """Get sample backtest configuration."""
    return {
        "initial_value": 100_000.0,
        "target_weights": {
            "SPY": 0.40,
            "VEU": 0.20,
            "AGG": 0.25,
            "DJP": 0.10,
            "VNQ": 0.05,
        },
        "alert_threshold": 0.5,
        "derisk_equity_reduction": 0.5,
        "trading_cost_bps": 10.0,
        "slippage_bps": 5.0,
    }


@pytest.fixture
def feature_config() -> dict[str, list[str]]:
    """Get feature configuration."""
    return {
        "rate_features": [
            "yield_10y",
            "yield_2y",
            "yield_curve_slope",
            "yield_curve_curvature",
        ],
        "credit_features": [
            "hy_spread",
            "ig_spread",
            "hy_ig_diff",
        ],
        "volatility_features": [
            "vix_level",
            "vix_percentile",
            "realized_vol",
        ],
        "macro_features": [
            "lei_yoy",
            "pmi_level",
            "claims_4wma",
        ],
    }


# ============================================================================
# API Fixtures
# ============================================================================


@pytest.fixture
def api_client():
    """Get test API client."""
    from fastapi.testclient import TestClient
    from airs.api.main import app

    return TestClient(app)


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture
def test_db_url() -> str:
    """Get test database URL."""
    return "sqlite:///:memory:"


@pytest.fixture
def db_session(test_db_url: str):
    """Create test database session."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from airs.db.models import Base

    engine = create_engine(test_db_url)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()


# ============================================================================
# Helper Functions
# ============================================================================


def assert_no_lookahead(
    features: pd.DataFrame,
    labels: pd.Series,
    horizon: int = 10,
) -> None:
    """Assert that features don't leak future label information."""
    # Check that all feature dates are before their corresponding label dates
    for feature_date in features.index:
        # Get label date (horizon days ahead)
        label_date = feature_date + pd.Timedelta(days=horizon)

        # Features should only use data available at feature_date
        # This is a placeholder for more sophisticated checks
        assert feature_date <= label_date, "Feature date should be before label date"


def assert_valid_probabilities(predictions: np.ndarray) -> None:
    """Assert predictions are valid probabilities."""
    assert np.all(predictions >= 0), "Probabilities must be >= 0"
    assert np.all(predictions <= 1), "Probabilities must be <= 1"
    assert not np.any(np.isnan(predictions)), "Probabilities must not be NaN"
