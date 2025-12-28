"""
Feature endpoints for AIRS API.
"""

from datetime import date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from airs.api.schemas import (
    FeatureValue,
    FeatureResponse,
    FeatureHistoryRequest,
    FeatureHistoryResponse,
    TimeframeEnum,
)
from airs.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Feature metadata
FEATURE_METADATA = {
    # Rate features
    "yield_10y": {"category": "rates", "description": "10-year Treasury yield"},
    "yield_2y": {"category": "rates", "description": "2-year Treasury yield"},
    "yield_curve_slope": {"category": "rates", "description": "Yield curve slope (10Y-2Y)"},
    "yield_curve_curvature": {"category": "rates", "description": "Yield curve curvature"},
    "curve_inversion_flag": {"category": "rates", "description": "Yield curve inversion indicator"},
    "rate_momentum_10d": {"category": "rates", "description": "10-day rate momentum"},
    # Credit features
    "hy_spread": {"category": "credit", "description": "High-yield credit spread (bps)"},
    "ig_spread": {"category": "credit", "description": "Investment-grade spread (bps)"},
    "hy_ig_diff": {"category": "credit", "description": "HY-IG spread differential"},
    "spread_momentum": {"category": "credit", "description": "Credit spread momentum"},
    "credit_stress_flag": {"category": "credit", "description": "Credit stress indicator"},
    # Volatility features
    "vix_level": {"category": "volatility", "description": "VIX index level"},
    "vix_percentile": {"category": "volatility", "description": "VIX percentile (2Y)"},
    "vix_term_structure": {"category": "volatility", "description": "VIX term structure"},
    "realized_vol": {"category": "volatility", "description": "Realized volatility (21d)"},
    "vol_risk_premium": {"category": "volatility", "description": "Volatility risk premium"},
    # Macro features
    "lei_yoy": {"category": "macro", "description": "Leading Economic Index YoY %"},
    "pmi_level": {"category": "macro", "description": "ISM Manufacturing PMI"},
    "claims_4wma": {"category": "macro", "description": "Jobless claims 4-week avg (000s)"},
    "consumer_sentiment": {"category": "macro", "description": "Consumer sentiment index"},
    "recession_prob": {"category": "macro", "description": "Recession probability"},
    # Cross-asset features
    "equity_bond_corr": {"category": "cross_asset", "description": "Equity-bond correlation"},
    "risk_on_off_score": {"category": "cross_asset", "description": "Risk-on/off score"},
    # Composite features
    "composite_stress_index": {"category": "composite", "description": "Composite stress index"},
    "early_warning_score": {"category": "composite", "description": "Early warning score"},
}


def get_current_features() -> list[dict[str, Any]]:
    """Get current feature values from feature store."""
    # TODO: Replace with actual feature store query
    # Mock current feature values
    import numpy as np

    features = []
    for name, meta in FEATURE_METADATA.items():
        # Generate mock values based on feature type
        if "yield" in name:
            value = np.random.uniform(3.5, 5.0)
        elif "spread" in name and "flag" not in name:
            value = np.random.uniform(100, 500)
        elif "vix" in name and "level" in name:
            value = np.random.uniform(15, 30)
        elif "percentile" in name or "prob" in name:
            value = np.random.uniform(0, 100)
        elif "flag" in name:
            value = float(np.random.choice([0, 1]))
        elif "momentum" in name:
            value = np.random.uniform(-0.5, 0.5)
        elif "corr" in name:
            value = np.random.uniform(-0.5, 0.5)
        else:
            value = np.random.uniform(-2, 2)

        features.append({
            "name": name,
            "value": round(value, 4),
            "percentile": round(np.random.uniform(10, 90), 1),
            "z_score": round(np.random.uniform(-2, 2), 2),
            "category": meta["category"],
        })

    return features


def get_feature_history(
    feature_names: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, list[dict[str, Any]]]:
    """Get historical feature values."""
    # TODO: Replace with actual database query
    import numpy as np

    history = {}
    for name in feature_names:
        history[name] = []
        current = start_date
        base_value = np.random.uniform(0, 100)

        while current <= end_date:
            # Random walk with trend
            base_value += np.random.uniform(-2, 2)
            history[name].append({
                "date": current.isoformat(),
                "value": round(base_value, 4),
            })
            current += timedelta(days=1)

    return history


@router.get("/current", response_model=FeatureResponse)
async def get_current_feature_values(
    category: str | None = Query(None, description="Filter by feature category"),
) -> FeatureResponse:
    """
    Get current values of all features.

    Returns the latest computed feature values used for prediction.
    """
    try:
        features = get_current_features()

        # Filter by category if specified
        if category:
            features = [f for f in features if f.get("category") == category]

        return FeatureResponse(
            timestamp=datetime.utcnow(),
            version="1.0",
            features=[FeatureValue(**f) for f in features],
            feature_date=date.today(),
        )
    except Exception as e:
        logger.error(f"Error getting current features: {e}")
        raise HTTPException(status_code=500, detail="Failed to get features")


@router.get("/history", response_model=FeatureHistoryResponse)
async def get_feature_history_endpoint(
    feature_names: str = Query(
        ..., description="Comma-separated list of feature names"
    ),
    start_date: date | None = Query(None, description="Start date"),
    end_date: date | None = Query(None, description="End date"),
    timeframe: TimeframeEnum = Query(
        TimeframeEnum.month_1, description="Predefined timeframe"
    ),
) -> FeatureHistoryResponse:
    """
    Get historical feature values.

    Returns feature values over the specified date range.
    """
    # Parse feature names
    names = [n.strip() for n in feature_names.split(",")]

    # Validate feature names
    valid_names = [n for n in names if n in FEATURE_METADATA]
    if not valid_names:
        raise HTTPException(
            status_code=400,
            detail=f"No valid feature names provided. Available: {list(FEATURE_METADATA.keys())}",
        )

    # Calculate date range
    if end_date is None:
        end_date = date.today()

    if start_date is None:
        timeframe_days = {
            TimeframeEnum.day_1: 1,
            TimeframeEnum.week_1: 7,
            TimeframeEnum.month_1: 30,
            TimeframeEnum.month_3: 90,
            TimeframeEnum.month_6: 180,
            TimeframeEnum.year_1: 365,
            TimeframeEnum.year_5: 1825,
            TimeframeEnum.all: 3650,
        }
        days = timeframe_days.get(timeframe, 30)
        start_date = end_date - timedelta(days=days)

    try:
        history = get_feature_history(valid_names, start_date, end_date)

        return FeatureHistoryResponse(
            timestamp=datetime.utcnow(),
            version="1.0",
            features=history,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        logger.error(f"Error getting feature history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feature history")


@router.get("/metadata")
async def get_feature_metadata(
    category: str | None = Query(None, description="Filter by category"),
) -> dict[str, Any]:
    """
    Get metadata for all features.

    Returns descriptions and categories for available features.
    """
    metadata = FEATURE_METADATA.copy()

    if category:
        metadata = {k: v for k, v in metadata.items() if v["category"] == category}

    # Get unique categories
    categories = list(set(m["category"] for m in FEATURE_METADATA.values()))

    return {
        "features": metadata,
        "categories": categories,
        "total_count": len(metadata),
    }


@router.get("/importance")
async def get_feature_importance() -> dict[str, Any]:
    """
    Get feature importance from the current model.

    Returns relative importance scores for all features.
    """
    # TODO: Get actual feature importance from model
    import numpy as np

    importance = {}
    for name in FEATURE_METADATA:
        # Mock importance scores
        importance[name] = round(np.random.uniform(0, 0.1), 4)

    # Normalize
    total = sum(importance.values())
    importance = {k: round(v / total, 4) for k, v in importance.items()}

    # Sort by importance
    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "feature_importance": sorted_importance,
        "top_10": dict(list(sorted_importance.items())[:10]),
        "model_id": "ensemble_v1",
        "last_updated": datetime.utcnow().isoformat(),
    }


@router.get("/correlations")
async def get_feature_correlations(
    features: str | None = Query(
        None, description="Comma-separated list of features (default: top 10)"
    ),
) -> dict[str, Any]:
    """
    Get correlation matrix for features.

    Returns pairwise correlations between features.
    """
    import numpy as np

    # Get feature list
    if features:
        feature_list = [f.strip() for f in features.split(",")]
        feature_list = [f for f in feature_list if f in FEATURE_METADATA]
    else:
        # Default to first 10 features
        feature_list = list(FEATURE_METADATA.keys())[:10]

    if len(feature_list) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 valid features required for correlation",
        )

    # Generate mock correlation matrix
    n = len(feature_list)
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            corr = np.random.uniform(-0.5, 0.8)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    # Convert to dictionary format
    correlations = {}
    for i, f1 in enumerate(feature_list):
        correlations[f1] = {}
        for j, f2 in enumerate(feature_list):
            correlations[f1][f2] = round(corr_matrix[i, j], 3)

    return {
        "features": feature_list,
        "correlations": correlations,
        "computed_at": datetime.utcnow().isoformat(),
    }
