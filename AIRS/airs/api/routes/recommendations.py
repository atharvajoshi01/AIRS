"""
Recommendation endpoints for AIRS API.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from airs.api.schemas import (
    RecommendationResponse,
    AssetRecommendation,
    KeyDriver,
    AlertLevelEnum,
)
from airs.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def get_current_recommendation() -> dict[str, Any]:
    """Get current recommendation from engine."""
    # TODO: Replace with actual recommendation engine call
    return {
        "alert_level": AlertLevelEnum.moderate,
        "probability": 0.62,
        "confidence": 0.78,
        "headline": "Risk indicators suggest caution",
        "summary": (
            "Our risk assessment system has flagged moderate risk conditions with a "
            "62% probability of drawdown. While not yet critical, these conditions "
            "warrant attention and potentially gradual defensive positioning. "
            "Key risk factors include: VIX level, credit spreads, yield curve slope. "
            "Estimated portfolio turnover: 12.5%."
        ),
        "asset_recommendations": [
            {
                "symbol": "SPY",
                "current_weight": 0.40,
                "target_weight": 0.28,
                "action": "sell",
                "urgency": "gradual",
                "rationale": "Reduce equity exposure by 12% to limit drawdown risk",
            },
            {
                "symbol": "VEU",
                "current_weight": 0.20,
                "target_weight": 0.14,
                "action": "sell",
                "urgency": "gradual",
                "rationale": "Reduce international equity exposure",
            },
            {
                "symbol": "AGG",
                "current_weight": 0.25,
                "target_weight": 0.30,
                "action": "buy",
                "urgency": "gradual",
                "rationale": "Increase bond allocation for defensive positioning",
            },
            {
                "symbol": "CASH",
                "current_weight": 0.05,
                "target_weight": 0.18,
                "action": "buy",
                "urgency": "gradual",
                "rationale": "Raise cash to 18% for risk protection",
            },
            {
                "symbol": "DJP",
                "current_weight": 0.10,
                "target_weight": 0.07,
                "action": "sell",
                "urgency": "optional",
                "rationale": "Reduce commodity exposure during risk-off period",
            },
            {
                "symbol": "VNQ",
                "current_weight": 0.05,
                "target_weight": 0.03,
                "action": "sell",
                "urgency": "optional",
                "rationale": "Reduce REIT exposure (correlated with equity risk)",
            },
        ],
        "key_drivers": [
            {
                "feature": "vix_level",
                "contribution": 0.15,
                "direction": "increasing risk",
                "magnitude": "high",
            },
            {
                "feature": "hy_spread",
                "contribution": 0.12,
                "direction": "increasing risk",
                "magnitude": "high",
            },
            {
                "feature": "yield_curve_slope",
                "contribution": 0.08,
                "direction": "increasing risk",
                "magnitude": "moderate",
            },
            {
                "feature": "equity_bond_corr",
                "contribution": -0.05,
                "direction": "decreasing risk",
                "magnitude": "moderate",
            },
        ],
        "historical_context": (
            "Current indicators resemble pre-correction periods such as late 2018 "
            "and early 2022. Historical analysis suggests elevated but not extreme risk."
        ),
        "suggested_timeline": (
            "Execute 12.5% turnover over 3-5 trading days. Daily rebalancing of ~3% recommended."
        ),
        "estimated_turnover": 0.125,
    }


@router.get("/current", response_model=RecommendationResponse)
async def get_current_recommendations() -> RecommendationResponse:
    """
    Get current portfolio recommendations.

    Returns actionable recommendations based on the latest model prediction.
    """
    try:
        rec_data = get_current_recommendation()

        return RecommendationResponse(
            timestamp=datetime.utcnow(),
            version="1.0",
            alert_level=rec_data["alert_level"],
            probability=rec_data["probability"],
            confidence=rec_data["confidence"],
            headline=rec_data["headline"],
            summary=rec_data["summary"],
            asset_recommendations=[
                AssetRecommendation(**ar) for ar in rec_data["asset_recommendations"]
            ],
            key_drivers=[KeyDriver(**kd) for kd in rec_data["key_drivers"]],
            historical_context=rec_data["historical_context"],
            suggested_timeline=rec_data["suggested_timeline"],
            estimated_turnover=rec_data["estimated_turnover"],
        )
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")


@router.get("/action-plan")
async def get_action_plan() -> dict[str, Any]:
    """
    Get executable action plan.

    Returns ordered list of trades to execute the recommendation.
    """
    try:
        rec = get_current_recommendation()

        # Generate action plan
        actions = []

        # Sort: sells first, then buys
        sells = [
            ar for ar in rec["asset_recommendations"] if ar["action"] == "sell"
        ]
        buys = [ar for ar in rec["asset_recommendations"] if ar["action"] == "buy"]

        for ar in sells:
            if abs(ar["target_weight"] - ar["current_weight"]) >= 0.01:
                actions.append({
                    "order": len(actions) + 1,
                    "action": "SELL",
                    "symbol": ar["symbol"],
                    "from_weight": f"{ar['current_weight']*100:.1f}%",
                    "to_weight": f"{ar['target_weight']*100:.1f}%",
                    "change": f"{(ar['target_weight'] - ar['current_weight'])*100:+.1f}%",
                    "urgency": ar["urgency"],
                    "rationale": ar["rationale"],
                })

        for ar in buys:
            if abs(ar["target_weight"] - ar["current_weight"]) >= 0.01:
                actions.append({
                    "order": len(actions) + 1,
                    "action": "BUY",
                    "symbol": ar["symbol"],
                    "from_weight": f"{ar['current_weight']*100:.1f}%",
                    "to_weight": f"{ar['target_weight']*100:.1f}%",
                    "change": f"{(ar['target_weight'] - ar['current_weight'])*100:+.1f}%",
                    "urgency": ar["urgency"],
                    "rationale": ar["rationale"],
                })

        return {
            "alert_level": rec["alert_level"].value,
            "generated_at": datetime.utcnow().isoformat(),
            "estimated_turnover": f"{rec['estimated_turnover']*100:.1f}%",
            "suggested_timeline": rec["suggested_timeline"],
            "actions": actions,
            "execution_notes": [
                "Execute sells before buys to free up cash",
                "Use limit orders where possible to minimize market impact",
                "Monitor for significant market moves during execution",
            ],
        }
    except Exception as e:
        logger.error(f"Error generating action plan: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate action plan")


@router.get("/explain")
async def get_recommendation_explanation() -> dict[str, Any]:
    """
    Get detailed explanation of current recommendation.

    Returns narrative explanation with feature attributions.
    """
    try:
        rec = get_current_recommendation()

        # Format driver explanations
        driver_explanations = []
        for driver in rec["key_drivers"]:
            feature = driver["feature"].replace("_", " ").title()
            direction = "increasing" if "increasing" in driver["direction"] else "decreasing"

            explanation = {
                "feature": driver["feature"],
                "display_name": feature,
                "contribution": driver["contribution"],
                "interpretation": f"{feature} is {direction} overall risk by {abs(driver['contribution'])*100:.1f}%",
            }
            driver_explanations.append(explanation)

        return {
            "probability": rec["probability"],
            "confidence": rec["confidence"],
            "alert_level": rec["alert_level"].value,
            "narrative": {
                "headline": rec["headline"],
                "summary": rec["summary"],
                "historical_context": rec["historical_context"],
            },
            "key_drivers": driver_explanations,
            "risk_by_category": {
                "rates": "elevated",
                "credit": "elevated",
                "volatility": "high",
                "macro": "moderate",
            },
            "model_info": {
                "model_id": "ensemble_v1",
                "last_trained": "2024-01-15",
                "validation_auc": 0.82,
            },
        }
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate explanation")


@router.post("/simulate")
async def simulate_recommendation(
    probability: float = Query(..., ge=0, le=1, description="Simulated probability"),
    current_weights: dict[str, float] | None = None,
) -> RecommendationResponse:
    """
    Simulate a recommendation for a given probability.

    Useful for testing and understanding the recommendation logic.
    """
    # TODO: Implement actual recommendation simulation
    # For now, modify the mock data based on probability

    rec = get_current_recommendation()

    # Adjust alert level based on probability
    if probability < 0.3:
        alert_level = AlertLevelEnum.none
        headline = "Market conditions stable - maintain allocation"
        summary = f"Our models indicate a {probability*100:.0f}% probability of significant drawdown. No action recommended."
        turnover = 0.0
    elif probability < 0.5:
        alert_level = AlertLevelEnum.low
        headline = "Minor stress signals detected - monitor closely"
        summary = f"Early warning indicators show a {probability*100:.0f}% drawdown probability. Monitoring recommended."
        turnover = 0.02
    elif probability < 0.7:
        alert_level = AlertLevelEnum.moderate
        headline = "Risk indicators suggest caution"
        summary = f"Risk models indicate a {probability*100:.0f}% probability of drawdown. Gradual de-risking recommended."
        turnover = 0.10
    elif probability < 0.85:
        alert_level = AlertLevelEnum.high
        headline = "Significant drawdown risk identified"
        summary = f"WARNING: Elevated {probability*100:.0f}% probability of significant drawdown. De-risking recommended."
        turnover = 0.20
    else:
        alert_level = AlertLevelEnum.critical
        headline = "ALERT: High probability of market stress"
        summary = f"CRITICAL ALERT: {probability*100:.0f}% probability of major drawdown. Immediate action advised."
        turnover = 0.30

    return RecommendationResponse(
        timestamp=datetime.utcnow(),
        version="1.0",
        alert_level=alert_level,
        probability=probability,
        confidence=rec["confidence"],
        headline=headline,
        summary=summary,
        asset_recommendations=[
            AssetRecommendation(**ar) for ar in rec["asset_recommendations"]
        ],
        key_drivers=[KeyDriver(**kd) for kd in rec["key_drivers"]],
        historical_context=rec["historical_context"],
        suggested_timeline=rec["suggested_timeline"],
        estimated_turnover=turnover,
    )
