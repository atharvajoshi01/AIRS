"""
Alert endpoints for AIRS API.
"""

from datetime import date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from airs.api.schemas import (
    AlertResponse,
    AlertHistoryItem,
    AlertHistoryResponse,
    AlertLevelEnum,
    TimeframeEnum,
)
from airs.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Mock data for demonstration - replace with actual database/model calls
def get_current_alert_data() -> dict[str, Any]:
    """Get current alert data from model/database."""
    # TODO: Replace with actual model prediction
    return {
        "alert_level": AlertLevelEnum.moderate,
        "probability": 0.62,
        "confidence": 0.78,
        "headline": "Elevated risk indicators - consider reducing exposure",
        "summary": (
            "Our risk assessment system has flagged moderate risk conditions with a "
            "62% probability of drawdown. Key factors include elevated VIX levels, "
            "credit spread widening, and deteriorating momentum."
        ),
        "last_updated": datetime.utcnow(),
    }


def get_alert_history(
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    """Get historical alerts from database."""
    # TODO: Replace with actual database query
    # Generate mock history
    history = []
    current = start_date
    while current <= end_date:
        # Mock data with varying alert levels
        day_of_year = current.timetuple().tm_yday
        if day_of_year % 30 < 5:
            level = AlertLevelEnum.high
            prob = 0.75
        elif day_of_year % 15 < 3:
            level = AlertLevelEnum.moderate
            prob = 0.55
        elif day_of_year % 7 < 2:
            level = AlertLevelEnum.low
            prob = 0.35
        else:
            level = AlertLevelEnum.none
            prob = 0.15

        history.append({
            "date": current,
            "alert_level": level,
            "probability": prob,
            "action_taken": "derisk" if level in [AlertLevelEnum.high, AlertLevelEnum.critical] else None,
        })
        current += timedelta(days=1)

    return history


@router.get("/current", response_model=AlertResponse)
async def get_current_alert() -> AlertResponse:
    """
    Get current risk alert status.

    Returns the latest model prediction and alert level.
    """
    try:
        alert_data = get_current_alert_data()
        return AlertResponse(**alert_data)
    except Exception as e:
        logger.error(f"Error getting current alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to get current alert")


@router.get("/history", response_model=AlertHistoryResponse)
async def get_alerts_history(
    start_date: date | None = Query(None, description="Start date for history"),
    end_date: date | None = Query(None, description="End date for history"),
    timeframe: TimeframeEnum = Query(
        TimeframeEnum.month_1, description="Predefined timeframe"
    ),
    alert_level: AlertLevelEnum | None = Query(
        None, description="Filter by alert level"
    ),
) -> AlertHistoryResponse:
    """
    Get historical alert data.

    Returns alerts within the specified date range or timeframe.
    """
    # Calculate date range
    if end_date is None:
        end_date = date.today()

    if start_date is None:
        # Use timeframe to calculate start date
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
        history = get_alert_history(start_date, end_date)

        # Filter by alert level if specified
        if alert_level:
            history = [h for h in history if h["alert_level"] == alert_level]

        return AlertHistoryResponse(
            timestamp=datetime.utcnow(),
            version="1.0",
            alerts=[AlertHistoryItem(**h) for h in history],
            total_count=len(history),
        )
    except Exception as e:
        logger.error(f"Error getting alert history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alert history")


@router.get("/statistics")
async def get_alert_statistics(
    timeframe: TimeframeEnum = Query(
        TimeframeEnum.year_1, description="Timeframe for statistics"
    ),
) -> dict[str, Any]:
    """
    Get alert statistics for the specified timeframe.

    Returns counts and percentages of alerts by level.
    """
    # Calculate date range
    end_date = date.today()
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
    days = timeframe_days.get(timeframe, 365)
    start_date = end_date - timedelta(days=days)

    try:
        history = get_alert_history(start_date, end_date)

        # Calculate statistics
        total = len(history)
        by_level = {}
        for level in AlertLevelEnum:
            count = sum(1 for h in history if h["alert_level"] == level)
            by_level[level.value] = {
                "count": count,
                "percentage": count / total * 100 if total > 0 else 0,
            }

        # Calculate action statistics
        actions_taken = sum(1 for h in history if h.get("action_taken"))

        return {
            "timeframe": timeframe.value,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_days": total,
            "alerts_by_level": by_level,
            "actions_taken": actions_taken,
            "action_rate": actions_taken / total * 100 if total > 0 else 0,
        }
    except Exception as e:
        logger.error(f"Error getting alert statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@router.get("/thresholds")
async def get_alert_thresholds() -> dict[str, Any]:
    """
    Get current alert thresholds.

    Returns the probability thresholds for each alert level.
    """
    return {
        "thresholds": {
            AlertLevelEnum.none.value: {"min": 0.0, "max": 0.3},
            AlertLevelEnum.low.value: {"min": 0.3, "max": 0.5},
            AlertLevelEnum.moderate.value: {"min": 0.5, "max": 0.7},
            AlertLevelEnum.high.value: {"min": 0.7, "max": 0.85},
            AlertLevelEnum.critical.value: {"min": 0.85, "max": 1.0},
        },
        "description": (
            "Alert levels are determined by the model's predicted probability "
            "of a significant (5%+) portfolio drawdown over the next 10-20 days."
        ),
    }
