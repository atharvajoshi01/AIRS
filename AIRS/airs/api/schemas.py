"""
Pydantic schemas for AIRS API.

Request and response models for API endpoints.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# Enums
class AlertLevelEnum(str, Enum):
    """Alert severity levels."""

    none = "none"
    low = "low"
    moderate = "moderate"
    high = "high"
    critical = "critical"


class TimeframeEnum(str, Enum):
    """Data timeframe options."""

    day_1 = "1d"
    week_1 = "1w"
    month_1 = "1m"
    month_3 = "3m"
    month_6 = "6m"
    year_1 = "1y"
    year_5 = "5y"
    all = "all"


# Base schemas
class TimestampedResponse(BaseModel):
    """Base response with timestamp."""

    timestamp: datetime = Field(..., description="Response timestamp")
    version: str = Field(default="1.0", description="API version")


# Alert schemas
class AlertResponse(BaseModel):
    """Current alert status."""

    alert_level: AlertLevelEnum
    probability: float = Field(..., ge=0, le=1, description="Drawdown probability")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    headline: str
    summary: str
    last_updated: datetime


class AlertHistoryItem(BaseModel):
    """Historical alert record."""

    date: date
    alert_level: AlertLevelEnum
    probability: float
    action_taken: str | None = None


class AlertHistoryResponse(TimestampedResponse):
    """Historical alerts response."""

    alerts: list[AlertHistoryItem]
    total_count: int


# Feature schemas
class FeatureValue(BaseModel):
    """Single feature value."""

    name: str
    value: float
    percentile: float | None = None
    z_score: float | None = None
    category: str | None = None


class FeatureResponse(TimestampedResponse):
    """Current features response."""

    features: list[FeatureValue]
    feature_date: date


class FeatureHistoryRequest(BaseModel):
    """Request for feature history."""

    feature_names: list[str] | None = None
    start_date: date | None = None
    end_date: date | None = None
    timeframe: TimeframeEnum = TimeframeEnum.month_1


class FeatureHistoryResponse(TimestampedResponse):
    """Feature history response."""

    features: dict[str, list[dict[str, Any]]]
    start_date: date
    end_date: date


# Recommendation schemas
class AssetRecommendation(BaseModel):
    """Single asset recommendation."""

    symbol: str
    current_weight: float = Field(..., ge=0, le=1)
    target_weight: float = Field(..., ge=0, le=1)
    action: str  # buy, sell, hold
    urgency: str  # immediate, gradual, optional
    rationale: str


class KeyDriver(BaseModel):
    """Key risk driver."""

    feature: str
    contribution: float
    direction: str
    magnitude: str


class RecommendationResponse(TimestampedResponse):
    """Full recommendation response."""

    alert_level: AlertLevelEnum
    probability: float
    confidence: float
    headline: str
    summary: str
    asset_recommendations: list[AssetRecommendation]
    key_drivers: list[KeyDriver]
    historical_context: str
    suggested_timeline: str
    estimated_turnover: float


# Backtest schemas
class BacktestRequest(BaseModel):
    """Backtest configuration request."""

    start_date: date | None = Field(
        None, description="Backtest start date (default: 5 years ago)"
    )
    end_date: date | None = Field(
        None, description="Backtest end date (default: today)"
    )
    initial_value: float = Field(
        default=100000.0, ge=1000, description="Initial portfolio value"
    )
    target_weights: dict[str, float] | None = Field(
        None, description="Target asset allocation"
    )
    alert_threshold: float = Field(
        default=0.5, ge=0, le=1, description="Alert trigger threshold"
    )
    derisk_equity_reduction: float = Field(
        default=0.5, ge=0, le=1, description="Equity reduction on alert"
    )
    trading_cost_bps: float = Field(
        default=10.0, ge=0, description="Trading cost in basis points"
    )


class BacktestMetrics(BaseModel):
    """Backtest performance metrics."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    var_95: float


class BacktestComparison(BaseModel):
    """Strategy vs benchmark comparison."""

    strategy: BacktestMetrics
    benchmark: BacktestMetrics
    excess_return: float
    excess_sharpe: float
    drawdown_improvement: float
    n_alerts: int
    n_trades: int


class BacktestResponse(TimestampedResponse):
    """Backtest results response."""

    config: BacktestRequest
    comparison: BacktestComparison
    stress_periods: list[dict[str, Any]]
    portfolio_values: list[dict[str, Any]] | None = None


# Health schemas
class ServiceStatus(BaseModel):
    """Individual service status."""

    name: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: float | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    """System health response."""

    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    services: list[ServiceStatus]
    version: str
    uptime_seconds: float


# Model info schemas
class ModelInfo(BaseModel):
    """Model metadata."""

    model_id: str
    model_type: str
    trained_date: datetime
    features_count: int
    training_samples: int
    validation_metrics: dict[str, float]


class ModelInfoResponse(TimestampedResponse):
    """Model information response."""

    current_model: ModelInfo
    available_models: list[ModelInfo]


# Error schemas
class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Portfolio schemas
class PortfolioPosition(BaseModel):
    """Portfolio position."""

    symbol: str
    shares: float
    price: float
    value: float
    weight: float
    unrealized_pnl: float


class PortfolioResponse(TimestampedResponse):
    """Current portfolio state."""

    total_value: float
    cash: float
    positions: list[PortfolioPosition]
    weights: dict[str, float]
    total_pnl: float
    total_return_pct: float
