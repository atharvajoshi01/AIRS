"""
Data schemas and models for AIRS.

Defines Pydantic models for data validation and type safety.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class MarketDataPoint(BaseModel):
    """Single market data observation."""

    date: date
    symbol: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float
    volume: int | None = None
    adjusted_close: float | None = None

    @field_validator("close", "open", "high", "low", "adjusted_close", mode="before")
    @classmethod
    def validate_price(cls, v):
        if v is not None and v < 0:
            raise ValueError("Price cannot be negative")
        return v


class EconomicIndicator(BaseModel):
    """Economic indicator observation."""

    date: date
    series_id: str
    value: float
    release_date: date | None = None
    units: str | None = None
    frequency: Literal["daily", "weekly", "monthly", "quarterly"] = "daily"


class YieldCurvePoint(BaseModel):
    """Yield curve observation."""

    date: date
    maturity_3m: float | None = Field(None, alias="DTB3")
    maturity_2y: float | None = Field(None, alias="DGS2")
    maturity_5y: float | None = Field(None, alias="DGS5")
    maturity_10y: float | None = Field(None, alias="DGS10")
    maturity_30y: float | None = Field(None, alias="DGS30")


class CreditSpread(BaseModel):
    """Credit spread observation."""

    date: date
    ig_spread: float | None = Field(None, description="Investment Grade OAS")
    hy_spread: float | None = Field(None, description="High Yield OAS")
    hy_ig_diff: float | None = Field(None, description="HY - IG spread")


class VolatilityData(BaseModel):
    """Volatility index observation."""

    date: date
    vix: float
    vix_3m: float | None = None
    vix_6m: float | None = None
    vvix: float | None = None
    realized_vol_21d: float | None = None


class PortfolioAllocation(BaseModel):
    """Portfolio allocation specification."""

    us_equity: float = Field(0.4, ge=0, le=1)
    intl_equity: float = Field(0.2, ge=0, le=1)
    bonds: float = Field(0.25, ge=0, le=1)
    commodities: float = Field(0.1, ge=0, le=1)
    reits: float = Field(0.05, ge=0, le=1)

    @field_validator("us_equity", "intl_equity", "bonds", "commodities", "reits")
    @classmethod
    def validate_weight(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Weight must be between 0 and 1")
        return v

    def total_weight(self) -> float:
        return (
            self.us_equity
            + self.intl_equity
            + self.bonds
            + self.commodities
            + self.reits
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "SPY": self.us_equity,
            "VEU": self.intl_equity,
            "AGG": self.bonds,
            "DJP": self.commodities,
            "VNQ": self.reits,
        }


class FeatureSet(BaseModel):
    """Feature set for model input."""

    date: date

    # Rate features
    yield_curve_slope: float | None = None
    yield_curve_curvature: float | None = None
    rate_10y_zscore: float | None = None
    curve_inversion_flag: bool = False

    # Credit features
    ig_spread_pct: float | None = None
    hy_spread_pct: float | None = None
    credit_momentum: float | None = None

    # Volatility features
    vix_level: float | None = None
    vix_percentile: float | None = None
    vix_term_structure: float | None = None

    # Macro features
    lei_momentum: float | None = None
    financial_conditions: float | None = None

    # Regime
    regime: Literal["low_vol", "high_vol", "transition"] | None = None
    regime_probability: float | None = None


class PredictionOutput(BaseModel):
    """Model prediction output."""

    date: date
    probability: float = Field(ge=0, le=1)
    threshold_triggered: bool
    confidence: float | None = Field(None, ge=0, le=1)
    top_features: list[tuple[str, float]] | None = None
    regime: str | None = None
    model_version: str | None = None


class AlertLevel(BaseModel):
    """Alert level specification."""

    level: Literal["low", "medium", "high", "critical"]
    probability: float
    message: str
    recommendations: list[str] = []
    triggered_at: datetime = Field(default_factory=datetime.now)


class BacktestResult(BaseModel):
    """Backtest result summary."""

    start_date: date
    end_date: date
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float | None = None
    max_drawdown: float
    calmar_ratio: float | None = None
    win_rate: float | None = None
    total_trades: int = 0
    benchmark_return: float | None = None
    excess_return: float | None = None


class DataFetchRequest(BaseModel):
    """Request for data fetching."""

    symbols: list[str]
    start_date: date
    end_date: date
    source: Literal["fred", "yahoo", "alpha_vantage"]
    include_cache: bool = True


class DataFetchResponse(BaseModel):
    """Response from data fetching."""

    success: bool
    rows_fetched: int
    symbols_fetched: list[str]
    date_range: tuple[date, date] | None = None
    errors: list[str] = []
    cached: bool = False
