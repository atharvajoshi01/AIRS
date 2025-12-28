"""
SQLAlchemy ORM models for AIRS database.

Defines tables for market data, economic indicators, features, and predictions.
"""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class MarketData(Base):
    """
    Market price data for assets.

    Stores OHLCV data for ETFs, indices, and other securities.
    """

    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    open: Mapped[Optional[float]] = mapped_column(Float)
    high: Mapped[Optional[float]] = mapped_column(Float)
    low: Mapped[Optional[float]] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[Optional[int]] = mapped_column(Integer)
    adjusted_close: Mapped[Optional[float]] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(50), default="yahoo")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("date", "symbol", name="uq_market_data_date_symbol"),
        Index("ix_market_data_symbol", "symbol"),
        Index("ix_market_data_date", "date"),
        Index("ix_market_data_symbol_date", "symbol", "date"),
    )

    def __repr__(self) -> str:
        return f"<MarketData({self.symbol}, {self.date}, close={self.close})>"


class EconomicIndicator(Base):
    """
    Economic indicator data from FRED and other sources.

    Stores macro indicators, yields, spreads, and other economic data.
    """

    __tablename__ = "economic_indicators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    series_id: Mapped[str] = mapped_column(String(50), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    release_date: Mapped[Optional[date]] = mapped_column(Date)
    units: Mapped[Optional[str]] = mapped_column(String(50))
    frequency: Mapped[str] = mapped_column(String(20), default="daily")
    source: Mapped[str] = mapped_column(String(50), default="fred")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("date", "series_id", name="uq_econ_date_series"),
        Index("ix_econ_series_id", "series_id"),
        Index("ix_econ_date", "date"),
        Index("ix_econ_series_date", "series_id", "date"),
    )

    def __repr__(self) -> str:
        return f"<EconomicIndicator({self.series_id}, {self.date}, {self.value})>"


class Feature(Base):
    """
    Computed features for model input.

    Stores daily feature values computed from raw data.
    """

    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    feature_name: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    feature_group: Mapped[Optional[str]] = mapped_column(String(50))
    version: Mapped[str] = mapped_column(String(20), default="v1")
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "date", "feature_name", "version", name="uq_feature_date_name_version"
        ),
        Index("ix_feature_name", "feature_name"),
        Index("ix_feature_date", "date"),
        Index("ix_feature_group", "feature_group"),
    )

    def __repr__(self) -> str:
        return f"<Feature({self.feature_name}, {self.date}, {self.value})>"


class Prediction(Base):
    """
    Model predictions and alerts.

    Stores daily predictions, alert levels, and associated metadata.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, default=0.5)
    alert_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_level: Mapped[Optional[str]] = mapped_column(String(20))
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    top_features: Mapped[Optional[str]] = mapped_column(Text)  # JSON string
    regime: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "date", "model_id", "model_version", name="uq_pred_date_model"
        ),
        Index("ix_pred_date", "date"),
        Index("ix_pred_model", "model_id"),
        Index("ix_pred_alert", "alert_triggered"),
    )

    def __repr__(self) -> str:
        return f"<Prediction({self.model_id}, {self.date}, prob={self.probability:.2f})>"


class PortfolioPosition(Base):
    """
    Portfolio positions and values.

    Tracks daily portfolio state for backtesting and live monitoring.
    """

    __tablename__ = "portfolio_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    portfolio_id: Mapped[str] = mapped_column(String(50), default="default")
    asset: Mapped[str] = mapped_column(String(20), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    shares: Mapped[Optional[float]] = mapped_column(Float)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    cost_basis: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "date", "portfolio_id", "asset", name="uq_position_date_portfolio_asset"
        ),
        Index("ix_position_date", "date"),
        Index("ix_position_portfolio", "portfolio_id"),
    )

    def __repr__(self) -> str:
        return f"<PortfolioPosition({self.asset}, {self.date}, weight={self.weight:.2%})>"


class BacktestRun(Base):
    """
    Backtest run metadata and results.

    Stores configuration and performance metrics for backtests.
    """

    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    config: Mapped[str] = mapped_column(Text)  # JSON string
    total_return: Mapped[float] = mapped_column(Float)
    annualized_return: Mapped[Optional[float]] = mapped_column(Float)
    volatility: Mapped[Optional[float]] = mapped_column(Float)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float)
    benchmark_return: Mapped[Optional[float]] = mapped_column(Float)
    excess_return: Mapped[Optional[float]] = mapped_column(Float)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_backtest_model", "model_id"),
        Index("ix_backtest_date_range", "start_date", "end_date"),
    )

    def __repr__(self) -> str:
        return f"<BacktestRun({self.run_id}, sharpe={self.sharpe_ratio:.2f})>"


class Alert(Base):
    """
    Generated alerts and notifications.

    Stores alert history for monitoring and analysis.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    alert_level: Mapped[str] = mapped_column(String(20), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    recommendations: Mapped[Optional[str]] = mapped_column(Text)  # JSON string
    top_drivers: Mapped[Optional[str]] = mapped_column(Text)  # JSON string
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_alert_date", "date"),
        Index("ix_alert_level", "alert_level"),
        Index("ix_alert_acknowledged", "acknowledged"),
    )

    def __repr__(self) -> str:
        return f"<Alert({self.alert_level}, {self.date}, prob={self.probability:.2f})>"
