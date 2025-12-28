"""
Repository pattern for database operations.

Provides clean interface for CRUD operations on database models.
"""

from datetime import date, datetime
from typing import Generic, TypeVar, Type, Sequence

import pandas as pd
from sqlalchemy import select, delete, and_, func
from sqlalchemy.orm import Session

from airs.db.models import (
    Base,
    MarketData,
    EconomicIndicator,
    Feature,
    Prediction,
    PortfolioPosition,
    BacktestRun,
    Alert,
)
from airs.db.session import get_session
from airs.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=Base)


class Repository(Generic[T]):
    """Generic repository for database operations."""

    def __init__(self, model: Type[T], session: Session | None = None):
        """
        Initialize repository.

        Args:
            model: SQLAlchemy model class
            session: Optional session (will create one if not provided)
        """
        self.model = model
        self._session = session

    @property
    def session(self) -> Session:
        """Get current session or raise if not in context."""
        if self._session is None:
            raise RuntimeError("Repository must be used within a session context")
        return self._session

    def get_by_id(self, id: int) -> T | None:
        """Get record by ID."""
        return self.session.get(self.model, id)

    def get_all(self, limit: int = 1000) -> Sequence[T]:
        """Get all records (with limit)."""
        stmt = select(self.model).limit(limit)
        return self.session.execute(stmt).scalars().all()

    def add(self, entity: T) -> T:
        """Add a new record."""
        self.session.add(entity)
        self.session.flush()
        return entity

    def add_all(self, entities: list[T]) -> list[T]:
        """Add multiple records."""
        self.session.add_all(entities)
        self.session.flush()
        return entities

    def delete(self, entity: T) -> None:
        """Delete a record."""
        self.session.delete(entity)
        self.session.flush()

    def delete_by_id(self, id: int) -> bool:
        """Delete record by ID."""
        entity = self.get_by_id(id)
        if entity:
            self.delete(entity)
            return True
        return False

    def count(self) -> int:
        """Count all records."""
        stmt = select(func.count()).select_from(self.model)
        return self.session.execute(stmt).scalar() or 0


class MarketDataRepository(Repository[MarketData]):
    """Repository for market data operations."""

    def __init__(self, session: Session | None = None):
        super().__init__(MarketData, session)

    def get_by_symbol(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> Sequence[MarketData]:
        """Get market data for a symbol within date range."""
        stmt = select(self.model).where(self.model.symbol == symbol)

        if start_date:
            stmt = stmt.where(self.model.date >= start_date)
        if end_date:
            stmt = stmt.where(self.model.date <= end_date)

        stmt = stmt.order_by(self.model.date)
        return self.session.execute(stmt).scalars().all()

    def get_symbols(self) -> list[str]:
        """Get list of unique symbols."""
        stmt = select(self.model.symbol).distinct()
        return list(self.session.execute(stmt).scalars().all())

    def get_date_range(self, symbol: str) -> tuple[date, date] | None:
        """Get date range for a symbol."""
        stmt = select(
            func.min(self.model.date),
            func.max(self.model.date),
        ).where(self.model.symbol == symbol)

        result = self.session.execute(stmt).first()
        if result and result[0] and result[1]:
            return (result[0], result[1])
        return None

    def upsert_from_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        source: str = "yahoo",
    ) -> int:
        """
        Upsert market data from DataFrame.

        Args:
            df: DataFrame with OHLCV data (index is date)
            symbol: Symbol name
            source: Data source name

        Returns:
            Number of records upserted
        """
        count = 0

        for idx, row in df.iterrows():
            record_date = idx.date() if hasattr(idx, "date") else idx

            # Check if record exists
            stmt = select(self.model).where(
                and_(
                    self.model.date == record_date,
                    self.model.symbol == symbol,
                )
            )
            existing = self.session.execute(stmt).scalar_one_or_none()

            if existing:
                # Update
                existing.open = row.get("open")
                existing.high = row.get("high")
                existing.low = row.get("low")
                existing.close = row["close"]
                existing.volume = row.get("volume")
                existing.adjusted_close = row.get("adjusted_close", row.get("close"))
                existing.updated_at = datetime.utcnow()
            else:
                # Insert
                record = MarketData(
                    date=record_date,
                    symbol=symbol,
                    open=row.get("open"),
                    high=row.get("high"),
                    low=row.get("low"),
                    close=row["close"],
                    volume=row.get("volume"),
                    adjusted_close=row.get("adjusted_close", row.get("close")),
                    source=source,
                )
                self.session.add(record)
                count += 1

        self.session.flush()
        return count

    def to_dataframe(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Convert market data to DataFrame."""
        records = self.get_by_symbol(symbol, start_date, end_date)

        if not records:
            return pd.DataFrame()

        data = [
            {
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "adjusted_close": r.adjusted_close,
            }
            for r in records
        ]

        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        return df


class EconomicIndicatorRepository(Repository[EconomicIndicator]):
    """Repository for economic indicator data."""

    def __init__(self, session: Session | None = None):
        super().__init__(EconomicIndicator, session)

    def get_by_series(
        self,
        series_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> Sequence[EconomicIndicator]:
        """Get indicators for a series within date range."""
        stmt = select(self.model).where(self.model.series_id == series_id)

        if start_date:
            stmt = stmt.where(self.model.date >= start_date)
        if end_date:
            stmt = stmt.where(self.model.date <= end_date)

        stmt = stmt.order_by(self.model.date)
        return self.session.execute(stmt).scalars().all()

    def get_series_ids(self) -> list[str]:
        """Get list of unique series IDs."""
        stmt = select(self.model.series_id).distinct()
        return list(self.session.execute(stmt).scalars().all())

    def to_dataframe(
        self,
        series_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Convert series to DataFrame."""
        records = self.get_by_series(series_id, start_date, end_date)

        if not records:
            return pd.DataFrame()

        data = [{"date": r.date, series_id: r.value} for r in records]

        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        return df


class FeatureRepository(Repository[Feature]):
    """Repository for computed features."""

    def __init__(self, session: Session | None = None):
        super().__init__(Feature, session)

    def get_by_date(
        self,
        target_date: date,
        version: str = "v1",
    ) -> Sequence[Feature]:
        """Get all features for a specific date."""
        stmt = select(self.model).where(
            and_(
                self.model.date == target_date,
                self.model.version == version,
            )
        )
        return self.session.execute(stmt).scalars().all()

    def get_feature_matrix(
        self,
        start_date: date,
        end_date: date,
        feature_names: list[str] | None = None,
        version: str = "v1",
    ) -> pd.DataFrame:
        """Get feature matrix as DataFrame."""
        stmt = select(self.model).where(
            and_(
                self.model.date >= start_date,
                self.model.date <= end_date,
                self.model.version == version,
            )
        )

        if feature_names:
            stmt = stmt.where(self.model.feature_name.in_(feature_names))

        records = self.session.execute(stmt).scalars().all()

        if not records:
            return pd.DataFrame()

        # Pivot to wide format
        data = [
            {"date": r.date, "feature": r.feature_name, "value": r.value}
            for r in records
        ]
        df = pd.DataFrame(data)
        df = df.pivot(index="date", columns="feature", values="value")

        return df

    def save_features(
        self,
        df: pd.DataFrame,
        feature_group: str | None = None,
        version: str = "v1",
    ) -> int:
        """
        Save feature DataFrame to database.

        Args:
            df: DataFrame with features (index is date, columns are feature names)
            feature_group: Optional feature group name
            version: Feature version

        Returns:
            Number of features saved
        """
        count = 0

        for idx, row in df.iterrows():
            feature_date = idx.date() if hasattr(idx, "date") else idx

            for feature_name, value in row.items():
                if pd.isna(value):
                    continue

                # Delete existing
                stmt = delete(self.model).where(
                    and_(
                        self.model.date == feature_date,
                        self.model.feature_name == feature_name,
                        self.model.version == version,
                    )
                )
                self.session.execute(stmt)

                # Insert new
                feature = Feature(
                    date=feature_date,
                    feature_name=feature_name,
                    value=float(value),
                    feature_group=feature_group,
                    version=version,
                )
                self.session.add(feature)
                count += 1

        self.session.flush()
        return count


class PredictionRepository(Repository[Prediction]):
    """Repository for model predictions."""

    def __init__(self, session: Session | None = None):
        super().__init__(Prediction, session)

    def get_latest(self, model_id: str | None = None) -> Prediction | None:
        """Get most recent prediction."""
        stmt = select(self.model).order_by(self.model.date.desc())

        if model_id:
            stmt = stmt.where(self.model.model_id == model_id)

        stmt = stmt.limit(1)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_alerts(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> Sequence[Prediction]:
        """Get predictions where alert was triggered."""
        stmt = select(self.model).where(self.model.alert_triggered == True)

        if start_date:
            stmt = stmt.where(self.model.date >= start_date)
        if end_date:
            stmt = stmt.where(self.model.date <= end_date)

        stmt = stmt.order_by(self.model.date.desc())
        return self.session.execute(stmt).scalars().all()


class AlertRepository(Repository[Alert]):
    """Repository for alerts."""

    def __init__(self, session: Session | None = None):
        super().__init__(Alert, session)

    def get_unacknowledged(self) -> Sequence[Alert]:
        """Get all unacknowledged alerts."""
        stmt = (
            select(self.model)
            .where(self.model.acknowledged == False)
            .order_by(self.model.created_at.desc())
        )
        return self.session.execute(stmt).scalars().all()

    def acknowledge(self, alert_id: int) -> bool:
        """Acknowledge an alert."""
        alert = self.get_by_id(alert_id)
        if alert:
            alert.acknowledged = True
            alert.acknowledged_at = datetime.utcnow()
            self.session.flush()
            return True
        return False

    def get_history(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        level: str | None = None,
    ) -> Sequence[Alert]:
        """Get alert history."""
        stmt = select(self.model)

        if start_date:
            stmt = stmt.where(self.model.date >= start_date)
        if end_date:
            stmt = stmt.where(self.model.date <= end_date)
        if level:
            stmt = stmt.where(self.model.alert_level == level)

        stmt = stmt.order_by(self.model.date.desc())
        return self.session.execute(stmt).scalars().all()
