"""
Database session management.

Provides connection pooling, session management, and database initialization.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from airs.config.settings import get_settings
from airs.db.models import Base
from airs.utils.logging import get_logger

logger = get_logger(__name__)

# Global engine and session factory
_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def get_engine(database_url: str | None = None) -> Engine:
    """
    Get or create the database engine.

    Args:
        database_url: Database connection URL (defaults to settings)

    Returns:
        SQLAlchemy Engine
    """
    global _engine

    if _engine is not None:
        return _engine

    settings = get_settings()
    url = database_url or settings.database_dsn

    _engine = create_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        echo=settings.is_development,
    )

    # Add event listeners for connection handling
    @event.listens_for(_engine, "connect")
    def on_connect(dbapi_connection, connection_record):
        logger.debug("Database connection established")

    @event.listens_for(_engine, "checkout")
    def on_checkout(dbapi_connection, connection_record, connection_proxy):
        logger.debug("Connection checked out from pool")

    return _engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker:
    """
    Get or create the session factory.

    Args:
        engine: SQLAlchemy engine (defaults to global engine)

    Returns:
        Session factory
    """
    global _session_factory

    if _session_factory is not None:
        return _session_factory

    if engine is None:
        engine = get_engine()

    _session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )

    return _session_factory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Handles commit/rollback and session cleanup automatically.

    Yields:
        SQLAlchemy Session
    """
    factory = get_session_factory()
    session = factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(drop_existing: bool = False) -> None:
    """
    Initialize the database schema.

    Args:
        drop_existing: If True, drop all existing tables first
    """
    engine = get_engine()

    if drop_existing:
        logger.warning("Dropping all existing tables")
        Base.metadata.drop_all(engine)

    logger.info("Creating database tables")
    Base.metadata.create_all(engine)

    # Verify tables were created
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
        )
        tables = [row[0] for row in result]
        logger.info(f"Created tables: {tables}")


def check_connection() -> bool:
    """
    Check if database connection is working.

    Returns:
        True if connection is successful
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def close_connections() -> None:
    """Close all database connections and dispose of the engine."""
    global _engine, _session_factory

    if _engine is not None:
        _engine.dispose()
        _engine = None

    _session_factory = None
    logger.info("Database connections closed")
