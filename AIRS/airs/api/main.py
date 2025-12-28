"""
FastAPI application for AIRS.

Main application factory and configuration.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from airs.api.routes import alerts, features, recommendations, backtest, health
from airs.config.settings import get_settings
from airs.utils.logging import get_logger

logger = get_logger(__name__)

# Track startup time for uptime calculation
_startup_time: datetime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _startup_time

    # Startup
    logger.info("Starting AIRS API server...")
    _startup_time = datetime.utcnow()

    # Initialize resources
    await initialize_resources()

    logger.info("AIRS API server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down AIRS API server...")
    await cleanup_resources()
    logger.info("AIRS API server stopped")


async def initialize_resources():
    """Initialize application resources."""
    # Initialize database connection pool
    # Initialize model cache
    # Initialize feature store connection
    logger.info("Resources initialized")


async def cleanup_resources():
    """Cleanup application resources."""
    # Close database connections
    # Clear caches
    logger.info("Resources cleaned up")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI application
    """
    settings = get_settings()

    app = FastAPI(
        title="AIRS API",
        description=(
            "AI-Driven Early-Warning System for Portfolio Drawdown Risk. "
            "Provides real-time risk alerts, feature monitoring, and "
            "portfolio recommendations."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add exception handlers
    app.add_exception_handler(Exception, global_exception_handler)

    # Include routers
    app.include_router(
        health.router,
        prefix="/api/v1",
        tags=["Health"],
    )
    app.include_router(
        alerts.router,
        prefix="/api/v1/alerts",
        tags=["Alerts"],
    )
    app.include_router(
        features.router,
        prefix="/api/v1/features",
        tags=["Features"],
    )
    app.include_router(
        recommendations.router,
        prefix="/api/v1/recommendations",
        tags=["Recommendations"],
    )
    app.include_router(
        backtest.router,
        prefix="/api/v1/backtest",
        tags=["Backtest"],
    )

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "AIRS API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if get_settings().debug else None,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


def get_uptime() -> float:
    """Get server uptime in seconds."""
    if _startup_time is None:
        return 0.0
    return (datetime.utcnow() - _startup_time).total_seconds()


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "airs.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
