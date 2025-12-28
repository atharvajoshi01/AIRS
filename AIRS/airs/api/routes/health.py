"""
Health check endpoints for AIRS API.
"""

from datetime import datetime
import time

from fastapi import APIRouter, Depends

from airs.api.schemas import HealthResponse, ServiceStatus
from airs.config.settings import get_settings, Settings
from airs.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


async def check_database_health() -> ServiceStatus:
    """Check database connection health."""
    start = time.time()
    try:
        # TODO: Implement actual database health check
        # from airs.db.session import get_session
        # async with get_session() as session:
        #     await session.execute("SELECT 1")
        latency = (time.time() - start) * 1000
        return ServiceStatus(
            name="database",
            status="healthy",
            latency_ms=latency,
            message="Connected",
        )
    except Exception as e:
        return ServiceStatus(
            name="database",
            status="unhealthy",
            latency_ms=None,
            message=str(e),
        )


async def check_model_health() -> ServiceStatus:
    """Check ML model availability."""
    try:
        # TODO: Implement actual model health check
        # Check if model is loaded and can make predictions
        return ServiceStatus(
            name="model",
            status="healthy",
            latency_ms=None,
            message="Model loaded",
        )
    except Exception as e:
        return ServiceStatus(
            name="model",
            status="unhealthy",
            latency_ms=None,
            message=str(e),
        )


async def check_data_freshness() -> ServiceStatus:
    """Check if data is up to date."""
    try:
        # TODO: Implement data freshness check
        # Check if last data update was within acceptable window
        return ServiceStatus(
            name="data",
            status="healthy",
            latency_ms=None,
            message="Data up to date",
        )
    except Exception as e:
        return ServiceStatus(
            name="data",
            status="degraded",
            latency_ms=None,
            message=str(e),
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """
    Check system health.

    Returns status of all services and overall system health.
    """
    from airs.api.main import get_uptime

    # Check all services
    services = [
        await check_database_health(),
        await check_model_health(),
        await check_data_freshness(),
    ]

    # Determine overall status
    statuses = [s.status for s in services]
    if all(s == "healthy" for s in statuses):
        overall_status = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        services=services,
        version="1.0.0",
        uptime_seconds=get_uptime(),
    )


@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe.

    Returns 200 if the service is alive.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_check():
    """
    Kubernetes readiness probe.

    Returns 200 if the service is ready to accept traffic.
    """
    # Check critical services
    db_status = await check_database_health()
    model_status = await check_model_health()

    if db_status.status == "unhealthy" or model_status.status == "unhealthy":
        return {"status": "not ready", "reason": "Critical service unavailable"}

    return {"status": "ready"}
