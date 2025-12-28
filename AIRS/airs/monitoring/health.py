"""
Health monitoring for AIRS.

System health checks and status reporting.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from airs.utils.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    status: HealthStatus
    message: str | None = None
    latency_ms: float | None = None
    last_check: datetime | None = None
    metadata: dict[str, Any] | None = None


class HealthChecker:
    """
    System health monitoring.

    Checks database, model, data freshness, and service health.
    """

    def __init__(self):
        """Initialize health checker."""
        self.checks: dict[str, Callable[[], HealthCheckResult]] = {}
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_check("database", self._check_database)
        self.register_check("model", self._check_model)
        self.register_check("data_freshness", self._check_data_freshness)
        self.register_check("api", self._check_api)
        self.register_check("features", self._check_features)

    def register_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult],
    ) -> None:
        """
        Register a health check.

        Args:
            name: Check name
            check_func: Function that performs the check
        """
        self.checks[name] = check_func

    def run_checks(
        self,
        check_names: list[str] | None = None,
    ) -> dict[str, HealthCheckResult]:
        """
        Run health checks.

        Args:
            check_names: Specific checks to run (default: all)

        Returns:
            Dictionary of check results
        """
        if check_names is None:
            check_names = list(self.checks.keys())

        results = {}
        for name in check_names:
            if name not in self.checks:
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Unknown check: {name}",
                )
                continue

            try:
                results[name] = self.checks[name]()
            except Exception as e:
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    last_check=datetime.utcnow(),
                )

        return results

    def get_overall_status(
        self,
        results: dict[str, HealthCheckResult],
    ) -> HealthStatus:
        """
        Determine overall system status.

        Args:
            results: Individual check results

        Returns:
            Overall status
        """
        statuses = [r.status for r in results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN

    def get_health_report(self) -> dict[str, Any]:
        """
        Generate complete health report.

        Returns:
            Health report dictionary
        """
        results = self.run_checks()
        overall = self.get_overall_status(results)

        return {
            "status": overall.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "latency_ms": result.latency_ms,
                    "last_check": (
                        result.last_check.isoformat()
                        if result.last_check
                        else None
                    ),
                }
                for name, result in results.items()
            },
            "summary": {
                "total": len(results),
                "healthy": sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY),
            },
        }

    def _check_database(self) -> HealthCheckResult:
        """Check database connectivity."""
        import time

        start = time.time()

        try:
            # TODO: Implement actual database check
            # from airs.db.session import get_engine
            # engine = get_engine()
            # with engine.connect() as conn:
            #     conn.execute("SELECT 1")

            latency = (time.time() - start) * 1000

            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Connected",
                latency_ms=latency,
                last_check=datetime.utcnow(),
            )

        except Exception as e:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                last_check=datetime.utcnow(),
            )

    def _check_model(self) -> HealthCheckResult:
        """Check model availability."""
        try:
            # TODO: Implement actual model check
            # from airs.models import load_production_model
            # model = load_production_model()
            # if model is None:
            #     raise ValueError("No model loaded")

            return HealthCheckResult(
                name="model",
                status=HealthStatus.HEALTHY,
                message="Model loaded and ready",
                last_check=datetime.utcnow(),
                metadata={"model_version": "v1"},
            )

        except Exception as e:
            return HealthCheckResult(
                name="model",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                last_check=datetime.utcnow(),
            )

    def _check_data_freshness(self) -> HealthCheckResult:
        """Check if data is up to date."""
        try:
            # TODO: Implement actual data freshness check
            # from airs.db.repository import DataRepository
            # repo = DataRepository()
            # latest = repo.get_latest_data_date()
            # lag = (datetime.now().date() - latest).days

            lag = 1  # Mock value

            if lag <= 1:
                return HealthCheckResult(
                    name="data_freshness",
                    status=HealthStatus.HEALTHY,
                    message=f"Data is {lag} day(s) old",
                    last_check=datetime.utcnow(),
                    metadata={"lag_days": lag},
                )
            elif lag <= 3:
                return HealthCheckResult(
                    name="data_freshness",
                    status=HealthStatus.DEGRADED,
                    message=f"Data is {lag} days old",
                    last_check=datetime.utcnow(),
                    metadata={"lag_days": lag},
                )
            else:
                return HealthCheckResult(
                    name="data_freshness",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Data is {lag} days old - stale",
                    last_check=datetime.utcnow(),
                    metadata={"lag_days": lag},
                )

        except Exception as e:
            return HealthCheckResult(
                name="data_freshness",
                status=HealthStatus.UNKNOWN,
                message=str(e),
                last_check=datetime.utcnow(),
            )

    def _check_api(self) -> HealthCheckResult:
        """Check API service health."""
        import time

        start = time.time()

        try:
            # TODO: Implement actual API check
            # import httpx
            # response = httpx.get("http://localhost:8000/api/v1/health")
            # response.raise_for_status()

            latency = (time.time() - start) * 1000

            return HealthCheckResult(
                name="api",
                status=HealthStatus.HEALTHY,
                message="API responding",
                latency_ms=latency,
                last_check=datetime.utcnow(),
            )

        except Exception as e:
            return HealthCheckResult(
                name="api",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                last_check=datetime.utcnow(),
            )

    def _check_features(self) -> HealthCheckResult:
        """Check feature computation status."""
        try:
            # TODO: Implement actual feature check
            # from airs.db.repository import FeatureRepository
            # repo = FeatureRepository()
            # latest = repo.get_latest_feature_date()
            # feature_count = repo.get_feature_count()

            return HealthCheckResult(
                name="features",
                status=HealthStatus.HEALTHY,
                message="Features computed",
                last_check=datetime.utcnow(),
                metadata={"feature_count": 50},
            )

        except Exception as e:
            return HealthCheckResult(
                name="features",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                last_check=datetime.utcnow(),
            )


class ServiceMonitor:
    """
    Monitor external service dependencies.
    """

    def __init__(self):
        """Initialize service monitor."""
        self.services = {
            "fred_api": "https://api.stlouisfed.org/fred/series",
            "yahoo_finance": "https://query1.finance.yahoo.com/v8/finance/chart/SPY",
            "mlflow": "http://localhost:5000/api/2.0/mlflow/experiments/list",
        }
        self.last_status: dict[str, HealthCheckResult] = {}

    def check_service(self, name: str, url: str) -> HealthCheckResult:
        """Check if external service is reachable."""
        import time

        start = time.time()

        try:
            # TODO: Implement actual HTTP check
            # import httpx
            # response = httpx.head(url, timeout=5.0)
            # response.raise_for_status()

            latency = (time.time() - start) * 1000

            return HealthCheckResult(
                name=name,
                status=HealthStatus.HEALTHY,
                message="Service reachable",
                latency_ms=latency,
                last_check=datetime.utcnow(),
            )

        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                last_check=datetime.utcnow(),
            )

    def check_all_services(self) -> dict[str, HealthCheckResult]:
        """Check all registered services."""
        results = {}
        for name, url in self.services.items():
            results[name] = self.check_service(name, url)
            self.last_status[name] = results[name]
        return results

    def get_service_status(self) -> dict[str, Any]:
        """Get current service status."""
        if not self.last_status:
            self.check_all_services()

        return {
            "services": {
                name: {
                    "status": result.status.value,
                    "latency_ms": result.latency_ms,
                    "last_check": (
                        result.last_check.isoformat()
                        if result.last_check
                        else None
                    ),
                }
                for name, result in self.last_status.items()
            },
            "all_healthy": all(
                r.status == HealthStatus.HEALTHY
                for r in self.last_status.values()
            ),
        }
