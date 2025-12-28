"""
Alert management for AIRS.

Handles alert generation, routing, and delivery.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from airs.utils.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""

    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class AlertConfig:
    """Alert configuration."""

    # Thresholds
    probability_warning: float = 0.5
    probability_error: float = 0.7
    probability_critical: float = 0.85

    # Drift thresholds
    drift_warning_psi: float = 0.1
    drift_error_psi: float = 0.2

    # Performance thresholds
    performance_degradation_warning: float = 0.05
    performance_degradation_error: float = 0.10

    # Channels
    channels: list[AlertChannel] = field(
        default_factory=lambda: [AlertChannel.LOG, AlertChannel.EMAIL]
    )

    # Rate limiting
    min_interval_minutes: int = 60
    max_alerts_per_day: int = 10

    # Recipients
    email_recipients: list[str] = field(
        default_factory=lambda: ["alerts@example.com"]
    )
    slack_webhook: str | None = None
    pagerduty_key: str | None = None


@dataclass
class Alert:
    """Alert record."""

    id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


class AlertManager:
    """
    Manage alert generation and delivery.

    Handles deduplication, rate limiting, and multi-channel delivery.
    """

    def __init__(self, config: AlertConfig | None = None):
        """
        Initialize alert manager.

        Args:
            config: Alert configuration
        """
        self.config = config or AlertConfig()
        self.alert_history: list[Alert] = []
        self.handlers: dict[AlertChannel, Callable] = {
            AlertChannel.LOG: self._log_alert,
            AlertChannel.EMAIL: self._send_email,
            AlertChannel.SLACK: self._send_slack,
            AlertChannel.WEBHOOK: self._send_webhook,
        }

    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> Alert | None:
        """
        Create and send an alert.

        Args:
            severity: Alert severity
            title: Alert title
            message: Alert message
            source: Alert source
            metadata: Additional metadata

        Returns:
            Created alert or None if rate limited
        """
        # Check rate limiting
        if self._is_rate_limited(source, severity):
            logger.debug(f"Alert rate limited: {title}")
            return None

        # Create alert
        alert = Alert(
            id=f"{source}_{datetime.utcnow().timestamp()}",
            severity=severity,
            title=title,
            message=message,
            source=source,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )

        # Store in history
        self.alert_history.append(alert)

        # Send to channels
        self._deliver_alert(alert)

        return alert

    def _is_rate_limited(self, source: str, severity: AlertSeverity) -> bool:
        """Check if alert should be rate limited."""
        # Critical alerts are never rate limited
        if severity == AlertSeverity.CRITICAL:
            return False

        now = datetime.utcnow()
        min_interval = timedelta(minutes=self.config.min_interval_minutes)

        # Check recent alerts from same source
        recent = [
            a for a in self.alert_history
            if a.source == source and (now - a.timestamp) < min_interval
        ]

        if recent:
            return True

        # Check daily limit
        today = now.date()
        today_alerts = [
            a for a in self.alert_history
            if a.timestamp.date() == today
        ]

        return len(today_alerts) >= self.config.max_alerts_per_day

    def _deliver_alert(self, alert: Alert) -> None:
        """Deliver alert to configured channels."""
        for channel in self.config.channels:
            handler = self.handlers.get(channel)
            if handler:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Failed to deliver alert via {channel}: {e}")

    def _log_alert(self, alert: Alert) -> None:
        """Log alert to logging system."""
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(alert.severity, logger.info)

        log_func(f"[{alert.source}] {alert.title}: {alert.message}")

    def _send_email(self, alert: Alert) -> None:
        """Send alert via email."""
        # TODO: Implement email sending
        # import smtplib
        # from email.mime.text import MIMEText

        logger.info(f"Would send email alert: {alert.title}")

    def _send_slack(self, alert: Alert) -> None:
        """Send alert to Slack."""
        if not self.config.slack_webhook:
            return

        # TODO: Implement Slack webhook
        # import requests
        # payload = {
        #     "text": f"*{alert.severity.value.upper()}*: {alert.title}\n{alert.message}"
        # }
        # requests.post(self.config.slack_webhook, json=payload)

        logger.info(f"Would send Slack alert: {alert.title}")

    def _send_webhook(self, alert: Alert) -> None:
        """Send alert to generic webhook."""
        # TODO: Implement webhook
        logger.info(f"Would send webhook alert: {alert.title}")

    def create_risk_alert(
        self,
        probability: float,
        key_drivers: list[dict[str, Any]] | None = None,
    ) -> Alert | None:
        """
        Create risk-level alert based on probability.

        Args:
            probability: Current risk probability
            key_drivers: Key risk drivers

        Returns:
            Created alert or None
        """
        # Determine severity
        if probability >= self.config.probability_critical:
            severity = AlertSeverity.CRITICAL
            title = "CRITICAL: High drawdown probability"
        elif probability >= self.config.probability_error:
            severity = AlertSeverity.ERROR
            title = "High risk level detected"
        elif probability >= self.config.probability_warning:
            severity = AlertSeverity.WARNING
            title = "Elevated risk indicators"
        else:
            return None  # No alert needed

        # Build message
        message = f"Current drawdown probability: {probability*100:.1f}%"

        if key_drivers:
            driver_text = ", ".join([d.get("feature", "unknown") for d in key_drivers[:3]])
            message += f"\nKey drivers: {driver_text}"

        return self.create_alert(
            severity=severity,
            title=title,
            message=message,
            source="risk_model",
            metadata={
                "probability": probability,
                "key_drivers": key_drivers,
            },
        )

    def create_drift_alert(
        self,
        drift_results: dict[str, Any],
    ) -> Alert | None:
        """
        Create drift alert.

        Args:
            drift_results: Drift detection results

        Returns:
            Created alert or None
        """
        feature_drift = drift_results.get("feature_drift", {})
        prediction_drift = drift_results.get("prediction_drift", {})

        issues = []

        if feature_drift.get("high_severity", 0) > 0:
            issues.append(
                f"High-severity feature drift in {feature_drift['high_severity']} features"
            )

        if prediction_drift.get("is_drifted"):
            issues.append(
                f"Prediction drift detected (mean shift: {prediction_drift.get('mean_shift', 0):.3f})"
            )

        if not issues:
            return None

        severity = (
            AlertSeverity.ERROR
            if feature_drift.get("high_severity", 0) > 2
            else AlertSeverity.WARNING
        )

        return self.create_alert(
            severity=severity,
            title="Data/Model drift detected",
            message="\n".join(issues),
            source="drift_monitor",
            metadata=drift_results,
        )

    def create_performance_alert(
        self,
        performance_results: dict[str, Any],
    ) -> Alert | None:
        """
        Create performance degradation alert.

        Args:
            performance_results: Performance monitoring results

        Returns:
            Created alert or None
        """
        if not performance_results.get("is_degraded"):
            return None

        precision_deg = performance_results.get("precision_degradation_pct", 0)
        recall_deg = performance_results.get("recall_degradation_pct", 0)

        severity = (
            AlertSeverity.ERROR
            if max(precision_deg, recall_deg) > self.config.performance_degradation_error * 100
            else AlertSeverity.WARNING
        )

        message = (
            f"Model performance degradation detected.\n"
            f"Precision: -{precision_deg:.1f}%\n"
            f"Recall: -{recall_deg:.1f}%"
        )

        return self.create_alert(
            severity=severity,
            title="Model performance degradation",
            message=message,
            source="performance_monitor",
            metadata=performance_results,
        )

    def get_active_alerts(self) -> list[Alert]:
        """Get all unresolved alerts."""
        return [a for a in self.alert_history if not a.resolved]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alert_history:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alert_history:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False

    def get_alert_summary(self, days: int = 7) -> dict[str, Any]:
        """Get alert summary for time period."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [a for a in self.alert_history if a.timestamp >= cutoff]

        by_severity = {}
        for severity in AlertSeverity:
            by_severity[severity.value] = len(
                [a for a in recent if a.severity == severity]
            )

        by_source = {}
        for alert in recent:
            by_source[alert.source] = by_source.get(alert.source, 0) + 1

        return {
            "period_days": days,
            "total_alerts": len(recent),
            "unresolved": len([a for a in recent if not a.resolved]),
            "by_severity": by_severity,
            "by_source": by_source,
        }
