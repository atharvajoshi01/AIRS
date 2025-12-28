"""
Data quality monitoring DAG for AIRS.

Runs daily data quality checks and drift detection.
"""

from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup


# Default arguments
default_args = {
    "owner": "airs",
    "depends_on_past": False,
    "email": ["data-alerts@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}


def check_data_freshness(**context) -> dict[str, Any]:
    """Check if data is up to date."""
    from datetime import date

    execution_date = context["execution_date"]

    # TODO: Query database for latest data dates
    # from airs.db.repository import DataRepository
    # repo = DataRepository()

    # Mock data freshness check
    issues = []

    # Check market data freshness
    # latest_market = repo.get_latest_market_date()
    latest_market = date.today() - timedelta(days=1)
    market_lag = (execution_date.date() - latest_market).days

    if market_lag > 1:
        issues.append(f"Market data is {market_lag} days old")

    # Check FRED data freshness (can have longer lag)
    # latest_fred = repo.get_latest_fred_date()
    latest_fred = date.today() - timedelta(days=2)
    fred_lag = (execution_date.date() - latest_fred).days

    if fred_lag > 3:
        issues.append(f"FRED data is {fred_lag} days old")

    return {
        "market_lag_days": market_lag,
        "fred_lag_days": fred_lag,
        "issues": issues,
        "is_fresh": len(issues) == 0,
    }


def check_missing_values(**context) -> dict[str, Any]:
    """Check for missing values in recent data."""
    # TODO: Query for missing value counts
    # from airs.data.quality import DataQualityChecker
    # checker = DataQualityChecker()
    # results = checker.check_missing_values(lookback_days=7)

    # Mock results
    missing_counts = {
        "SPY_close": 0,
        "VIX": 0,
        "DGS10": 1,  # Treasury data can have gaps
        "BAMLH0A0HYM2": 0,
    }

    total_missing = sum(missing_counts.values())
    critical_missing = missing_counts.get("SPY_close", 0) + missing_counts.get("VIX", 0)

    return {
        "missing_counts": missing_counts,
        "total_missing": total_missing,
        "critical_missing": critical_missing,
        "has_issues": critical_missing > 0,
    }


def check_outliers(**context) -> dict[str, Any]:
    """Detect outliers in recent data."""
    # TODO: Run outlier detection
    # from airs.data.quality import DataQualityChecker
    # checker = DataQualityChecker()
    # results = checker.check_outliers(lookback_days=7, z_threshold=4)

    # Mock results
    outliers = {
        "VIX": {"count": 0, "max_z": 1.5},
        "SPY_returns": {"count": 0, "max_z": 2.1},
        "HY_spread": {"count": 1, "max_z": 4.2},
    }

    total_outliers = sum(o["count"] for o in outliers.values())

    return {
        "outliers": outliers,
        "total_outliers": total_outliers,
        "has_issues": total_outliers > 3,
    }


def check_feature_drift(**context) -> dict[str, Any]:
    """Check for feature distribution drift."""
    import numpy as np

    # TODO: Calculate drift metrics
    # from airs.monitoring.drift import FeatureDriftDetector
    # detector = FeatureDriftDetector()
    # results = detector.calculate_drift(reference_window=252, current_window=21)

    # Mock drift results (PSI - Population Stability Index)
    feature_psi = {
        "vix_level": 0.05,
        "yield_curve_slope": 0.08,
        "hy_spread": 0.12,
        "equity_bond_corr": 0.15,
        "composite_stress_index": 0.07,
    }

    # PSI thresholds: <0.1 = stable, 0.1-0.25 = some change, >0.25 = significant shift
    drifted_features = [f for f, psi in feature_psi.items() if psi > 0.1]
    significant_drift = [f for f, psi in feature_psi.items() if psi > 0.25]

    return {
        "feature_psi": feature_psi,
        "drifted_features": drifted_features,
        "significant_drift": significant_drift,
        "avg_psi": np.mean(list(feature_psi.values())),
        "has_issues": len(significant_drift) > 0,
    }


def check_prediction_drift(**context) -> dict[str, Any]:
    """Check for prediction distribution drift."""
    # TODO: Compare recent predictions to historical
    # from airs.monitoring.drift import PredictionDriftDetector
    # detector = PredictionDriftDetector()
    # results = detector.check_drift(lookback_days=30)

    # Mock results
    historical_mean = 0.35
    recent_mean = 0.42
    mean_shift = recent_mean - historical_mean

    historical_std = 0.15
    recent_std = 0.18
    std_change = (recent_std - historical_std) / historical_std

    return {
        "historical_mean": historical_mean,
        "recent_mean": recent_mean,
        "mean_shift": mean_shift,
        "std_change_pct": std_change * 100,
        "has_issues": abs(mean_shift) > 0.1 or abs(std_change) > 0.3,
    }


def check_model_performance(**context) -> dict[str, Any]:
    """Check recent model performance."""
    # TODO: Calculate recent performance metrics
    # from airs.monitoring.drift import ModelPerformanceMonitor
    # monitor = ModelPerformanceMonitor()
    # results = monitor.check_performance(lookback_days=30)

    # Mock results - compare recent to historical performance
    historical_precision = 0.72
    recent_precision = 0.68
    precision_degradation = (historical_precision - recent_precision) / historical_precision

    historical_recall = 0.65
    recent_recall = 0.60
    recall_degradation = (historical_recall - recent_recall) / historical_recall

    return {
        "historical_precision": historical_precision,
        "recent_precision": recent_precision,
        "precision_degradation_pct": precision_degradation * 100,
        "historical_recall": historical_recall,
        "recent_recall": recent_recall,
        "recall_degradation_pct": recall_degradation * 100,
        "has_issues": precision_degradation > 0.1 or recall_degradation > 0.1,
    }


def aggregate_quality_results(**context) -> dict[str, Any]:
    """Aggregate all quality check results."""
    ti = context["ti"]

    freshness = ti.xcom_pull(task_ids="data_checks.check_freshness")
    missing = ti.xcom_pull(task_ids="data_checks.check_missing")
    outliers = ti.xcom_pull(task_ids="data_checks.check_outliers")
    feature_drift = ti.xcom_pull(task_ids="drift_checks.check_feature_drift")
    prediction_drift = ti.xcom_pull(task_ids="drift_checks.check_prediction_drift")
    performance = ti.xcom_pull(task_ids="drift_checks.check_model_performance")

    all_issues = []

    if not freshness.get("is_fresh", True):
        all_issues.extend(freshness.get("issues", []))

    if missing.get("has_issues", False):
        all_issues.append(f"Critical missing values: {missing['critical_missing']}")

    if outliers.get("has_issues", False):
        all_issues.append(f"Outliers detected: {outliers['total_outliers']}")

    if feature_drift.get("has_issues", False):
        all_issues.append(f"Feature drift: {feature_drift['significant_drift']}")

    if prediction_drift.get("has_issues", False):
        all_issues.append(f"Prediction drift: mean shift {prediction_drift['mean_shift']:.2f}")

    if performance.get("has_issues", False):
        all_issues.append(
            f"Performance degradation: precision -{performance['precision_degradation_pct']:.1f}%"
        )

    # Determine severity
    if len(all_issues) == 0:
        severity = "healthy"
    elif len(all_issues) <= 2:
        severity = "warning"
    else:
        severity = "critical"

    return {
        "severity": severity,
        "issue_count": len(all_issues),
        "issues": all_issues,
        "checks": {
            "freshness": freshness.get("is_fresh", True),
            "missing_values": not missing.get("has_issues", False),
            "outliers": not outliers.get("has_issues", False),
            "feature_drift": not feature_drift.get("has_issues", False),
            "prediction_drift": not prediction_drift.get("has_issues", False),
            "model_performance": not performance.get("has_issues", False),
        },
    }


def decide_alerting(**context) -> str:
    """Decide whether to send alerts."""
    ti = context["ti"]
    results = ti.xcom_pull(task_ids="aggregate_results")

    if results["severity"] == "critical":
        return "send_critical_alert"
    elif results["severity"] == "warning":
        return "send_warning_alert"
    else:
        return "no_alert_needed"


def send_critical_alert(**context) -> dict[str, Any]:
    """Send critical data quality alert."""
    ti = context["ti"]
    results = ti.xcom_pull(task_ids="aggregate_results")

    # TODO: Send critical alert
    # from airs.utils.notifications import send_critical_alert
    # send_critical_alert(results)

    return {
        "alert_sent": True,
        "severity": "critical",
        "recipients": ["on-call@example.com", "ml-team@example.com"],
    }


def send_warning_alert(**context) -> dict[str, Any]:
    """Send warning data quality alert."""
    ti = context["ti"]
    results = ti.xcom_pull(task_ids="aggregate_results")

    # TODO: Send warning alert
    # from airs.utils.notifications import send_warning_alert
    # send_warning_alert(results)

    return {
        "alert_sent": True,
        "severity": "warning",
        "recipients": ["ml-team@example.com"],
    }


def store_quality_metrics(**context) -> dict[str, Any]:
    """Store quality metrics in database."""
    ti = context["ti"]
    results = ti.xcom_pull(task_ids="aggregate_results")

    # TODO: Store metrics
    # from airs.db.repository import MetricsRepository
    # repo = MetricsRepository()
    # repo.store_quality_metrics(results)

    return {"stored": True}


# Create DAG
with DAG(
    dag_id="airs_data_quality",
    default_args=default_args,
    description="Daily data quality and drift monitoring",
    schedule_interval="0 17 * * 1-5",  # 5 PM UTC on weekdays (before daily pipeline)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["airs", "monitoring", "data-quality"],
    max_active_runs=1,
) as dag:

    # Start
    start = EmptyOperator(task_id="start")

    # Data quality checks
    with TaskGroup("data_checks") as data_checks:
        freshness = PythonOperator(
            task_id="check_freshness",
            python_callable=check_data_freshness,
        )

        missing = PythonOperator(
            task_id="check_missing",
            python_callable=check_missing_values,
        )

        outliers = PythonOperator(
            task_id="check_outliers",
            python_callable=check_outliers,
        )

    # Drift checks
    with TaskGroup("drift_checks") as drift_checks:
        feature_drift = PythonOperator(
            task_id="check_feature_drift",
            python_callable=check_feature_drift,
        )

        prediction_drift = PythonOperator(
            task_id="check_prediction_drift",
            python_callable=check_prediction_drift,
        )

        model_perf = PythonOperator(
            task_id="check_model_performance",
            python_callable=check_model_performance,
        )

    # Aggregate results
    aggregate = PythonOperator(
        task_id="aggregate_results",
        python_callable=aggregate_quality_results,
    )

    # Decide alerting
    decide = BranchPythonOperator(
        task_id="decide_alerting",
        python_callable=decide_alerting,
    )

    # Alert paths
    critical_alert = PythonOperator(
        task_id="send_critical_alert",
        python_callable=send_critical_alert,
    )

    warning_alert = PythonOperator(
        task_id="send_warning_alert",
        python_callable=send_warning_alert,
    )

    no_alert = EmptyOperator(task_id="no_alert_needed")

    # Store metrics
    store_metrics = PythonOperator(
        task_id="store_metrics",
        python_callable=store_quality_metrics,
        trigger_rule="none_failed_min_one_success",
    )

    # End
    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    # Define dependencies
    start >> [data_checks, drift_checks]
    [data_checks, drift_checks] >> aggregate >> decide
    decide >> critical_alert >> store_metrics >> end
    decide >> warning_alert >> store_metrics >> end
    decide >> no_alert >> store_metrics >> end
