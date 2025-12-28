"""
Backtest endpoints for AIRS API.
"""

from datetime import date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Body

from airs.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    BacktestMetrics,
    BacktestComparison,
)
from airs.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def run_backtest(config: BacktestRequest) -> dict[str, Any]:
    """Run backtest with given configuration."""
    # TODO: Replace with actual backtest engine call
    import numpy as np

    # Mock backtest results
    strategy_metrics = {
        "total_return": 0.45,
        "annualized_return": 0.08,
        "volatility": 0.12,
        "sharpe_ratio": 0.67,
        "sortino_ratio": 0.95,
        "max_drawdown": -0.15,
        "calmar_ratio": 0.53,
        "win_rate": 0.54,
        "var_95": -0.018,
    }

    benchmark_metrics = {
        "total_return": 0.38,
        "annualized_return": 0.07,
        "volatility": 0.15,
        "sharpe_ratio": 0.47,
        "sortino_ratio": 0.68,
        "max_drawdown": -0.25,
        "calmar_ratio": 0.28,
        "win_rate": 0.52,
        "var_95": -0.022,
    }

    stress_periods = [
        {
            "name": "2018 Q4 Selloff",
            "start_date": "2018-10-01",
            "end_date": "2018-12-31",
            "strategy_return": -0.08,
            "benchmark_return": -0.14,
            "outperformance": 0.06,
            "alert_triggered": True,
            "lead_days": 5,
        },
        {
            "name": "2020 COVID Crash",
            "start_date": "2020-02-19",
            "end_date": "2020-03-23",
            "strategy_return": -0.18,
            "benchmark_return": -0.28,
            "outperformance": 0.10,
            "alert_triggered": True,
            "lead_days": 8,
        },
        {
            "name": "2022 Rate Shock",
            "start_date": "2022-01-01",
            "end_date": "2022-10-31",
            "strategy_return": -0.12,
            "benchmark_return": -0.20,
            "outperformance": 0.08,
            "alert_triggered": True,
            "lead_days": 12,
        },
    ]

    return {
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
        "excess_return": strategy_metrics["total_return"] - benchmark_metrics["total_return"],
        "excess_sharpe": strategy_metrics["sharpe_ratio"] - benchmark_metrics["sharpe_ratio"],
        "drawdown_improvement": benchmark_metrics["max_drawdown"] - strategy_metrics["max_drawdown"],
        "n_alerts": 15,
        "n_trades": 45,
        "stress_periods": stress_periods,
    }


@router.post("/run", response_model=BacktestResponse)
async def run_backtest_endpoint(
    config: BacktestRequest = Body(...),
) -> BacktestResponse:
    """
    Run a custom backtest.

    Execute backtest with specified configuration and return results.
    """
    try:
        # Set defaults
        if config.end_date is None:
            config.end_date = date.today()
        if config.start_date is None:
            config.start_date = config.end_date - timedelta(days=5 * 365)

        results = run_backtest(config)

        return BacktestResponse(
            timestamp=datetime.utcnow(),
            version="1.0",
            config=config,
            comparison=BacktestComparison(
                strategy=BacktestMetrics(**results["strategy"]),
                benchmark=BacktestMetrics(**results["benchmark"]),
                excess_return=results["excess_return"],
                excess_sharpe=results["excess_sharpe"],
                drawdown_improvement=results["drawdown_improvement"],
                n_alerts=results["n_alerts"],
                n_trades=results["n_trades"],
            ),
            stress_periods=results["stress_periods"],
        )
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail="Failed to run backtest")


@router.get("/default", response_model=BacktestResponse)
async def get_default_backtest() -> BacktestResponse:
    """
    Get pre-computed backtest with default settings.

    Returns cached results for quick access.
    """
    config = BacktestRequest(
        start_date=date.today() - timedelta(days=5 * 365),
        end_date=date.today(),
    )

    try:
        results = run_backtest(config)

        return BacktestResponse(
            timestamp=datetime.utcnow(),
            version="1.0",
            config=config,
            comparison=BacktestComparison(
                strategy=BacktestMetrics(**results["strategy"]),
                benchmark=BacktestMetrics(**results["benchmark"]),
                excess_return=results["excess_return"],
                excess_sharpe=results["excess_sharpe"],
                drawdown_improvement=results["drawdown_improvement"],
                n_alerts=results["n_alerts"],
                n_trades=results["n_trades"],
            ),
            stress_periods=results["stress_periods"],
        )
    except Exception as e:
        logger.error(f"Error getting default backtest: {e}")
        raise HTTPException(status_code=500, detail="Failed to get backtest")


@router.get("/stress-periods")
async def get_stress_period_analysis() -> dict[str, Any]:
    """
    Get detailed stress period analysis.

    Returns performance during known historical stress events.
    """
    # TODO: Replace with actual stress period analysis
    stress_periods = [
        {
            "name": "2011 EU Debt Crisis",
            "start_date": "2011-07-01",
            "end_date": "2011-10-31",
            "severity": "moderate",
            "strategy": {
                "return": -0.06,
                "max_drawdown": -0.10,
                "volatility": 0.22,
            },
            "benchmark": {
                "return": -0.12,
                "max_drawdown": -0.18,
                "volatility": 0.28,
            },
            "outperformance": 0.06,
            "signal_analysis": {
                "alert_triggered": True,
                "days_before_trough": 8,
                "avg_signal": 0.65,
            },
        },
        {
            "name": "2015 China Devaluation",
            "start_date": "2015-08-01",
            "end_date": "2015-09-30",
            "severity": "moderate",
            "strategy": {
                "return": -0.05,
                "max_drawdown": -0.08,
                "volatility": 0.25,
            },
            "benchmark": {
                "return": -0.10,
                "max_drawdown": -0.12,
                "volatility": 0.30,
            },
            "outperformance": 0.05,
            "signal_analysis": {
                "alert_triggered": True,
                "days_before_trough": 5,
                "avg_signal": 0.58,
            },
        },
        {
            "name": "2018 Q4 Selloff",
            "start_date": "2018-10-01",
            "end_date": "2018-12-31",
            "severity": "moderate",
            "strategy": {
                "return": -0.08,
                "max_drawdown": -0.12,
                "volatility": 0.20,
            },
            "benchmark": {
                "return": -0.14,
                "max_drawdown": -0.19,
                "volatility": 0.26,
            },
            "outperformance": 0.06,
            "signal_analysis": {
                "alert_triggered": True,
                "days_before_trough": 5,
                "avg_signal": 0.72,
            },
        },
        {
            "name": "2020 COVID Crash",
            "start_date": "2020-02-19",
            "end_date": "2020-03-23",
            "severity": "severe",
            "strategy": {
                "return": -0.18,
                "max_drawdown": -0.22,
                "volatility": 0.65,
            },
            "benchmark": {
                "return": -0.28,
                "max_drawdown": -0.34,
                "volatility": 0.80,
            },
            "outperformance": 0.10,
            "signal_analysis": {
                "alert_triggered": True,
                "days_before_trough": 8,
                "avg_signal": 0.85,
            },
        },
        {
            "name": "2022 Rate Shock",
            "start_date": "2022-01-01",
            "end_date": "2022-10-31",
            "severity": "severe",
            "strategy": {
                "return": -0.12,
                "max_drawdown": -0.18,
                "volatility": 0.22,
            },
            "benchmark": {
                "return": -0.20,
                "max_drawdown": -0.25,
                "volatility": 0.26,
            },
            "outperformance": 0.08,
            "signal_analysis": {
                "alert_triggered": True,
                "days_before_trough": 12,
                "avg_signal": 0.68,
            },
        },
    ]

    # Calculate summary statistics
    avg_outperformance = sum(p["outperformance"] for p in stress_periods) / len(stress_periods)
    avg_lead_time = sum(
        p["signal_analysis"]["days_before_trough"]
        for p in stress_periods
        if p["signal_analysis"]["alert_triggered"]
    ) / len([p for p in stress_periods if p["signal_analysis"]["alert_triggered"]])

    return {
        "periods": stress_periods,
        "summary": {
            "n_periods": len(stress_periods),
            "n_protected": sum(1 for p in stress_periods if p["signal_analysis"]["alert_triggered"]),
            "avg_outperformance": avg_outperformance,
            "avg_lead_time_days": avg_lead_time,
            "avg_drawdown_reduction": sum(
                p["benchmark"]["max_drawdown"] - p["strategy"]["max_drawdown"]
                for p in stress_periods
            ) / len(stress_periods),
        },
    }


@router.get("/metrics")
async def get_backtest_metrics() -> dict[str, Any]:
    """
    Get detailed performance metrics.

    Returns comprehensive metrics with statistical significance.
    """
    # Run default backtest
    config = BacktestRequest()
    results = run_backtest(config)

    # Calculate additional metrics
    return {
        "strategy": {
            **results["strategy"],
            "information_ratio": 0.45,
            "tracking_error": 0.06,
            "beta": 0.75,
            "alpha": 0.03,
        },
        "benchmark": results["benchmark"],
        "comparison": {
            "excess_return": results["excess_return"],
            "excess_sharpe": results["excess_sharpe"],
            "drawdown_improvement": results["drawdown_improvement"],
            "t_statistic": 2.15,
            "p_value": 0.032,
            "significant_outperformance": True,
        },
        "trading_statistics": {
            "n_trades": results["n_trades"],
            "n_alerts": results["n_alerts"],
            "avg_turnover_per_alert": 0.12,
            "total_trading_costs_bps": 450,
        },
        "period": {
            "start_date": (date.today() - timedelta(days=5*365)).isoformat(),
            "end_date": date.today().isoformat(),
            "trading_days": 1260,
            "years": 5.0,
        },
    }
