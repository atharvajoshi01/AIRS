"""
Stress period analysis for backtesting.

Analyzes performance during known market stress events.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from airs.backtest.metrics import PerformanceMetrics
from airs.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StressPeriod:
    """Definition of a market stress period."""

    name: str
    start_date: str
    end_date: str
    description: str
    severity: str  # "mild", "moderate", "severe"


# Known historical stress periods
STRESS_PERIODS = [
    StressPeriod(
        name="2011 EU Debt Crisis",
        start_date="2011-07-01",
        end_date="2011-10-31",
        description="European sovereign debt crisis, US debt ceiling",
        severity="moderate",
    ),
    StressPeriod(
        name="2015 China Devaluation",
        start_date="2015-08-01",
        end_date="2015-09-30",
        description="Chinese yuan devaluation, global market selloff",
        severity="moderate",
    ),
    StressPeriod(
        name="2018 Q4 Selloff",
        start_date="2018-10-01",
        end_date="2018-12-31",
        description="Fed tightening concerns, trade war",
        severity="moderate",
    ),
    StressPeriod(
        name="2020 COVID Crash",
        start_date="2020-02-19",
        end_date="2020-03-23",
        description="COVID-19 pandemic market crash",
        severity="severe",
    ),
    StressPeriod(
        name="2022 Rate Shock",
        start_date="2022-01-01",
        end_date="2022-10-31",
        description="Fed rate hikes, inflation concerns",
        severity="severe",
    ),
]


class StressPeriodAnalyzer:
    """
    Analyze portfolio performance during stress periods.
    """

    def __init__(
        self,
        stress_periods: list[StressPeriod] | None = None,
    ):
        """
        Initialize analyzer.

        Args:
            stress_periods: List of stress periods to analyze
        """
        self.stress_periods = stress_periods or STRESS_PERIODS
        self.metrics_calculator = PerformanceMetrics()

    def analyze_period(
        self,
        strategy_values: pd.Series,
        benchmark_values: pd.Series,
        period: StressPeriod,
        signals: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Analyze performance during a specific stress period.

        Args:
            strategy_values: Strategy portfolio values
            benchmark_values: Benchmark portfolio values
            period: Stress period definition
            signals: Trading signals (optional)

        Returns:
            Analysis results
        """
        start = pd.Timestamp(period.start_date)
        end = pd.Timestamp(period.end_date)

        # Filter to period
        strat = strategy_values[(strategy_values.index >= start) & (strategy_values.index <= end)]
        bench = benchmark_values[(benchmark_values.index >= start) & (benchmark_values.index <= end)]

        if len(strat) == 0 or len(bench) == 0:
            return {
                "period": period.name,
                "available": False,
                "message": "No data available for this period",
            }

        # Calculate returns
        strat_return = (strat.iloc[-1] / strat.iloc[0]) - 1
        bench_return = (bench.iloc[-1] / bench.iloc[0]) - 1

        # Calculate max drawdown
        strat_dd = self._max_drawdown(strat)
        bench_dd = self._max_drawdown(bench)

        # Volatility
        strat_vol = strat.pct_change().std() * np.sqrt(252)
        bench_vol = bench.pct_change().std() * np.sqrt(252)

        # Signal analysis
        signal_analysis = None
        if signals is not None:
            period_signals = signals[(signals.index >= start) & (signals.index <= end)]
            if len(period_signals) > 0:
                signal_analysis = {
                    "avg_signal": period_signals.mean(),
                    "max_signal": period_signals.max(),
                    "days_alerted": (period_signals >= 0.5).sum(),
                    "alert_before_trough": self._check_early_warning(
                        period_signals, strat, start, end
                    ),
                }

        return {
            "period": period.name,
            "available": True,
            "start_date": start,
            "end_date": end,
            "duration_days": (end - start).days,
            "severity": period.severity,
            "strategy": {
                "return": strat_return,
                "max_drawdown": strat_dd,
                "volatility": strat_vol,
            },
            "benchmark": {
                "return": bench_return,
                "max_drawdown": bench_dd,
                "volatility": bench_vol,
            },
            "outperformance": {
                "return_diff": strat_return - bench_return,
                "drawdown_improvement": bench_dd - strat_dd,
                "protected": strat_return > bench_return,
            },
            "signal_analysis": signal_analysis,
        }

    def analyze_all_periods(
        self,
        strategy_values: pd.Series,
        benchmark_values: pd.Series,
        signals: pd.Series | None = None,
    ) -> list[dict[str, Any]]:
        """
        Analyze performance across all stress periods.

        Args:
            strategy_values: Strategy values
            benchmark_values: Benchmark values
            signals: Trading signals

        Returns:
            List of period analyses
        """
        results = []

        for period in self.stress_periods:
            try:
                analysis = self.analyze_period(
                    strategy_values, benchmark_values, period, signals
                )
                results.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing {period.name}: {e}")
                results.append({
                    "period": period.name,
                    "available": False,
                    "error": str(e),
                })

        return results

    def generate_summary(
        self,
        analyses: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Generate summary statistics across all stress periods.

        Args:
            analyses: List of period analyses

        Returns:
            Summary statistics
        """
        available = [a for a in analyses if a.get("available", False)]

        if not available:
            return {"n_periods": 0, "message": "No periods available for analysis"}

        return {
            "n_periods": len(available),
            "n_protected": sum(1 for a in available if a["outperformance"]["protected"]),
            "avg_outperformance": np.mean([
                a["outperformance"]["return_diff"] for a in available
            ]),
            "avg_drawdown_improvement": np.mean([
                a["outperformance"]["drawdown_improvement"] for a in available
            ]),
            "periods": {a["period"]: a["outperformance"] for a in available},
        }

    def _max_drawdown(self, values: pd.Series) -> float:
        """Calculate max drawdown for a series."""
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return drawdown.min()

    def _check_early_warning(
        self,
        signals: pd.Series,
        values: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, Any]:
        """
        Check if signals provided early warning before trough.

        Returns:
            Early warning analysis
        """
        # Find trough date
        trough_date = values.idxmin()
        trough_value = values.min()

        # Check signals before trough
        signals_before_trough = signals[signals.index < trough_date]

        if len(signals_before_trough) == 0:
            return {"provided_warning": False, "lead_time_days": None}

        # First alert date (signal > 0.5)
        alerts = signals_before_trough[signals_before_trough >= 0.5]

        if len(alerts) == 0:
            return {"provided_warning": False, "lead_time_days": None}

        first_alert = alerts.index[0]
        lead_time = (trough_date - first_alert).days

        return {
            "provided_warning": True,
            "first_alert_date": first_alert,
            "trough_date": trough_date,
            "lead_time_days": lead_time,
        }

    def generate_report(self, analyses: list[dict[str, Any]]) -> str:
        """Generate human-readable stress period report."""
        summary = self.generate_summary(analyses)

        report = [
            f"\n{'='*60}",
            "Stress Period Analysis Report",
            f"{'='*60}",
            "",
            f"Periods Analyzed: {summary.get('n_periods', 0)}",
            f"Periods Protected: {summary.get('n_protected', 0)}",
            f"Avg Outperformance: {summary.get('avg_outperformance', 0)*100:.2f}%",
            f"Avg Drawdown Improvement: {summary.get('avg_drawdown_improvement', 0)*100:.2f}%",
            "",
        ]

        for analysis in analyses:
            if not analysis.get("available", False):
                continue

            report.extend([
                f"\n{'-'*40}",
                f"{analysis['period']} ({analysis['severity']})",
                f"{'-'*40}",
                f"Duration: {analysis['duration_days']} days",
                f"Strategy Return: {analysis['strategy']['return']*100:>8.2f}%",
                f"Benchmark Return: {analysis['benchmark']['return']*100:>8.2f}%",
                f"Outperformance: {analysis['outperformance']['return_diff']*100:>8.2f}%",
                f"Strategy DD: {analysis['strategy']['max_drawdown']*100:>8.2f}%",
                f"Benchmark DD: {analysis['benchmark']['max_drawdown']*100:>8.2f}%",
                f"DD Improvement: {analysis['outperformance']['drawdown_improvement']*100:>8.2f}%",
            ])

            if analysis.get("signal_analysis"):
                sig = analysis["signal_analysis"]
                report.append(f"Days Alerted: {sig['days_alerted']}")
                if sig.get("alert_before_trough", {}).get("provided_warning"):
                    report.append(
                        f"Lead Time: {sig['alert_before_trough']['lead_time_days']} days"
                    )

        report.append("\n" + "=" * 60)

        return "\n".join(report)
