# Backtesting Methodology

This document describes the backtesting framework and methodology used in AIRS.

## Overview

The AIRS backtest simulates a systematic de-risking strategy that:
1. Monitors model signals daily
2. Reduces risk when alerts are triggered
3. Gradually returns to normal allocation
4. Accounts for realistic transaction costs

## Portfolio Configuration

### Default Allocation

| Asset | Weight | Type |
|-------|--------|------|
| SPY | 40% | US Equity |
| VEU | 20% | Intl Equity |
| AGG | 25% | US Bonds |
| DJP | 10% | Commodities |
| VNQ | 5% | REITs |

### De-risked Allocation

When alerts trigger:

| Asset | Normal | De-risked | Change |
|-------|--------|-----------|--------|
| SPY | 40% | 20% | -50% |
| VEU | 20% | 10% | -50% |
| AGG | 25% | 30% | +20% |
| DJP | 10% | 7% | -30% |
| VNQ | 5% | 3% | -40% |
| CASH | 0% | 30% | +30% |

## Signal Logic

### Alert Thresholds

| Level | Probability | Action |
|-------|-------------|--------|
| None | 0-30% | No action |
| Low | 30-50% | Monitor |
| Moderate | 50-70% | Consider de-risking |
| High | 70-85% | De-risk recommended |
| Critical | 85%+ | Immediate de-risk |

### Signal Parameters

```python
{
    "alert_threshold": 0.5,      # Probability to trigger
    "min_signal_days": 3,        # Consecutive days required
    "signal_lag": 1,             # Days between signal and trade
    "rerisk_days": 5,            # Days to return to normal
}
```

### De-risk/Re-risk Logic

```
Day 1-3:  Signal > 0.5, consecutive_days++
Day 4:    consecutive_days >= 3, TRIGGER DE-RISK
          Execute trades to reduce equity, raise cash

Day 5+:   If signal < 0.5 for 5 consecutive days:
          Begin gradual re-risking (20% per day)
```

## Transaction Costs

### Cost Model

| Component | Value | Description |
|-----------|-------|-------------|
| Trading Cost | 10 bps | Commission + fees |
| Slippage | 5 bps | Bid-ask spread |
| Total One-way | 15 bps | Per trade |
| Round-trip | 30 bps | Buy + sell |

### Asset-Specific Spreads

| Asset | Spread (bps) |
|-------|--------------|
| SPY | 1 |
| AGG | 3 |
| VEU | 5 |
| VNQ | 5 |
| DJP | 10 |

### Cost Calculation

```python
cost = trade_value * (trading_cost_bps + slippage_bps) / 10000

# With market impact (large orders)
if participation_rate > 0.01:
    cost += trade_value * 0.1 * sqrt(participation_rate)
```

## Backtest Engine

### Daily Simulation Loop

```python
for date in trading_days:
    # 1. Get lagged signal
    signal = signals[date - 1]  # T-1 signal

    # 2. Check for de-risk trigger
    if signal >= threshold:
        consecutive_alert_days += 1
    else:
        consecutive_alert_days = 0

    # 3. Execute de-risking if needed
    if not in_derisk_mode and consecutive_alert_days >= min_signal_days:
        enter_derisk_mode()
        rebalance_to_defensive()

    # 4. Handle re-risking
    if in_derisk_mode and signal < threshold:
        rerisk_days_remaining -= 1
        if rerisk_days_remaining == 0:
            exit_derisk_mode()
            rebalance_to_normal()

    # 5. Update portfolio values
    portfolio.update(prices[date])

    # 6. Periodic rebalancing (if not in de-risk)
    if should_rebalance(date):
        rebalance_to_targets()
```

### Rebalancing

Normal rebalancing:
- Frequency: Monthly (first trading day)
- Threshold: 5% deviation from target
- Excludes de-risk periods

## Performance Metrics

### Return Metrics

| Metric | Calculation |
|--------|-------------|
| Total Return | (Final Value / Initial Value) - 1 |
| Annualized Return | (1 + Total Return)^(1/years) - 1 |
| Excess Return | Strategy Return - Benchmark Return |
| Alpha | Excess return vs CAPM |

### Risk Metrics

| Metric | Calculation |
|--------|-------------|
| Volatility | Annualized std of daily returns |
| Max Drawdown | Largest peak-to-trough decline |
| VaR (95%) | 5th percentile of daily returns |
| CVaR (95%) | Mean of worst 5% returns |

### Risk-Adjusted Metrics

| Metric | Formula |
|--------|---------|
| Sharpe Ratio | (Return - Rf) / Volatility |
| Sortino Ratio | (Return - Rf) / Downside Vol |
| Calmar Ratio | Return / |Max Drawdown| |
| Information Ratio | Excess Return / Tracking Error |

## Stress Period Analysis

### Historical Stress Events

| Period | Dates | Severity |
|--------|-------|----------|
| 2011 EU Crisis | Jul-Oct 2011 | Moderate |
| 2015 China Deval | Aug-Sep 2015 | Moderate |
| 2018 Q4 Selloff | Oct-Dec 2018 | Moderate |
| 2020 COVID | Feb-Mar 2020 | Severe |
| 2022 Rate Shock | Jan-Oct 2022 | Severe |

### Analysis Metrics

For each stress period:
- Strategy return vs benchmark
- Max drawdown comparison
- Days alerted before trough
- Average signal level during period

## Avoiding Lookahead Bias

### Safeguards

1. **Signal Lag:** 1-day delay between signal and trade
2. **Point-in-Time Data:** Economic releases use actual dates
3. **Walk-Forward Training:** Model trained only on past data
4. **Embargo Period:** Gap between train and test sets

### Validation

```python
# Assert no future information
for date in backtest_dates:
    features_used = get_features(date)
    assert all(f.date <= date for f in features_used)

    signal_used = get_signal(date - signal_lag)
    assert signal_used.date < date
```

## Benchmark Comparison

### Buy-and-Hold Benchmark

Same asset allocation without de-risking:
- Monthly rebalancing only
- Same transaction costs
- No signal-based trading

### Comparison Metrics

| Metric | Strategy | Benchmark | Difference |
|--------|----------|-----------|------------|
| Total Return | X% | Y% | X-Y% |
| Sharpe Ratio | S1 | S2 | S1-S2 |
| Max Drawdown | D1% | D2% | D2-D1% |

## Expected Results

Based on historical backtests (2014-2024):

| Metric | Target | Typical |
|--------|--------|---------|
| Excess Return | >0% | +2-4% p.a. |
| Sharpe Improvement | +0.15 | +0.15-0.25 |
| Drawdown Reduction | >20% | 20-35% |
| Alerts/Year | <10 | 3-6 |

## Running Backtests

### Command Line

```bash
# Default backtest
python scripts/run_backtest.py

# Custom date range
python scripts/run_backtest.py --start 2020-01-01 --end 2024-01-01

# With specific config
python scripts/run_backtest.py --config configs/aggressive.yaml
```

### API

```python
from airs.backtest.engine import BacktestEngine, BacktestConfig

config = BacktestConfig(
    initial_value=100_000,
    alert_threshold=0.6,
    derisk_equity_reduction=0.5,
)

engine = BacktestEngine(config)
results = engine.run(prices, signals)
```

See `airs/backtest/` for implementation details.
