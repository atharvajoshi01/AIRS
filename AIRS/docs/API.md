# API Reference

This document describes the AIRS REST API endpoints.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently no authentication required for local development. Production deployments should implement API key or OAuth2 authentication.

## Common Response Format

All responses follow this structure:

```json
{
  "timestamp": "2024-01-15T18:00:00Z",
  "version": "1.0",
  "data": { ... }
}
```

Error responses:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2024-01-15T18:00:00Z"
}
```

## Endpoints

### Health

#### GET /health

System health check.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T18:00:00Z",
  "services": [
    {
      "name": "database",
      "status": "healthy",
      "latency_ms": 5.2
    },
    {
      "name": "model",
      "status": "healthy"
    },
    {
      "name": "data",
      "status": "healthy"
    }
  ],
  "version": "1.0.0",
  "uptime_seconds": 3600.5
}
```

#### GET /health/live

Kubernetes liveness probe.

**Response:**

```json
{
  "status": "alive"
}
```

#### GET /health/ready

Kubernetes readiness probe.

**Response:**

```json
{
  "status": "ready"
}
```

---

### Alerts

#### GET /alerts/current

Get current risk alert status.

**Response:**

```json
{
  "alert_level": "moderate",
  "probability": 0.62,
  "confidence": 0.78,
  "headline": "Risk indicators suggest caution",
  "summary": "Our risk assessment system has flagged moderate risk conditions...",
  "last_updated": "2024-01-15T18:00:00Z"
}
```

**Alert Levels:**
- `none` (0-30%)
- `low` (30-50%)
- `moderate` (50-70%)
- `high` (70-85%)
- `critical` (85%+)

#### GET /alerts/history

Get historical alerts.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| start_date | date | Start date (YYYY-MM-DD) |
| end_date | date | End date (YYYY-MM-DD) |
| timeframe | string | Predefined period: 1d, 1w, 1m, 3m, 6m, 1y, 5y, all |
| alert_level | string | Filter by level |

**Response:**

```json
{
  "timestamp": "2024-01-15T18:00:00Z",
  "version": "1.0",
  "alerts": [
    {
      "date": "2024-01-15",
      "alert_level": "moderate",
      "probability": 0.62,
      "action_taken": null
    },
    {
      "date": "2024-01-14",
      "alert_level": "low",
      "probability": 0.35,
      "action_taken": null
    }
  ],
  "total_count": 30
}
```

#### GET /alerts/statistics

Get alert statistics for a time period.

**Response:**

```json
{
  "timeframe": "1y",
  "start_date": "2023-01-15",
  "end_date": "2024-01-15",
  "total_days": 252,
  "alerts_by_level": {
    "none": { "count": 180, "percentage": 71.4 },
    "low": { "count": 40, "percentage": 15.9 },
    "moderate": { "count": 25, "percentage": 9.9 },
    "high": { "count": 5, "percentage": 2.0 },
    "critical": { "count": 2, "percentage": 0.8 }
  },
  "actions_taken": 7,
  "action_rate": 2.8
}
```

---

### Features

#### GET /features/current

Get current feature values.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| category | string | Filter by category: rates, credit, volatility, macro, cross_asset, composite |

**Response:**

```json
{
  "timestamp": "2024-01-15T18:00:00Z",
  "version": "1.0",
  "features": [
    {
      "name": "vix_level",
      "value": 18.5,
      "percentile": 45.2,
      "z_score": -0.3,
      "category": "volatility"
    },
    {
      "name": "hy_spread",
      "value": 380.5,
      "percentile": 55.8,
      "z_score": 0.2,
      "category": "credit"
    }
  ],
  "feature_date": "2024-01-15"
}
```

#### GET /features/history

Get historical feature values.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| feature_names | string | Comma-separated list of features (required) |
| start_date | date | Start date |
| end_date | date | End date |
| timeframe | string | Predefined period |

**Response:**

```json
{
  "timestamp": "2024-01-15T18:00:00Z",
  "version": "1.0",
  "features": {
    "vix_level": [
      { "date": "2024-01-15", "value": 18.5 },
      { "date": "2024-01-14", "value": 17.8 }
    ],
    "hy_spread": [
      { "date": "2024-01-15", "value": 380.5 },
      { "date": "2024-01-14", "value": 375.2 }
    ]
  },
  "start_date": "2024-01-01",
  "end_date": "2024-01-15"
}
```

#### GET /features/metadata

Get feature metadata and descriptions.

**Response:**

```json
{
  "features": {
    "vix_level": {
      "category": "volatility",
      "description": "VIX index level"
    },
    "hy_spread": {
      "category": "credit",
      "description": "High-yield credit spread (bps)"
    }
  },
  "categories": ["rates", "credit", "volatility", "macro", "cross_asset", "composite"],
  "total_count": 70
}
```

#### GET /features/importance

Get feature importance from the current model.

**Response:**

```json
{
  "feature_importance": {
    "vix_level": 0.12,
    "hy_spread": 0.10,
    "yield_curve_slope": 0.08
  },
  "top_10": { ... },
  "model_id": "ensemble_v1",
  "last_updated": "2024-01-15T06:00:00Z"
}
```

---

### Recommendations

#### GET /recommendations/current

Get current portfolio recommendations.

**Response:**

```json
{
  "timestamp": "2024-01-15T18:00:00Z",
  "version": "1.0",
  "alert_level": "moderate",
  "probability": 0.62,
  "confidence": 0.78,
  "headline": "Risk indicators suggest caution",
  "summary": "Our risk assessment system has flagged moderate risk conditions...",
  "asset_recommendations": [
    {
      "symbol": "SPY",
      "current_weight": 0.40,
      "target_weight": 0.28,
      "action": "sell",
      "urgency": "gradual",
      "rationale": "Reduce equity exposure by 12% to limit drawdown risk"
    },
    {
      "symbol": "AGG",
      "current_weight": 0.25,
      "target_weight": 0.30,
      "action": "buy",
      "urgency": "gradual",
      "rationale": "Increase bond allocation for defensive positioning"
    }
  ],
  "key_drivers": [
    {
      "feature": "vix_level",
      "contribution": 0.15,
      "direction": "increasing risk",
      "magnitude": "high"
    }
  ],
  "historical_context": "Current indicators resemble pre-correction periods...",
  "suggested_timeline": "Execute 12.5% turnover over 3-5 trading days",
  "estimated_turnover": 0.125
}
```

#### GET /recommendations/action-plan

Get executable action plan.

**Response:**

```json
{
  "alert_level": "moderate",
  "generated_at": "2024-01-15T18:00:00Z",
  "estimated_turnover": "12.5%",
  "suggested_timeline": "Execute over 3-5 trading days",
  "actions": [
    {
      "order": 1,
      "action": "SELL",
      "symbol": "SPY",
      "from_weight": "40.0%",
      "to_weight": "28.0%",
      "change": "-12.0%",
      "urgency": "gradual",
      "rationale": "Reduce equity exposure"
    },
    {
      "order": 2,
      "action": "BUY",
      "symbol": "AGG",
      "from_weight": "25.0%",
      "to_weight": "30.0%",
      "change": "+5.0%",
      "urgency": "gradual",
      "rationale": "Increase defensive allocation"
    }
  ],
  "execution_notes": [
    "Execute sells before buys to free up cash",
    "Use limit orders where possible"
  ]
}
```

#### GET /recommendations/explain

Get detailed explanation of current recommendation.

**Response:**

```json
{
  "probability": 0.62,
  "confidence": 0.78,
  "alert_level": "moderate",
  "narrative": {
    "headline": "Risk indicators suggest caution",
    "summary": "...",
    "historical_context": "..."
  },
  "key_drivers": [
    {
      "feature": "vix_level",
      "display_name": "VIX Level",
      "contribution": 0.15,
      "interpretation": "VIX Level is increasing overall risk by 15.0%"
    }
  ],
  "risk_by_category": {
    "rates": "elevated",
    "credit": "elevated",
    "volatility": "high",
    "macro": "moderate"
  },
  "model_info": {
    "model_id": "ensemble_v1",
    "last_trained": "2024-01-15",
    "validation_auc": 0.82
  }
}
```

#### POST /recommendations/simulate

Simulate a recommendation for testing.

**Request Body:**

```json
{
  "probability": 0.75,
  "current_weights": {
    "SPY": 0.40,
    "VEU": 0.20,
    "AGG": 0.25,
    "DJP": 0.10,
    "VNQ": 0.05
  }
}
```

**Response:** Same format as `/recommendations/current`

---

### Backtest

#### POST /backtest/run

Run a custom backtest.

**Request Body:**

```json
{
  "start_date": "2019-01-01",
  "end_date": "2024-01-01",
  "initial_value": 100000,
  "target_weights": {
    "SPY": 0.40,
    "VEU": 0.20,
    "AGG": 0.25,
    "DJP": 0.10,
    "VNQ": 0.05
  },
  "alert_threshold": 0.5,
  "derisk_equity_reduction": 0.5,
  "trading_cost_bps": 10.0
}
```

**Response:**

```json
{
  "timestamp": "2024-01-15T18:00:00Z",
  "version": "1.0",
  "config": { ... },
  "comparison": {
    "strategy": {
      "total_return": 0.45,
      "annualized_return": 0.08,
      "volatility": 0.12,
      "sharpe_ratio": 0.67,
      "sortino_ratio": 0.95,
      "max_drawdown": -0.15,
      "calmar_ratio": 0.53,
      "win_rate": 0.54,
      "var_95": -0.018
    },
    "benchmark": {
      "total_return": 0.38,
      "annualized_return": 0.07,
      "volatility": 0.15,
      "sharpe_ratio": 0.47,
      "max_drawdown": -0.25
    },
    "excess_return": 0.07,
    "excess_sharpe": 0.20,
    "drawdown_improvement": 0.10,
    "n_alerts": 15,
    "n_trades": 45
  },
  "stress_periods": [
    {
      "name": "2020 COVID Crash",
      "strategy_return": -0.18,
      "benchmark_return": -0.28,
      "outperformance": 0.10
    }
  ]
}
```

#### GET /backtest/default

Get pre-computed backtest with default settings.

**Response:** Same format as POST /backtest/run

#### GET /backtest/stress-periods

Get detailed stress period analysis.

**Response:**

```json
{
  "periods": [
    {
      "name": "2020 COVID Crash",
      "start_date": "2020-02-19",
      "end_date": "2020-03-23",
      "severity": "severe",
      "strategy": {
        "return": -0.18,
        "max_drawdown": -0.22,
        "volatility": 0.65
      },
      "benchmark": {
        "return": -0.28,
        "max_drawdown": -0.34,
        "volatility": 0.80
      },
      "outperformance": 0.10,
      "signal_analysis": {
        "alert_triggered": true,
        "days_before_trough": 8,
        "avg_signal": 0.85
      }
    }
  ],
  "summary": {
    "n_periods": 5,
    "n_protected": 5,
    "avg_outperformance": 0.078,
    "avg_lead_time_days": 7.6,
    "avg_drawdown_reduction": 0.08
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Resource doesn't exist |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Dependency down |

## Rate Limiting

No rate limiting in development. Production: 100 requests/minute per IP.

## OpenAPI Specification

Full OpenAPI spec available at `/openapi.json`.

Interactive docs at `/docs` (Swagger UI) and `/redoc` (ReDoc).
