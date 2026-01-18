# AIRS: AI-Driven Early-Warning System for Portfolio Drawdown Risk

A production-ready machine learning system that detects early warning signs of portfolio stress and generates actionable recommendations for risk mitigation.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

AIRS monitors multi-asset portfolios for signs of impending drawdowns by analyzing:
- Interest rate dynamics and yield curve shape
- Credit market stress indicators
- Volatility patterns and term structure
- Macroeconomic leading indicators
- Cross-asset correlations and regime changes

When elevated risk is detected, the system generates portfolio de-risking recommendations with explanations and suggested action timelines.

## Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Precision | >70% | Proportion of alerts that precede actual drawdowns |
| Lead Time | 7-14 days | Warning before significant drawdown begins |
| False Positive Rate | <30% | Avoiding excessive trading from false signals |
| Sharpe Improvement | +0.15 | Risk-adjusted return improvement vs buy-and-hold |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- API Keys: FRED, Alpha Vantage (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/atharvajoshi01/AIRS.git
cd AIRS/AIRS

# Create environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Running with Docker

```bash
cd AIRS
docker-compose up -d
```

### Running Locally

```bash
# Start API server
uvicorn airs.api.main:app --reload

# Run tests
pytest tests/ -v

# Run backtest
python scripts/run_backtest.py
```

## Project Structure

```
AIRS/
├── airs/
│   ├── config/          # Configuration management
│   ├── data/            # Data fetching and quality
│   ├── db/              # Database models and repository
│   ├── features/        # Feature engineering
│   ├── targets/         # Target variable definition
│   ├── models/          # ML models
│   ├── backtest/        # Backtesting framework
│   ├── recommendations/ # Recommendation engine
│   ├── api/             # REST API
│   ├── monitoring/      # Drift detection and alerts
│   └── utils/           # Utilities
├── dags/                # Airflow DAGs
├── dashboard/           # Next.js dashboard
├── tests/               # Test suite
├── docs/                # Documentation
└── scripts/             # CLI scripts
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | System health check |
| `/api/v1/alerts/current` | GET | Current risk alert |
| `/api/v1/alerts/history` | GET | Historical alerts |
| `/api/v1/features/current` | GET | Current feature values |
| `/api/v1/recommendations/current` | GET | Portfolio recommendations |
| `/api/v1/backtest/run` | POST | Run custom backtest |

## Portfolio Allocation

Default multi-asset allocation:

| Asset | Weight | Description |
|-------|--------|-------------|
| SPY | 40% | US Equity |
| VEU | 20% | International Equity |
| AGG | 25% | US Bonds |
| DJP | 10% | Commodities |
| VNQ | 5% | REITs |

## De-risking Strategy

When high-risk conditions are detected:

1. **Reduce Equity Exposure**: Cut equity positions by 50%
2. **Increase Bonds**: Add to high-quality bond allocation
3. **Raise Cash**: Move to cash for protection
4. **Gradual Re-entry**: Return to normal allocation over 5 days when conditions improve

## Development

### Running Tests

```bash
pytest tests/ -v --cov=airs --cov-report=html
```

### Code Quality

```bash
black airs/ tests/
isort airs/ tests/
mypy airs/
ruff check airs/
```

## Documentation

- [Data Sources](AIRS/docs/DATA.md) - Data collection and schemas
- [Feature Engineering](AIRS/docs/FEATURES.md) - Feature definitions
- [Models](AIRS/docs/MODELS.md) - Model architecture
- [Backtesting](AIRS/docs/BACKTEST.md) - Backtest methodology
- [API Reference](AIRS/docs/API.md) - API documentation

## License

MIT License - see LICENSE file for details.
