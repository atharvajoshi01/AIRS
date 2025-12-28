# Data Sources

This document describes the data sources, schemas, and update frequencies for AIRS.

## Data Sources Overview

| Source | Data Types | Update Frequency | Rate Limits |
|--------|------------|------------------|-------------|
| FRED | Economic indicators, yields, spreads | Daily/Weekly | 120 req/min |
| Yahoo Finance | ETF prices, VIX | Daily | Generous |
| Alpha Vantage | Sector performance | Daily | 5 req/min |

## FRED Data

Federal Reserve Economic Data provides macroeconomic indicators.

### Treasury Yields

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| DGS3MO | 3-Month Treasury | Daily |
| DGS2 | 2-Year Treasury | Daily |
| DGS5 | 5-Year Treasury | Daily |
| DGS10 | 10-Year Treasury | Daily |
| DGS30 | 30-Year Treasury | Daily |

### Credit Spreads

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| BAMLC0A0CM | ICE BofA US Corp Index OAS | Daily |
| BAMLH0A0HYM2 | ICE BofA US High Yield OAS | Daily |
| TEDRATE | TED Spread | Daily |

### Macro Indicators

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| USSLIND | Leading Economic Index | Monthly |
| UNRATE | Unemployment Rate | Monthly |
| ICSA | Initial Claims | Weekly |
| UMCSENT | Consumer Sentiment | Monthly |
| NFCI | Financial Conditions | Weekly |

### Volatility

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| VIXCLS | CBOE VIX | Daily |

## Yahoo Finance Data

### ETF Prices

| Symbol | Description | Asset Class |
|--------|-------------|-------------|
| SPY | S&P 500 ETF | US Equity |
| VEU | FTSE All-World ex-US | Intl Equity |
| AGG | US Aggregate Bond | Bonds |
| DJP | Commodity Index | Commodities |
| VNQ | Real Estate ETF | REITs |
| GLD | Gold ETF | Commodities |
| TLT | 20+ Year Treasury | Bonds |
| HYG | High Yield Corp Bond | Credit |

### Market Indices

| Symbol | Description |
|--------|-------------|
| ^VIX | CBOE VIX Index |
| ^VIX9D | 9-Day VIX |
| ^VIX3M | 3-Month VIX |

## Database Schema

### market_data

```sql
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DECIMAL(18, 6),
    high DECIMAL(18, 6),
    low DECIMAL(18, 6),
    close DECIMAL(18, 6),
    adjusted_close DECIMAL(18, 6),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(date, symbol)
);

CREATE INDEX idx_market_data_date ON market_data(date);
CREATE INDEX idx_market_data_symbol ON market_data(symbol);
```

### economic_indicators

```sql
CREATE TABLE economic_indicators (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    series_id VARCHAR(50) NOT NULL,
    value DECIMAL(18, 6),
    release_date DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(date, series_id)
);

CREATE INDEX idx_econ_date ON economic_indicators(date);
CREATE INDEX idx_econ_series ON economic_indicators(series_id);
```

### features

```sql
CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    value DECIMAL(18, 6),
    computed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(date, feature_name)
);

CREATE INDEX idx_features_date ON features(date);
```

### predictions

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    model_id VARCHAR(50) NOT NULL,
    probability DECIMAL(8, 6),
    alert_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(date, model_id)
);
```

## Point-in-Time Considerations

Economic data has release delays that must be modeled:

| Data Type | Typical Delay |
|-----------|---------------|
| Treasury Yields | 0 days (real-time) |
| Credit Spreads | 0 days |
| VIX | 0 days |
| LEI | ~20 days |
| Employment Data | ~3 days |
| Consumer Sentiment | ~3 days |

The system uses `release_date` to ensure no lookahead bias in feature engineering.

## Data Quality Checks

The system performs automated checks:

1. **Freshness**: Data updated within expected window
2. **Completeness**: No unexpected missing values
3. **Validity**: Values within reasonable ranges
4. **Consistency**: Cross-source validation

See `airs/data/quality.py` for implementation.

## Data Refresh Schedule

| Time (UTC) | Process |
|------------|---------|
| 17:00 | Data quality checks |
| 18:00 | Daily data ingestion |
| 18:30 | Feature computation |
| 19:00 | Prediction generation |

Managed by Airflow DAGs in `dags/daily_pipeline.py`.
