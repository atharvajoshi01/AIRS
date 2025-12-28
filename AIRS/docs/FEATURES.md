# Feature Engineering

This document describes all features used in the AIRS prediction model.

## Feature Categories

| Category | Count | Description |
|----------|-------|-------------|
| Interest Rate | 15 | Yield curve dynamics |
| Credit | 12 | Credit spread indicators |
| Volatility | 15 | Market volatility signals |
| Macro | 10 | Economic indicators |
| Cross-Asset | 8 | Correlation and divergence |
| Regime | 5 | Market regime detection |
| Composite | 5 | Aggregate stress indices |

**Total: ~70 features**

## Interest Rate Features

### Yield Curve Level & Shape

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `yield_10y` | 10-year Treasury yield | Raw value |
| `yield_2y` | 2-year Treasury yield | Raw value |
| `yield_curve_slope` | Curve slope | 10Y - 2Y yield (bps) |
| `yield_curve_curvature` | Butterfly spread | 2*5Y - 2Y - 10Y |
| `curve_inversion_flag` | Inversion indicator | 1 if 2Y > 10Y, else 0 |
| `curve_inversion_depth` | Inversion magnitude | max(0, 2Y - 10Y) |

### Rate Dynamics

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `rate_momentum_5d` | 5-day rate change | 10Y yield change |
| `rate_momentum_10d` | 10-day rate change | 10Y yield change |
| `rate_momentum_21d` | 21-day rate change | 10Y yield change |
| `rate_zscore_21d` | Standardized rate level | Z-score vs 21-day |
| `rate_zscore_63d` | Standardized rate level | Z-score vs 63-day |
| `rate_vol_21d` | Rate volatility | Std of daily changes |

### Term Premium

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `slope_zscore` | Standardized slope | Z-score of curve slope |
| `slope_momentum` | Slope change | 21-day change in slope |
| `real_rate_proxy` | Real rate estimate | 10Y - 10Y breakeven |

## Credit Features

### Spread Levels

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `ig_spread` | Investment grade spread | ICE BofA IG OAS (bps) |
| `hy_spread` | High yield spread | ICE BofA HY OAS (bps) |
| `hy_ig_diff` | HY-IG differential | HY spread - IG spread |

### Spread Dynamics

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `ig_spread_percentile` | IG percentile | 2-year percentile |
| `hy_spread_percentile` | HY percentile | 2-year percentile |
| `spread_momentum_10d` | Spread momentum | 10-day change |
| `spread_velocity` | Spread acceleration | Change in momentum |
| `spread_zscore` | Standardized spread | Z-score vs 63-day |

### Stress Indicators

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `credit_stress_flag` | Stress indicator | 1 if HY > 500bps |
| `spread_spike_flag` | Spike detection | 1 if 2-sigma move |
| `spread_regime` | Credit regime | High/Normal/Low |

## Volatility Features

### VIX Levels

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `vix_level` | VIX index | Raw value |
| `vix_percentile_1y` | 1-year percentile | Percentile rank |
| `vix_percentile_2y` | 2-year percentile | Percentile rank |
| `vix_zscore` | Standardized VIX | Z-score vs 63-day |

### VIX Term Structure

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `vix_term_structure` | Term structure slope | VIX3M - VIX |
| `vix_contango_flag` | Contango indicator | 1 if VIX3M > VIX |
| `vix_backwardation_flag` | Backwardation | 1 if VIX > VIX3M |

### Volatility Dynamics

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `vix_momentum_5d` | VIX momentum | 5-day change |
| `vix_momentum_10d` | VIX momentum | 10-day change |
| `realized_vol_21d` | Realized volatility | 21-day annualized |
| `realized_vol_63d` | Realized volatility | 63-day annualized |
| `vol_risk_premium` | VRP | VIX - realized vol |

### Volatility Regime

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `vol_regime` | Volatility regime | Low/Normal/High |
| `vvix_level` | Vol of vol | VVIX index |

## Macro Features

### Leading Indicators

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `lei_level` | LEI level | Conference Board LEI |
| `lei_yoy` | LEI year-over-year | 12-month change % |
| `lei_momentum` | LEI momentum | 6-month change |
| `lei_diffusion` | LEI breadth | Positive components % |

### Employment

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `claims_level` | Initial claims | Weekly claims |
| `claims_4wma` | Smoothed claims | 4-week average |
| `claims_momentum` | Claims trend | 4-week change |

### Sentiment

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `consumer_sentiment` | Consumer sentiment | U of M index |
| `sentiment_momentum` | Sentiment change | 3-month change |

### Financial Conditions

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `nfci` | Financial conditions | Chicago Fed NFCI |
| `recession_prob` | Recession probability | Model-based estimate |

## Cross-Asset Features

### Correlations

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `equity_bond_corr_21d` | Stock-bond correlation | 21-day rolling |
| `equity_bond_corr_63d` | Stock-bond correlation | 63-day rolling |
| `cross_asset_corr` | Average correlation | Mean pairwise corr |

### Risk Appetite

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `risk_on_off_score` | Risk appetite | Composite score |
| `safe_haven_demand` | Flight to quality | Gold/equity ratio |
| `equity_momentum` | Market momentum | SPY 20-day return |

### Divergence

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `sector_dispersion` | Sector spread | Cross-sector vol |
| `factor_momentum` | Factor performance | Value/Growth spread |

## Regime Features

### Hidden Markov Model

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `regime_low_vol_prob` | Low-vol regime | HMM state prob |
| `regime_high_vol_prob` | High-vol regime | HMM state prob |
| `regime_transition_prob` | Transition risk | Prob of state change |

### Threshold-Based

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `market_regime` | Current regime | Rule-based classification |
| `regime_persistence` | Regime duration | Days in current regime |

## Composite Features

### Stress Indices

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `composite_stress_index` | Aggregate stress | Weighted combination |
| `rate_stress_score` | Rate stress | Rate feature composite |
| `credit_stress_score` | Credit stress | Credit feature composite |
| `vol_stress_score` | Vol stress | Volatility composite |

### Early Warning

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `early_warning_score` | EWS | Multi-factor score 0-1 |

## Feature Importance (Typical)

Top 10 most important features:

1. `vix_level` - VIX absolute level
2. `hy_spread` - High yield spread
3. `yield_curve_slope` - Curve slope
4. `composite_stress_index` - Aggregate stress
5. `credit_stress_score` - Credit composite
6. `vix_momentum_10d` - VIX momentum
7. `equity_bond_corr_21d` - Cross-asset correlation
8. `regime_high_vol_prob` - Volatility regime
9. `lei_yoy` - Leading indicator trend
10. `spread_momentum_10d` - Credit momentum

## Feature Engineering Notes

### Lookback Windows

Standard lookback periods:
- Short-term: 5, 10, 21 days
- Medium-term: 63 days (~3 months)
- Long-term: 252 days (~1 year)

### Normalization

Features are standardized using:
- Z-scores for continuous features
- Percentile ranks for bounded features
- Min-max scaling where appropriate

### Missing Data

Handling strategies:
- Forward fill for prices (up to 5 days)
- Interpolation for economic indicators
- NaN propagation for calculated features

See `airs/features/` for implementation details.
