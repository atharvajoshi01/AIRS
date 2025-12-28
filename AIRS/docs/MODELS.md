# Model Architecture

This document describes the machine learning models used in AIRS.

## Model Overview

AIRS uses a stacking ensemble approach:

```
                    ┌─────────────────────────────────────┐
                    │         Meta-Learner (LogReg)       │
                    │       with Regime Weighting         │
                    └─────────────────────────────────────┘
                                      ▲
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
              ┌─────┴─────┐     ┌─────┴─────┐     ┌─────┴─────┐
              │  XGBoost  │     │ LightGBM  │     │    RF     │
              └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▲
                              ┌───────┴───────┐
                              │   Features    │
                              │   (~70 dim)   │
                              └───────────────┘
```

## Base Models

### XGBoost

Primary gradient boosting model.

**Hyperparameters:**
```python
{
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 10,  # Class imbalance
}
```

**Strengths:**
- Handles feature interactions well
- Good with numerical features
- Built-in regularization

### LightGBM

Fast gradient boosting with leaf-wise growth.

**Hyperparameters:**
```python
{
    "n_estimators": 200,
    "max_depth": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "is_unbalance": True,
}
```

**Strengths:**
- Faster training than XGBoost
- Lower memory usage
- Handles categorical features natively

### Random Forest

Bagging ensemble for stability.

**Hyperparameters:**
```python
{
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": "balanced",
}
```

**Strengths:**
- Robust to overfitting
- Provides feature importance
- Stable predictions

### Logistic Regression (Baseline)

Regularized linear model.

**Hyperparameters:**
```python
{
    "penalty": "l1",
    "C": 0.1,
    "solver": "saga",
    "max_iter": 1000,
    "class_weight": "balanced",
}
```

**Strengths:**
- Interpretable coefficients
- Fast training
- Works with limited data

## Stacking Ensemble

### Architecture

1. **Layer 1 (Base Models):**
   - Train base models on training data
   - Generate out-of-fold predictions

2. **Layer 2 (Meta-Learner):**
   - Train on base model predictions
   - Logistic regression with L2 regularization
   - Optionally includes regime features

### Training Process

```python
# Pseudo-code for stacking
for train_idx, val_idx in cv.split(X):
    for model in base_models:
        model.fit(X[train_idx], y[train_idx])
        meta_features[val_idx, model_idx] = model.predict_proba(X[val_idx])

meta_model.fit(meta_features, y)
```

### Regime-Aware Weighting

The ensemble adjusts weights based on market regime:

| Regime | XGBoost Weight | LightGBM Weight | RF Weight |
|--------|----------------|-----------------|-----------|
| Low Vol | 0.35 | 0.35 | 0.30 |
| High Vol | 0.30 | 0.30 | 0.40 |
| Transition | 0.33 | 0.33 | 0.34 |

## Model Validation

### Walk-Forward Cross-Validation

Time-series aware validation:

```
Train: [2010─────2017] Test: [2018]  ← Fold 1
Train: [2010──────2018] Test: [2019]  ← Fold 2
Train: [2010───────2019] Test: [2020] ← Fold 3
...
```

**Parameters:**
- Initial training: 7 years
- Test period: 1 year
- Embargo gap: 5 days (prevent leakage)
- Expanding window (no sliding)

### Metrics

Primary metrics:

| Metric | Target | Description |
|--------|--------|-------------|
| Precision | >0.70 | Avoiding false positives |
| Recall | >0.60 | Catching drawdowns |
| F1 Score | >0.65 | Balanced measure |
| AUC-ROC | >0.80 | Ranking quality |

### Calibration

Probability calibration using:
- Platt scaling (sigmoid)
- Isotonic regression (non-parametric)

## Target Variable

### Definition

Binary classification target:

```
y = 1 if max_drawdown(t, t+horizon) < -threshold else 0
```

**Parameters:**
- Horizon: 10-20 trading days
- Threshold: -5% to -7.5%

### Class Distribution

Typical distribution:
- Positive class (drawdown): ~5%
- Negative class (normal): ~95%

### Handling Imbalance

Techniques used:
1. Class weights in loss function
2. Focal loss (reduces easy negative weight)
3. SMOTE for training data (optional)
4. Threshold tuning for precision/recall trade-off

## Feature Importance

### Methods

1. **Permutation Importance:**
   - Model-agnostic
   - Accounts for feature interactions

2. **SHAP Values:**
   - Additive feature attribution
   - Local explanations per prediction

3. **Gain-Based (Trees):**
   - Split information gain
   - Fast computation

### Typical Top Features

1. VIX level and momentum
2. Credit spreads (HY, IG)
3. Yield curve metrics
4. Composite stress index
5. Cross-asset correlations

## Model Retraining

### Schedule

- **Monthly:** Full retraining with walk-forward
- **Quarterly:** Hyperparameter tuning
- **Ad-hoc:** If performance degrades >10%

### Triggers

Automatic retraining when:
- Precision drops below 0.60
- Feature drift PSI > 0.25
- Prediction drift detected

### MLflow Tracking

All experiments tracked:
- Hyperparameters
- Metrics (train/val/test)
- Artifacts (models, feature importance)
- Tags (regime, data version)

## Production Considerations

### Model Serving

- Model loaded at API startup
- Cached in memory
- Prediction latency: <50ms

### Version Management

```
models/
├── ensemble_v1/        # Production
├── ensemble_v2/        # Staging
└── ensemble_archive/   # Historical
```

### Fallback Strategy

If primary model fails:
1. Use threshold-based rules
2. Return conservative (high) risk estimate
3. Alert operations team

See `airs/models/` for implementation.
