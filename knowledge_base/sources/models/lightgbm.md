# LightGBM Guide

## When to Use
- Large datasets (fastest GBDT implementation)
- High-dimensional data with many features
- When training speed is critical
- Great for datasets > 100K rows

## Key Parameters
- `n_estimators`: 100–5000. Use early stopping
- `num_leaves`: 20–300. Main complexity param (replaces max_depth)
- `max_depth`: -1 (unlimited) or 3–12. Usually control via num_leaves instead
- `learning_rate`: 0.01–0.3
- `subsample` (bagging_fraction): 0.6–1.0
- `colsample_bytree` (feature_fraction): 0.6–1.0
- `min_child_samples`: 5–100. Prevents overfitting on small leaf nodes
- `reg_alpha`: 0–1. L1 regularization
- `reg_lambda`: 0–1. L2 regularization

## Speed Advantages
- Histogram-based splitting (much faster than exact)
- Leaf-wise growth (vs level-wise in XGBoost) — deeper trees, better accuracy
- Native support for categorical features via `categorical_feature`
- Efficient multi-threading with `n_jobs=-1`

## Best Practices
- Set `num_leaves` < 2^max_depth to prevent overfitting
- Use `callbacks=[lgb.early_stopping(50)]` for early stopping
- Handle missing values natively (uses NaN as a split point)
- For imbalanced: `is_unbalance=True` or `scale_pos_weight`
- Use `verbose=-1` to suppress training output

## Code Template
```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=2000,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
```
