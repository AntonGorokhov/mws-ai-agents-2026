# XGBoost Guide

## When to Use
- Tabular data with mixed feature types
- Medium to large datasets (1K–10M rows)
- When you need high accuracy with reasonable training time
- Works well for both classification and regression

## Key Parameters
- `n_estimators`: 100–3000. Start with 500, use early stopping
- `max_depth`: 3–10. Default 6. Deeper = more complex, risk overfitting
- `learning_rate` (eta): 0.01–0.3. Lower = more trees needed but better generalization
- `subsample`: 0.6–1.0. Row sampling per tree. 0.8 is good default
- `colsample_bytree`: 0.6–1.0. Feature sampling per tree. 0.8 is good default
- `min_child_weight`: 1–10. Higher = more conservative
- `reg_alpha` (L1): 0–1. For feature selection / sparsity
- `reg_lambda` (L2): 1–10. Regularization strength
- `gamma`: 0–5. Minimum loss reduction for split

## Imbalanced Data
- Use `scale_pos_weight = count_negative / count_positive`
- Or set `sample_weight` in fit()
- For multi-class: use `objective='multi:softprob'`

## Best Practices
- Always use early stopping with validation set: `early_stopping_rounds=50`
- Use `tree_method='hist'` for speed (default in modern XGBoost)
- For GPU: `tree_method='gpu_hist', device='cuda'`
- Handle missing values natively — XGBoost learns optimal split direction
- Feature importance: use `importance_type='gain'` (not 'weight')

## Hyperparameter Tuning Order
1. Fix `learning_rate=0.1`, tune `n_estimators` with early stopping
2. Tune `max_depth` and `min_child_weight`
3. Tune `subsample` and `colsample_bytree`
4. Tune `reg_alpha` and `reg_lambda`
5. Lower `learning_rate` to 0.01–0.05, increase `n_estimators`

## Code Template
```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score

model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```
