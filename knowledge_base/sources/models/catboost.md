# CatBoost Guide

## When to Use
- Datasets with many categorical features (handles them natively)
- When you want minimal preprocessing
- When you need good defaults out of the box
- Ordered boosting helps prevent target leakage

## Key Parameters
- `iterations`: 100–5000. Use early stopping
- `depth`: 4–10. Default 6. CatBoost uses symmetric trees
- `learning_rate`: 0.01–0.3. Auto-selected if not set
- `l2_leaf_reg`: 1–10. L2 regularization
- `bagging_temperature`: 0–10. Controls Bayesian bootstrap intensity
- `random_strength`: 0–10. Randomness for scoring splits
- `border_count`: 32–255. Number of bins for numerical features

## Native Categorical Handling
- Pass categorical column indices via `cat_features` parameter
- No need to one-hot encode — CatBoost uses ordered target statistics
- Works with string categories directly
- Much better than one-hot for high-cardinality features

## Best Practices
- Use `verbose=100` to see progress every 100 iterations
- Use `eval_set` with early stopping: `early_stopping_rounds=50`
- For text features: use `text_features` parameter
- CatBoost handles NaN natively — treated as a separate category
- Use `task_type='GPU'` for GPU acceleration

## Code Template
```python
from catboost import CatBoostRegressor

cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']

model = CatBoostRegressor(
    iterations=2000,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
    cat_features=cat_features,
)
model.fit(X_train, y_train, eval_set=(X_val, y_val))
```
