# Hyperparameter Tuning Guide

## Default Starting Points by Dataset Size

### Small Dataset (<5K rows)
```python
xgb_params = {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 300, "subsample": 0.8}
lgb_params = {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 300, "min_child_samples": 20}
cat_params = {"depth": 4, "iterations": 500, "learning_rate": 0.1}
```

### Medium Dataset (5K–100K rows)
```python
xgb_params = {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 1000, "subsample": 0.8}
lgb_params = {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 1000, "min_child_samples": 20}
cat_params = {"depth": 6, "iterations": 2000, "learning_rate": 0.05}
```

### Large Dataset (>100K rows)
```python
xgb_params = {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 2000, "subsample": 0.7}
lgb_params = {"num_leaves": 127, "learning_rate": 0.03, "n_estimators": 3000, "min_child_samples": 50}
cat_params = {"depth": 8, "iterations": 3000, "learning_rate": 0.03}
```

## Tuning Strategy
1. **Random Search** over coarse grid — faster than grid search
2. **Optuna** for Bayesian optimization — best for fine-tuning
3. Always use cross-validation, never tune on a single split

## Common Param Ranges for Search
```python
param_space = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [0.1, 1, 5, 10],
}
```

## Tips
- Tune one group at a time: tree structure → sampling → regularization
- Use early stopping to auto-select n_estimators
- Lower learning rate and increase n_estimators for final model
- Always fix random_state for reproducibility
