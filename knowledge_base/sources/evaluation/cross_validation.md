# Cross-Validation Guide

## Stratified K-Fold (Classification)
- Preserves class distribution in each fold
- Default choice for classification
```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
```

## K-Fold (Regression)
- Standard K-Fold for regression tasks
```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

## Time Series Split
- Never use random split for time series — future data would leak into training
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

## Group K-Fold
- When samples from same group (user, entity) must stay together
```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=df['user_id']):
    pass
```

## Number of Folds
- 5 folds: standard, good balance of bias/variance
- 10 folds: lower bias, more computation
- 3 folds: faster, for very large datasets (>500K rows)

## Best Practices
- Use same random_state everywhere for reproducibility
- Report mean ± std of CV scores
- If std is high → model is unstable, consider simpler model or more data
- Never tune hyperparameters on test set
- Nested CV for unbiased model selection (outer for eval, inner for tuning)

## Quick CV Score
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
print(f"RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")
```
