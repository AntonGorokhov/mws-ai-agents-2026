# Feature Selection Guide

## Correlation-Based
- Drop features with >0.95 correlation to each other (keep the one with higher target correlation)
```python
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
```

## Importance-Based
- Train model → get feature importances → drop low-importance features
```python
import lightgbm as lgb
model = lgb.LGBMRegressor(n_estimators=500, verbose=-1)
model.fit(X_train, y_train)
importances = pd.Series(model.feature_importances_, index=X_train.columns)
important_features = importances[importances > 0].index.tolist()
```

## Mutual Information
- Non-linear relationship measurement
```python
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X, y, random_state=42)
mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
```

## Recursive Feature Elimination
- Iteratively removes least important features
- Slow but thorough
```python
from sklearn.feature_selection import RFECV
selector = RFECV(estimator=model, step=1, cv=5, scoring='neg_mean_squared_error')
selector.fit(X, y)
selected = X.columns[selector.support_]
```

## Tips
- Start with all features, then prune
- Remove constant and near-constant features first
- Remove duplicated features
- For GBDT models: they handle irrelevant features fairly well, so aggressive pruning not always needed
- Always validate: compare CV score with and without dropped features
