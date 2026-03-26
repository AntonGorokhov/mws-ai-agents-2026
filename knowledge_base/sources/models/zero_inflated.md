# Zero-Inflated / Two-Stage Modeling

## When to Use

Use a two-stage model when the target variable has an excess of zeros (typically >20%).
Common scenarios:
- Insurance claims (many zero claims)
- Sales prediction (many days with zero sales)
- Availability/occupancy (many fully booked or fully available)
- Count data with structural zeros

## Two-Stage Approach

### Stage 1: Binary Classifier
- Predict whether target is zero or non-zero
- Use LightGBM/XGBoost classifier with `binary` objective
- Outputs probability P(non_zero)

### Stage 2: Regression on Non-Zero Samples
- Train only on samples where target > 0
- Use LightGBM/XGBoost/CatBoost regressor
- Predicts the value given that it's non-zero

### Combining Predictions
```python
# Final prediction = P(non_zero) * regression_prediction
final_pred = classifier_proba * regressor_pred
```

This naturally handles the zero-inflation: if the classifier is confident the value is zero,
the final prediction will be close to zero regardless of the regressor's output.

## Implementation Pattern

```python
import lightgbm as lgb
from sklearn.model_selection import KFold

# Stage 1: Classify zero vs non-zero
y_binary = (y_train > 0).astype(int)
clf = lgb.LGBMClassifier(objective='binary', n_estimators=1000)

# Stage 2: Regress on non-zero only
mask_nonzero = y_train > 0
reg = lgb.LGBMRegressor(objective='regression', n_estimators=2000)

# Cross-validation with both stages
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X_train):
    # Train classifier on all data
    clf.fit(X_train[train_idx], y_binary[train_idx])
    proba = clf.predict_proba(X_train[val_idx])[:, 1]

    # Train regressor on non-zero subset only
    nz_mask = y_train[train_idx] > 0
    reg.fit(X_train[train_idx][nz_mask], y_train[train_idx][nz_mask])
    reg_pred = reg.predict(X_train[val_idx])

    # Combine
    final_pred = proba * reg_pred
```

## Tips

- Use the SAME features for both classifier and regressor
- The classifier doesn't need as many estimators (500-1000 is usually enough)
- The regressor benefits from more estimators (1500-2000) since non-zero values have more variance
- Consider using `log1p` transform on the non-zero target for the regressor if the distribution is right-skewed
- Clip final predictions to valid range (e.g., [0, 365] for availability days)
- Early stopping is important for both stages

## When NOT to Use

- If zeros are <15% of data — standard regression works fine
- If zeros are noise rather than structural (measurement error)
- If the task explicitly penalizes zero predictions differently

## Comparison with Single-Stage

| Approach | Pros | Cons |
|----------|------|------|
| Single regression | Simple, fewer hyperparams | Struggles with zero-inflation |
| Two-stage | Better handles zero-heavy targets | More complex, needs tuning of two models |
| Hurdle model | Statistically principled | Harder to implement with GBDT |

For Kaggle competitions with zero-inflated targets, two-stage typically improves score by 3-10%.
