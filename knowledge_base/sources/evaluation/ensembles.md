# Ensemble Methods Guide

## Simple Averaging
- Average predictions from multiple models
- Works best when models are diverse (different algorithms)
```python
pred_final = (pred_lgbm + pred_xgb + pred_catboost) / 3
```

## Weighted Averaging
- Weight by CV performance — better models get higher weight
```python
# Weights from CV scores (inverse RMSE)
w1, w2, w3 = 1/rmse_lgbm, 1/rmse_xgb, 1/rmse_catboost
total = w1 + w2 + w3
pred_final = (w1*pred_lgbm + w2*pred_xgb + w3*pred_catboost) / total
```

## Stacking
- Use model predictions as features for a meta-model
- Level 0: base models generate out-of-fold predictions
- Level 1: meta-model (usually Ridge or LogReg) trained on these predictions
```python
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge

# Generate OOF predictions
oof_lgbm = cross_val_predict(lgbm, X, y, cv=5)
oof_xgb = cross_val_predict(xgb_model, X, y, cv=5)
oof_cat = cross_val_predict(catboost, X, y, cv=5)

# Stack
stack_X = np.column_stack([oof_lgbm, oof_xgb, oof_cat])
meta_model = Ridge()
meta_model.fit(stack_X, y)
```

## Blending
- Simpler than stacking: use a holdout set instead of cross-validation
- Split train into train1 (80%) and blend (20%)
- Train base models on train1, predict on blend set
- Train meta-model on blend predictions

## Tips
- Diversity matters more than individual accuracy
- Combine different model families (trees, linear, neural)
- Don't ensemble too many models — 3-5 is usually optimal
- For Kaggle: start with 3-model weighted average, add stacking if competitive
- Use same random_state and CV folds across all models for fair comparison
