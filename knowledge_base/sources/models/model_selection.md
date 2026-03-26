# Model Selection Guide

## Decision Tree: Task Type → Model

### Regression
1. **Start with**: LightGBM or XGBoost — fast, accurate, handle mixed types
2. **If many categoricals**: CatBoost (native handling, no encoding needed)
3. **If linear relationships dominate**: Ridge/Lasso regression as baseline
4. **If small dataset (<1K rows)**: RandomForest or Ridge — less prone to overfitting

### Binary Classification
1. **Start with**: LightGBM or XGBoost with `objective='binary'`
2. **If interpretability needed**: LogisticRegression as baseline
3. **If imbalanced**: XGBoost with `scale_pos_weight` or CatBoost
4. **Always**: compare with RandomForest as sanity check

### Multi-class Classification
1. **Start with**: LightGBM with `objective='multiclass'`
2. **If many classes (>20)**: Use CatBoost or XGBoost with softprob
3. **Baseline**: RandomForest or LogisticRegression(multi_class='multinomial')

## Dataset Size Guidelines
| Rows | Recommended |
|------|-------------|
| <500 | Ridge/Lasso, RandomForest, simple models |
| 500–10K | XGBoost, CatBoost, RandomForest |
| 10K–1M | LightGBM, XGBoost, CatBoost |
| >1M | LightGBM (fastest), XGBoost with hist |

## Ensemble Strategy
- Train 3 models (LightGBM, XGBoost, CatBoost)
- Weighted average predictions: `0.4*lgbm + 0.3*xgb + 0.3*catboost`
- Use CV scores as weights: better CV = higher weight
- Stacking: use predictions as features for a meta-model (Ridge)

## Quick Baseline Approach
1. Minimal preprocessing (drop high-cardinality strings, fill NaN with median)
2. Train LightGBM with defaults
3. Evaluate with 5-fold CV
4. This gives you a baseline score to improve upon
