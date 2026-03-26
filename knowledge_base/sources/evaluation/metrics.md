# Evaluation Metrics Guide

## Regression Metrics

### RMSE (Root Mean Squared Error)
- Most common regression metric
- Penalizes large errors heavily (squared)
- Same units as target
```python
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_true, y_pred, squared=False)
```

### MAE (Mean Absolute Error)
- Robust to outliers (no squaring)
- Use when outliers should not dominate
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

### R² (Coefficient of Determination)
- 1.0 = perfect, 0 = predicting mean, <0 = worse than mean
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

### RMSLE (Root Mean Squared Log Error)
- Good for right-skewed targets (prices, counts)
- Penalizes under-prediction more than over-prediction
```python
rmsle = mean_squared_error(np.log1p(y_true), np.log1p(y_pred), squared=False)
```

## Classification Metrics

### Accuracy
- Fraction of correct predictions. Misleading for imbalanced data

### ROC-AUC
- Area under ROC curve. Threshold-independent
- Use for binary classification, especially imbalanced

### F1 Score
- Harmonic mean of precision and recall
- Use when both false positives and negatives matter

### Log Loss
- Measures calibration of probabilities
- Lower is better. Use when probability output matters

## Choosing the Right Metric
| Task | Metric |
|------|--------|
| Regression, normal target | RMSE |
| Regression, skewed target | RMSLE or log-transform + RMSE |
| Regression, many outliers | MAE |
| Binary classification | ROC-AUC |
| Multi-class | Log Loss or Macro F1 |
| Imbalanced binary | ROC-AUC or F1 |

## Kaggle Tips
- Always match the competition metric exactly
- If metric is RMSE but target is skewed → train on log(target), predict, then expm1
- Use same metric in CV as competition leaderboard
