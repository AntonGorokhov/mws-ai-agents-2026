# Data Leakage Detection & Prevention

## What Is Data Leakage
Data from outside the training set used to create the model. Leads to overly optimistic evaluation but poor real-world performance.

## Common Types

### Target Leakage
- Feature that is derived from or directly correlates with the target
- Example: using "claim_paid" to predict "is_fraudulent"
- Detection: feature with AUC ≈ 1.0 or correlation ≈ 1.0 with target
```python
# Check for suspiciously high correlations
correlations = df.corr()[target_col].abs().sort_values(ascending=False)
suspect = correlations[correlations > 0.95]
```

### Train-Test Contamination
- Using test data statistics for preprocessing
- Example: fitting scaler on full dataset, then splitting
- Fix: fit on train, transform on test
```python
# WRONG
scaler.fit(pd.concat([X_train, X_test]))

# CORRECT
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Temporal Leakage
- Using future data to predict the past
- Fix: time-series split, no shuffling

### Target Encoding Leakage
- Encoding categories with target statistics computed on full training set
- Fix: use cross-validated target encoding (per-fold statistics)

## Detection Checklist
1. Any feature with very high importance and correlation > 0.9 with target?
2. Model accuracy suspiciously high (>0.99)?
3. Large gap between CV and leaderboard score?
4. Feature that wouldn't be available at prediction time?

## Prevention Rules
1. Split FIRST, preprocess SECOND
2. Use pipelines to ensure fit/transform separation
3. Always validate with CV, never on training data
4. For feature engineering: only use information that would be available in production
5. For target encoding: always cross-validate
