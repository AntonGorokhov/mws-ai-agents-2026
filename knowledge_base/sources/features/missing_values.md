# Missing Value Strategies

## Diagnosis
```python
# Check missing percentages
missing = df.isnull().sum() / len(df) * 100
print(missing[missing > 0].sort_values(ascending=False))
```

## Strategy by Column Type

### Numerical Columns
- **Median** imputation: robust to outliers, good default
- **Mean** imputation: only if distribution is symmetric
- **Zero fill**: when missing genuinely means zero (e.g., count of events)
- **KNN imputation**: uses similar rows, slower but more accurate
```python
from sklearn.impute import SimpleImputer, KNNImputer
df['col'] = SimpleImputer(strategy='median').fit_transform(df[['col']])
```

### Categorical Columns
- **Mode** (most frequent): safe default
- **"missing" category**: treat missing as its own category — often very informative
```python
df['col'] = df['col'].fillna('missing')
```

### DateTime Columns
- Fill with median date or create `is_missing` flag
- Missing date is often informative (e.g., no last review = new listing)

## When to Drop
- Column has >70% missing → usually drop unless domain knowledge says otherwise
- Row-level drops only if very few rows (<1%) have missing values

## Missing Indicator Feature
- Create binary flag: `col_is_missing = col.isna().astype(int)`
- Captures the pattern of missingness itself
- Especially useful for tree models
```python
df['col_missing'] = df['col'].isna().astype(int)
```

## Tips
- NEVER impute test set using test statistics — always use train set statistics
- For tree models: XGBoost/LightGBM/CatBoost handle NaN natively
- Missing values in target → drop those rows (never impute target)
- Check if missingness correlates with target (informative missingness)
