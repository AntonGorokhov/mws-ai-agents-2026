# Categorical Feature Engineering

## Encoding Methods

### Label Encoding
- Assigns integer to each category
- Use for: ordinal features, tree-based models
- Warning: implies order — only use when order exists or with trees
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['col_encoded'] = le.fit_transform(df['col'].astype(str))
```

### One-Hot Encoding
- Creates binary column per category
- Use for: low-cardinality features (<15 unique values)
- Warning: high cardinality → feature explosion
```python
df = pd.get_dummies(df, columns=['col'], drop_first=True)
```

### Frequency/Count Encoding
- Replace category with its frequency in the dataset
- Good for: high-cardinality features, preserves distribution info
```python
freq = df['col'].value_counts(normalize=True)
df['col_freq'] = df['col'].map(freq)
```

### Target Encoding (Mean Encoding)
- Replace category with mean of target for that category
- Very powerful but HIGH RISK of data leakage
- MUST use cross-validation scheme: encode fold N using folds 1..N-1
- Use smoothing to regularize rare categories: smoothed = (count * mean + global_count * global_mean) / (count + global_count)
```python
from sklearn.model_selection import KFold

def target_encode_cv(train_df, test_df, col, target_col, n_splits=5, smoothing=10):
    """CV-based target encoding with smoothing. No leakage."""
    global_mean = train_df[target_col].mean()
    train_encoded = pd.Series(np.nan, index=train_df.index)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train_df):
        tr = train_df.iloc[tr_idx]
        agg = tr.groupby(col)[target_col].agg(['mean', 'count'])
        smooth = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
        train_encoded.iloc[val_idx] = train_df.iloc[val_idx][col].map(smooth)
    train_encoded.fillna(global_mean, inplace=True)

    # For test: use full training data
    agg = train_df.groupby(col)[target_col].agg(['mean', 'count'])
    smooth = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
    test_encoded = test_df[col].map(smooth).fillna(global_mean)

    return train_encoded, test_encoded
```

### Host-Level Aggregation
- When multiple rows share same entity (e.g. host), aggregate target stats
- Use CV-based encoding to avoid leakage (same as target encoding)
- Features: mean target per host, count listings per host, std target per host
```python
# host_name target encoding = avg availability per host
train_df['host_target_enc'], test_df['host_target_enc'] = target_encode_cv(
    train_df, test_df, 'host_name', 'target', smoothing=20
)
```

### Binary Encoding
- Encode category integer as binary digits, each in separate column
- Good for: medium cardinality (15–100 unique values)

## When to Use What
| Cardinality | Method |
|-------------|--------|
| 2 values | Label encoding |
| 3–15 | One-hot encoding |
| 15–100 | Frequency or binary encoding |
| >100 | Target encoding or frequency encoding |

## For Tree Models
- CatBoost: pass as `cat_features`, no manual encoding needed
- LightGBM: pass as `categorical_feature`, uses gradient-based one-vs-other
- XGBoost: requires manual encoding (label or one-hot)
