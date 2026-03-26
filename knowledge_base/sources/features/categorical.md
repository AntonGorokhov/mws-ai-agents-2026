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
```python
from sklearn.model_selection import KFold
def target_encode(df, col, target, n_splits=5):
    encoded = pd.Series(index=df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(df):
        means = df.iloc[tr_idx].groupby(col)[target].mean()
        encoded.iloc[val_idx] = df.iloc[val_idx][col].map(means)
    encoded.fillna(df[target].mean(), inplace=True)
    return encoded
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
