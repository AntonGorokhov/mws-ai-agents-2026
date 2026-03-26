# DateTime Feature Engineering

## Basic Extraction
```python
df['dt'] = pd.to_datetime(df['date_col'])
df['year'] = df['dt'].dt.year
df['month'] = df['dt'].dt.month
df['day'] = df['dt'].dt.day
df['dayofweek'] = df['dt'].dt.dayofweek  # 0=Mon, 6=Sun
df['hour'] = df['dt'].dt.hour
df['is_weekend'] = df['dt'].dt.dayofweek.isin([5, 6]).astype(int)
```

## Cyclical Encoding (sin/cos)
- Preserves cyclical nature (Jan is close to Dec)
```python
import numpy as np
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
```

## Time Since Features
- Days since last event, days until next event
```python
df['days_since_last_review'] = (pd.Timestamp.now() - df['last_review_dt']).dt.days
```

## Lag Features (Time Series)
```python
df['target_lag1'] = df.groupby('entity')['target'].shift(1)
df['target_rolling7'] = df.groupby('entity')['target'].transform(
    lambda x: x.shift(1).rolling(7).mean()
)
```

## Holiday / Special Date Flags
```python
us_holidays = [...]  # list of holiday dates
df['is_holiday'] = df['dt'].dt.date.isin(us_holidays).astype(int)
```

## Critical: Feature Extraction Order
- ALWAYS extract all features from a datetime column BEFORE dropping it
- Wrong: drop last_dt → try to extract year/month → fail (column gone)
- Right: extract year/month/days_since → then drop last_dt
- Put datetime source columns in a separate "drop_after_features" list, not in "drop_columns"

## Missing Datetime as Signal
- Missing dates are often highly informative
- Example: no last_review_date = listing has zero reviews = likely different availability pattern
- Create `has_date` binary feature BEFORE filling NaN
- Missing datetime correlates with other missing fields (e.g., avg_reviews)
- This pattern is especially important for zero-inflated targets

## Tips
- Always check for timezone issues
- For tree models: raw month/day/hour features work well without cyclical encoding
- Use a fixed reference date for "days since" features (e.g., '2020-01-01'), not pd.Timestamp.now()
- Drop the original datetime column AFTER feature extraction (not before)
