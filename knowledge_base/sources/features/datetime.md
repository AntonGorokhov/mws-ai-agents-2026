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

## Tips
- Always check for timezone issues
- Missing dates may be informative — create `has_date` binary feature
- For tree models: raw month/day/hour features work well without cyclical encoding
- Drop the original datetime column after feature extraction
