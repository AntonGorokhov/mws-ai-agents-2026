# Numerical Feature Engineering

## Transformations

### Log Transform
- Use when: right-skewed distributions (prices, counts, areas)
- Compresses large values, reduces outlier impact
```python
import numpy as np
df['col_log'] = np.log1p(df['col'])  # log(1+x) handles zeros
```

### Square Root Transform
- Milder than log, good for count data
```python
df['col_sqrt'] = np.sqrt(df['col'])
```

### Box-Cox / Yeo-Johnson
- Automatic power transform to make data more Gaussian
- Yeo-Johnson handles negative values
```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df[['col']] = pt.fit_transform(df[['col']])
```

## Scaling
### StandardScaler
- Mean=0, Std=1. Use for: linear models, SVMs, neural nets
- Not needed for tree-based models

### RobustScaler
- Uses median/IQR. Better when outliers present

### MinMaxScaler
- Scale to [0,1]. Use for: neural nets, distance-based methods

## Binning
- Convert continuous to categorical bins
- Useful when relationship is non-linear and step-like
```python
df['col_bin'] = pd.qcut(df['col'], q=10, labels=False, duplicates='drop')
```

## Interaction Features
- Multiply/divide related features
- Example: price_per_sqft = price / area
```python
df['price_per_review'] = df['price'] / (df['reviews'] + 1)
df['area_rooms'] = df['area'] * df['rooms']
```

## Outlier Handling
- Clip to percentiles (e.g., 1st–99th)
```python
lower, upper = df['col'].quantile([0.01, 0.99])
df['col_clipped'] = df['col'].clip(lower, upper)
```

## For Tree Models
- Scaling NOT needed (trees split on thresholds)
- Log/sqrt transforms still help with very skewed distributions
- Binning can help if the model struggles with continuous features
