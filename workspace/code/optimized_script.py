"""Optimized pipeline script for NYC Airbnb availability prediction.
Two-stage model + target encoding + proper feature engineering.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

TRAIN_INPUT = '/Users/antongorokhov/Documents/mws-ai-agents-2026/data/train.csv'
TEST_INPUT = '/Users/antongorokhov/Documents/mws-ai-agents-2026/data/test.csv'
SUBMISSION_PATH = '/Users/antongorokhov/Documents/mws-ai-agents-2026/workspace/submission.csv'
MODEL_PATH = '/Users/antongorokhov/Documents/mws-ai-agents-2026/workspace/model.pkl'
TARGET_COLUMN = 'target'

# ===== Load data =====
train_df = pd.read_csv(TRAIN_INPUT)
test_df = pd.read_csv(TEST_INPUT)

train_len = len(train_df)
all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# ===== Feature Engineering =====
# Datetime features (BEFORE dropping last_dt)
all_df['last_dt_parsed'] = pd.to_datetime(all_df['last_dt'], errors='coerce')
all_df['last_dt_year'] = all_df['last_dt_parsed'].dt.year
all_df['last_dt_month'] = all_df['last_dt_parsed'].dt.month
all_df['last_dt_dayofweek'] = all_df['last_dt_parsed'].dt.dayofweek
all_df['days_since_last_review'] = (pd.Timestamp('2020-01-01') - all_df['last_dt_parsed']).dt.days
all_df['has_last_review'] = all_df['last_dt_parsed'].notna().astype(int)

# Host-level features (BEFORE dropping host_name)
all_df['host_listing_count'] = all_df.groupby('host_name')['host_name'].transform('count')

# Numeric features
all_df['log_sum'] = np.log1p(all_df['sum'])
all_df['log_amt_reviews'] = np.log1p(all_df['amt_reviews'])
all_df['log_min_days'] = np.log1p(all_df['min_days'])
all_df['reviews_per_host'] = all_df['amt_reviews'] / (all_df['total_host'] + 1)
all_df['price_per_min_nights'] = all_df['sum'] / all_df['min_days'].clip(lower=1)
all_df['distance_from_center'] = np.sqrt((all_df['lat'] - 40.7128)**2 + (all_df['lon'] + 74.0060)**2)
all_df['lat_lon_interaction'] = all_df['lat'] * all_df['lon']
all_df['avg_reviews_missing'] = all_df['avg_reviews'].isna().astype(int)
all_df['sum_x_min_days'] = all_df['sum'] * all_df['min_days']
all_df['total_host_x_type'] = all_df['total_host'] * all_df['type_house'].astype('category').cat.codes
all_df['name_length'] = all_df['name'].astype(str).str.len()
all_df['name_word_count'] = all_df['name'].astype(str).str.split().str.len()

# Frequency encoding for location (how common is the neighborhood)
loc_freq = all_df['location'].value_counts(normalize=True)
all_df['location_freq'] = all_df['location'].map(loc_freq)

# Cyclical encoding for month and day of week
all_df['month_sin'] = np.sin(2 * np.pi * all_df['last_dt_month'] / 12)
all_df['month_cos'] = np.cos(2 * np.pi * all_df['last_dt_month'] / 12)
all_df['dow_sin'] = np.sin(2 * np.pi * all_df['last_dt_dayofweek'] / 7)
all_df['dow_cos'] = np.cos(2 * np.pi * all_df['last_dt_dayofweek'] / 7)

# Drop columns we no longer need
all_df.drop(columns=['name', 'last_dt', 'last_dt_parsed'], inplace=True, errors='ignore')

# ===== Label encoding for low-cardinality categoricals =====
le_type = LabelEncoder()
all_df['type_house'] = le_type.fit_transform(all_df['type_house'].astype(str))

le_cluster = LabelEncoder()
all_df['location_cluster'] = le_cluster.fit_transform(all_df['location_cluster'].astype(str))

# ===== Fill NaN =====
all_df['avg_reviews'] = all_df['avg_reviews'].fillna(0)
all_df['last_dt_year'] = all_df['last_dt_year'].fillna(all_df['last_dt_year'].median())
all_df['last_dt_month'] = all_df['last_dt_month'].fillna(all_df['last_dt_month'].median())
all_df['last_dt_dayofweek'] = all_df['last_dt_dayofweek'].fillna(all_df['last_dt_dayofweek'].median())
all_df['days_since_last_review'] = all_df['days_since_last_review'].fillna(all_df['days_since_last_review'].median())
# fillna only numeric columns
for col in all_df.select_dtypes(include=['number']).columns:
    all_df[col] = all_df[col].fillna(0)

# ===== Split back =====
train_part = all_df.iloc[:train_len].copy()
test_part = all_df.iloc[train_len:].copy()

# ===== CV-based target encoding for location and host_name =====
y_full = train_df[TARGET_COLUMN].values
global_mean = y_full.mean()

def target_encode_cv(train_vals, test_vals, y, smoothing=10, n_splits=5):
    """CV-based target encoding with smoothing. Returns encoded train and test arrays."""
    train_encoded = np.full(len(train_vals), np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, val_idx in kf.split(train_vals):
        # Build mapping from training fold
        tr_cats = train_vals[tr_idx]
        tr_y = y[tr_idx]
        mapping = {}
        for cat in np.unique(tr_cats):
            mask = tr_cats == cat
            count = mask.sum()
            cat_mean = tr_y[mask].mean()
            mapping[cat] = (count * cat_mean + smoothing * global_mean) / (count + smoothing)

        # Apply to validation fold
        val_cats = train_vals[val_idx]
        train_encoded[val_idx] = np.array([mapping.get(c, global_mean) for c in val_cats])

    # For test: use all training data
    mapping_full = {}
    for cat in np.unique(train_vals):
        mask = train_vals == cat
        count = mask.sum()
        cat_mean = y[mask].mean()
        mapping_full[cat] = (count * cat_mean + smoothing * global_mean) / (count + smoothing)

    test_encoded = np.array([mapping_full.get(c, global_mean) for c in test_vals])

    return train_encoded, test_encoded

# Target encode location (220 unique values)
train_part['location'] = train_part['location'].fillna('__MISSING__')
test_part['location'] = test_part['location'].fillna('__MISSING__')
train_loc = train_part['location'].values.astype(str)
test_loc = test_part['location'].values.astype(str)
train_part['location_target_enc'], test_part['location_target_enc'] = target_encode_cv(
    train_loc, test_loc, y_full, smoothing=10
)

# Target encode host_name (9629 unique values)
train_part['host_name'] = train_part['host_name'].fillna('__MISSING__')
test_part['host_name'] = test_part['host_name'].fillna('__MISSING__')
train_host = train_part['host_name'].values.astype(str)
test_host = test_part['host_name'].values.astype(str)
train_part['host_target_enc'], test_part['host_target_enc'] = target_encode_cv(
    train_host, test_host, y_full, smoothing=20
)

# Drop original categorical columns
train_part.drop(columns=['location', 'host_name'], inplace=True, errors='ignore')
test_part.drop(columns=['location', 'host_name'], inplace=True, errors='ignore')

# Drop remaining object columns
train_part = train_part.select_dtypes(exclude=['object'])
test_part = test_part.select_dtypes(exclude=['object'])

# Align columns
common_cols = sorted(set(train_part.columns) & set(test_part.columns) - {TARGET_COLUMN})
X_train = train_part[common_cols].copy()
y_train = train_part[TARGET_COLUMN].values
X_test = test_part[common_cols].copy()

print(f"Features: {len(common_cols)}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Target: mean={y_train.mean():.2f}, zeros={(y_train==0).sum()} ({(y_train==0).mean()*100:.1f}%)")

# ===== STAGE 1: Binary Classifier (zero vs non-zero) =====
y_binary = (y_train > 0).astype(int)
clf_oof = np.zeros(len(X_train))
clf_test = np.zeros(len(X_test))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    clf = lgb.LGBMClassifier(
        objective='binary', n_estimators=1000, num_leaves=63,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1
    )
    clf.fit(
        X_train.iloc[tr_idx], y_binary[tr_idx],
        eval_set=[(X_train.iloc[val_idx], y_binary[val_idx])],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    clf_oof[val_idx] = clf.predict_proba(X_train.iloc[val_idx])[:, 1]
    clf_test += clf.predict_proba(X_test)[:, 1] / 5

clf_mse = mean_squared_error(y_binary, clf_oof)
print(f"Classifier OOF MSE: {clf_mse:.6f}")

# ===== STAGE 2: Regressor on non-zero samples =====
mask = y_train > 0
X_train_reg = X_train[mask].copy()
y_train_reg = y_train[mask].copy()

print(f"Non-zero samples: {mask.sum()} ({mask.mean()*100:.1f}%)")

# Model configs
models_config = {
    'lgbm': {
        'params': dict(num_leaves=63, learning_rate=0.05, n_estimators=2000,
                       subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                       min_child_samples=20, random_state=42, n_jobs=-1),
    },
    'xgb': {
        'params': dict(max_depth=6, learning_rate=0.05, n_estimators=2000,
                       subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                       min_child_weight=5, random_state=42, n_jobs=-1, early_stopping_rounds=100),
    },
    'catboost': {
        'params': dict(depth=6, learning_rate=0.05, iterations=2000,
                       l2_leaf_reg=3, random_strength=1, bagging_temperature=0.8,
                       random_seed=42, verbose=0, early_stopping_rounds=100),
    },
}

reg_results = {}  # model_name -> (test_preds, oof_mse)

for name, cfg in models_config.items():
    try:
        oof_preds = np.zeros(len(X_train_reg))
        test_preds = np.zeros(len(X_test))
        kf_reg = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (tr_idx, val_idx) in enumerate(kf_reg.split(X_train_reg)):
            X_tr = X_train_reg.iloc[tr_idx]
            X_val = X_train_reg.iloc[val_idx]
            y_tr = y_train_reg[tr_idx]
            y_val = y_train_reg[val_idx]

            if name == 'lgbm':
                model = lgb.LGBMRegressor(**cfg['params'])
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            elif name == 'xgb':
                model = xgb.XGBRegressor(**cfg['params'])
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
            elif name == 'catboost':
                model = CatBoostRegressor(**cfg['params'])
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

            oof_preds[val_idx] = model.predict(X_val)
            test_preds += model.predict(X_test) / 5

        oof_mse = mean_squared_error(y_train_reg, oof_preds)
        reg_results[name] = (test_preds, oof_mse)
        print(f"MSE ({name} regressor): {oof_mse:.2f}")
    except Exception as e:
        print(f"{name} regressor failed: {e}")

# ===== Weighted Ensemble =====
if len(reg_results) > 0:
    # Weight by inverse MSE
    weights = {name: 1.0 / mse for name, (_, mse) in reg_results.items()}
    total_w = sum(weights.values())
    weights = {name: w / total_w for name, w in weights.items()}

    print(f"Ensemble weights: {weights}")

    reg_pred = sum(weights[name] * preds for name, (preds, _) in reg_results.items())
else:
    reg_pred = np.full(len(X_test), y_train_reg.mean())

# ===== Combine: P(non_zero) * regression_pred =====
final_pred = clf_test * reg_pred
final_pred = np.clip(final_pred, 0, 365)

print(f"Final predictions: mean={final_pred.mean():.2f}, min={final_pred.min():.2f}, max={final_pred.max():.2f}")

# ===== Save =====
submission = pd.DataFrame({
    'index': range(len(test_df)),
    'prediction': final_pred
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved: {SUBMISSION_PATH}")

with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'models': list(reg_results.keys()), 'weights': weights}, f)
print("Done!")
