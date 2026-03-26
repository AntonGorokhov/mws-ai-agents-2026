# === INJECTED BY EXECUTOR (do not modify) ===
import warnings; warnings.filterwarnings('ignore')
TRAIN_INPUT = '/Users/antongorokhov/Documents/mws-ai-agents-2026/data/train.csv'
TEST_INPUT = '/Users/antongorokhov/Documents/mws-ai-agents-2026/data/test.csv'
TRAIN_OUTPUT = '/Users/antongorokhov/Documents/mws-ai-agents-2026/workspace/cleaned_train.csv'
TEST_OUTPUT = '/Users/antongorokhov/Documents/mws-ai-agents-2026/workspace/cleaned_test.csv'
SUBMISSION_PATH = '/Users/antongorokhov/Documents/mws-ai-agents-2026/workspace/submission.csv'
MODEL_PATH = '/Users/antongorokhov/Documents/mws-ai-agents-2026/workspace/model.pkl'
TARGET_COLUMN = 'target'

# Safety helper: drop non-numeric columns before model training
def _safe_drop_non_numeric(df):
    return df.select_dtypes(exclude=['object', 'datetime64', 'datetime64[ns]'])

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle

train_df = pd.read_csv(TRAIN_INPUT)
test_df = pd.read_csv(TEST_INPUT)

# Combine for feature engineering
all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Feature Engineering
all_df['has_last_review'] = all_df['last_dt'].notna().astype(int)
all_df['last_review_year'] = pd.to_datetime(all_df['last_dt'], errors='coerce').dt.year
all_df['last_review_month'] = pd.to_datetime(all_df['last_dt'], errors='coerce').dt.month
all_df['last_review_day_of_week'] = pd.to_datetime(all_df['last_dt'], errors='coerce').dt.dayofweek
all_df['days_since_last_review'] = (pd.Timestamp('2020-01-01') - pd.to_datetime(all_df['last_dt'], errors='coerce')).dt.days
all_df['host_listing_count'] = all_df.groupby('host_name')['host_name'].transform('count')
all_df['lat_lon_interaction'] = all_df['lat'] * all_df['lon']
all_df['distance_from_center'] = np.sqrt((all_df['lat'] - 40.7128)**2 + (all_df['lon'] - (-74.0060))**2)
all_df['price_per_day'] = all_df['sum'] / (all_df['min_days'] + 1)
all_df['review_density'] = all_df['amt_reviews'] / (all_df['total_host'] + 1)
all_df['avg_reviews_missing'] = all_df['avg_reviews'].isna().astype(int)

# Target Encoding for location and host_name with CV and smoothing
global_mean = train_df[TARGET_COLUMN].mean()

def target_encode_cv(train_df, test_df, column, target_col, smoothing):
    # Initialize new columns
    train_df[f'{column}_target_enc'] = np.nan
    test_df[f'{column}_target_enc'] = np.nan
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_df = train_df.reset_index(drop=True)
    
    for train_idx, val_idx in kf.split(train_df):
        # Compute means from out-of-fold data
        means = train_df.iloc[train_idx].groupby(column)[target_col].agg(['mean', 'count'])
        # Apply smoothing
        smoothed = (means['count'] * means['mean'] + smoothing * global_mean) / (means['count'] + smoothing)
        smoothed_dict = smoothed.to_dict()
        
        # Map to validation set
        train_df.loc[val_idx, f'{column}_target_enc'] = train_df.loc[val_idx, column].map(smoothed_dict).fillna(global_mean)
    
    # For test set, use full train data
    means_full = train_df.groupby(column)[target_col].agg(['mean', 'count'])
    smoothed_full = (means_full['count'] * means_full['mean'] + smoothing * global_mean) / (means_full['count'] + smoothing)
    smoothed_full_dict = smoothed_full.to_dict()
    test_df[f'{column}_target_enc'] = test_df[column].map(smoothed_full_dict).fillna(global_mean)
    
    return train_df, test_df

# Apply target encoding BEFORE dropping the columns
train_part = all_df.iloc[:len(train_df)].copy()
test_part = all_df.iloc[len(train_df):].copy()

# For location
train_part, test_part = target_encode_cv(train_part, test_part, 'location', TARGET_COLUMN, smoothing=10)
# For host_name
train_part, test_part = target_encode_cv(train_part, test_part, 'host_name', TARGET_COLUMN, smoothing=20)

# Combine back
all_df = pd.concat([train_part, test_part], axis=0, ignore_index=True)

# Drop specified columns
drop_cols = ["_id"]
all_df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Drop after features (now safe since we've already done target encoding)
drop_after_features = ["name", "host_name", "last_dt"]
all_df.drop(columns=drop_after_features, inplace=True, errors='ignore')

# Categorical Encoding
# Label Encoding for location_cluster and type_house
for col in ['location_cluster', 'type_house']:
    le = LabelEncoder()
    all_df[col] = le.fit_transform(all_df[col].astype(str))

# Frequency Encoding for location
freq_map = all_df['location'].value_counts(normalize=True).to_dict()
all_df['location_freq'] = all_df['location'].map(freq_map)

# Fill NaN values
fill_na_config = {
    "last_review_year": "median",
    "last_review_month": "median",
    "last_review_day_of_week": "median",
    "days_since_last_review": "median",
    "avg_reviews": "median",
    "location": "missing",
    "type_house": "missing"
}

for col, method in fill_na_config.items():
    if method == "median":
        all_df[col] = all_df[col].fillna(all_df[col].median())
    elif method == "missing":
        all_df[col] = all_df[col].fillna("missing")

# Drop any remaining object columns
numeric_df = all_df.select_dtypes(exclude=['object'])
# Ensure we have the target column
if TARGET_COLUMN not in numeric_df.columns and TARGET_COLUMN in all_df.columns:
    numeric_df[TARGET_COLUMN] = all_df[TARGET_COLUMN]

# Align train/test columns
common_cols = sorted(list(set(numeric_df.columns)))
train_df_processed = numeric_df.iloc[:len(train_df)][common_cols]
test_df_processed = numeric_df.iloc[len(train_df):][common_cols]

# Prepare final datasets
X_train = train_df_processed.drop(columns=[TARGET_COLUMN])
y_train = train_df_processed[TARGET_COLUMN]
X_test = test_df_processed.drop(columns=[TARGET_COLUMN])

# Two-stage modeling
# Stage 1: Classifier
y_binary = (y_train > 0).astype(int)
classifier_probas = np.zeros(len(X_test))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

clf_params = {
    'objective': 'binary',
    'n_estimators': 1000,
    'random_state': 42,
    'n_jobs': -1
}

mse_scores_clf = []
try:
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_binary.iloc[train_idx], y_binary.iloc[val_idx]
        
        clf = lgb.LGBMClassifier(**clf_params)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        preds = clf.predict_proba(X_val)[:, 1]
        mse = mean_squared_error(y_val, preds)
        mse_scores_clf.append(mse)
        
        classifier_probas += clf.predict_proba(X_test)[:, 1] / 5
        
    print(f"MSE (Classifier): {np.mean(mse_scores_clf)}")
except Exception as e:
    print(f"Classifier failed: {e}")

# Stage 2: Regressor on non-zero samples only
mask = y_train > 0
X_train_reg = X_train[mask]
y_train_reg = y_train[mask]

reg_models = {}
predictions = []

# LGBM Regressor
try:
    lgb_params = {
        'num_leaves': 63,
        'learning_rate': 0.05,
        'n_estimators': 2000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    lgb_preds = np.zeros(len(X_test))
    kf_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores_lgb = []
    
    for train_idx, val_idx in kf_reg.split(X_train_reg):
        X_tr, X_val = X_train_reg.iloc[train_idx], X_train_reg.iloc[val_idx]
        y_tr, y_val = y_train_reg.iloc[train_idx], y_train_reg.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        mse_scores_lgb.append(mse)
        
        lgb_preds += model.predict(X_test) / 5
        
    print(f"MSE (LGBM Regressor): {np.mean(mse_scores_lgb)}")
    predictions.append(lgb_preds)
    reg_models['lgb'] = model
except Exception as e:
    print(f"LGBM Regressor failed: {e}")

# XGBoost Regressor
try:
    xgb_params = {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 2000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 100
    }
    
    xgb_preds = np.zeros(len(X_test))
    kf_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores_xgb = []
    
    for train_idx, val_idx in kf_reg.split(X_train_reg):
        X_tr, X_val = X_train_reg.iloc[train_idx], X_train_reg.iloc[val_idx]
        y_tr, y_val = y_train_reg.iloc[train_idx], y_train_reg.iloc[val_idx]
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=0
        )
        
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        mse_scores_xgb.append(mse)
        
        xgb_preds += model.predict(X_test) / 5
        
    print(f"MSE (XGB Regressor): {np.mean(mse_scores_xgb)}")
    predictions.append(xgb_preds)
    reg_models['xgb'] = model
except Exception as e:
    print(f"XGB Regressor failed: {e}")

# CatBoost Regressor
try:
    cb_params = {
        'depth': 6,
        'learning_rate': 0.05,
        'iterations': 2000,
        'l2_leaf_reg': 3,
        'random_strength': 1,
        'bagging_temperature': 0.8,
        'random_seed': 42,
        'thread_count': -1,
        'verbose': 0,
        'early_stopping_rounds': 100
    }
    
    cb_preds = np.zeros(len(X_test))
    kf_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores_cb = []
    
    for train_idx, val_idx in kf_reg.split(X_train_reg):
        X_tr, X_val = X_train_reg.iloc[train_idx], X_train_reg.iloc[val_idx]
        y_tr, y_val = y_train_reg.iloc[train_idx], y_train_reg.iloc[val_idx]
        
        model = CatBoostRegressor(**cb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            verbose=0
        )
        
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        mse_scores_cb.append(mse)
        
        cb_preds += model.predict(X_test) / 5
        
    print(f"MSE (CatBoost Regressor): {np.mean(mse_scores_cb)}")
    predictions.append(cb_preds)
    reg_models['catboost'] = model
except Exception as e:
    print(f"CatBoost Regressor failed: {e}")

# Ensemble regression predictions (simple average)
if len(predictions) > 0:
    reg_pred = np.mean(np.array(predictions), axis=0)
else:
    # Fallback if all regressors fail
    reg_pred = np.full(len(X_test), y_train_reg.mean())

# Combine classifier and regressor
final_pred = classifier_probas * reg_pred

# Clip predictions
final_pred = np.clip(final_pred, 0, 365)

# Save submission
submission = pd.DataFrame({TARGET_COLUMN: final_pred})
submission.to_csv(SUBMISSION_PATH, index=False)

# Save models
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({
        'classifier': clf if 'clf' in locals() else None,
        'regressors': reg_models,
        'columns': list(X_train.columns)
    }, f)