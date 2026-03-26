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
import pickle

train_df = pd.read_csv(TRAIN_INPUT)
test_df = pd.read_csv(TEST_INPUT)

# Feature Engineering on both datasets
for df in [train_df, test_df]:
    df['last_dt_year'] = pd.to_datetime(df['last_dt'], errors='coerce').dt.year
    df['last_dt_month'] = pd.to_datetime(df['last_dt'], errors='coerce').dt.month
    df['last_dt_day_of_week'] = pd.to_datetime(df['last_dt'], errors='coerce').dt.dayofweek
    df['days_since_last_review'] = (pd.Timestamp('2020-01-01') - pd.to_datetime(df['last_dt'], errors='coerce')).dt.days
    df['has_last_review'] = df['last_dt'].notna().astype(int)
    df['lat_lon_interaction'] = df['lat'] * df['lon']
    df['reviews_per_day'] = df['amt_reviews'] / (df['days_since_last_review'] + 1)
    df['price_per_person_est'] = df['sum'] / (df['total_host'] + 1)
    df['host_experience'] = df['total_host'] * df['amt_reviews']

# Drop specified columns
drop_cols = ["name", "_id"]
train_df.drop(columns=drop_cols, inplace=True, errors='ignore')
test_df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Drop datetime source columns after feature extraction
drop_after = ["last_dt"]
train_df.drop(columns=drop_after, inplace=True, errors='ignore')
test_df.drop(columns=drop_after, inplace=True, errors='ignore')

# Categorical Encoding
freq_enc_cols = []
label_enc_cols = []

for col, method in {
    "host_name": "frequency",
    "location_cluster": "label",
    "location": "frequency",
    "type_house": "label"
}.items():
    if method == "frequency":
        freq_map = train_df[col].value_counts().to_dict()
        train_df[col + '_freq'] = train_df[col].map(freq_map)
        test_df[col + '_freq'] = test_df[col].map(freq_map)
        freq_enc_cols.append(col + '_freq')
    elif method == "label":
        combined = pd.concat([train_df[[col]], test_df[[col]]], ignore_index=True)
        labels = combined[col].astype('category').cat.codes
        label_map = dict(zip(combined[col].values, labels))
        train_df[col] = train_df[col].map(label_map)
        test_df[col] = test_df[col].map(label_map)
        label_enc_cols.append(col)

# Fill NaN values
fill_na_config = {
    "last_dt_year": "median",
    "last_dt_month": "median",
    "last_dt_day_of_week": "median",
    "days_since_last_review": "median",
    "avg_reviews": "zero"
}

for col, strategy in fill_na_config.items():
    if strategy == "median":
        median_val = train_df[col].median()
        train_df[col].fillna(median_val, inplace=True)
        test_df[col].fillna(median_val, inplace=True)
    elif strategy == "zero":
        train_df[col].fillna(0, inplace=True)
        test_df[col].fillna(0, inplace=True)

# Remove all object columns
train_df = train_df.select_dtypes(exclude=['object'])
test_df = test_df.select_dtypes(exclude=['object'])

# Align columns
common_cols = sorted(list(set(train_df.columns) & set(test_df.columns)))
if TARGET_COLUMN in common_cols:
    common_cols.remove(TARGET_COLUMN)
train_df = train_df[common_cols + [TARGET_COLUMN]]
test_df = test_df[common_cols]

X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]
X_test = test_df.copy()

# Two-stage modeling
# Stage 1: Classifier to predict zero vs non-zero
y_binary = (y_train > 0).astype(int)

lgbm_clf = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=1000,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
clf_scores = []
clf_oof_preds = np.zeros(len(X_train))
clf_test_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_binary.iloc[train_idx], y_binary.iloc[val_idx]
    
    lgbm_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    val_pred = lgbm_clf.predict_proba(X_val)[:, 1]
    clf_oof_preds[val_idx] = val_pred
    clf_test_preds += lgbm_clf.predict_proba(X_test)[:, 1] / 5
    
    mse = mean_squared_error(y_val, val_pred)
    clf_scores.append(mse)

print(f"MSE (Classifier): {np.mean(clf_scores)}")

# Stage 2: Regression on non-zero samples only
mask = y_train > 0
X_reg_train = X_train[mask]
y_reg_train = y_train[mask]

models = {}
reg_test_preds = []

# LightGBM Regressor
try:
    lgb_model = lgb.LGBMRegressor(**{
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42
    })
    
    lgb_scores = []
    lgb_oof_preds = np.zeros(len(X_reg_train))
    lgb_test_pred = np.zeros(len(X_test))
    
    kf_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf_reg.split(X_reg_train):
        X_tr, X_val = X_reg_train.iloc[train_idx], X_reg_train.iloc[val_idx]
        y_tr, y_val = y_reg_train.iloc[train_idx], y_reg_train.iloc[val_idx]
        
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        val_pred = lgb_model.predict(X_val)
        lgb_oof_preds[val_idx] = val_pred
        lgb_test_pred += lgb_model.predict(X_test) / 5
        
        mse = mean_squared_error(y_val, val_pred)
        lgb_scores.append(mse)
    
    print(f"MSE (LGBM Regressor): {np.mean(lgb_scores)}")
    models['lgb'] = lgb_model
    reg_test_preds.append(lgb_test_pred)
except Exception as e:
    print("Error training LGBM:", e)

# XGBoost Regressor
try:
    xgb_model = xgb.XGBRegressor(**{
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "early_stopping_rounds": 100,
        "verbosity": 0
    })
    
    xgb_scores = []
    xgb_oof_preds = np.zeros(len(X_reg_train))
    xgb_test_pred = np.zeros(len(X_test))
    
    kf_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf_reg.split(X_reg_train):
        X_tr, X_val = X_reg_train.iloc[train_idx], X_reg_train.iloc[val_idx]
        y_tr, y_val = y_reg_train.iloc[train_idx], y_reg_train.iloc[val_idx]
        
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=0
        )
        
        val_pred = xgb_model.predict(X_val)
        xgb_oof_preds[val_idx] = val_pred
        xgb_test_pred += xgb_model.predict(X_test) / 5
        
        mse = mean_squared_error(y_val, val_pred)
        xgb_scores.append(mse)
    
    print(f"MSE (XGB Regressor): {np.mean(xgb_scores)}")
    models['xgb'] = xgb_model
    reg_test_preds.append(xgb_test_pred)
except Exception as e:
    print("Error training XGBoost:", e)

# CatBoost Regressor
try:
    cb_model = CatBoostRegressor(**{
        "depth": 6,
        "learning_rate": 0.05,
        "iterations": 2000,
        "l2_leaf_reg": 3,
        "random_strength": 1,
        "random_state": 42,
        "verbose": 0,
        "early_stopping_rounds": 100
    })
    
    cb_scores = []
    cb_oof_preds = np.zeros(len(X_reg_train))
    cb_test_pred = np.zeros(len(X_test))
    
    kf_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf_reg.split(X_reg_train):
        X_tr, X_val = X_reg_train.iloc[train_idx], X_reg_train.iloc[val_idx]
        y_tr, y_val = y_reg_train.iloc[train_idx], y_reg_train.iloc[val_idx]
        
        cb_model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            verbose=0
        )
        
        val_pred = cb_model.predict(X_val)
        cb_oof_preds[val_idx] = val_pred
        cb_test_pred += cb_model.predict(X_test) / 5
        
        mse = mean_squared_error(y_val, val_pred)
        cb_scores.append(mse)
    
    print(f"MSE (CatBoost Regressor): {np.mean(cb_scores)}")
    models['catboost'] = cb_model
    reg_test_preds.append(cb_test_pred)
except Exception as e:
    print("Error training CatBoost:", e)

# Ensemble regression predictions
if len(reg_test_preds) > 0:
    avg_reg_pred = np.mean(np.array(reg_test_preds), axis=0)
else:
    # fallback if no regressors succeeded
    avg_reg_pred = np.zeros(len(X_test))

# Final prediction combining classifier and regressor
final_pred = clf_test_preds * avg_reg_pred
final_pred = np.clip(final_pred, 0, 365)

submission = pd.DataFrame({
    'index': test_df.index,
    TARGET_COLUMN: final_pred
})
submission.to_csv(SUBMISSION_PATH, index=False)

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(models, f)