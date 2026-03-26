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

# Read data
train_df = pd.read_csv(TRAIN_INPUT)
test_df = pd.read_csv(TEST_INPUT)

# Drop columns
drop_cols = ["name", "_id", "host_name", "location", "last_dt"]
train_df = train_df.drop(columns=drop_cols, errors='ignore')
test_df = test_df.drop(columns=drop_cols, errors='ignore')

# Combine for feature engineering
all_df = pd.concat([train_df, test_df], axis=0)

# Feature engineering
features = [
    {"name": "last_dt_year", "formula": "pd.to_datetime(df['last_dt'], errors='coerce').dt.year"},
    {"name": "last_dt_month", "formula": "pd.to_datetime(df['last_dt'], errors='coerce').dt.month"},
    {"name": "last_dt_day_of_week", "formula": "pd.to_datetime(df['last_dt'], errors='coerce').dt.dayofweek"},
    {"name": "days_since_last_review", "formula": "(pd.Timestamp('2020-01-01') - pd.to_datetime(df['last_dt'], errors='coerce')).dt.days"},
    {"name": "has_last_review", "formula": "df['last_dt'].notna().astype(int)"},
    {"name": "lat_lon_interaction", "formula": "df['lat'] * df['lon']"},
    {"name": "price_per_review", "formula": "df['sum'] / (df['amt_reviews'] + 1)"},
    {"name": "review_density", "formula": "df['amt_reviews'] / (df['total_host'] + 1)"},
    {"name": "min_days_to_price_ratio", "formula": "df['min_days'] / (df['sum'] + 1)"}
]

for feature in features:
    try:
        all_df[feature["name"]] = eval(feature["formula"].replace("df", "all_df"))
    except Exception as e:
        print(f"Failed to create feature {feature['name']}: {e}")

# Drop original datetime column after feature extraction
all_df = all_df.drop(columns=["last_dt"], errors='ignore')

# Fill missing values
fill_na = {
    "last_dt_year": "median",
    "last_dt_month": "median",
    "last_dt_day_of_week": "median",
    "days_since_last_review": "median",
    "avg_reviews": "median"
}

for col, method in fill_na.items():
    if col in all_df.columns:
        if method == "median":
            all_df[col] = all_df[col].fillna(all_df[col].median())
        else:
            all_df[col] = all_df[col].fillna(0)

# Encode categorical variables
# One-hot encoding for 'location_cluster'
if 'location_cluster' in all_df.columns:
    all_df = pd.get_dummies(all_df, columns=['location_cluster'], prefix='loc_cluster')

# Label encoding for 'type_house'
if 'type_house' in all_df.columns:
    all_df['type_house'] = pd.Categorical(all_df['type_house']).codes

# Frequency encoding for 'host_name'
if 'host_name' in all_df.columns:
    freq_map = all_df['host_name'].value_counts().to_dict()
    all_df['host_name_freq'] = all_df['host_name'].map(freq_map)
    all_df = all_df.drop(columns=['host_name'])

# Split back to train and test
train_df = all_df.iloc[:len(train_df)]
test_df = all_df.iloc[len(train_df):].drop(columns=[TARGET_COLUMN], errors='ignore')

# Ensure consistent columns
common_cols = list(set(train_df.columns) & set(test_df.columns) - {TARGET_COLUMN})
train_df = train_df[common_cols + [TARGET_COLUMN]]
test_df = test_df[common_cols]

# Drop non-numeric columns
X_train = train_df.drop(columns=[TARGET_COLUMN])
X_train = X_train.select_dtypes(exclude=['object', 'datetime64'])
y_train = train_df[TARGET_COLUMN]
X_test = test_df.select_dtypes(exclude=['object', 'datetime64'])

# Align train and test columns
X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

# Model parameters
lgb_params = {
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_samples": 20,
    "verbose": -1
}

xgb_params = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "eval_metric": "rmse"
}

catboost_params = {
    "depth": 6,
    "learning_rate": 0.05,
    "iterations": 2000,
    "l2_leaf_reg": 3,
    "random_strength": 1,
    "bagging_temperature": 0.8,
    "verbose": 0
}

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store models and predictions
models = {}
predictions = {}
scores = {}

# LightGBM
lgb_preds = []
lgb_scores = []
for train_index, val_index in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    pred = model.predict(X_val)
    lgb_scores.append(np.sqrt(mean_squared_error(y_val, pred)))
    lgb_preds.append(model.predict(X_test))

models['lgb'] = [model]
predictions['lgb'] = np.mean(lgb_preds, axis=0)
scores['lgb'] = np.mean(lgb_scores)
print(f"RMSE LGBM: {scores['lgb']}")

# XGBoost
xgb_preds = []
xgb_scores = []
for train_index, val_index in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=0
    )
    pred = model.predict(X_val)
    xgb_scores.append(np.sqrt(mean_squared_error(y_val, pred)))
    xgb_preds.append(model.predict(X_test))

models['xgb'] = [model]
predictions['xgb'] = np.mean(xgb_preds, axis=0)
scores['xgb'] = np.mean(xgb_scores)
print(f"RMSE XGBoost: {scores['xgb']}")

# CatBoost
cat_preds = []
cat_scores = []
for train_index, val_index in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model = CatBoostRegressor(**catboost_params)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=0
    )
    pred = model.predict(X_val)
    cat_scores.append(np.sqrt(mean_squared_error(y_val, pred)))
    cat_preds.append(model.predict(X_test))

models['catboost'] = [model]
predictions['catboost'] = np.mean(cat_preds, axis=0)
scores['catboost'] = np.mean(cat_scores)
print(f"RMSE CatBoost: {scores['catboost']}")

# Ensemble predictions (weighted by inverse RMSE)
weights = {k: 1/v for k, v in scores.items()}
total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

final_pred = np.zeros(len(X_test))
for model_name in predictions:
    final_pred += weights[model_name] * predictions[model_name]

# Clip predictions
clip_range = [0, 365]
final_pred = np.clip(final_pred, clip_range[0], clip_range[1])

# Save submission
submission = pd.DataFrame({
    'index': test_df.index,
    'prediction': final_pred
})
submission.to_csv(SUBMISSION_PATH, index=False)

# Save model (save last model of each type)
model_dict = {}
for name, model_list in models.items():
    model_dict[name] = model_list[-1]
    
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_dict, f)

# Print final ensemble RMSE (approximation using weighted average of individual scores)
ensemble_rmse = sum([scores[name] * weights[name] for name in scores])
print(f"RMSE Ensemble: {ensemble_rmse}")