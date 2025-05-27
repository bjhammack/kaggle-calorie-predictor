import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor, Pool

# --- Load data ---
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# --- Feature Engineering ---
for df in [train, test]:
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["Intensity"] = df["Heart_Rate"] / df["Duration"]

# --- Features to use ---
features = [
    "Sex",
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
    "Intensity",
    "BMI",
]
cat_features = ["Sex"]

# --- Prepare target ---
X = train[features]
y = np.log1p(train["Calories"])  # log1p for RMSLE alignment
X_test = test[features]

# --- Cross-validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        early_stopping_rounds=50,
        verbose=0,
    )

    model.fit(train_pool, eval_set=val_pool)

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred

    test_preds += model.predict(X_test) / kf.n_splits

# --- Evaluation ---
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_preds)))
print(f"OOF RMSLE: {rmsle:.5f}")

# --- Final Submission ---
test["Calories"] = np.expm1(test_preds)  # inverse log1p
submission = test[["id", "Calories"]]
submission.to_csv("data/submissions/catboost_v1_submission.csv", index=False)
