from xgboost import XGBRegressor
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np
from pathlib import Path


def tune_xgboost_params(X, y, n_trials=30, n_splits=5):
    def objective(trial):
        params = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "objective": "reg:squarederror",
            "random_state": 42,
            "early_stopping_rounds": 50,
            "verbosity": 0,
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y))

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            oof_preds[val_idx] = model.predict(X_val)

        rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_preds)))
        return rmsle

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    log_path = Path("data/tuning_results_xgboost.csv")
    log_entry = pd.DataFrame(
        [
            {
                "best_rmsle": study.best_value,
                "params": study.best_params,
                "timestamp": pd.Timestamp.now(),
            }
        ]
    )

    if log_path.exists():
        prev = pd.read_csv(log_path)
        log_entry = pd.concat([prev, log_entry], ignore_index=True)

    log_entry.to_csv(log_path, index=False)

    print(f"[✓] Best XGBoost RMSLE: {study.best_value:.5f}")
    print(f"[📌] Best Params: {study.best_params}")
    return study.best_params


def train_xgboost_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    val_idx,
    oof_xgb,
    test_xgb,
    kf_n_splits,
    fold,
    params=None,
):
    params["verbosity"] = 0  # Ensure verbosity is set to 0 for silent mode
    if params is None:
        params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": fold,
            "objective": "reg:squarederror",
            "early_stopping_rounds": 50,
            "verbosity": 0,
        }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    oof_xgb[val_idx] = model.predict(X_val)
    test_xgb += model.predict(X_test) / kf_n_splits

    return oof_xgb, test_xgb
