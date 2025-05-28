import optuna
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np
from pathlib import Path


def tune_catboost_params(X, y, cat_features, n_trials=30, n_splits=5):
    def objective(trial):
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "loss_function": "RMSE",
            "verbose": 0,
            "early_stopping_rounds": 50,
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y))

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)

            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=val_pool)

            preds = model.predict(X_val)
            oof_preds[val_idx] = preds

        rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_preds)))
        return rmsle

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Logging results
    log_path = Path("data/tuning_results_catboost.csv")
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

    print(f"[✓] Best CatBoost RMSLE: {study.best_value:.5f}")
    print(f"[📌] Best Params: {study.best_params}")

    return study.best_params


def train_catboost_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    cat_features,
    val_idx,
    oof_cat,
    test_cat,
    kf_n_splits,
    params=None,
):
    # Default parameters if none provided
    if params is None:
        params = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "early_stopping_rounds": 50,
            "verbose": 0,
        }
    params["verbose"] = 0

    # Pools
    cat_train = Pool(X_train, y_train, cat_features=cat_features)
    cat_val = Pool(X_val, y_val, cat_features=cat_features)

    # Model
    model = CatBoostRegressor(**params)
    model.fit(cat_train, eval_set=cat_val)

    # Predictions
    oof_cat[val_idx] = model.predict(X_val)
    test_cat += model.predict(X_test) / kf_n_splits

    return oof_cat, test_cat, model
