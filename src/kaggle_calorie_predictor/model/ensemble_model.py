import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def data_prep():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    for df in [train, test]:
        df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
        df["Intensity"] = df["Heart_Rate"] / df["Duration"]
        df["Effort"] = df["Heart_Rate"] * df["Duration"]
        df["HR_per_kg"] = df["Heart_Rate"] / df["Weight"]
        df["Temp_Above_Basal"] = (
            df["Body_Temp"] - 37.0
        )  # Assuming 37.0 is the basal body temperature

    sex_map = {val: idx for idx, val in enumerate(train["Sex"].unique())}
    train["Sex"] = train["Sex"].map(sex_map)
    test["Sex"] = test["Sex"].map(sex_map)

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
        # "Effort",
        # "HR_per_kg",
        # "Temp_Above_Basal",
    ]
    cat_features = ["Sex"]

    X = train[features]
    y = np.log1p(train["Calories"])
    X_test = test[features]
    return X, y, X_test, cat_features, train, test


def train_ensemble(train, test, X, y, X_test, cat_features):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_cat = np.zeros(len(train))
    oof_lgb = np.zeros(len(train))
    oof_xgb = np.zeros(len(train))

    test_cat = np.zeros(len(test))
    test_lgb = np.zeros(len(test))
    test_xgb = np.zeros(len(test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{kf.n_splits}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        cat_train = Pool(X_train, y_train, cat_features=cat_features)
        cat_val = Pool(X_val, y_val, cat_features=cat_features)

        cat_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            eval_metric="RMSE",
            early_stopping_rounds=50,
            verbose=0,
        )
        cat_model.fit(cat_train, eval_set=cat_val)
        oof_cat[val_idx] = cat_model.predict(X_val)
        test_cat += cat_model.predict(X_test) / kf.n_splits

        lgb_model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=fold,
            early_stopping_rounds=50,
            verbose=0,
            verbosity=-1,
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        test_lgb += lgb_model.predict(X_test) / kf.n_splits

        xgb_model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=fold,
            objective="reg:squarederror",
            early_stopping_rounds=50,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict(X_val)
        test_xgb += xgb_model.predict(X_test) / kf.n_splits

    return oof_cat, oof_lgb, oof_xgb, test_cat, test_lgb, test_xgb


def refine_weights(oof_cat, oof_lgb, oof_xgb, test_cat, test_lgb, test_xgb, y):
    best_score = float("inf")
    best_weights = None

    for w1 in np.arange(0.4, 0.61, 0.05):
        for w2 in np.arange(0.2, 0.41, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0 or w3 > 0.4:
                continue

            blended = w1 * oof_cat + w2 * oof_lgb + w3 * oof_xgb
            score = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(blended)))
            if score < best_score:
                best_score = score
                best_weights = (w1, w2, w3)

    print(f"Best RMSLE: {best_score:.5f}, weights: {best_weights}")

    oof_blend = (
        best_weights[0] * oof_cat
        + best_weights[1] * oof_lgb
        + best_weights[2] * oof_xgb
    )
    test_preds_blend = (
        best_weights[0] * test_cat
        + best_weights[1] * test_lgb
        + best_weights[2] * test_xgb
    )

    return oof_blend, test_preds_blend, best_score


def prediction_flattening(oof_blend, test_preds_blend, train):
    reg = LinearRegression()
    reg.fit(np.expm1(oof_blend).reshape(-1, 1), train["Calories"])

    final_test_preds = reg.predict(np.expm1(test_preds_blend).reshape(-1, 1))
    return final_test_preds


def final_submission(submission_name, final_test_preds, test):
    test["Calories"] = final_test_preds
    submission = test[["id", "Calories"]]
    submission.to_csv(f"data/submissions/{submission_name}_submission.csv", index=False)


def update_model_results(version_name, best_rmsle):
    import pandas as pd
    from pathlib import Path

    path = Path("data/ensemble_results.csv")

    # Load existing results or create empty DataFrame
    if path.exists():
        results = pd.read_csv(path)
    else:
        results = pd.DataFrame(columns=["version", "rmsle", "date"])

    # Create new row
    new_result = pd.DataFrame(
        [{"version": version_name, "rmsle": best_rmsle, "date": pd.Timestamp.now()}]
    )

    # Concatenate and save
    results = pd.concat([results, new_result], ignore_index=True)
    results = results.sort_values(by="rmsle")
    results.to_csv(path, index=False)


def main(version_name, submission=True):
    X, y, X_test, cat_features, train, test = data_prep()
    oof_cat, oof_lgb, oof_xgb, test_cat, test_lgb, test_xgb = train_ensemble(
        train, test, X, y, X_test, cat_features
    )
    oof_blend, test_preds_blend, best_rmsle = refine_weights(
        oof_cat, oof_lgb, oof_xgb, test_cat, test_lgb, test_xgb, y
    )
    final_test_preds = prediction_flattening(oof_blend, test_preds_blend, train)
    update_model_results(version_name, best_rmsle)
    if submission:
        final_submission(version_name, final_test_preds, test)


if __name__ == "__main__":
    main("ensemble_v3", False)
    print("Ensemble model training and submission completed.")
