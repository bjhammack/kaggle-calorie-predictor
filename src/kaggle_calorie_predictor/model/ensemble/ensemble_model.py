import pandas as pd
import numpy as np
from model.ensemble.catboost_model import train_catboost_fold, tune_catboost_params
from model.ensemble.lightgbm_model import train_lightgbm_fold, tune_lightgbm_params
from model.ensemble.xgboost_model import train_xgboost_fold, tune_xgboost_params
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from model.ensemble.params import PARAMS
from model.ensemble.weights import WEIGHTS
import matplotlib.pyplot as plt
import shap
from catboost import Pool
from pathlib import Path


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
        df["HR_by_Age"] = df["Heart_Rate"] / df["Age"]
        df["Temp_Increase_Rate"] = df["Body_Temp"] / df["Duration"]
        df["Heat_Generation"] = df["Duration"] * df["Temp_Above_Basal"]
        df["Effort_by_BMI"] = df["Effort"] / df["BMI"]

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
        "Effort",
        "HR_per_kg",
        "Temp_Above_Basal",
        "HR_by_Age",
        "Temp_Increase_Rate",
        "Heat_Generation",
        "Effort_by_BMI",
    ]
    cat_features = ["Sex"]

    X = train[features]
    y = np.log1p(train["Calories"])
    X_test = test[features]
    return X, y, X_test, cat_features, train, test


def train_ensemble(
    train,
    test,
    X,
    y,
    X_test,
    cat_features,
    params=None,
    models=None,
):
    n_trials = 30
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_cat = np.zeros(len(train))
    oof_lgb = np.zeros(len(train))
    oof_xgb = np.zeros(len(train))

    test_cat = np.zeros(len(test))
    test_lgb = np.zeros(len(test))
    test_xgb = np.zeros(len(test))

    if not params:
        print("Tuning hyperparameters for CatBoost, LightGBM, and XGBoost...")
        catboost_params = tune_catboost_params(
            X, y, cat_features, n_trials=n_trials, n_splits=kf.n_splits
        )
        lgbm_params = tune_lightgbm_params(
            X, y, n_trials=n_trials, n_splits=kf.n_splits
        )
        xgb_params = tune_xgboost_params(X, y, n_trials=n_trials, n_splits=kf.n_splits)
    else:
        print("Using provided hyperparameters for CatBoost, LightGBM, and XGBoost...")
        catboost_params = params.get("catboost", {})
        lgbm_params = params.get("lightgbm", {})
        xgb_params = params.get("xgboost", {})

    for fold, (train_idx, val_idx) in tqdm(
        enumerate(kf.split(X)), total=kf.get_n_splits(X)
    ):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        trained_models = []
        if "catboost" in models or not models:
            oof_cat, test_cat, cat_model = train_catboost_fold(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                cat_features,
                val_idx,
                oof_cat,
                test_cat,
                kf_n_splits=kf.n_splits,
                params=catboost_params,
            )
            trained_models.append(cat_model)
        if "lightgbm" in models or not models:
            oof_lgb, test_lgb, lgb_model = train_lightgbm_fold(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                val_idx,
                oof_lgb,
                test_lgb,
                kf_n_splits=kf.n_splits,
                fold=fold,
                params=lgbm_params,
            )
            trained_models.append(lgb_model)
        if "xgboost" in models or not models:
            oof_xgb, test_xgb, xgb_model = train_xgboost_fold(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                val_idx,
                oof_xgb,
                test_xgb,
                kf_n_splits=kf.n_splits,
                fold=fold,
                params=xgb_params,
            )
            trained_models.append(xgb_model)

    return oof_cat, oof_lgb, oof_xgb, test_cat, test_lgb, test_xgb, trained_models


def refine_weights(
    oof_cat, oof_lgb, oof_xgb, test_cat, test_lgb, test_xgb, y, weights=None
):
    if weights:
        oof_blend = weights[0] * oof_cat + weights[1] * oof_lgb + weights[2] * oof_xgb
        test_preds_blend = (
            weights[0] * test_cat + weights[1] * test_lgb + weights[2] * test_xgb
        )
        best_score = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_blend)))
        return oof_blend, test_preds_blend, best_score

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

    if path.exists():
        results = pd.read_csv(path)
    else:
        results = pd.DataFrame(columns=["version", "rmsle", "date"])

    new_result = pd.DataFrame(
        [{"version": version_name, "rmsle": best_rmsle, "date": pd.Timestamp.now()}]
    )

    results = pd.concat([results, new_result], ignore_index=True)
    results = results.sort_values(by="rmsle")
    results.to_csv(path, index=False)


def analyze_feature_importance(
    models, trained_models, X, y, cat_features, version_name
):
    print("[🔍] Analyzing feature importance...")
    output_dir = Path(f"model/ensemble/feature_analysis/{version_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in zip(models, trained_models):
        shap_values = model.get_feature_importance(
            type="ShapValues", data=Pool(X, label=y, cat_features=cat_features)
        )

        shap.summary_plot(shap_values[:, :-1], X, show=False, plot_size=(10, 6))
        summary_path = output_dir / f"{model_name}_shap_summary.png"
        plt.savefig(summary_path, bbox_inches="tight")
        plt.clf()

        mean_abs_shap = np.abs(shap_values[:, :-1]).mean(axis=0)
        feature_names = X.columns

        shap_df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
        ).sort_values(by="mean_abs_shap", ascending=False)

        shap_df.plot.barh(x="feature", y="mean_abs_shap", figsize=(8, 6), legend=False)
        plt.title(f"{model_name} - Mean Absolute SHAP Value per Feature")
        plt.gca().invert_yaxis()

        bar_path = output_dir / f"{model_name}_shap_bar.png"
        plt.savefig(bar_path, bbox_inches="tight")
        plt.clf()

        print(f"[📂] Saved feature importance analysis to {output_dir}.")


def main(
    version_name,
    submission=True,
    flatten=True,
    params=None,
    weights=None,
    models=None,
    feature_importance=False,
):
    X, y, X_test, cat_features, train, test = data_prep()
    oof_cat, oof_lgb, oof_xgb, test_cat, test_lgb, test_xgb, trained_models = (
        train_ensemble(train, test, X, y, X_test, cat_features, params, models)
    )
    oof_blend, test_preds_blend, best_rmsle = refine_weights(
        oof_cat, oof_lgb, oof_xgb, test_cat, test_lgb, test_xgb, y, weights=weights
    )
    if flatten:
        print("Flattening predictions...")
        final_test_preds = prediction_flattening(oof_blend, test_preds_blend, train)
    else:
        final_test_preds = np.expm1(test_preds_blend)
    if feature_importance:
        analyze_feature_importance(
            models, trained_models, X, y, cat_features, version_name
        )
    update_model_results(version_name, best_rmsle)
    if submission:
        final_submission(version_name, final_test_preds, test)


if __name__ == "__main__":
    models = [
        "catboost",
        # "lightgbm",
        # "xgboost",
    ]
    main(
        "ensemble_v5.3",
        submission=False,
        flatten=False,
        params=PARAMS["v5.2"],
        weights=WEIGHTS["v5.2"],
        models=models,
        feature_importance=True,
    )
    print("Ensemble model training and submission completed.")
