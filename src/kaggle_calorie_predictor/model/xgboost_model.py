import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib


df = pd.read_csv("data/train.csv")
features = ["Age", "Duration", "Heart_Rate", "Body_Temp"]
target = "Calories"

X = df[features].values
y = df[target].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "model/saved_models/xgb_scaler.pkl")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.8, 1.0]
}

model = XGBRegressor(
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    eval_metric="rmse",
    early_stopping_rounds=10,
)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # Use an appropriate scoring metric
    cv=5,  # 5-fold cross-validation
    verbose=1,
    n_jobs=-1  # Use all available cores
)

grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# model.fit(
#     X_train,
#     y_train,
#     eval_set=[(X_val, y_val)],
#     verbose=False,
# )
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", -grid_search.best_score_)

# Evaluate on the test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_val, y_val)
print("Test Set Score:", test_score)

# preds = model.predict(X_val)
# rmse = root_mean_squared_error(y_val, preds)
# mae = mean_absolute_error(y_val, preds)
# r2 = r2_score(y_val, preds)

# print(f"RMSE: {rmse:.2f}")
# print(f"MAE: {mae:.2f}")
# print(f"R²:  {r2:.3f}")

best_model.save_model("model/saved_models/xgboost_model.json")
