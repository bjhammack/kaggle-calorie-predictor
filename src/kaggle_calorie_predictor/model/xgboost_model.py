import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
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

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    eval_metric="rmse",
    early_stopping_rounds=10,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

preds = model.predict(X_val)
rmse = root_mean_squared_error(y_val, preds)
mae = mean_absolute_error(y_val, preds)
r2 = r2_score(y_val, preds)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²:  {r2:.3f}")

model.save_model("model/saved_models/xgboost_model.json")
