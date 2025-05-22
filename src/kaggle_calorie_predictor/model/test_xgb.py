from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=4, noise=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="rmse",  # ✅ must be set here
    early_stopping_rounds=10,  # ✅ must be set here
)

model.fit(
    X_train, y_train, eval_set=[(X_val, y_val)], verbose=False  # ✅ still works here
)
