import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import joblib

df = pd.read_csv("formwork_dataset_enhanced.csv")

feature_cols = [
    "length_m",
    "width_m",
    "height_m",
    "quantity_elements",
    "wastage_percent",
    "unit_cost_per_sqm",
    "total_formwork_area_sqm",
]

X = df[feature_cols]
y = df["actual_cost"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.3333, random_state=42
)

# Parameter grid
param_dist = {
    "n_estimators": [100, 150, 200, 300],
    "max_depth": [None, 10, 15, 20],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt"]
}

base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

search.fit(X_train, y_train)

print("\nBest params:", search.best_params_)

best_model = search.best_estimator_

# Validation performance
y_val_pred = best_model.predict(X_val)
print("\n=== Tuned Model (Validation) ===")
print("MAE:", mean_absolute_error(y_val, y_val_pred))
print("R2 :", r2_score(y_val, y_val_pred))

# Test performance
y_test_pred = best_model.predict(X_test)
print("\n=== Tuned Model (Test) ===")
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("R2 :", r2_score(y_test, y_test_pred))

joblib.dump(best_model, "cost_model_tuned.pkl")
print("\nTuned cost model saved as cost_model_tuned.pkl")