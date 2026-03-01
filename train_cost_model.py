import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
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

# 70 / 20 / 10 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.3333, random_state=42
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# Base model
model = RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Validation
from sklearn.metrics import mean_absolute_percentage_error

y_val_pred = model.predict(X_val)
print("\n=== Validation Metrics ===")
print("MAE:", mean_absolute_error(y_val, y_val_pred))
print("MAPE:", mean_absolute_percentage_error(y_val, y_val_pred))
print("R2 :", r2_score(y_val, y_val_pred))

# Test
y_test_pred = model.predict(X_test)
print("\n=== Test Metrics ===")
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("MAPE:", mean_absolute_percentage_error(y_test, y_test_pred))
print("R2 :", r2_score(y_test, y_test_pred))

# Save model
joblib.dump(model, "cost_model.pkl")
print("\nCost model saved as cost_model.pkl")