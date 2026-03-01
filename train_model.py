import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("formwork_dataset.csv")

# Select features
features = [
    "length_m",
    "width_m",
    "height_m",
    "quantity_elements",
    "wastage_percent",
    "unit_cost_per_sqm",
]

X = df[features]
y = df["total_formwork_area_sqm"]

# Split data (70 train / 20 val / 10 test)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.3333, random_state=42
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validation evaluation
y_val_pred = model.predict(X_val)
print("Validation MAE:", mean_absolute_error(y_val, y_val_pred))
print("Validation R2:", r2_score(y_val, y_val_pred))

# Test evaluation
y_test_pred = model.predict(X_test)
print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
print("Test R2:", r2_score(y_test, y_test_pred))

# Plot results
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

import joblib

# Save the trained model
joblib.dump(model, "formwork_model.pkl")

print("Model saved successfully!")