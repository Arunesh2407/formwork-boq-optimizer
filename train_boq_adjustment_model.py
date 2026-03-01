import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("formwork_dataset_enhanced.csv")

# Features include element type & floor also
df_encoded = pd.get_dummies(df, columns=["element_type", "material_type"], drop_first=True)

feature_cols = [
    "length_m",
    "width_m",
    "height_m",
    "quantity_elements",
    "wastage_percent",
    "total_formwork_area_sqm",
    "boq_area_sqm",
    "floor",
    "planned_start_week",
    "planned_end_week",
]

# Add all dummy columns too
dummy_cols = [c for c in df_encoded.columns if c.startswith("element_type_") or c.startswith("material_type_")]
feature_cols += dummy_cols

X = df_encoded[feature_cols]
y = df_encoded["boq_adjustment_factor"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.3333, random_state=42
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

model = RandomForestRegressor(
    n_estimators=120,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
print("\n=== Validation ===")
print("MAE:", mean_absolute_error(y_val, y_val_pred))
print("R2 :", r2_score(y_val, y_val_pred))

y_test_pred = model.predict(X_test)
print("\n=== Test ===")
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("R2 :", r2_score(y_test, y_test_pred))

joblib.dump(model, "boq_adjustment_model.pkl")
print("\nBoQ adjustment model saved as boq_adjustment_model.pkl")