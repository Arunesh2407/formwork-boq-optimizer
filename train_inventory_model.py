import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("formwork_dataset_enhanced.csv")

# Expand each row into weeks (approx) -> small synthetic TS
rows = []
for _, row in df.iterrows():
    for week in range(int(row["planned_start_week"]), int(row["planned_end_week"]) + 1):
        rows.append({
            "week": week,
            "floor": row["floor"],
            "weekly_area_demand_sqm": row["weekly_area_demand_sqm"]
        })

weekly_df = pd.DataFrame(rows)

# Aggregate by week (total demand)
weekly_agg = weekly_df.groupby("week", as_index=False)["weekly_area_demand_sqm"].sum()
weekly_agg.rename(columns={"weekly_area_demand_sqm": "total_weekly_area_demand_sqm"}, inplace=True)

print(weekly_agg.head())

X = weekly_agg[["week"]]
y = weekly_agg["total_weekly_area_demand_sqm"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Inventory Demand Model ===")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 :", r2_score(y_test, y_pred))

joblib.dump(model, "inventory_weekly_demand_model.pkl")
print("\nInventory model saved as inventory_weekly_demand_model.pkl")