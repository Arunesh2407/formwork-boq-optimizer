import pandas as pd
import numpy as np

# Original dataset load
df = pd.read_csv("formwork_dataset.csv")

rng = np.random.default_rng(42)

# ----- 1) BoQ planned quantity (thoda over/under estimate) -----
df["boq_area_sqm"] = df["total_formwork_area_sqm"] * rng.uniform(
    0.9, 1.2, size=len(df)
)

# Assume actual area = total_formwork_area_sqm (ground truth)
df["actual_area_sqm"] = df["total_formwork_area_sqm"]

# Adjustment factor: actual / BoQ (ideal: < 1 means BoQ overestimated)
df["boq_adjustment_factor"] = df["actual_area_sqm"] / df["boq_area_sqm"]

# ----- 2) Actual cost (procurement + wastage + misc) -----
# base cost = area * unit_cost
base_cost = df["actual_area_sqm"] * df["unit_cost_per_sqm"]

# Random overhead  -5% to +15% (site inefficiencies)
overhead_factor = rng.uniform(0.95, 1.15, size=len(df))
df["actual_cost"] = base_cost * overhead_factor

# ----- 3) Weekly demand create karne ke liye helper column -----
# Roughly assume demand is spread uniformly across planned weeks
df["duration_weeks"] = (
    df["planned_end_week"] - df["planned_start_week"] + 1
).clip(lower=1)

# Per-week area demand (approx)
df["weekly_area_demand_sqm"] = df["actual_area_sqm"] / df["duration_weeks"]

# Save enhanced dataset
df.to_csv("formwork_dataset_enhanced.csv", index=False)
print("Enhanced dataset saved as formwork_dataset_enhanced.csv")
print(df.head())