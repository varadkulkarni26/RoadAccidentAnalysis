"""
data_cleaning.py
-----------------
Loads the raw Indian Road Accident CSV, cleans it, and saves a
processed version that the dashboard and modeling scripts will use.
 
ADAPT THESE COLUMN NAMES to match your actual downloaded CSV.
Common variants across the Kaggle Indian accident datasets:
    State / State_Name / State/UT
    Year, Month, Day_of_Week / Day
    Time / Accident_Time
    Severity / Accident_Severity   (values like Fatal/Serious/Minor
                                     OR numeric 1/2/3)
    Vehicle_Type / Vehicles_Involved
    No_of_Vehicles / Number_of_Vehicles
"""
 
import pandas as pd
import numpy as np
import os
 
RAW_PATH = os.path.join("data", "india_road_accidents_raw.csv")
CLEAN_PATH = os.path.join("data", "india_road_accidents_clean.csv")
 
# ---- 1. Load ----
df = pd.read_csv(RAW_PATH)
print(f"Raw shape: {df.shape}")
print(df.columns.tolist())
 
# ---- 2. Standardize column names (lowercase, no spaces) ----
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
 
# ---- 3. Rename to a consistent internal schema ----
# EDIT the left-hand keys below to match whatever df.columns printed above
rename_map = {
    "state": "state",
    "state_name": "state",
    "state/ut": "state",
    "year": "year",
    "month": "month",
    "day_of_week": "day_of_week",
    "day": "day_of_week",
    "time": "time",
    "accident_time": "time",
    "severity": "severity",
    "accident_severity": "severity",
    "vehicle_type": "vehicle_type",
    "no_of_vehicles": "num_vehicles",
    "number_of_vehicles": "num_vehicles",
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
 
# ---- 4. Drop rows missing critical fields ----
required = ["state", "severity"]
required = [c for c in required if c in df.columns]
before = len(df)
df = df.dropna(subset=required)
print(f"Dropped {before - len(df)} rows missing {required}")
 
# ---- 5. Standardize state names (common inconsistencies) ----
state_fixes = {
    "orissa": "Odisha",
    "pondicherry": "Puducherry",
    "nct of delhi": "Delhi",
    "delhi (ut)": "Delhi",
    "jammu & kashmir": "Jammu and Kashmir",
    "andaman & nicobar islands": "Andaman and Nicobar Islands",
}
if "state" in df.columns:
    df["state"] = df["state"].astype(str).str.strip().str.title()
    df["state"] = df["state"].str.lower().replace(state_fixes).str.title()
 
# ---- 6. Handle missing values in non-critical columns ----
if "num_vehicles" in df.columns:
    df["num_vehicles"] = pd.to_numeric(df["num_vehicles"], errors="coerce")
    df["num_vehicles"] = df["num_vehicles"].fillna(df["num_vehicles"].median())
 
if "vehicle_type" in df.columns:
    df["vehicle_type"] = df["vehicle_type"].fillna("Unknown")
 
# ---- 7. Parse / derive time features ----
if "time" in df.columns:
    parsed_time = pd.to_datetime(df["time"], errors="coerce", format=None)
    df["hour"] = parsed_time.dt.hour
 
if "day_of_week" in df.columns:
    df["is_weekend"] = df["day_of_week"].astype(str).str.lower().isin(
        ["saturday", "sunday"]
    )
 
if "month" in df.columns:
    month_map = {m: i + 1 for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november", "december"]
    )}
    if df["month"].dtype == object:
        df["month_num"] = df["month"].astype(str).str.lower().map(month_map)
    else:
        df["month_num"] = df["month"]
 
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Summer", 4: "Summer", 5: "Summer",
                  6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
                  10: "Post-Monsoon", 11: "Post-Monsoon"}
    df["season"] = df["month_num"].map(season_map)
 
# ---- 8. Encode severity as an ordinal numeric column too (for t-tests/ANOVA) ----
if "severity" in df.columns:
    sev_order = {"minor": 1, "serious": 2, "fatal": 3}
    df["severity_clean"] = df["severity"].astype(str).str.lower().str.strip()
    df["severity_score"] = df["severity_clean"].map(sev_order)
    # If severity was already numeric in the raw file, keep that too
    if df["severity_score"].isna().all():
        df["severity_score"] = pd.to_numeric(df["severity"], errors="coerce")
 
# ---- 9. Remove obvious outliers in num_vehicles using Z-score ----
if "num_vehicles" in df.columns:
    z = (df["num_vehicles"] - df["num_vehicles"].mean()) / df["num_vehicles"].std()
    df = df[z.abs() < 3]
 
# ---- 10. Save cleaned data ----
os.makedirs("data", exist_ok=True)
df.to_csv(CLEAN_PATH, index=False)
print(f"Clean shape: {df.shape}")
print(f"Saved to {CLEAN_PATH}")