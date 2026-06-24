from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AQICN_FILE = PROJECT_ROOT / "data" / "processed" / "aqicn_all_stations.csv"

df = pd.read_csv(AQICN_FILE)

print("=" * 80)
print("AQICN COMBINED DATA CHECK")
print("=" * 80)

print("\nShape:")
print(df.shape)

print("\nColumns:")
for col in df.columns:
    print(" -", col)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\n Missing Values:")
print(df.isna().sum())

print("\nStations Names:")
if "station_name" in df.columns:
    print(df["station_name"].value_counts())
else:
    print("station_name column not found.")

print("\nPossible Datetime columns:")
for col in df.columns:
    if "time" in col.lower() or "date" in col.lower():
        print("-", col)
print("\nPossible pollution columns:")
for col in df.columns:
    name = col.lower()
    if "pm" in name or "aqi" in name or "pollut" in name:
        print(" -", col)