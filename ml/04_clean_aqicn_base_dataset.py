from pathlib import Path
import pandas as pd

#1. Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "aqicn_all_stations.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "aqicn_clean_base.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

#2. Load Data

df = pd.read_csv(INPUT_FILE)
print("Original Columns:")
for col in df.columns:
    print("-", col)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(".", "_")
    .str.replace("-", "_")
)
print("\nCleaned Columns: ")
for col in df.columns:
    print("-", col)

#3. Function to find columns

def find_column(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

#4. Find Important Columns

datetime_col = find_column([
    "recorded_at_local_time",
    "recorded_at",
    "api_time",
    "datetime",
    "date_time",
    "time",
    "date"
])

station_col = find_column([
    "station_name",
    "station",
    "city",
    "location"
])

lat_col = find_column([
    "station_latitude",
    "latitude",
    "lat"
])

lon_col = find_column([
    "station_longitude",
    "longitude",
    "lon",
    "lng"
])

pm25_col = find_column([
    "pm25",
    "pm2_5",
    "pm2_5_value",
    "iaqi_pm25"
])

pm10_col = find_column([
    "pm10",
    "pm10_value",
    "iaqi_pm10"
])

aqi_col = find_column([
    "aqi",
    "aqi_value"
])


print("\nDetected columns:")
print("datetime_col:", datetime_col)
print("station_col :", station_col)
print("lat_col     :", lat_col)
print("lon_col     :", lon_col)
print("pm25_col    :", pm25_col)
print("pm10_col    :", pm10_col)
print("aqi_col     :", aqi_col)

#5. Safety Checks

if datetime_col is None:
    raise ValueError("No datetime column found. Check column names.")
if station_col is None:
    raise ValueError("No station column found. Check column names.")
if pm25_col is None:
    raise ValueError("No pollution column found. Check column names.")

#6. Create Clean data-sheet

clean_df = pd.DataFrame()

clean_df["datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")
clean_df["station_name"] = df[station_col]

if lat_col is not None:
    clean_df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")

if lon_col is not None:
    clean_df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")

if pm25_col is not None:
    clean_df["pm25"] = pd.to_numeric(df[pm25_col], errors="coerce")

if pm10_col is not None:
    clean_df["pm10"] = pd.to_numeric(df[pm10_col], errors="coerce")

if aqi_col is not None:
    clean_df["aqi"] = pd.to_numeric(df[aqi_col], errors="coerce")

#7. Basic Cleaning

clean_df = clean_df.dropna(subset = ["datetime"])
clean_df = clean_df.drop_duplicates()
clean_df.sort_values(["station_name", "datetime"])

print("\nClean dataset shape:")
print(clean_df.shape)

print("\nClean dataset columns:")
for col in clean_df.columns:
    print(" -", col)

print("\nFirst 10 rows:")
print(clean_df.head(10))

print("\nMissing values:")
print(clean_df.isna().sum())

#8. Save Clean Dataset

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("\nTrying to save file to:")
print(OUTPUT_FILE)
print("Output folder exists?", OUTPUT_FILE.parent.exists())

try:
    clean_df.to_csv(OUTPUT_FILE, index=False)
    print("\nSaved clean AQICN base dataset to:")
    print(OUTPUT_FILE)

except PermissionError:
    print("\nPermissionError: Python could not save the file.")
    print("Most likely reason: aqicn_clean_base.csv is open in Excel.")
    print("Close Excel and run again.")

except Exception as e:
    print("\nSome other error happened while saving:")
    print(type(e).__name__)
    print(e)