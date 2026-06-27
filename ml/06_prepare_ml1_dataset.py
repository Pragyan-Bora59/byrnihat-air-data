from pathlib import Path
import pandas as pd
import numpy as np

#1. Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

AQICN_FILE = PROCESSED_DIR / "aqicn_clean_base.csv"
OUTPUT_FILE = PROCESSED_DIR / "ml1_environment_pollution_dataset.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

#2. Load clean AQICN data

aq = pd.read_csv(AQICN_FILE)

aq["datetime"] = pd.to_datetime(aq["datetime"], errors = "coerce")
aq = aq.dropna(subset = ["datetime"])
aq = aq.sort_values("datetime")

print("AQICN shape:", aq.shape)
print("AQICN columns:")
print(aq.columns.tolist())

#3. Read NASA Power Files

def read_power_file(file_path):

    with open(file_path, "r", encoding = "utf-8") as f:
        lines = f.readlines()
    
    header_row = None

    for i, line in enumerate(lines):
        if line.startswith("YEAR"):
            header_row = i
            break
    
    if header_row is None:
        return pd.read_csv(file_path)
    
    return pd.read_csv(file_path, skiprows = header_row)

weather_files = list(DATA_DIR.glob("POWER*.csv"))

print("\nNASA file found:") 

for file in weather_files:
    print("-", file.name)

if not weather_files:
    raise FileNotFoundError("No NASA  POWER CSV files found in data folder.")

weather_dfs = []

for file in weather_files:
    w = read_power_file(file)

    w.columns = (
        w.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )

    print("\nWeather file:", file.name)
    print("Columns:", w.columns.tolist())

    required_time_cols = ["YEAR", "MO", "DY", "HR"]

    for col in required_time_cols:
        if col not in w.columns:
            raise ValueError(f"{col} not found in {file.name}")

    w["datetime"] = pd.to_datetime(
        {
            "year": w["YEAR"],
            "month": w["MO"],
            "day": w["DY"],
            "hour": w["HR"]
        },
        errors="coerce"
    )

    weather_dfs.append(w)

weather = pd.concat(weather_dfs, ignore_index=True)

weather = weather.dropna(subset=["datetime"])
weather = weather.sort_values("datetime")

# 4. Select useful weather columns

def find_col(possible_names, columns):
    for name in possible_names:
        if name in columns:
            return name
    return None

temp_col = find_col(["T2M"], weather.columns)
humidity_col = find_col(["RH2M"], weather.columns)
rain_col = find_col(["PRECTOTCORR", "PRECTOT", "PRECTOTCORR_SUM"], weather.columns)
wind_speed_col = find_col(["WS10M", "WS2M"], weather.columns)
wind_dir_col = find_col(["WD10M", "WD2M"], weather.columns)

print("\nDetected weather columns:")
print("temperature:", temp_col)
print("humidity   :", humidity_col)
print("rainfall   :", rain_col)
print("wind_speed :", wind_speed_col)
print("wind_dir   :", wind_dir_col)

clean_weather = pd.DataFrame()
clean_weather["datetime"] = weather["datetime"]

if temp_col:
    clean_weather["Temperature"] = pd.to_numeric(weather[temp_col], errors = "coerce")
if humidity_col:
    clean_weather["humidity"] = pd.to_numeric(weather[humidity_col], errors="coerce")

if rain_col:
    clean_weather["rainfall"] = pd.to_numeric(weather[rain_col], errors="coerce")

if wind_speed_col:
    clean_weather["wind_speed"] = pd.to_numeric(weather[wind_speed_col], errors="coerce")

if wind_dir_col:
    clean_weather["wind_direction"] = pd.to_numeric(weather[wind_dir_col], errors="coerce")

# If multiple NASA files have same datetime, average them for now
clean_weather = clean_weather.groupby("datetime", as_index=False).mean()

print("\nClean weather shape:", clean_weather.shape)
print(clean_weather.head())

#5. Merge AQICN + weather

aq = aq.sort_values("datetime")
clean_weather = clean_weather.sort_values("datetime")

ml1 = pd.merge_asof(
    aq,
    clean_weather,
    on = "datetime",
    direction = "nearest",
    tolerance = pd.Timedelta("6h")
)

print("\nMerged ML1 shape before fallback:", ml1.shape)

matched_rows = ml1["wind_speed"].notna().sum()
total_rows = len(ml1)

if total_rows > 0:
    match_percent = (matched_rows / total_rows) * 100
else:
    match_percent = 0

print(f"Weather maatched rows: {matched_rows} / {total_rows} ({match_percent :.2f}%)")

#Fall back id AQICN Dates and NASA dates do not match

if match_percent < 50:
    print("\nLow datetime match detected.")
    print("Using prototype fallback: matching by month, day, and hour.")

    aq_fallback = aq.copy()
    weather_fallback = clean_weather.copy()

    aq_fallback["month"] = aq_fallback["datetime"].dt.month
    aq_fallback["day"] = aq_fallback["datetime"].dt.day
    aq_fallback["hour"] = aq_fallback["datetime"].dt.hour

    weather_fallback["month"] = weather_fallback["datetime"].dt.month
    weather_fallback["day"] = weather_fallback["datetime"].dt.day
    weather_fallback["hour"] = weather_fallback["datetime"].dt.hour

    weather_key = (
    weather_fallback
    .groupby(["month", "day", "hour"], as_index=False)
    .mean(numeric_only=True)
)

    ml1 = pd.merge(
        aq_fallback,
        weather_key,
        on = ["month", "day", "hour"],
        how = "left"
    )

    ml1["weather_merge_mode"] = "month_day_hour_prototype"

else:
    ml1["weather_merge_mode"] = "nearest_datetime"


print("\nMerged ML1 shape after fallback/check:", ml1.shape)

#6. Wind Vector Features

if "wind_speed" in ml1.columns and "wind_direction" in ml1.columns:
    theta = np.deg2rad(ml1["wind_direction"])

    ml1["wind_u"] = -ml1["wind_speed"] * np.sin(theta)
    ml1["wind_v"] = ml1["wind_speed"] * np.cos(theta)

else:
    ml1["wind_u"] = np.nan
    ml1["wind_v"] = np.nan

#7. Create Time Features

ml1["hour"] = ml1["datetime"].dt.hour
ml1["day"] = ml1["datetime"].dt.day
ml1["month"] = ml1["datetime"].dt.month
ml1["day_of_week"] = ml1["datetime"].dt.dayofweek

ml1["hour_sin"] = np.sin(2 * np.pi * ml1["hour"] / 24)
ml1["hour_cos"] = np.cos(2 * np.pi * ml1["hour"] / 24)

#8. Final cleaning

ml1 = ml1.drop_duplicates()
ml1 = ml1.sort_values(["station_name", "datetime"])

print("\nFinal ML1 shape:", ml1.shape)

print("\nFinal ML1 columns:")
for col in ml1.columns:
    print(" -", col)

print("\nMissing values:")
print(ml1.isna().sum())

print("\nFirst 10 rows:")
print(ml1.head(10))


#9. Save ML1 dataset

ml1.to_csv(OUTPUT_FILE, index=False)

print("\nSaved ML1 dataset to:")
print(OUTPUT_FILE)