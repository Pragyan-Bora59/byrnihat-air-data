from pathlib import Path
import pandas as pd
import numpy as np

# 1. Paths and configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

AQICN_FILE = PROCESSED_DIR / "aqicn_clean_base.csv"
NASA_TRACKER_FILE = RAW_DIR / "nasa_power_hourly_tracker.csv"

OUTPUT_FILE = PROCESSED_DIR / "ml1_environment_pollution_dataset.csv"
MISSING_REPORT_FILE = REPORTS_DIR / "ml1_dataset_missing_report.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_TOLERANCE_HOURS = 6
FALLBACK_MATCH_THRESHOLD_PERCENT = 50

# 2. Helper functions

def clean_column_names(df):
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(".", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    return df


def standardize_pollutant_names(df):
    """
    Standardizes common pollutant column variations.
    Example:
    pm2_5 -> pm25
    pm2_5_value -> pm25 if present
    """

    df = df.copy()

    rename_map = {
        "pm2_5": "pm25",
        "pm_2_5": "pm25",
        "pm2_5_value": "pm25",
        "pm10_value": "pm10",
        "aqi_value": "aqi",
        "sulfur_dioxide": "so2",
        "sulphur_dioxide": "so2",
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
        "ozone": "o3",
        "ammonia": "nh3"
    }

    existing_rename_map = {}

    for old_col, new_col in rename_map.items():
        if old_col in df.columns and new_col not in df.columns:
            existing_rename_map[old_col] = new_col

    df = df.rename(columns=existing_rename_map)

    return df


def read_power_file(file_path):
    """
    NASA POWER CSV files often contain metadata before the real table.
    This function finds the real table header starting with YEAR.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_row = None

    for i, line in enumerate(lines):
        if line.strip().startswith("YEAR"):
            header_row = i
            break

    if header_row is None:
        return pd.read_csv(file_path)

    return pd.read_csv(file_path, skiprows=header_row)


def infer_station_from_filename(file_name):
    name = file_name.lower()

    if "byrnihat" in name:
        return "Byrnihat"
    if "guwahati" in name:
        return "Guwahati"
    if "shillong" in name:
        return "Shillong"

    return "regional_average"


def add_wind_vectors(df):
    """
    NASA wind direction is usually the direction FROM which wind comes.
    For pollution movement, we want the direction TO which air/pollutants move.

    wind_u:
        positive = eastward movement
        negative = westward movement

    wind_v:
        positive = northward movement
        negative = southward movement
    """

    df = df.copy()

    if "wind_speed" not in df.columns:
        df["wind_speed"] = np.nan

    if "wind_direction" not in df.columns:
        df["wind_direction"] = np.nan

    theta = np.deg2rad(df["wind_direction"])

    df["wind_u"] = -df["wind_speed"] * np.sin(theta)
    df["wind_v"] = -df["wind_speed"] * np.cos(theta)

    return df


def reconstruct_wind_from_vectors(df):
    """
    After averaging wind_u and wind_v, reconstruct:
    - wind_speed
    - wind_to_direction
    - wind_direction, meaning wind FROM direction
    """

    df = df.copy()

    if "wind_u" in df.columns and "wind_v" in df.columns:
        df["wind_speed"] = np.sqrt(df["wind_u"] ** 2 + df["wind_v"] ** 2)

        # Direction TO which pollutant moves, degrees clockwise from North
        df["wind_to_direction"] = (
            np.degrees(np.arctan2(df["wind_u"], df["wind_v"])) + 360
        ) % 360

        # Direction FROM which wind comes
        df["wind_direction"] = (df["wind_to_direction"] + 180) % 360

    return df



# 3. Load AQICN clean base dataset

print("=" * 80)
print("LOADING AQICN CLEAN BASE DATA")
print("=" * 80)

if not AQICN_FILE.exists():
    raise FileNotFoundError(f"AQICN clean file not found: {AQICN_FILE}")

aq = pd.read_csv(AQICN_FILE)
aq = clean_column_names(aq)
aq = standardize_pollutant_names(aq)

if "datetime" not in aq.columns:
    raise ValueError("AQICN dataset must contain a datetime column.")

if "station_name" not in aq.columns:
    raise ValueError("AQICN dataset must contain a station_name column.")

aq["datetime"] = pd.to_datetime(aq["datetime"], errors="coerce")
aq = aq.dropna(subset=["datetime"])
aq = aq.sort_values("datetime")

# Convert pollutant columns to numeric if present
pollutant_columns = [
    "pm25",
    "pm10",
    "aqi",
    "so2",
    "co",
    "no2",
    "o3",
    "nh3",
    "no",
    "nox"
]

for col in pollutant_columns:
    if col in aq.columns:
        aq[col] = pd.to_numeric(aq[col], errors="coerce")

# Convert location columns to numeric if present
for col in ["latitude", "longitude"]:
    if col in aq.columns:
        aq[col] = pd.to_numeric(aq[col], errors="coerce")

print("AQICN shape:", aq.shape)
print("AQICN columns:")
print(aq.columns.tolist())

available_pollutants = [col for col in pollutant_columns if col in aq.columns]

print("\nAvailable pollutant columns preserved from AQICN:")
print(available_pollutants)

missing_pollutants = [col for col in pollutant_columns if col not in aq.columns]

print("\nPollutant columns not found in AQICN clean base:")
print(missing_pollutants)

# 4. Load old static NASA POWER files

def load_old_nasa_power_files():
    power_files = list(DATA_DIR.glob("POWER*.csv")) + list(RAW_DIR.glob("POWER*.csv"))

    print("\n" + "=" * 80)
    print("LOADING OLD STATIC NASA POWER FILES")
    print("=" * 80)

    if not power_files:
        print("No old NASA POWER files found.")
        return pd.DataFrame()

    weather_dfs = []

    print("NASA POWER files found:")
    for file in power_files:
        print("-", file.name)

    for file in power_files:
        w = read_power_file(file)
        w = clean_column_names(w)

        required_time_cols = ["year", "mo", "dy", "hr"]

        missing_time_cols = [col for col in required_time_cols if col not in w.columns]

        if missing_time_cols:
            print(f"Skipping {file.name}. Missing time columns: {missing_time_cols}")
            continue

        w["datetime"] = pd.to_datetime(
            {
                "year": w["year"],
                "month": w["mo"],
                "day": w["dy"],
                "hour": w["hr"]
            },
            errors="coerce"
        )

        weather = pd.DataFrame()
        weather["datetime"] = w["datetime"]
        weather["weather_station_name"] = infer_station_from_filename(file.name)
        weather["weather_source"] = file.name

        if "t2m" in w.columns:
            weather["temperature"] = pd.to_numeric(w["t2m"], errors="coerce")

        if "rh2m" in w.columns:
            weather["humidity"] = pd.to_numeric(w["rh2m"], errors="coerce")

        if "prectotcorr" in w.columns:
            weather["rainfall"] = pd.to_numeric(w["prectotcorr"], errors="coerce")
        elif "prectot" in w.columns:
            weather["rainfall"] = pd.to_numeric(w["prectot"], errors="coerce")

        if "ws10m" in w.columns:
            weather["wind_speed"] = pd.to_numeric(w["ws10m"], errors="coerce")
        elif "ws2m" in w.columns:
            weather["wind_speed"] = pd.to_numeric(w["ws2m"], errors="coerce")

        if "wd10m" in w.columns:
            weather["wind_direction"] = pd.to_numeric(w["wd10m"], errors="coerce")
        elif "wd2m" in w.columns:
            weather["wind_direction"] = pd.to_numeric(w["wd2m"], errors="coerce")

        weather = weather.dropna(subset=["datetime"])

        weather_dfs.append(weather)

    if not weather_dfs:
        return pd.DataFrame()

    old_weather = pd.concat(weather_dfs, ignore_index=True)

    return old_weather

# 5. Load NASA tracker data

def load_nasa_tracker_file():
    print("\n" + "=" * 80)
    print("LOADING NASA POWER TRACKER FILE")
    print("=" * 80)

    if not NASA_TRACKER_FILE.exists():
        print("NASA tracker file not found.")
        return pd.DataFrame()

    tracker = pd.read_csv(NASA_TRACKER_FILE)
    tracker = clean_column_names(tracker)

    print("NASA tracker found:", NASA_TRACKER_FILE)
    print("Tracker shape:", tracker.shape)
    print("Tracker columns:")
    print(tracker.columns.tolist())

    weather = pd.DataFrame()

    if "datetime_ist" in tracker.columns:
        weather["datetime"] = pd.to_datetime(tracker["datetime_ist"], errors="coerce")
    elif "datetime_utc" in tracker.columns:
        weather["datetime"] = pd.to_datetime(tracker["datetime_utc"], errors="coerce")
    else:
        raise ValueError("NASA tracker must contain datetime_ist or datetime_utc.")

    if "station_name" in tracker.columns:
        weather["weather_station_name"] = tracker["station_name"].astype(str)
    else:
        weather["weather_station_name"] = "regional_average"

    weather["weather_source"] = "nasa_power_hourly_tracker.csv"

    for col in ["temperature", "humidity", "rainfall", "wind_speed", "wind_direction"]:
        if col in tracker.columns:
            weather[col] = pd.to_numeric(tracker[col], errors="coerce")
        else:
            weather[col] = np.nan

    weather = weather.dropna(subset=["datetime"])

    return weather


# 6. Combine and clean weather data

old_weather = load_old_nasa_power_files()
tracker_weather = load_nasa_tracker_file()

weather_sources = []

if not old_weather.empty:
    weather_sources.append(old_weather)

if not tracker_weather.empty:
    weather_sources.append(tracker_weather)

if not weather_sources:
    raise FileNotFoundError(
        "No NASA weather data found. Need old POWER*.csv files or nasa_power_hourly_tracker.csv."
    )

weather = pd.concat(weather_sources, ignore_index=True)

weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
weather = weather.dropna(subset=["datetime"])

for col in ["temperature", "humidity", "rainfall", "wind_speed", "wind_direction"]:
    if col not in weather.columns:
        weather[col] = np.nan

    weather[col] = pd.to_numeric(weather[col], errors="coerce")

weather = add_wind_vectors(weather)

# Average duplicate weather rows using vector-safe wind representation
weather_numeric_cols = [
    "temperature",
    "humidity",
    "rainfall",
    "wind_u",
    "wind_v"
]

weather_grouped = (
    weather
    .groupby(["weather_station_name", "datetime"], as_index=False)[weather_numeric_cols]
    .mean()
)

weather_grouped = reconstruct_wind_from_vectors(weather_grouped)

weather_grouped["weather_datetime"] = weather_grouped["datetime"]

weather_grouped = weather_grouped.sort_values(["weather_station_name", "datetime"])

print("\n" + "=" * 80)
print("COMBINED CLEAN WEATHER DATA")
print("=" * 80)
print("Weather shape:", weather_grouped.shape)
print("Weather date range:")
print("Start:", weather_grouped["datetime"].min())
print("End  :", weather_grouped["datetime"].max())
print("\nWeather stations:")
print(weather_grouped["weather_station_name"].unique().tolist())

print("\nWeather missing values:")
print(weather_grouped.isna().sum())

# 7. Nearest datetime merge by station

def nearest_weather_merge_by_station(aq_df, weather_df, tolerance_hours):
    merged_parts = []

    weather_station_names_lower = (
        weather_df["weather_station_name"]
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )

    for station_name, aq_station_df in aq_df.groupby("station_name", dropna=False):
        station_string = str(station_name)
        station_lower = station_string.lower()

        aq_part = aq_station_df.sort_values("datetime").copy()

        if station_lower in weather_station_names_lower:
            weather_part = weather_df[
                weather_df["weather_station_name"].astype(str).str.lower() == station_lower
            ].copy()
        else:
            weather_part = weather_df.copy()
            weather_part["weather_station_name"] = "regional_average"

        weather_part = weather_part.sort_values("datetime")

        merged = pd.merge_asof(
            aq_part,
            weather_part,
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta(hours=tolerance_hours)
        )

        merged_parts.append(merged)

    return pd.concat(merged_parts, ignore_index=True)


print("\n" + "=" * 80)
print("MERGING AQICN + NASA WEATHER")
print("=" * 80)

ml1 = nearest_weather_merge_by_station(
    aq_df=aq,
    weather_df=weather_grouped,
    tolerance_hours=WEATHER_TOLERANCE_HOURS
)

ml1["weather_merge_mode"] = "nearest_datetime"
ml1["is_prototype_weather_match"] = False

if "weather_datetime" in ml1.columns:
    ml1["weather_gap_hours"] = (
        (ml1["datetime"] - ml1["weather_datetime"])
        .abs()
        .dt.total_seconds() / 3600
    )
else:
    ml1["weather_gap_hours"] = np.nan

matched_rows = ml1["wind_speed"].notna().sum()
total_rows = len(ml1)

if total_rows > 0:
    match_percent = (matched_rows / total_rows) * 100
else:
    match_percent = 0

print(f"Nearest datetime weather matched rows: {matched_rows}/{total_rows} ({match_percent:.2f}%)")

print("\nWeather gap summary:")
print(ml1["weather_gap_hours"].describe())


# 8. Final-ready fallback for old NASA data

if match_percent < FALLBACK_MATCH_THRESHOLD_PERCENT:
    print("\nLow nearest-datetime match detected.")
    print("Using final-compatible fallback: station-aware month-day-hour weather matching.")
    print("This keeps the code final-ready, but the fallback data should be replaced for final scientific validation.")

    aq_fallback = aq.copy()
    weather_fallback = weather_grouped.copy()

    aq_fallback["month"] = aq_fallback["datetime"].dt.month
    aq_fallback["day"] = aq_fallback["datetime"].dt.day
    aq_fallback["hour"] = aq_fallback["datetime"].dt.hour

    weather_fallback["month"] = weather_fallback["datetime"].dt.month
    weather_fallback["day"] = weather_fallback["datetime"].dt.day
    weather_fallback["hour"] = weather_fallback["datetime"].dt.hour

    fallback_parts = []

    weather_station_names_lower = (
        weather_fallback["weather_station_name"]
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )

    for station_name, aq_station_df in aq_fallback.groupby("station_name", dropna=False):
        station_string = str(station_name)
        station_lower = station_string.lower()

        if station_lower in weather_station_names_lower:
            weather_part = weather_fallback[
                weather_fallback["weather_station_name"].astype(str).str.lower() == station_lower
            ].copy()
            used_weather_station = station_string
        else:
            weather_part = weather_fallback.copy()
            used_weather_station = "regional_average"

        fallback_numeric_cols = [
            "temperature",
            "humidity",
            "rainfall",
            "wind_u",
            "wind_v"
        ]

        weather_key = (
            weather_part
            .groupby(["month", "day", "hour"], as_index=False)[fallback_numeric_cols]
            .mean()
        )

        weather_key = reconstruct_wind_from_vectors(weather_key)

        merged_part = pd.merge(
            aq_station_df,
            weather_key,
            on=["month", "day", "hour"],
            how="left"
        )

        merged_part["weather_station_name"] = used_weather_station
        merged_part["weather_datetime"] = pd.NaT
        merged_part["weather_gap_hours"] = np.nan
        merged_part["weather_merge_mode"] = "month_day_hour_fallback"
        merged_part["is_prototype_weather_match"] = True

        fallback_parts.append(merged_part)

    ml1 = pd.concat(fallback_parts, ignore_index=True)

    matched_rows = ml1["wind_speed"].notna().sum()
    total_rows = len(ml1)

    if total_rows > 0:
        match_percent = (matched_rows / total_rows) * 100
    else:
        match_percent = 0

    print(f"Fallback weather matched rows: {matched_rows}/{total_rows} ({match_percent:.2f}%)")

# 9. Final time features

ml1["datetime"] = pd.to_datetime(ml1["datetime"], errors="coerce")

ml1["hour"] = ml1["datetime"].dt.hour
ml1["day"] = ml1["datetime"].dt.day
ml1["month"] = ml1["datetime"].dt.month
ml1["day_of_week"] = ml1["datetime"].dt.dayofweek

# Cyclic time features
ml1["hour_sin"] = np.sin(2 * np.pi * ml1["hour"] / 24)
ml1["hour_cos"] = np.cos(2 * np.pi * ml1["hour"] / 24)

ml1["month_sin"] = np.sin(2 * np.pi * ml1["month"] / 12)
ml1["month_cos"] = np.cos(2 * np.pi * ml1["month"] / 12)

# 10. Final numeric cleanup

final_numeric_cols = [
    "latitude",
    "longitude",
    "pm25",
    "pm10",
    "aqi",
    "so2",
    "co",
    "no2",
    "o3",
    "nh3",
    "no",
    "nox",
    "temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "wind_to_direction",
    "wind_u",
    "wind_v",
    "weather_gap_hours",
    "hour",
    "day",
    "month",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos"
]

for col in final_numeric_cols:
    if col in ml1.columns:
        ml1[col] = pd.to_numeric(ml1[col], errors="coerce")



# 11. Final ordering and save

preferred_columns = [
    "datetime",
    "station_name",
    "latitude",
    "longitude",
    "pm25",
    "pm10",
    "aqi",
    "so2",
    "co",
    "no2",
    "o3",
    "nh3",
    "no",
    "nox",
    "temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "wind_to_direction",
    "wind_u",
    "wind_v",
    "hour",
    "day",
    "month",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "weather_station_name",
    "weather_datetime",
    "weather_gap_hours",
    "weather_merge_mode",
    "is_prototype_weather_match"
]

existing_preferred_columns = [col for col in preferred_columns if col in ml1.columns]
remaining_columns = [col for col in ml1.columns if col not in existing_preferred_columns]

ml1 = ml1[existing_preferred_columns + remaining_columns]

ml1 = ml1.drop_duplicates()
ml1 = ml1.sort_values(["station_name", "datetime"])

ml1.to_csv(OUTPUT_FILE, index=False)

# 12. Reports

missing_report = pd.DataFrame(
    {
        "column": ml1.columns,
        "missing_count": ml1.isna().sum().values,
        "missing_percent": (ml1.isna().mean().values * 100).round(2)
    }
)

missing_report.to_csv(MISSING_REPORT_FILE, index=False)


print("\n" + "=" * 80)
print("FINAL ML1 DATASET CREATED")
print("=" * 80)

print("Saved ML1 dataset to:")
print(OUTPUT_FILE)

print("\nFinal ML1 shape:")
print(ml1.shape)

print("\nFinal columns:")
print(ml1.columns.tolist())

print("\nAvailable pollutant columns in final ML1:")
print([col for col in pollutant_columns if col in ml1.columns])

print("\nWeather merge mode counts:")
print(ml1["weather_merge_mode"].value_counts(dropna=False))

print("\nPrototype weather match counts:")
print(ml1["is_prototype_weather_match"].value_counts(dropna=False))

print("\nMissing report saved to:")
print(MISSING_REPORT_FILE)

print("\nMissing values summary:")
print(missing_report.sort_values("missing_percent", ascending=False).head(20))

print("\nFirst 10 rows:")
print(ml1.head(10))

print("\nML1 dataset preparation completed.")