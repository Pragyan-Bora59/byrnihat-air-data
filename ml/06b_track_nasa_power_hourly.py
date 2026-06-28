from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. Paths and configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

NASA_TRACKER_FILE = RAW_DIR / "nasa_power_hourly_tracker.csv"

OUTPUT_FILE = PROCESSED_DIR / "ml1_environment_pollution_dataset.csv"
MISSING_REPORT_FILE = REPORTS_DIR / "ml1_dataset_missing_report.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_TOLERANCE_HOURS = 6
FALLBACK_MATCH_THRESHOLD_PERCENT = 50

POLLUTANT_COLUMNS = [
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
]

AQICN_FILE_PAIRS = [
    {
        "processed": PROCESSED_DIR / "byrnihat_aqicn_observations.csv",
        "legacy": DATA_DIR / "byrnihat_aqicn_data.csv",
    },
    {
        "processed": PROCESSED_DIR / "guwahati_aqicn_observations.csv",
        "legacy": DATA_DIR / "guwahati_aqicn_data.csv",
    },
    {
        "processed": PROCESSED_DIR / "shillong_aqicn_observations.csv",
        "legacy": DATA_DIR / "shillong_aqicn_data.csv",
    },
]


# =============================================================================
# 2. Helper functions
# =============================================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(".", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    return df


def drop_noise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    columns_to_drop = []

    for column in df.columns:
        column_string = str(column).strip().lower()

        if column_string == "":
            columns_to_drop.append(column)
        elif column_string == "0":
            columns_to_drop.append(column)
        elif column_string.startswith("unnamed"):
            columns_to_drop.append(column)

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    return df


def standardize_pollutant_names(df: pd.DataFrame) -> pd.DataFrame:
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
        "ammonia": "nh3",
    }

    existing_rename_map = {
        old_column: new_column
        for old_column, new_column in rename_map.items()
        if old_column in df.columns and new_column not in df.columns
    }

    return df.rename(columns=existing_rename_map)


def standardize_weather_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "t2m": "temperature",
        "temperature_c": "temperature",
        "temperature_2m": "temperature",
        "temp": "temperature",
        "rh2m": "humidity",
        "relative_humidity": "humidity",
        "prectotcorr": "rainfall",
        "prectot": "rainfall",
        "precipitation": "rainfall",
        "rain": "rainfall",
        "rain_mm": "rainfall",
        "ws10m": "wind_speed",
        "ws2m": "wind_speed",
        "wind_speed_10m": "wind_speed",
        "wd10m": "wind_direction",
        "wd2m": "wind_direction",
        "wind_direction_10m": "wind_direction",
    }

    existing_rename_map = {
        old_column: new_column
        for old_column, new_column in rename_map.items()
        if old_column in df.columns and new_column not in df.columns
    }

    return df.rename(columns=existing_rename_map)


def parse_datetime_naive(values: pd.Series) -> pd.Series:
    """
    Converts datetime-like values into timezone-naive pandas datetime values.

    This avoids merge errors between timezone-aware and timezone-naive columns.
    """

    parsed = pd.to_datetime(values, errors="coerce")

    try:
        if parsed.dt.tz is not None:
            return parsed.dt.tz_localize(None)

        return parsed

    except AttributeError:
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        return parsed.dt.tz_localize(None)


def read_power_file(file_path: Path) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    header_row = None

    for index, line in enumerate(lines):
        if line.strip().startswith("YEAR"):
            header_row = index
            break

    if header_row is None:
        return pd.read_csv(file_path)

    return pd.read_csv(file_path, skiprows=header_row)


def infer_station_from_filename(file_name: str) -> str:
    name = file_name.lower()

    if "byrnihat" in name:
        return "Byrnihat"

    if "guwahati" in name:
        return "Guwahati"

    if "shillong" in name:
        return "Shillong"

    return "regional_average"


def add_wind_vectors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "wind_speed" not in df.columns:
        df["wind_speed"] = np.nan

    if "wind_direction" not in df.columns:
        df["wind_direction"] = np.nan

    theta = np.deg2rad(df["wind_direction"])

    df["wind_u"] = -df["wind_speed"] * np.sin(theta)
    df["wind_v"] = -df["wind_speed"] * np.cos(theta)

    return df


def reconstruct_wind_from_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """
    After averaging wind_u and wind_v, reconstruct:
    - wind_speed
    - wind_to_direction
    - wind_direction, meaning wind FROM direction
    """

    df = df.copy()

    if "wind_u" in df.columns and "wind_v" in df.columns:
        df["wind_speed"] = np.sqrt(df["wind_u"] ** 2 + df["wind_v"] ** 2)

        df["wind_to_direction"] = (
            np.degrees(np.arctan2(df["wind_u"], df["wind_v"])) + 360
        ) % 360

        df["wind_direction"] = (df["wind_to_direction"] + 180) % 360

    return df


def make_regional_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    weather_numeric_columns = [
        "temperature",
        "humidity",
        "rainfall",
        "wind_u",
        "wind_v",
    ]

    existing_numeric_columns = [
        column for column in weather_numeric_columns if column in weather_df.columns
    ]

    regional_weather = (
        weather_df
        .groupby("datetime", as_index=False)[existing_numeric_columns]
        .mean()
    )

    regional_weather = reconstruct_wind_from_vectors(regional_weather)
    regional_weather["weather_station_name"] = "regional_average"
    regional_weather["weather_datetime"] = regional_weather["datetime"]

    return regional_weather


# =============================================================================
# 3. Load AQICN processed observation data
# =============================================================================

def load_aqicn_data() -> pd.DataFrame:
    print("=" * 80)
    print("LOADING AQICN PROCESSED OBSERVATION DATA")
    print("=" * 80)

    selected_files: list[Path] = []

    for file_pair in AQICN_FILE_PAIRS:
        processed_file = file_pair["processed"]
        legacy_file = file_pair["legacy"]

        if processed_file.exists():
            selected_files.append(processed_file)
        elif legacy_file.exists():
            selected_files.append(legacy_file)

    if not selected_files:
        raise FileNotFoundError(
            "No AQICN files found. Expected processed files in data/processed/ "
            "or legacy station CSV files in data/."
        )

    print("AQICN files selected:")
    for file_path in selected_files:
        print(f"- {file_path}")

    aqicn_parts = []

    for file_path in selected_files:
        df = pd.read_csv(file_path)

        df = clean_column_names(df)
        df = drop_noise_columns(df)
        df = standardize_pollutant_names(df)

        if "api_time" in df.columns and "datetime" not in df.columns:
            df = df.rename(columns={"api_time": "datetime"})

        if "recorded_at_local_time" in df.columns and "datetime" not in df.columns:
            df = df.rename(columns={"recorded_at_local_time": "datetime"})

        if "station_latitude" in df.columns and "latitude" not in df.columns:
            df = df.rename(columns={"station_latitude": "latitude"})

        if "station_longitude" in df.columns and "longitude" not in df.columns:
            df = df.rename(columns={"station_longitude": "longitude"})

        if "lat" in df.columns and "latitude" not in df.columns:
            df = df.rename(columns={"lat": "latitude"})

        if "lon" in df.columns and "longitude" not in df.columns:
            df = df.rename(columns={"lon": "longitude"})

        if "lng" in df.columns and "longitude" not in df.columns:
            df = df.rename(columns={"lng": "longitude"})

        if "target_city" in df.columns:
            if "station_name" in df.columns:
                df["aqicn_station_name"] = df["station_name"]

            df["station_name"] = df["target_city"]

        else:
            inferred_station = infer_station_from_filename(file_path.name)
            df["target_city"] = inferred_station

            if "station_name" in df.columns:
                df["aqicn_station_name"] = df["station_name"]

            df["station_name"] = inferred_station

        aqicn_weather_rename_map = {
            "temperature": "aqicn_temperature",
            "humidity": "aqicn_humidity",
            "pressure": "aqicn_pressure",
            "wind": "aqicn_wind_indicator",
        }

        existing_aqicn_weather_rename_map = {
            old_column: new_column
            for old_column, new_column in aqicn_weather_rename_map.items()
            if old_column in df.columns and new_column not in df.columns
        }

        df = df.rename(columns=existing_aqicn_weather_rename_map)

        aqicn_parts.append(df)

    aq = pd.concat(aqicn_parts, ignore_index=True)
    aq = aq.dropna(how="all")

    if "datetime" not in aq.columns:
        raise ValueError("AQICN dataset must contain datetime, api_time, or recorded_at_local_time.")

    if "station_name" not in aq.columns:
        raise ValueError("AQICN dataset must contain station_name or target_city.")

    aq["datetime"] = parse_datetime_naive(aq["datetime"])
    aq = aq.dropna(subset=["datetime"])

    aq["station_name"] = aq["station_name"].astype(str).str.strip()

    for column in ["latitude", "longitude"]:
        if column in aq.columns:
            aq[column] = pd.to_numeric(aq[column], errors="coerce")

    for column in POLLUTANT_COLUMNS:
        if column in aq.columns:
            aq[column] = pd.to_numeric(aq[column], errors="coerce")

    for column in ["aqicn_temperature", "aqicn_humidity", "aqicn_pressure", "aqicn_wind_indicator"]:
        if column in aq.columns:
            aq[column] = pd.to_numeric(aq[column], errors="coerce")

    aq = aq.sort_values(["station_name", "datetime"])

    print("AQICN shape:", aq.shape)
    print("AQICN columns:")
    print(aq.columns.tolist())

    print("\nAvailable pollutant columns preserved from AQICN:")
    print([column for column in POLLUTANT_COLUMNS if column in aq.columns])

    print("\nMissing pollutant columns from AQICN:")
    print([column for column in POLLUTANT_COLUMNS if column not in aq.columns])

    return aq


# =============================================================================
# 4. Load old static NASA POWER files
# =============================================================================

def load_old_nasa_power_files() -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("LOADING OLD STATIC NASA POWER FILES")
    print("=" * 80)

    power_files = list(DATA_DIR.glob("*POWER*.csv")) + list(RAW_DIR.glob("*POWER*.csv"))

    power_files = sorted(
        {
            file_path
            for file_path in power_files
            if file_path.name != NASA_TRACKER_FILE.name
        }
    )

    if not power_files:
        print("No old NASA POWER files found.")
        return pd.DataFrame()

    print("NASA POWER files found:")
    for file_path in power_files:
        print(f"- {file_path}")

    weather_parts = []

    for file_path in power_files:
        weather_raw = read_power_file(file_path)

        if weather_raw.empty:
            print(f"Skipping {file_path.name}: file is empty.")
            continue

        weather_raw = clean_column_names(weather_raw)
        weather_raw = drop_noise_columns(weather_raw)
        weather_raw = standardize_weather_names(weather_raw)

        required_time_columns = ["year", "mo", "dy", "hr"]

        missing_time_columns = [
            column for column in required_time_columns if column not in weather_raw.columns
        ]

        if missing_time_columns:
            print(f"Skipping {file_path.name}. Missing time columns: {missing_time_columns}")
            continue

        weather_raw["datetime"] = pd.to_datetime(
            {
                "year": weather_raw["year"],
                "month": weather_raw["mo"],
                "day": weather_raw["dy"],
                "hour": weather_raw["hr"],
            },
            errors="coerce",
        )

        weather = pd.DataFrame()
        weather["datetime"] = weather_raw["datetime"]
        weather["weather_station_name"] = infer_station_from_filename(file_path.name)
        weather["weather_source"] = f"static_power_file:{file_path.name}"

        for column in ["temperature", "humidity", "rainfall", "wind_speed", "wind_direction"]:
            if column in weather_raw.columns:
                weather[column] = pd.to_numeric(weather_raw[column], errors="coerce")
            else:
                weather[column] = np.nan

        weather = weather.dropna(subset=["datetime"])

        if not weather.empty:
            weather_parts.append(weather)

    if not weather_parts:
        print("No usable old NASA POWER rows found.")
        return pd.DataFrame()

    old_weather = pd.concat(weather_parts, ignore_index=True)

    print("Old NASA POWER weather shape:", old_weather.shape)

    return old_weather


# =============================================================================
# 5. Load NASA POWER hourly tracker file
# =============================================================================

def load_nasa_tracker_file() -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("LOADING NASA POWER TRACKER FILE")
    print("=" * 80)

    if not NASA_TRACKER_FILE.exists():
        print("NASA tracker file not found.")
        return pd.DataFrame()

    tracker = pd.read_csv(NASA_TRACKER_FILE)

    if tracker.empty:
        print("NASA tracker file exists but is empty.")
        return pd.DataFrame()

    tracker = clean_column_names(tracker)
    tracker = drop_noise_columns(tracker)
    tracker = standardize_weather_names(tracker)

    print("NASA tracker found:", NASA_TRACKER_FILE)
    print("Tracker shape:", tracker.shape)
    print("Tracker columns:")
    print(tracker.columns.tolist())

    datetime_column = None

    datetime_candidates = [
        "datetime_ist",
        "datetime_local",
        "datetime",
        "date_time",
        "timestamp",
        "time",
        "datetime_utc",
    ]

    for candidate in datetime_candidates:
        if candidate in tracker.columns:
            datetime_column = candidate
            break

    if datetime_column is None:
        raise ValueError(
            "NASA tracker must contain one datetime column, such as datetime_ist, "
            "datetime_utc, datetime, timestamp, or time."
        )

    weather = pd.DataFrame()
    weather["datetime"] = parse_datetime_naive(tracker[datetime_column])

    station_column = None

    station_candidates = [
        "station_name",
        "target_city",
        "city",
        "location",
        "weather_station_name",
    ]

    for candidate in station_candidates:
        if candidate in tracker.columns:
            station_column = candidate
            break

    if station_column is not None:
        weather["weather_station_name"] = tracker[station_column].astype(str).str.strip()
    else:
        weather["weather_station_name"] = "regional_average"

    weather["weather_source"] = "nasa_power_hourly_tracker.csv"

    for column in ["temperature", "humidity", "rainfall", "wind_speed", "wind_direction"]:
        if column in tracker.columns:
            weather[column] = pd.to_numeric(tracker[column], errors="coerce")
        else:
            weather[column] = np.nan

    weather = weather.dropna(subset=["datetime"])

    print("NASA tracker weather shape:", weather.shape)

    return weather


# =============================================================================
# 6. Combine and clean NASA weather data
# =============================================================================

def load_combined_weather_data() -> pd.DataFrame:
    old_weather = load_old_nasa_power_files()
    tracker_weather = load_nasa_tracker_file()

    weather_sources = []

    if not old_weather.empty:
        weather_sources.append(old_weather)

    if not tracker_weather.empty:
        weather_sources.append(tracker_weather)

    if not weather_sources:
        raise FileNotFoundError(
            "No NASA weather data found. Need old POWER*.csv files or "
            "data/raw/nasa_power_hourly_tracker.csv."
        )

    weather = pd.concat(weather_sources, ignore_index=True)
    weather = clean_column_names(weather)
    weather = drop_noise_columns(weather)
    weather = standardize_weather_names(weather)

    weather["datetime"] = parse_datetime_naive(weather["datetime"])
    weather = weather.dropna(subset=["datetime"])

    if "weather_station_name" not in weather.columns:
        weather["weather_station_name"] = "regional_average"

    weather["weather_station_name"] = weather["weather_station_name"].astype(str).str.strip()

    for column in ["temperature", "humidity", "rainfall", "wind_speed", "wind_direction"]:
        if column not in weather.columns:
            weather[column] = np.nan

        weather[column] = pd.to_numeric(weather[column], errors="coerce")

    weather = add_wind_vectors(weather)

    weather_numeric_columns = [
        "temperature",
        "humidity",
        "rainfall",
        "wind_u",
        "wind_v",
    ]

    weather_grouped = (
        weather
        .groupby(["weather_station_name", "datetime"], as_index=False)[weather_numeric_columns]
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

    return weather_grouped


# =============================================================================
# 7. Merge AQICN + NASA weather data
# =============================================================================

def nearest_weather_merge_by_station(
    aq_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    tolerance_hours: int,
) -> pd.DataFrame:
    merged_parts = []

    weather_station_names_lower = (
        weather_df["weather_station_name"]
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )

    regional_weather = make_regional_weather(weather_df)

    for station_name, aq_station_df in aq_df.groupby("station_name", dropna=False):
        station_string = str(station_name).strip()
        station_lower = station_string.lower()

        aq_part = aq_station_df.sort_values("datetime").copy()

        if station_lower in weather_station_names_lower:
            weather_part = weather_df[
                weather_df["weather_station_name"].astype(str).str.lower() == station_lower
            ].copy()
        else:
            weather_part = regional_weather.copy()

        weather_part = weather_part.sort_values("datetime")

        merged = pd.merge_asof(
            aq_part,
            weather_part,
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta(hours=tolerance_hours),
        )

        merged_parts.append(merged)

    if not merged_parts:
        return pd.DataFrame()

    return pd.concat(merged_parts, ignore_index=True)


def fallback_month_day_hour_merge(
    aq_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    print("\nLow nearest-datetime match detected.")
    print("Using final-compatible fallback: station-aware month-day-hour weather matching.")
    print("This fallback is useful for prototype continuity but should be replaced with matched final data.")

    aq_fallback = aq_df.copy()
    weather_fallback = weather_df.copy()
    regional_weather = make_regional_weather(weather_df)

    aq_fallback["month"] = aq_fallback["datetime"].dt.month
    aq_fallback["day"] = aq_fallback["datetime"].dt.day
    aq_fallback["hour"] = aq_fallback["datetime"].dt.hour

    weather_fallback["month"] = weather_fallback["datetime"].dt.month
    weather_fallback["day"] = weather_fallback["datetime"].dt.day
    weather_fallback["hour"] = weather_fallback["datetime"].dt.hour

    regional_weather["month"] = regional_weather["datetime"].dt.month
    regional_weather["day"] = regional_weather["datetime"].dt.day
    regional_weather["hour"] = regional_weather["datetime"].dt.hour

    weather_station_names_lower = (
        weather_fallback["weather_station_name"]
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )

    fallback_parts = []

    fallback_numeric_columns = [
        "temperature",
        "humidity",
        "rainfall",
        "wind_u",
        "wind_v",
    ]

    for station_name, aq_station_df in aq_fallback.groupby("station_name", dropna=False):
        station_string = str(station_name).strip()
        station_lower = station_string.lower()

        if station_lower in weather_station_names_lower:
            weather_part = weather_fallback[
                weather_fallback["weather_station_name"].astype(str).str.lower() == station_lower
            ].copy()
            used_weather_station = station_string
        else:
            weather_part = regional_weather.copy()
            used_weather_station = "regional_average"

        weather_key = (
            weather_part
            .groupby(["month", "day", "hour"], as_index=False)[fallback_numeric_columns]
            .mean()
        )

        weather_key = reconstruct_wind_from_vectors(weather_key)

        merged_part = pd.merge(
            aq_station_df,
            weather_key,
            on=["month", "day", "hour"],
            how="left",
        )

        merged_part["weather_station_name"] = used_weather_station
        merged_part["weather_datetime"] = pd.NaT
        merged_part["weather_gap_hours"] = np.nan
        merged_part["weather_merge_mode"] = "month_day_hour_fallback"
        merged_part["is_prototype_weather_match"] = True

        fallback_parts.append(merged_part)

    if not fallback_parts:
        return pd.DataFrame()

    return pd.concat(fallback_parts, ignore_index=True)


# =============================================================================
# 8. Final ML1 dataset creation
# =============================================================================

def create_ml1_dataset() -> pd.DataFrame:
    aq = load_aqicn_data()
    weather_grouped = load_combined_weather_data()

    print("\n" + "=" * 80)
    print("MERGING AQICN + NASA WEATHER")
    print("=" * 80)

    ml1 = nearest_weather_merge_by_station(
        aq_df=aq,
        weather_df=weather_grouped,
        tolerance_hours=WEATHER_TOLERANCE_HOURS,
    )

    if ml1.empty:
        raise RuntimeError("AQICN + NASA merge produced an empty dataset.")

    ml1["weather_merge_mode"] = "nearest_datetime"
    ml1["is_prototype_weather_match"] = False

    if "weather_datetime" in ml1.columns:
        ml1["weather_gap_hours"] = (
            (ml1["datetime"] - ml1["weather_datetime"])
            .abs()
            .dt.total_seconds()
            / 3600
        )
    else:
        ml1["weather_gap_hours"] = np.nan

    matched_rows = ml1["wind_speed"].notna().sum()
    total_rows = len(ml1)

    if total_rows > 0:
        match_percent = (matched_rows / total_rows) * 100
    else:
        match_percent = 0

    print(
        f"Nearest datetime weather matched rows: "
        f"{matched_rows}/{total_rows} ({match_percent:.2f}%)"
    )

    print("\nWeather gap summary:")
    print(ml1["weather_gap_hours"].describe())

    if match_percent < FALLBACK_MATCH_THRESHOLD_PERCENT:
        ml1 = fallback_month_day_hour_merge(
            aq_df=aq,
            weather_df=weather_grouped,
        )

        matched_rows = ml1["wind_speed"].notna().sum()
        total_rows = len(ml1)

        if total_rows > 0:
            match_percent = (matched_rows / total_rows) * 100
        else:
            match_percent = 0

        print(
            f"Fallback weather matched rows: "
            f"{matched_rows}/{total_rows} ({match_percent:.2f}%)"
        )

    ml1["datetime"] = parse_datetime_naive(ml1["datetime"])

    ml1["hour"] = ml1["datetime"].dt.hour
    ml1["day"] = ml1["datetime"].dt.day
    ml1["month"] = ml1["datetime"].dt.month
    ml1["day_of_week"] = ml1["datetime"].dt.dayofweek

    ml1["hour_sin"] = np.sin(2 * np.pi * ml1["hour"] / 24)
    ml1["hour_cos"] = np.cos(2 * np.pi * ml1["hour"] / 24)

    ml1["month_sin"] = np.sin(2 * np.pi * ml1["month"] / 12)
    ml1["month_cos"] = np.cos(2 * np.pi * ml1["month"] / 12)

    final_numeric_columns = [
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
        "aqicn_temperature",
        "aqicn_humidity",
        "aqicn_pressure",
        "aqicn_wind_indicator",
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
        "month_cos",
    ]

    for column in final_numeric_columns:
        if column in ml1.columns:
            ml1[column] = pd.to_numeric(ml1[column], errors="coerce")

    preferred_columns = [
        "datetime",
        "target_city",
        "station_name",
        "aqicn_station_name",
        "station_code",
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
        "aqicn_temperature",
        "aqicn_humidity",
        "aqicn_pressure",
        "aqicn_wind_indicator",
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
        "is_prototype_weather_match",
    ]

    existing_preferred_columns = [
        column for column in preferred_columns if column in ml1.columns
    ]

    remaining_columns = [
        column for column in ml1.columns if column not in existing_preferred_columns
    ]

    ml1 = ml1[existing_preferred_columns + remaining_columns]

    ml1 = ml1.drop_duplicates()
    ml1 = ml1.sort_values(["station_name", "datetime"])

    return ml1


# =============================================================================
# 9. Save outputs and reports
# =============================================================================

def save_outputs(ml1: pd.DataFrame) -> None:
    ml1.to_csv(OUTPUT_FILE, index=False)

    missing_report = pd.DataFrame(
        {
            "column": ml1.columns,
            "missing_count": ml1.isna().sum().values,
            "missing_percent": (ml1.isna().mean().values * 100).round(2),
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
    print([column for column in POLLUTANT_COLUMNS if column in ml1.columns])

    print("\nWeather merge mode counts:")
    if "weather_merge_mode" in ml1.columns:
        print(ml1["weather_merge_mode"].value_counts(dropna=False))

    print("\nPrototype weather match counts:")
    if "is_prototype_weather_match" in ml1.columns:
        print(ml1["is_prototype_weather_match"].value_counts(dropna=False))

    print("\nMissing report saved to:")
    print(MISSING_REPORT_FILE)

    print("\nMissing values summary:")
    print(missing_report.sort_values("missing_percent", ascending=False).head(20))

    print("\nFirst 10 rows:")
    print(ml1.head(10))

    print("\nML1 dataset preparation completed.")


def main() -> None:
    ml1 = create_ml1_dataset()
    save_outputs(ml1)


if __name__ == "__main__":
    main()