# Record_AQICNdata.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


AQICN_API_URL_TEMPLATE = "https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}"

RAW_FETCH_LOG_FILE = Path("data/raw/aqicn_fetch_log.csv")

STATIONS = [
    {
        "target_city": "Byrnihat",
        "station_code": "A1364707",
        "station_name": "15th Mile-Nongthymmai, Byrnihat",
        "latitude": 26.05374,
        "longitude": 91.86796,
        "processed_file": Path("data/processed/byrnihat_aqicn_observations.csv"),
        "legacy_file": Path("data/byrnihat_aqicn_data.csv"),
    },
    {
        "target_city": "Guwahati",
        "station_code": "H11844",
        "station_name": "Railway Colony, Guwahati, India",
        "latitude": 26.181742,
        "longitude": 91.78063,
        "processed_file": Path("data/processed/guwahati_aqicn_observations.csv"),
        "legacy_file": Path("data/guwahati_aqicn_data.csv"),
    },
    {
        "target_city": "Shillong",
        "station_code": "H12740",
        "station_name": "Lumpyngngad, Shillong, India",
        "latitude": 25.5586,
        "longitude": 91.8985,
        "processed_file": Path("data/processed/shillong_aqicn_observations.csv"),
        "legacy_file": Path("data/shillong_aqicn_data.csv"),
    },
]


OBSERVATION_COLUMNS = [
    "first_collection_time_utc",
    "target_city",
    "station_code",
    "station_name",
    "station_latitude",
    "station_longitude",
    "aqicn_idx",
    "api_time",
    "aqi",
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "nh3",
    "no",
    "nox",
    "temperature",
    "humidity",
    "pressure",
    "wind",
]

RAW_COLUMNS = [
    "collection_time_utc",
    "fetch_success",
    "api_status",
    "error_message",
    "target_city",
    "station_code",
    "station_name",
    "station_latitude",
    "station_longitude",
    "aqicn_idx",
    "api_time",
    "aqi",
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "nh3",
    "no",
    "nox",
    "temperature",
    "humidity",
    "pressure",
    "wind",
]


def get_aqicn_token() -> str:
    token = (
        os.getenv("AQICN_TOKEN")
        or os.getenv("AQICN_API_TOKEN")
        or os.getenv("WAQI_TOKEN")
    )

    if not token:
        raise RuntimeError(
            "AQICN token not found. Add it as AQICN_TOKEN in GitHub Secrets or .env."
        )

    return token


def get_iaqi_value(iaqi: dict[str, Any], key: str) -> float | int | None:
    value = iaqi.get(key)

    if isinstance(value, dict):
        return value.get("v")

    return None


def get_api_time(data: dict[str, Any]) -> str | None:
    time_info = data.get("time", {})

    if isinstance(time_info, dict):
        return time_info.get("s") or time_info.get("iso")

    return None


def fetch_station_data(station: dict[str, Any], token: str) -> dict[str, Any]:
    url = AQICN_API_URL_TEMPLATE.format(
        lat=station["latitude"],
        lon=station["longitude"],
        token=token,
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    return response.json()


def build_base_row(station: dict[str, Any], collection_time_utc: str) -> dict[str, Any]:
    return {
        "collection_time_utc": collection_time_utc,
        "fetch_success": False,
        "api_status": None,
        "error_message": None,
        "target_city": station["target_city"],
        "station_code": station["station_code"],
        "station_name": station["station_name"],
        "station_latitude": station["latitude"],
        "station_longitude": station["longitude"],
        "aqicn_idx": None,
        "api_time": None,
        "aqi": None,
        "pm25": None,
        "pm10": None,
        "no2": None,
        "so2": None,
        "co": None,
        "o3": None,
        "nh3": None,
        "no": None,
        "nox": None,
        "temperature": None,
        "humidity": None,
        "pressure": None,
        "wind": None,
    }


def parse_successful_response(
    station: dict[str, Any],
    api_response: dict[str, Any],
    collection_time_utc: str,
) -> dict[str, Any]:
    row = build_base_row(station, collection_time_utc)

    api_status = api_response.get("status")
    row["api_status"] = api_status

    if api_status != "ok":
        row["error_message"] = str(api_response.get("data", "AQICN returned non-ok status"))
        return row

    data = api_response.get("data", {})

    if not isinstance(data, dict):
        row["error_message"] = "AQICN response data is not a dictionary."
        return row

    iaqi = data.get("iaqi", {})

    if not isinstance(iaqi, dict):
        iaqi = {}

    city_info = data.get("city", {})

    if not isinstance(city_info, dict):
        city_info = {}

    geo = city_info.get("geo", [])

    if isinstance(geo, list) and len(geo) >= 2:
        row["station_latitude"] = geo[0]
        row["station_longitude"] = geo[1]

    row.update(
        {
            "fetch_success": True,
            "error_message": None,
            "aqicn_idx": data.get("idx"),
            "api_time": get_api_time(data),
            "aqi": data.get("aqi"),
            "pm25": get_iaqi_value(iaqi, "pm25"),
            "pm10": get_iaqi_value(iaqi, "pm10"),
            "no2": get_iaqi_value(iaqi, "no2"),
            "so2": get_iaqi_value(iaqi, "so2"),
            "co": get_iaqi_value(iaqi, "co"),
            "o3": get_iaqi_value(iaqi, "o3"),
            "nh3": get_iaqi_value(iaqi, "nh3"),
            "no": get_iaqi_value(iaqi, "no"),
            "nox": get_iaqi_value(iaqi, "nox"),
            "temperature": get_iaqi_value(iaqi, "t"),
            "humidity": get_iaqi_value(iaqi, "h"),
            "pressure": get_iaqi_value(iaqi, "p"),
            "wind": get_iaqi_value(iaqi, "w"),
        }
    )

    if row["api_time"] is None:
        row["fetch_success"] = False
        row["error_message"] = "AQICN response did not contain api_time."

    return row


def append_raw_fetch_log(row: dict[str, Any]) -> None:
    RAW_FETCH_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw_row = {column: row.get(column) for column in RAW_COLUMNS}

    raw_data = pd.DataFrame([raw_row], columns=RAW_COLUMNS)

    raw_data.to_csv(
        RAW_FETCH_LOG_FILE,
        mode="a",
        header=not RAW_FETCH_LOG_FILE.exists(),
        index=False,
    )

    print(f"Raw fetch logged: {row['target_city']}")


def normalize_legacy_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "0" in df.columns:
        df = df.drop(columns=["0"])

    if "recorded_at_local_time" in df.columns and "first_collection_time_utc" not in df.columns:
        df = df.rename(columns={"recorded_at_local_time": "first_collection_time_utc"})

    for column in OBSERVATION_COLUMNS:
        if column not in df.columns:
            df[column] = None

    return df[OBSERVATION_COLUMNS]


def load_existing_processed_data(station: dict[str, Any]) -> pd.DataFrame:
    processed_file = station["processed_file"]
    legacy_file = station["legacy_file"]

    if processed_file.exists():
        existing_data = pd.read_csv(processed_file)
        return normalize_legacy_data(existing_data)

    if legacy_file.exists():
        legacy_data = pd.read_csv(legacy_file)
        return normalize_legacy_data(legacy_data)

    return pd.DataFrame(columns=OBSERVATION_COLUMNS)


def update_processed_observations(row: dict[str, Any], station: dict[str, Any]) -> int:
    if not row.get("fetch_success"):
        print(f"Processed skipped: {row['target_city']} fetch was not successful.")
        return 0

    if not row.get("api_time"):
        print(f"Processed skipped: {row['target_city']} has no api_time.")
        return 0

    processed_file = station["processed_file"]
    processed_file.parent.mkdir(parents=True, exist_ok=True)

    existing_data = load_existing_processed_data(station)

    observation_row = {
        "first_collection_time_utc": row["collection_time_utc"],
        "target_city": row["target_city"],
        "station_code": row["station_code"],
        "station_name": row["station_name"],
        "station_latitude": row["station_latitude"],
        "station_longitude": row["station_longitude"],
        "aqicn_idx": row["aqicn_idx"],
        "api_time": row["api_time"],
        "aqi": row["aqi"],
        "pm25": row["pm25"],
        "pm10": row["pm10"],
        "no2": row["no2"],
        "so2": row["so2"],
        "co": row["co"],
        "o3": row["o3"],
        "nh3": row["nh3"],
        "no": row["no"],
        "nox": row["nox"],
        "temperature": row["temperature"],
        "humidity": row["humidity"],
        "pressure": row["pressure"],
        "wind": row["wind"],
    }

    new_data = pd.DataFrame([observation_row], columns=OBSERVATION_COLUMNS)

    combined_data = pd.concat(
        [existing_data, new_data],
        ignore_index=True,
    )

    combined_data = combined_data.dropna(how="all")

    combined_data = combined_data.dropna(
        subset=["station_code", "api_time"],
        how="any",
    )

    before_deduplication = len(combined_data)

    combined_data = combined_data.drop_duplicates(
        subset=["station_code", "api_time"],
        keep="first",
    )

    after_deduplication = len(combined_data)

    combined_data = combined_data.sort_values(
        by=["station_code", "api_time"],
        ascending=True,
    )

    combined_data.to_csv(processed_file, index=False)

    new_unique_rows_added = after_deduplication - len(existing_data.dropna(how="all"))

    if before_deduplication == after_deduplication:
        print(f"Processed updated: {row['target_city']} new observation added.")
    else:
        print(f"Processed unchanged: {row['target_city']} duplicate api_time already exists.")

    print(f"Processed row count for {row['target_city']}: {after_deduplication}")

    return max(new_unique_rows_added, 0)


def collect_station(station: dict[str, Any], token: str) -> None:
    collection_time_utc = pd.Timestamp.utcnow().isoformat()

    try:
        api_response = fetch_station_data(station, token)
        row = parse_successful_response(
            station=station,
            api_response=api_response,
            collection_time_utc=collection_time_utc,
        )

    except Exception as error:
        row = build_base_row(station, collection_time_utc)
        row["api_status"] = "request_failed"
        row["error_message"] = str(error)

    append_raw_fetch_log(row)
    update_processed_observations(row, station)


def main() -> None:
    token = get_aqicn_token()

    print("Starting AQICN data collection...")
    print(f"Raw fetch log file: {RAW_FETCH_LOG_FILE}")

    for station in STATIONS:
        print("-" * 70)
        print(f"Collecting AQICN data for {station['target_city']}")
        collect_station(station, token)

    print("-" * 70)
    print("AQICN data collection completed.")


if __name__ == "__main__":
    main()
