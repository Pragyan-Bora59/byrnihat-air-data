import os
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# 1. Load AQICN API token
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

AQICN_TOKEN = os.getenv("AQICN_TOKEN")


# ============================================================
# 2. Stations
# ============================================================

STATIONS = [
    {
        "city": "Byrnihat",
        "station_name": "15th Mile-Nongthymmai, Byrnihat",
        "station_code": "A1364707",
        "url": "https://api.waqi.info/feed/A1364707/",
        "file": Path("data/byrnihat_aqicn_data.csv")
    },
    {
        "city": "Guwahati",
        "station_name": "Railway Colony, Guwahati",
        "station_code": "H11844",
        "url": "https://api.waqi.info/feed/@11844/",
        "file": Path("data/guwahati_aqicn_data.csv")
    },
    {
        "city": "Shillong",
        "station_name": "Lumpyngngad, Shillong",
        "station_code": "H12740",
        "url": "https://api.waqi.info/feed/@12740/",
        "file": Path("data/shillong_aqicn_data.csv")
    }
]


# ============================================================
# 3. Fetch AQICN data
# ============================================================

def fetch_aqicn_data(station):
    if not AQICN_TOKEN:
        raise ValueError(
            "AQICN_TOKEN not found. Add it as a GitHub Actions secret "
            "or in your local .env file."
        )

    params = {
        "token": AQICN_TOKEN
    }

    response = requests.get(
        station["url"],
        params=params,
        timeout=30
    )

    response.raise_for_status()

    api_response = response.json()

    if api_response.get("status") != "ok":
        raise ValueError(
            f"API returned an error for {station['city']}: {api_response}"
        )

    return api_response["data"]


# ============================================================
# 4. Safely get pollutant values
# ============================================================

def get_iaqi_value(iaqi_data, pollutant_name):
    pollutant_data = iaqi_data.get(pollutant_name)

    if pollutant_data is None:
        return None

    return pollutant_data.get("v")


# ============================================================
# 5. Convert API response into one clean row
# ============================================================

def convert_api_data_to_row(data, station):
    iaqi = data.get("iaqi", {})
    city = data.get("city", {})
    time_info = data.get("time", {})

    station_coordinates = city.get("geo", [None, None])

    row = {
        "recorded_at_local_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "target_city": station["city"],
        "station_code": station["station_code"],

        "station_name": city.get("name"),
        "station_latitude": station_coordinates[0],
        "station_longitude": station_coordinates[1],

        "api_time": time_info.get("s"),
        "aqi": data.get("aqi"),

        "pm25": get_iaqi_value(iaqi, "pm25"),
        "pm10": get_iaqi_value(iaqi, "pm10"),
        "no2": get_iaqi_value(iaqi, "no2"),
        "so2": get_iaqi_value(iaqi, "so2"),
        "co": get_iaqi_value(iaqi, "co"),
        "o3": get_iaqi_value(iaqi, "o3"),

        "temperature": get_iaqi_value(iaqi, "t"),
        "humidity": get_iaqi_value(iaqi, "h"),
        "pressure": get_iaqi_value(iaqi, "p"),
        "wind": get_iaqi_value(iaqi, "w"),
    }

    return row


# ============================================================
# 6. Save row to station CSV without losing old data
# ============================================================

def save_row_to_city_csv(row, station):
    data_file = station["file"]

    data_file.parent.mkdir(parents=True, exist_ok=True)

    new_data = pd.DataFrame([row])

    # This records when our GitHub/local script collected the row.
    new_data["collection_time_utc"] = pd.Timestamp.utcnow()

    if data_file.exists():
        old_data = pd.read_csv(data_file)

        combined_data = pd.concat(
            [old_data, new_data],
            ignore_index=True
        )
    else:
        combined_data = new_data.copy()

    # Standardize datetime-like columns.
    if "api_time" in combined_data.columns:
        combined_data["api_time"] = pd.to_datetime(
            combined_data["api_time"],
            errors="coerce"
        )

    if "recorded_at_local_time" in combined_data.columns:
        combined_data["recorded_at_local_time"] = pd.to_datetime(
            combined_data["recorded_at_local_time"],
            errors="coerce"
        )

    if "collection_time_utc" in combined_data.columns:
        combined_data["collection_time_utc"] = pd.to_datetime(
            combined_data["collection_time_utc"],
            errors="coerce"
        )

    # Same station_code + same api_time means same AQICN observation.
    duplicate_keys = []

    if "station_code" in combined_data.columns:
        duplicate_keys.append("station_code")

    if "api_time" in combined_data.columns:
        duplicate_keys.append("api_time")

    if duplicate_keys:
        combined_data = combined_data.drop_duplicates(
            subset=duplicate_keys,
            keep="last"
        )
    else:
        combined_data = combined_data.drop_duplicates(keep="last")

    sort_cols = []

    if "station_code" in combined_data.columns:
        sort_cols.append("station_code")

    if "api_time" in combined_data.columns:
        sort_cols.append("api_time")

    if sort_cols:
        combined_data = combined_data.sort_values(sort_cols)

    combined_data.to_csv(data_file, index=False)

    print(f"Saved {len(combined_data)} total rows to {data_file}")

    return len(combined_data)


# ============================================================
# 7. Main function
# ============================================================

def main():
    print("Fetching latest AQICN air quality data for all stations...")

    success_count = 0

    for station in STATIONS:
        print(f"\nFetching data for {station['city']}...")

        try:
            api_data = fetch_aqicn_data(station)

            row = convert_api_data_to_row(api_data, station)

            total_rows = save_row_to_city_csv(row, station)

            success_count += 1

            print(
                f"Success: {station['city']} | "
                f"AQI: {row['aqi']} | "
                f"PM2.5: {row['pm25']} | "
                f"PM10: {row['pm10']} | "
                f"API time: {row['api_time']} | "
                f"Total rows now: {total_rows} | "
                f"Saved to: {station['file']}"
            )

        except Exception as error:
            print(f"Failed for {station['city']}: {error}")

    if success_count == 0:
        raise RuntimeError(
            "No AQICN data was saved for any station. "
            "Check AQICN_TOKEN, GitHub Secrets, and API response."
        )


# ============================================================
# 8. Run program
# ============================================================

if __name__ == "__main__":
    main()

