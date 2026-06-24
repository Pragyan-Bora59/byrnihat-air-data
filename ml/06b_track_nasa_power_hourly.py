from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta, timezone
import requests

#1. Project Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

RAW_DIR.mkdir(parents = True, exist_ok = True)

OUTPUT_FILE = RAW_DIR / "nasa_power_hourly_tracker.csv"

#2. Station Locations

STATIONS = {
    "Byrnihat": {
        "latitude": 26.0715,
        "longitude": 91.8746
    },
    "Guwahati": {
        "latitude": 26.1445,
        "longitude": 91.7362
    },
    "Shillong": {
        "latitude": 25.5788,
        "longitude": 91.8933
    }
}

#3. NASA Power Parameters

PARAMETERS = [
    "T2M",
    "RH2M",
    "PRECTOTCORR",
    "WS10M",
    "WD10M"
]

#4. DATA Range

today = datetime.now(timezone.utc).date()
start_date = today - timedelta(days = 20)
end_date = today - timedelta(days = 7)

START = start_date.strftime("%Y%m%d")
END = end_date.strftime("%Y%m%d")

print("Fetching NASA data form:", START, "to", END)

#5. Function to fetch NASA POWER hourly data

def fethch_nasa_power_hourly(station_name, latitude, longitude):

    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"

    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": START,
        "end": END,
        "format": "JSON",
        "time-standard": "UTC"
    }

    print(f"\nRequesting NASA POWER data for {station_name}...")

    response = requests.get(base_url, params=params, timeout=120)

    if response.status_code != 200:
        print("Request failed.")
        print("Status code:", response.status_code)
        print("Response:", response.text[:500])
        return pd.DataFrame()

    data = response.json()

    # Correct place where NASA stores weather values
    if "properties" not in data:
        print("No properties key found in response.")
        return pd.DataFrame()

    if "parameter" not in data["properties"]:
        print("No parameter key found inside properties.")
        print("Available properties keys:", data["properties"].keys())
        return pd.DataFrame()

    parameter_data = data["properties"]["parameter"]

    if not parameter_data:
        print("Parameter data is empty.")
        return pd.DataFrame()

    # Get all time keys from the first available parameter
    first_parameter = next(iter(parameter_data.values()))
    all_times = sorted(first_parameter.keys())

    rows = []

    for time_key in all_times:
        row = {
            "datetime_utc": pd.to_datetime(time_key, format="%Y%m%d%H", errors="coerce"),
            "station_name": station_name,
            "latitude": latitude,
            "longitude": longitude,
            "temperature": parameter_data.get("T2M", {}).get(time_key),
            "humidity": parameter_data.get("RH2M", {}).get(time_key),
            "rainfall": parameter_data.get("PRECTOTCORR", {}).get(time_key),
            "wind_speed": parameter_data.get("WS10M", {}).get(time_key),
            "wind_direction": parameter_data.get("WD10M", {}).get(time_key)
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    print(f"Fetched {len(df)} rows for {station_name}")

    return df

#6. Fetch data for all the stations

all_weather_dfs = []

for station_name, info in STATIONS.items():
    station_df = fethch_nasa_power_hourly(
        station_name = station_name,
        latitude = info["latitude"],
        longitude = info["longitude"]
    )

    if not station_df.empty:
        all_weather_dfs.append(station_df)

if not all_weather_dfs:
    raise RuntimeError("No NASA Power data has been fetched.")

new_weather = pd.concat(all_weather_dfs, ignore_index = True)

#7. Add local datetime

new_weather["datetime_ist"] = new_weather["datetime_utc"] + pd.Timedelta(hours = 5, minutes = 30)
#Since UST is ahead of IST by 5 hours and 30 minuites

#8. Clean Numeric Columns

numeric_cols = [
    "temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "wind_direction"
]

for col in numeric_cols:
    new_weather[col] = pd.to_numeric(new_weather[col], errors = "coerce")

#9. Load Old Tracker if it exists

if OUTPUT_FILE.exists():
    old_weather = pd.read_csv(OUTPUT_FILE)
    old_weather["datetime_utc"] = pd.to_datetime(old_weather["datetime_utc"], errors = "coerce")
    old_weather["datetime_ist"] = pd.to_datetime(old_weather["datetime_ist"], errors = "coerce")

    combined = pd.concat([old_weather, new_weather], ignore_index = True)

else: 
    combined = new_weather.copy()

#10. Remove Duplicates

combined = combined.drop_duplicates(
    subset = ["station_name", "datetime_utc"],
    keep = "last"
)

combined = combined.sort_values(["station_name", "datetime_utc"])

#11. Save Tracker File

combined.to_csv(OUTPUT_FILE, index=False)

print("\nNASA POWER hourly tracker updated successfully.")
print("Saved to:")
print(OUTPUT_FILE)

print("\nFinal shape:")
print(combined.shape)

print("\nDate range:")
print("UTC start:", combined["datetime_utc"].min())
print("UTC end  :", combined["datetime_utc"].max())

print("\nLast few rows:")
print(combined.tail())