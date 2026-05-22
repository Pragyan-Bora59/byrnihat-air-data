import os
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
from  dotenv import load_dotenv

#1. Loading API from the .env file

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

AQICN_TOKEN = os.getenv("AQICN_TOKEN")

#2. Stations
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
#4. Function to fetch the data:

def fetch_aqicn_data(station):

    if not AQICN_TOKEN:
        raise ValueError("AQICN token not found. Please add it in your .env file.")

    params = {
        "token": AQICN_TOKEN
    }

    response = requests.get(station["url"], params=params, timeout=30)
    response.raise_for_status()

    api_response = response.json()

    if api_response.get("status") != "ok":
        raise ValueError(f"API returned an error for {station['city']}: {api_response}")

    return api_response["data"]

#5. Function to safely get pollutant values
def get_iaqi_value(iaqi_data, pollutant_name):
    pollutant_data = iaqi_data.get(pollutant_name)
    if pollutant_data is None:
        return None
    return pollutant_data.get("v")

#6. Converting API data into one clean row

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

#7. Saving data in the csv file without duplicates:

def save_row_to_city_csv(row, station):

    data_file = station["file"]

    data_file.parent.mkdir(exist_ok=True)

    new_data = pd.DataFrame([row])

    if data_file.exists():
        old_data = pd.read_csv(data_file)

        combined_data = pd.concat(
            [old_data, new_data],
            ignore_index=True
        )

        combined_data = combined_data.drop_duplicates(
            subset=["station_code", "api_time"],
            keep="last"
        )

    else:
        combined_data = new_data

    combined_data.to_csv(data_file, index=False)

#8 Main Function
def main():

    print("Fetching latest AQICN air quality data for all stations...")

    for station in STATIONS:
        print(f"Fetching data for {station['city']}...")

        try:
            api_data = fetch_aqicn_data(station)

            row = convert_api_data_to_row(api_data, station)

            save_row_to_city_csv(row, station)

            print(
                f"Success: {station['city']} | "
                f"AQI: {row['aqi']} | "
                f"PM2.5: {row['pm25']} | "
                f"Saved to: {station['file']}"
            )

        except Exception as error:
            print(f"Failed for {station['city']}: {error}")

 #9. Run the program


if __name__ == "__main__":
    main()

