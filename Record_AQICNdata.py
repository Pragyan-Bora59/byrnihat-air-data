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

#2. Byrnihat approximate coordinates

Latitude  = 26.0525
Longitude = 91.8700

#CSV file path
Data_File = Path("data/Byrnihat_AQICN_data.csv")

#4. Function to fetch the data:

def fetch_aqicn_data():
    
    if not AQICN_TOKEN:
        raise ValueError("AQICN token not found. Please add it in your env file.")
    
    url = "https://api.waqi.info/feed/A1364707/"
    
    params = {
        "token": AQICN_TOKEN
    }
    response = requests.get(url, params= params, timeout=30)
    response.raise_for_status()
    api_response = response.json()

    if api_response.get("status") != "ok":
        raise ValueError(f"API returned an error: { api_response}")
    
    return api_response["data"]

#5. Function to safely get pollutant values
def get_iaqi_value(iaqi_data, pollutant_name):
    pollutant_data = iaqi_data.get(pollutant_name)
    if pollutant_data is None:
        return None
    return pollutant_data.get("v")

#6. Converting API data into one clean row

def convert_api_data_to_row(data):
    iaqi = data.get("iaqi", {})
    city = data.get("city", {})
    time_info = data.get("time", {})

    station_coordinates = city.get("geo", [None, None])
    
    row = {
        "recorded_at_local_time": datetime.now().strftime("%Y-%m_%d %H:%M:%S"),

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

def save_row_to_csv(row):

    Data_File.parent.mkdir(exist_ok= True)
    new_data = pd.DataFrame([row])
    if Data_File.exists():
        old_data = pd.read_csv(Data_File)

        combined_data = pd.concat(
            [old_data, new_data],
            ignore_index=True
        )

        combined_data = combined_data.drop_duplicates(
            subset=["station_name", "api_time"],
            keep="last"
        )

    else:
        combined_data = new_data

    combined_data.to_csv(Data_File, index=False)

#8 Main Function
def main():
    print("Fetchin the latest air quality data near Byrnihat")

    aqi_data = fetch_aqicn_data()
    row = convert_api_data_to_row(aqi_data)
    save_row_to_csv(row)

    print("Data recorded successfully.")
    print("--------------------------------")
    print(f"Station name: {row['station_name']}")
    print(f"Station latitude: {row['station_latitude']}")
    print(f"Station longitude: {row['station_longitude']}")
    print(f"API time: {row['api_time']}")
    print(f"AQI: {row['aqi']}")
    print(f"PM2.5: {row['pm25']}")
    print(f"PM10: {row['pm10']}")
    print(f"Saved file: {Data_File}")

 #9. Run the program


if __name__ == "__main__":
    main()

