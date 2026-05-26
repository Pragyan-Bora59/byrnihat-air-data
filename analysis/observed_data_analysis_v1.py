import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#1. File Paths

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR/ "figures"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR/ "outputs"

FIGURES_DIR.mkdir(exist_ok= True)
OUTPUT_DIR.mkdir(exist_ok= True)

BYRNIHAT_FILE = DATA_DIR / "byrnihat_aqicn_data.csv"
GUWAHATI_FILE = DATA_DIR / "guwahati_aqicn_data.csv"
SHILLONG_FILE = DATA_DIR / "shillong_aqicn_data.csv"

#2. Functiom to read one city file

def read_city_file(file_path, city_name):
    data = pd.read_csv(file_path)

    unwanted_columns = [] #Remove unwanted columns like "unnanmaed" or "0"

    for column in data.columns:
        if column.startswith("Unnamed") or column == "0":
            unwanted_columns.append(column)
    data = data.drop(columns= unwanted_columns, errors = "ignore")

    if "city" not in data.columns:
        data["city"] = city_name
    
    if "api_time" in data.columns:
        data["api_time"] = pd.to_datetime(data["api_time"], errors= "coerce")

    if "recorded_at_local_time" in data.columns:
        data["recorded_at_local_time"] = pd.to_datetime(data["api_time"], errors = "coerce")
    
    numeric_columns = [ "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors= "coerce")
    
    if "api_time" in data.columns:
        data = data.dropna(subset=["api_time"])
    
    if "api_time" in data.columns:
        data = data.drop_duplicates(subset = ["city", "api_time"], keep = "last")
    
    return data

#3. Read all city files

byrnihat_data = read_city_file(BYRNIHAT_FILE, "Byrnihat")
guwahati_data = read_city_file(GUWAHATI_FILE, "Guwahati")
shillong_data = read_city_file(SHILLONG_FILE, "Shillong")

#4. Combine all city data
all_data = pd.concat([byrnihat_data, guwahati_data, shillong_data], ignore_index = True)
all_data = all_data.sort_values(["city", "api_time"])
print("Total number of rows:", len(all_data))
print("The last data line:", all_data.tail())

#5. Save cleaned combined data

cleaned_file = OUTPUT_DIR/ "observed_aqicn_combined_cleaned.csv"
all_data.to_csv(cleaned_file, index = False)
print("Saved combined cleaned file:", cleaned_file)

#6. Function to plot pollutant vs time 
def plot_pollutant_over_time(data, pollutant_column, ylabel, output_filename):

    if pollutant_column not in data.columns:
        print(f"Column {pollutant_column} not found. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))

    for city in data["city"].unique():
        city_data = data[data["city"] == city]

        plt.plot(
            city_data["api_time"],
            city_data[pollutant_column],
            marker="o",
            label=city
        )

    plt.title(f"{ylabel} Over Time")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_file = FIGURES_DIR / output_filename

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output_file)

#7. Observed Graphs

plot_pollutant_over_time(
    all_data,
    "aqi",
    "AQI",
    "observed_aqi_3cities.png"
)

plot_pollutant_over_time(
    all_data,
    "pm10",
    "PM10",
    "observed_pm10_3cities.png"
)

plot_pollutant_over_time(
    all_data,
    "pm25",
    "PM2.5",
    "observed_pm25_3cities.png"
)

#8. Create Summary Tables

summary_colunms = []
for column in ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co"]:
    if column in all_data.columns:
        summary_colunms.append(column)

summary_table = all_data.groupby("city")[summary_colunms].agg(
    ["count", "mean", "max", "min"]
)
summary_file = OUTPUT_DIR/ "observed_data_summary.csv"
summary_table.to_csv(summary_file)
print("Summary Table saved:", summary_file)

#9. Missing data report

missing_report = all_data.groupby("city").apply(
    lambda group : group.isna().sum()
)
missing_file = OUTPUT_DIR / "observed__missing_data_report.csv"
missing_report.to_csv(missing_file)
print("Saved missing data report:", missing_file)

#10.  Latest Values Report
latest_rows = []
for city in all_data["city"].unique():
    city_data = all_data[all_data["city"] == city].sort_values("api_time")

    if len(city_data) > 0:
        latest_rows.append(city_data.iloc[-1])
    latest_data = pd.DataFrame(latest_rows)

    latest_file = OUTPUT_DIR / "latest_observed_values.csv"
    latest_data.to_csv(latest_file, index = False)
    print("Saved latest observed values:", latest_file)







    





