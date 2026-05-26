import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#1. File Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = BASE_DIR/ "data" / "byrnihat_15_active_sources_pollutant_weights.csv"
FIGURES_DIR = BASE_DIR/ "figures"

FIGURES_DIR.mkdir(exist_ok = True)

#2. Reading the source file
sources = pd.read_csv(CSV_FILE)
print("Total number of industries in the csv file found", len(sources))
print(sources.head())

#3. Keeping only active sources for the simulation:
sources = sources[sources["active_for_simulation"].str.lower() == "yes"].copy()
print("Active sources:", len(sources))

#4. Creating 2D Grid
GRID_SIZE = 101
pm_10_grid = np.zeros((GRID_SIZE, GRID_SIZE))

#5. Defining the map boundary 

min_lat = sources["latitude"].min()
max_lat = sources["latitude"].max()
min_lon = sources["longitude"].min()
max_lon = sources["longitude"].max()

lat_padding = (max_lat - min_lat)*0.10
lon_padding = (max_lon - min_lon)*0.10

min_lat = min_lat - lat_padding
max_lat = max_lat + lat_padding
min_lon = min_lon - lon_padding
max_lon = max_lon + lon_padding

print("Letitude Range:", min_lat, "to", max_lat)
print("Longitude range is:", min_lon, "to", max_lon)

#Converting real life latitude and longitude values to a grid

def latlon_to_grid(lat, lon):

    row = int((lat - min_lat)/(max_lat - min_lat)*(GRID_SIZE - 1))
    col = int((lon - min_lon)/(max_lon - min_lon)*(GRID_SIZE - 1))

    return row, col

#7. Placing the sources on the grid

for index, source in sources.iterrows():
    row, col = latlon_to_grid(source["latitude"], source["longitude"])
    pm_10_grid[row, col] += source["pm10_weight"]

    print(
        source["industry_name"],
        "--> rows:", row,
        "col:", col,
        "PM10 weight:", source["pm10_weight"]
    )

# 8. Plotting the Heatmap

plt.figure(figsize=(9, 8))

plt.imshow(pm_10_grid, origin="lower")
plt.colorbar(label="PM10 source weight")

# Plot larger visible source markers
for index, source in sources.iterrows():
    row, col = latlon_to_grid(source["latitude"], source["longitude"])
    plt.scatter(col, row, s=40, marker="x")
    plt.text(col + 1, row + 1, str(index + 1), fontsize=8)

plt.title("Byrnihat Industrial Source Points - PM10")
plt.xlabel("Grid Column")
plt.ylabel("Grid Row")

output_file = FIGURES_DIR / "pm10_source_points.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print("Figure saved at:", output_file)

plt.show()