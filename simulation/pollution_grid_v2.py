import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# 1. File paths

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_FILE = BASE_DIR / "data" / "byrnihat_33_active_sources_pollutant_weights_updated_raksha_gmaps.csv"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

# 2. Read source data

sources = pd.read_csv(CSV_FILE)

sources = sources[sources["active_for_simulation"].str.lower() == "yes"].copy()

print("Active sources:", len(sources))

# 3. Create simulation grid 
GRID_SIZE = 101

pm10_grid = np.zeros((GRID_SIZE, GRID_SIZE))

# 4. Define map boundary with padding

min_lat_raw = sources["latitude"].min()
max_lat_raw = sources["latitude"].max()
min_lon_raw = sources["longitude"].min()
max_lon_raw = sources["longitude"].max()

lat_padding = (max_lat_raw - min_lat_raw) * 0.10
lon_padding = (max_lon_raw - min_lon_raw) * 0.10

min_lat = min_lat_raw - lat_padding
max_lat = max_lat_raw + lat_padding
min_lon = min_lon_raw - lon_padding
max_lon = max_lon_raw + lon_padding

print("Latitude range:", min_lat, "to", max_lat)
print("Longitude range:", min_lon, "to", max_lon)

# 5. Convert latitude-longitude to grid row-column

def latlon_to_grid(lat, lon):
    row = int((lat - min_lat) / (max_lat - min_lat) * (GRID_SIZE - 1))
    col = int((lon - min_lon) / (max_lon - min_lon) * (GRID_SIZE - 1))

    return row, col

#6. Add PM10 emissions from industries

def add_pm10_sources(input_grid, source_table):
    updated_grid = input_grid.copy()

    for index, source in source_table.iterrows():
        row, col = latlon_to_grid(source["latitude"], source["longitude"])
        updated_grid[row, col] += source["pm10_weight"]
    return updated_grid

#7. Diffusion Function:

def diffuse(input_grid, diffusion_rate):
    updated_grid = input_grid.copy()

    # Inner cells, excluding boundary
    center = input_grid[1:-1, 1:-1]

    # Neighbours of each inner cell
    row_plus_1 = input_grid[2:, 1:-1]       # row index + 1
    row_minus_1 = input_grid[:-2, 1:-1]     # row index - 1
    col_plus_1 = input_grid[1:-1, 2:]       # column index + 1
    col_minus_1 = input_grid[1:-1, :-2]     # column index - 1

    # 2D Laplacian
    laplacian = (
        row_plus_1
        + row_minus_1
        + col_plus_1
        + col_minus_1
        - 4 * center
    )

    # Update only inner cells
    updated_grid[1:-1, 1:-1] = center + diffusion_rate * laplacian

    return updated_grid

#8. Entropy Function

def calculate_entropy(input_grid):
    total_pollution = input_grid.sum()

    if total_pollution == 0:
        return 0
    probability_grid = input_grid / total_pollution

    probability_values = probability_grid[probability_grid > 0]

    entropy = -np.sum(probability_values * np.log(probability_values))

    return entropy

#9. Run Simulations

TIME_STEPS = 150
DIFFUSION_RATE = 0.15
DECAY_RATE = 0.995

entropy_values = []

for t in range(TIME_STEPS):

    pm10_grid = add_pm10_sources(pm10_grid, sources)
    pm10_grid = diffuse(pm10_grid, DIFFUSION_RATE)
    pm10_grid = pm10_grid*DECAY_RATE
    entropy = calculate_entropy(pm10_grid)
    entropy_values.append(entropy)

    if t%25 == 0:
        print("Time step:", t, "Entropy", entropy)

#10 Plot final diffusion heatmap

plt.figure(figsize = (9, 8))
plt.imshow(pm10_grid, origin = "lower")
plt.colorbar(label = "PM10 concentration")

for index, source in sources.iterrows():
    row, col  = latlon_to_grid(source["latitude"], source["longitude"])
    plt.scatter(col, row, s =35, marker = "x")
    plt.text(col + 1, row +1, str(index + 1), fontsize = 8)

plt.title("PM10 Diffusion Simulation - Byrnihat Industrial Sources")
plt.xlabel("Grid Column")
plt.ylabel("Grid Row")
output_file = FIGURES_DIR/ "pm10_diffusion_final.png"
plt.savefig(output_file, dpi = 300, bbox_inches = "tight")
print("Saved", output_file)

plt.close

#11. PLot Entropy vs Time

plt.figure(figsize=(8, 5))

plt.plot(entropy_values)
plt.title("Entropy Spread of PM10 overtime" )
plt.xlabel("Time Step")
plt.ylabel("Entropy")

output_file = FIGURES_DIR / "entropyvstime"
plt.savefig(output_file, dpi = 300, bbox_inches = "tight")
print("Saved", output_file)
plt.close






