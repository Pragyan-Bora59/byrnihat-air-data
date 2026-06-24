from pathlib import Path
import pandas as pd

#1. Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents = True, exist_ok = True)

#2. Finding AQICN station files

csv_files = list(DATA_DIR.rglob("*.csv"))

aqicn_files = []

for file in csv_files:
    name = file.name.lower()
    
    if "aq" in name and "industry" not in name:
        aqicn_files.append(file)

print("AQICN files found:")
for files in aqicn_files:
    print("-", files)

#3. Reading and combining all the files

all_dfs = []

for files in aqicn_files:
    df = pd.read_csv(files)

    file_name = files.name.lower()

    if "byrnihat" in file_name:
        df["station_name"] = "Byrnihat"
    if "guwahati" in file_name:
        df["station_name"] = "Guwahati"
    if "shillong" in file_name:
        df["station_name"] = "Shillong"
    else:
        df["station_name"] = "Unknown"

    df["source_file"] = files.name

    all_dfs.append(df)
    
combined_df = pd.concat(all_dfs, ignore_index  = True)

#4. Basic Cleaning

print("\nCombined shape before cleaning:", combined_df.shape)

print("\n Columns found:")
for col in combined_df.columns:
    print("-", col)

#Remove Duplicate rows if any
combined_df = combined_df.drop_duplicates()

print("\nCombined shape after removing duplicates:", combined_df.shape)

#5. Save Combined Dataset

output_path = PROCESSED_DIR / "aqicn_all_stations.csv"
combined_df.to_csv(output_path, index = False)

print("\nSaved combined AQICN dataset to:")
print(output_path)



