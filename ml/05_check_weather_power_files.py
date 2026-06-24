from pathlib import Path
import pandas as pd

# 1. Paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# 2. Find NASA POWER files


weather_files = list(DATA_DIR.glob("POWER*.csv"))

print("NASA POWER files found:")
for file in weather_files:
    print(" -", file)

if not weather_files:
    raise FileNotFoundError("No NASA POWER CSV files found in data folder.")


# 3. Helper function


def read_power_file(file_path):
    """
    NASA POWER CSV sometimes has metadata lines before the actual table.
    This function finds the row where the real data starts.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_row = None

    for i, line in enumerate(lines):
        if line.startswith("YEAR") or "YEAR" in line.split(",")[0]:
            header_row = i
            break

    if header_row is None:
        print("Could not automatically find header row.")
        print("Trying normal pd.read_csv...")
        return pd.read_csv(file_path)

    df = pd.read_csv(file_path, skiprows=header_row)
    return df



# 4. Inspect each file


for file in weather_files:
    print("\n" + "=" * 80)
    print("File:", file.name)
    print("=" * 80)

    df = read_power_file(file)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    for col in df.columns:
        print(" -", col)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isna().sum())