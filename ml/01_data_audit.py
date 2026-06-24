from pathlib import Path
import pandas as pd

#1. Projects Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT  / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents = True, exist_ok = True)


#2. Helper Function

def inspect_csv(file_path):
    print("\n"+ "=" * 80)
    print(f"File:{file_path}")
    print("=" * 80)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Could not read the file: {e}")
        return None
    
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\nCoulumns:")
    for col in df.columns:
        print(f"- {col}")
    
    print(f"\n Datatypes: {df.dtypes}")

    print(f"\n Missing Values:")
    missing = df.isna().sum()

    missing = df.isna().sum()

    if len(df) > 0:
        missing_percent = ((missing / len(df)) * 100).round(2)
    else:
        missing_percent = missing * 0 

    missing_table = pd.DataFrame({
        "columns" : df.columns,
        "missing_count" : missing.values,
        "missing_percent" : missing_percent.values,
        "data_type" : df.dtypes.astype(str).values
    })

    print(f"Missing Table:\n{missing_table}")

    print(f"First 5 rows:\n{df.head()}")

    numeric_cols = df.select_dtypes(include =["int64","float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include = ["object"]).columns.tolist()

    datetime_candidates = [
        col for col in df.columns
        if "time" in col.lower()
        or "date" in col.lower()
        or "datetime" in col.lower()
    ]

    summary = {
        "file_name" : file_path.name,
        "file_path" : str(file_path),
        "rows" : df.shape[0],
        "columns" : df.shape[1],
         "numeric_columns": ", ".join(numeric_cols),
        "categorical_columns": ", ".join(categorical_cols),
        "datetime_candidates": ", ".join(datetime_candidates),
    }
    return summary

#3. Find and Inspect the CSV Files

csv_files = list(DATA_DIR.rglob("*.csv"))

if not csv_files:
    print("No CSV files found inside data/ folder.")
else:
    print(f"Found {len(csv_files)} CSV files.")

all_summaries = []

for csv_file in csv_files:
    summary = inspect_csv(csv_file)
    if summary is not None:
        all_summaries.append(summary)

#4. Save Audit Report

if all_summaries:
    report_df = pd.DataFrame(all_summaries)
    report_path = OUTPUT_DIR / "data_audit_report.csv"
    report_df.to_csv(report_path, index = False)

    print("\n" + "=" * 80)
    print(f"Data audit report saved to: {report_path}")
    print("=" * 80)


