from pathlib import Path
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import joblib

#1. Project Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

INPUT_FILE = PROCESSED_DIR / "ml1_environment_pollution_dataset.csv"

METRICS_FILE = REPORTS_DIR / "final_ml1_metrics.csv"
PREDICTIONS_FILE = REPORTS_DIR / "final_ml1_predictions.csv"
FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "final_ml1_feature_importance.csv"
FEATURE_CONFIG_FILE = REPORTS_DIR / "final_ml1_feature_config.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_TARGET_ROWS = 8
MAX_FEATURE_MISSING_PERCENT = 70

#2. Load ML-1 dataset

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"ML1 dataset not found{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace("  ",  "_", regex = False)
    .str.replace(".", "_", regex=False)
    .str.replace("-", "_", regex=False)
    .str.replace("/", "_", regex=False)
)
print("="*80)
print('ML-1 Dataset loaded')
print("="*80)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.to_list())

df["datetime"] = pd.to_datetime(df["datetime"], errors = "coerce")
df = df.dropna(subset = ["datetime"])
df = df.sort_values("datetime")

print("\nDate Rnage:")
print("Start:", df["datetime"].min())
print("End:", df["datetime"].max())

#3. Column Names

pollutant_targets = [
    "pm25",
    "pm10",
    "so2",
    "co",
    "no2",
    "o3",
    "nh3",
    "no",
    "nox"
]

wind_targets = [
    "wind_u",
    "wind_v"
]

possible_targets = pollutant_targets + wind_targets

weather_features = [
    "temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "wind_to_direction",
    "wind_u",
    "wind_v",
    "weather_gap_hours"
]

time_features = [
    "hour",
    "day",
    "month",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos"
]

location_features = [
    "latitude",
    "longitude"
]

pollutant_features = [
    "pm25",
    "pm10",
    "so2",
    "co",
    "no2",
    "o3",
    "nh3",
    "no",
    "nox"
]

categorical_features_possible = [
    "station_name",
    "weather_station_name"
]

#Converting expected numeric columns

expected_numeric_columns = (
    pollutant_targets
    + wind_targets
    + weather_features
    + location_features
    + time_features
)

for col in expected_numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        print(f"Column not found, skipping numeric conversion: {col}")

#5. Helper Functions:

#a.
def remove_duplicates_keep_order(items):
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

#b.
def usable_numeric_columns(columns, dataframe, target):

    usable = []
    for col in columns:
        
        if col not in dataframe.columns:
            continue
        
        if col in dataframe.columns:
            continue

        missing_percent = dataframe[col].isna().mean() * 100

        if missing_percent > MAX_FEATURE_MISSING_PERCENT:
             print(f"Skipping numeric feature {col}: {missing_percent:.2f}% missing.")
             continue

        if dataframe[col].notna().sum() == 0:
            print(f"Skipping numeric feature {col}: all values missing.")
            continue

        usable.append(col)
    
    return usable

#c.
def usable_categorical_columns(columns, dataframe):
    usable = []

    for col in columns:
        if col not in dataframe.columns:
            print(f"Categorical column not found, skipping: {col}")
            continue

        missing_percent = dataframe[col].isna().mean() * 100

        if missing_percent > 95:
            print(f"Skipping categorical feature {col}: {missing_percent:.2f}% missing.")
            continue

        usable.append(col)

    return usable

#d.
def get_features_for_target(target, dataframe):

    if target in pollutant_targets:
        numeric_pool = (
            weather_features
            + time_features
            +location_features
            +pollutant_features
        )
        leakage_cols = [
            target,
            "aqi"
        ]
    elif target in wind_targets:
        numeric_pool = (
            [
                "temperature",
                "humidity",
                "rainfall",
                "weather_gap_hours"
            ]
            + time_features
            + location_features
        )

        leakage_cols = [
            "wind_speed",
            "wind_direction",
            "wind_to_direction",
            "wind_u",
            "wind_v"
        ]

    else:
        numeric_pool = (
            weather_features
            + time_features
            + location_features
            + pollutant_features
        )

        leakage_cols = [target, "aqi"]
    
    numeric_pool = remove_duplicates_keep_order(numeric_pool)

    numeric_pool = [
        col for col in numeric_pool
        if col not in leakage_cols
    ]

    numeric_features = usable_numeric_columns(
        columns = numeric_pool,
        dataframe = dataframe,
        target = target
    )

    categorical_features = usable_categorical_columns(
        columns=categorical_features_possible,
        dataframe=dataframe
    )

    return numeric_features, categorical_features

#e. 
def chronological_train_test_split(model_df, test_size):

    model_df = model_df.copy()

    if "datetime" in model_df.columns:
        model_df = model_df.sort_values("datetime")
    
    split_index = int(len(model_df) * (1 - test_size))

    train_df = model_df[:split_index].copy()
    test_df = model_df[split_index:].copy()

    return train_df, test_df

#f.

def build_preprocessor(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline(
            steps = [
                ("imputer", SimpleImputer(strategy = "median"))
            ]
        )
        transformers.append("num", numeric_transformer, numeric_features)

    if categorical_features:
        categorical_transformer = Pipeline(
            steps = [
                ("imputer", SimpleImputer(strategy = "most_frequent"))
                ("onehot", OneHotEncoder(handle_unknown = "ignore"))
            ]
        )
        transformers.append("cat", categorical_transformer, categorical_features)
    
    if not transformers:
        raise RuntimeError("No usable features available for preprocessing.")
    
    preprocessor = ColumnTransformer(
        transformers = transformers,
        remainder = "drop"
    )

    return preprocessor

#6. Selecting Valid Targets

print("\nAvailable dataframe columns:")
print(df.columns.tolist())

print("\nChecking possible targets:")

for target in possible_targets:
    if target in df.columns:
        print(
            target,
            "| valid rows =",
            df[target].notna().sum(),
            "| missing % =",
            round(df[target].isna().mean() * 100, 2)
        )
    else:
        print(target, "| not found")

targets = []

print("\n" + "=" * 80)
print("TARGET SELECTION")
print("=" * 80)

for target in possible_targets:
    if target not in df.columns:
        print(f"Skipping {target}: column not found")
        continue

    valid_rows = df[target].notna().sum()
    missing_percent = df[target].isna().mean()*100

    if valid_rows < MIN_TARGET_ROWS:
        print(
            f"Skipping {target}: only {valid_rows} valid rows "
            f"({missing_percent:.2f}% missing)."
        )
        continue
    
    targets.append(target)

print("\nSelected targets:")
print(targets)

if not targets:
    raise RuntimeError("No valid targets available for ML1 training.")

#7. Train Final ML-1 model

all_metrics = []
all_predictions = []
all_feature_importance = []
feature_config = {}

for target in targets:
    print("\n" + "="*80)
    print(f"TRAINING FINAL ML-1 MODEL FOR TARGET: {target}")
    print("="*80)

    numeric_features, categorical_features = get_features_for_target(
        target = target,
        dataframe = df
    )

    selected_features = numeric_features + categorical_features

    print("\nNumeric Features used:")
    print(numeric_features)

    print("\nCategorical Features used:")
    print(categorical_features)

    if len(selected_features) < 3:
        print(f"Skipping {target}: fewer than 3 usable features.")
        continue

    model_df = df.dropna(subset = [target]).copy()

    if len(model_df) < MIN_TARGET_ROWS:
        print(f"Skipping {target}: not enough rows after target cleanup.")
        continue

    train_df, test_df = chronological_train_test_split(
        model_df = model_df,
        test_size = TEST_SIZE
    )

    if len(train_df) < 20 or len(test_df) < 5:
        print(f"Skipping {target}: train/test split too small")
        continue

    X_train = train_df[selected_features]
    y_train = train_df[target]

    X_test = test_df[selected_features]
    y_test = test_df[target]

    print("\nTraining rows:", len(X_train))
    print("\nTesting rows:", len(y_train))

    preprocessor = build_preprocessor(
        numeric_features = numeric_features,
        categorical_features = categorical_features
    )

    model = ExtraTreesRegressor(
        n_estimators = 500,
        random_state = RANDOM_STATE,
        min_samples_leaf = 2,
        n_jobs = -1
    )

    pipeline = Pipeline(
        steps = [
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    if len(y_test) > 1:
        r2 = r2_score(y_test, y_pred)

    else:
        r2 = np.nan

    print("\nPerformance:")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))

    model_file = MODELS_DIR / f"final_ml1_{target}_model.joblib"
    joblib.dump(pipeline, model_file)

    print("\nSaved Model:")
    print(model_file)

    feature_config[target] = {
        "numeric_features": numeric_features,
        "categorical_features" : categorical_features,
        "selected_features" : selected_features,
        "target" : target,
        "model_file" : str(model_file)
    }

    prototype_train_count = None
    prototype_test_count = None

    if "is_prototype_weather_match" in train_df.columns:
        prototype_train_count = int(train_df["is_prototype_weather_match"].sum())

    if "is_prototype_weather_match" in test_df.columns:
        prototype_test_count = int(test_df["is_prototype_weather_match"].sum())

    all_metrics.append(
        {
            "target": target,
            "model": "ExtraTreesRegressor",
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "feature_count": len(selected_features),
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "target_missing_percent": df[target].isna().mean() * 100,
            "prototype_weather_rows_train": prototype_train_count,
            "prototype_weather_rows_test": prototype_test_count,
            "model_file": str(model_file),
            "features_used": ", ".join(selected_features)
        }
    )

    prediction_df = pd.DataFrame()

    if "datetime" in test_df.columns:
        prediction_df["datetime"] = test_df["datetime"].values
    
    if "station_name" in test_df.columns:
        prediction_df["station_name"] = test_df["station_name"].values
    
    prediction_df["target"] = target
    prediction_df["actual"] = y_test.values
    prediction_df["predicted"] = y_pred
    prediction_df["error"] = prediction_df["actual"] - prediction_df["predicted"]
    prediction_df["absolute_error"] = prediction_df["error"].abs()

    if "weather_merge_mode" in test_df.columns:
        prediction_df["weather_merge_mode"] = test_df["weather_merge_mode"].values

    if "is_prototype_weather_match" in test_df.columns:
        prediction_df["is_prototype_weather_match"] = test_df["is_prototype_weather_match"].values

    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_

        importance_df = pd.DataFrame(
            {
                "target" : target,
                "feature" : feature_names,
                "importance" : importances
            }
        )
        
        importance_df = importance_df.sort_values(
            "importance",
            ascending = False
        )

        all_feature_importance.append(importance_df)

        print("\nTop 15 features:")
        print(importance_df.head(15))

    except Exception as error:
        print("Feature importance extraction failed.")
        print("Reason:", error)

#8. Save Final Report

metrics_df = pd.DataFrame

if metrics_df.empty:
    raise RuntimeError("No model was trained successfully")

metrics_df.to_csv(METRICS_FILE, index = False)

print("\n" + "=" * 80)
print("SAVED FINAL ML1 METRICS")
print("=" * 80)
print(METRICS_FILE)
print(metrics_df)

if all_predictions:
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv(PREDICTIONS_FILE, index=False)

    print("\nSaved final ML1 predictions:")
    print(PREDICTIONS_FILE)

if all_feature_importance:
    feature_importance_df = pd.concat(
        all_feature_importance,
        ignore_index=True
    )

    feature_importance_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    print("\nSaved final ML1 feature importance:")
    print(FEATURE_IMPORTANCE_FILE)

with open(FEATURE_CONFIG_FILE, "w", encoding="utf-8") as f:
    json.dump(feature_config, f, indent=4)

print("\nSaved final ML1 feature configuration:")
print(FEATURE_CONFIG_FILE)

# 9. Final summary

print("\n" + "=" * 80)
print("FINAL ML1 TRAINING COMPLETED")
print("=" * 80)

print("\nModels trained for targets:")
print(metrics_df["target"].tolist())

print("\nModel files saved in:")
print(MODELS_DIR)

print("\nReports saved in:")
print(REPORTS_DIR)







