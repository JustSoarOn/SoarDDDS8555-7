# ============================================================
# Assignment 7: PCA and Cluster Analysis of Wine Data
#
# assignment_07_01_inspect_wine_data
#
# Purpose:
#   Inspect the wine-clustering.csv data and save durable
#   report-ready artifacts for Assignment 7.
# ============================================================

from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLE_DIR = OUTPUT_DIR / "tables"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Load wine dataset
# ------------------------------------------------------------

wine = pd.read_csv(
    DATA_DIR / "wine-clustering.csv"
)


# ------------------------------------------------------------
# Basic dimensions
# ------------------------------------------------------------

print("WINE CLUSTER DATA")
print("=" * 60)

print(
    f"Rows:    {wine.shape[0]:,}"
)

print(
    f"Columns: {wine.shape[1]:,}"
)


# ------------------------------------------------------------
# Variable names and data types
# ------------------------------------------------------------

data_types = pd.DataFrame(
    {
        "Variable": wine.columns,
        "Data_Type": [
            str(dtype)
            for dtype in wine.dtypes
        ]
    }
)

data_types.to_csv(
    TABLE_DIR / "wine_data_types.csv",
    index=False
)


print("\nWINE CLUSTER VARIABLES")
print("=" * 60)

for column in wine.columns:
    print(
        f"{column}: {wine[column].dtype}"
    )


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

missing_values = pd.DataFrame(
    {
        "Variable": wine.columns,
        "Missing_Count": [
            int(wine[column].isna().sum())
            for column in wine.columns
        ]
    }
)

missing_values["Missing_Percentage"] = (
    missing_values["Missing_Count"]
    / len(wine)
    * 100
)

missing_values.to_csv(
    TABLE_DIR / "wine_missing_values.csv",
    index=False
)


print("\nMISSING VALUES")
print("=" * 60)

print(
    wine.isna().sum()
)


# ------------------------------------------------------------
# Duplicate observations
# ------------------------------------------------------------

duplicate_count = int(
    wine.duplicated().sum()
)

duplicate_summary = pd.DataFrame(
    {
        "Measure": [
            "Duplicate wine observations"
        ],
        "Count": [
            duplicate_count
        ]
    }
)

duplicate_summary.to_csv(
    TABLE_DIR / "wine_duplicate_summary.csv",
    index=False
)


print("\nDUPLICATES")
print("=" * 60)

print(
    "Duplicate wine observations:",
    duplicate_count
)


# ------------------------------------------------------------
# Descriptive statistics
# ------------------------------------------------------------

descriptive_statistics = wine.describe().T

descriptive_statistics.to_csv(
    TABLE_DIR / "wine_descriptive_statistics.csv"
)


print("\nDESCRIPTIVE STATISTICS")
print("=" * 60)

print(
    wine.describe()
)


# ------------------------------------------------------------
# First five observations
# ------------------------------------------------------------

first_five = wine.head(5)

first_five.to_csv(
    TABLE_DIR / "wine_first_five_observations.csv",
    index=False
)


print("\nFIRST FIVE WINE OBSERVATIONS")
print("=" * 60)

print(
    first_five
)


# ------------------------------------------------------------
# Correlation matrix
# ------------------------------------------------------------

correlation_matrix = wine.corr()

correlation_matrix.to_csv(
    TABLE_DIR / "wine_correlation_matrix.csv"
)


# ------------------------------------------------------------
# Final artifact confirmation
# ------------------------------------------------------------

print("\nREPORT ARTIFACTS SAVED")
print("=" * 60)

for artifact in sorted(
    TABLE_DIR.glob("wine_*.csv")
):
    print(
        artifact.name
    )


print(
    "\nassignment_07_01_inspect_wine_data completed successfully."
)
