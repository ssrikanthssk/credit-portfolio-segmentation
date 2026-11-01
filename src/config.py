"""Project-wide settings: paths, random seed and feature lists."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "loans_granting_model_dataset.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEANED_PATH = PROCESSED_DIR / "cleaned_loans.csv"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

RANDOM_STATE = 42
SAMPLE_SIZE = 20_000          # for t-SNE, UMAP, hierarchical clustering
OUTLIER_QUANTILE = 0.99       # cap income and DTI at this percentile
RARE_PURPOSE_THRESHOLD = 0.01  # purposes below 1% are grouped as "other"

ID_COL = "id"
TARGET_COL = "Default"        # never used for clustering

# Columns loaded from the CSV (title, desc, experience_c, zip_code are dropped)
USE_COLS = [
    "id", "issue_d", "revenue", "dti_n", "loan_amnt", "fico_n",
    "emp_length", "purpose", "home_ownership_n", "addr_state", "Default",
]

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}  # "NI" (no information) is not in the map, so it becomes NaN

NUMERIC_FEATURES = [
    "revenue_log", "dti_n", "loan_amnt", "fico_n",
    "emp_length_years", "issue_year",
]
CATEGORICAL_FEATURES = ["purpose_grouped", "home_ownership_n"]
