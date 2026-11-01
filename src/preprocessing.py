"""Data loading, cleaning and feature preparation."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config


def load_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV with only the columns we need."""
    return pd.read_csv(path, usecols=config.USE_COLS, low_memory=False)


def cap_outliers(df: pd.DataFrame, columns, quantile: float) -> pd.DataFrame:
    """Cap the upper tail of the given columns at a quantile."""
    for col in columns:
        df[col] = df[col].clip(upper=df[col].quantile(quantile))
    return df


def group_rare_categories(series: pd.Series, threshold: float) -> pd.Series:
    """Replace categories rarer than the threshold with 'other'."""
    share = series.value_counts(normalize=True)
    rare = share[share < threshold].index
    return series.where(~series.isin(rare), "other")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw loans data and create the new features."""
    df = df.drop_duplicates().copy()

    # Dates -> proper datetime plus time-based features
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["issue_year"] = df["issue_d"].dt.year
    df["issue_month"] = df["issue_d"].dt.month

    # Employment length -> number of years; "NI" becomes NaN, then median
    df["emp_length_years"] = df["emp_length"].map(config.EMP_LENGTH_MAP)
    df["emp_length_missing"] = df["emp_length_years"].isna().astype(int)
    df["emp_length_years"] = df["emp_length_years"].fillna(
        df["emp_length_years"].median()
    )

    # Outliers: cap income and DTI, then log-transform the skewed income
    df = cap_outliers(df, ["revenue", "dti_n"], config.OUTLIER_QUANTILE)
    df["revenue_log"] = np.log1p(df["revenue"])

    # Group rare loan purposes (original column kept for default analysis)
    df["purpose_grouped"] = group_rare_categories(
        df["purpose"], config.RARE_PURPOSE_THRESHOLD
    )
    return df.reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame):
    """Encode categoricals and scale features. Default and id are excluded."""
    numeric = df[config.NUMERIC_FEATURES]
    categorical = pd.get_dummies(df[config.CATEGORICAL_FEATURES], dtype=int)
    features = pd.concat([numeric, categorical], axis=1)

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=df.index,
    )
    return scaled, scaler


def stratified_sample(df: pd.DataFrame, n: int = config.SAMPLE_SIZE):
    """Sample n rows, keeping the same default ratio as the full data."""
    frac = min(1.0, n / len(df))
    return df.groupby(config.TARGET_COL, group_keys=False).sample(
        frac=frac, random_state=config.RANDOM_STATE
    )
