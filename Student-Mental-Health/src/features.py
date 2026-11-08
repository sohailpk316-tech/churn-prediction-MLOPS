"""Shared cleaning / feature-engineering code.

Used by preprocess.py (training time) AND predict.py (new students), so the
exact same rules are applied in both places.

Two kinds of steps:
  * clean_data / engineer_features  - stateless row-wise rules (safe before the split)
  * build_preprocessor              - learns statistics (median, scaling, one-hot
                                      categories) -> must be fit on TRAIN only
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and fix inconsistent text (whitespace, casing, spelling variants)."""
    df = df.copy()
    if "Timestamp" in df.columns:                 # raw survey headers -> short names
        df = df.rename(columns=config.RENAME_MAP)

    df["course"] = (df["course"].astype(str).str.strip().str.lower()
                    .replace(config.COURSE_ALIASES))
    df["year_of_study"] = df["year_of_study"].astype(str).str.strip().str.lower()
    df["cgpa"] = df["cgpa"].astype(str).str.strip()
    for col in ["gender", "marital_status", "anxiety", "panic_attack",
                "sought_treatment", "depression"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    return df.drop(columns=[c for c in config.DROP_COLUMNS if c in df.columns])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric/ordinal/binary versions of the survey answers."""
    df = df.copy()
    df["year_of_study_num"] = df["year_of_study"].str.extract(r"(\d+)")[0].astype(float)
    df["cgpa_ordinal"] = df["cgpa"].map(config.CGPA_ORDER)
    df["gender_bin"] = df["gender"].map(config.GENDER_MAP)
    for col in ["marital_status", "anxiety", "panic_attack", "sought_treatment"]:
        df[col + "_bin"] = df[col].map(config.YES_NO)
    return df


def prepare_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw survey frame -> cleaned frame with engineered features (+ target if present)."""
    df = engineer_features(clean_data(raw))
    if config.TARGET in df.columns:
        df["target"] = df[config.TARGET].map(config.YES_NO)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Unfitted preprocessing.

    * numeric + binary columns: median-impute (covers the 1 missing age) then standardise.
      The 0/1 columns are scaled too: regularised models (Logistic Regression, SVM, KNN)
      are sensitive to feature scale, and this scored best in train-only cross-validation.
    * course: one-hot encoded (unseen courses at prediction time are ignored safely).
    """
    scaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", scaled, config.NUMERIC_FEATURES + config.BINARY_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])
