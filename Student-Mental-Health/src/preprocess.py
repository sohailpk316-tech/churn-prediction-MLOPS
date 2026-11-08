"""Step 2 - clean the raw data, engineer features, and make a stratified train/test split.

Outputs (data/processed/):
  student_mental_health_clean.csv, train.csv, test.csv, dataset_metadata.json

The learned transformations (median imputation, scaling, one-hot) are NOT applied here;
they live inside the model Pipeline (see features.build_preprocessor) so that they are
fit on training folds only - no data leakage, including during cross-validation.
"""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from features import prepare_dataframe


def run_preprocessing():
    print("[INFO] Starting preprocessing...")
    raw = pd.read_csv(config.RAW_DATA_PATH)
    df = prepare_dataframe(raw)
    print(f"[INFO] Cleaned data: {df.shape[0]} rows | target balance: "
          f"{df['target'].value_counts().to_dict()}")

    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=df["target"]
    )

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_DATA_PATH, index=False)
    train_df.to_csv(config.TRAIN_PATH, index=False)
    test_df.to_csv(config.TEST_PATH, index=False)

    metadata = {
        "dataset_name": "Student Mental Health Survey",
        "problem": "Binary classification - does the student report Depression?",
        "target": config.TARGET,
        "n_rows": int(len(df)),
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "class_balance_train": {str(k): int(v) for k, v in train_df["target"].value_counts().items()},
        "class_balance_test": {str(k): int(v) for k, v in test_df["target"].value_counts().items()},
        "numerical_features": config.NUMERIC_FEATURES,
        "binary_features": config.BINARY_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[INFO] Train: {train_df.shape} | Test: {test_df.shape}")
    print("[SUCCESS] Preprocessing completed. Files saved to data/processed/")


if __name__ == "__main__":
    run_preprocessing()
