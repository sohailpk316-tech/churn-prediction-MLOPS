"""Step 7 - sanity-check everything the pipeline produced.

Writes artifacts/output_validation_report.json; exits 1 if any check fails.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

import config


def validate_outputs() -> bool:
    print("[INFO] Validating pipeline outputs...")
    errors, checks = [], {}

    try:
        train = pd.read_csv(config.TRAIN_PATH)
        test = pd.read_csv(config.TEST_PATH)
        model = joblib.load(config.BEST_MODEL_PATH)
    except FileNotFoundError as e:
        print(f"[ERROR] Missing output: {e}")
        return False

    # Feature columns must be complete (age may be NaN: it is imputed inside the model)
    for name, df in [("train", train), ("test", test)]:
        non_age = [c for c in config.FEATURES if c != "age"]
        n_nan = int(df[non_age].isna().sum().sum())
        checks[f"{name}_missing_non_age_features"] = n_nan
        if n_nan:
            errors.append(f"{n_nan} NaN(s) in {name} features (excluding age)")

    # Split sanity
    if set(train.columns) != set(test.columns):
        errors.append("Train/test columns differ")
    total = len(train) + len(test)
    checks["train_rows"], checks["test_rows"] = int(len(train)), int(len(test))
    if total != 101:
        errors.append(f"Expected 101 rows in total, found {total}")
    gap = abs(train["target"].mean() - test["target"].mean())
    checks["class_ratio_gap_train_vs_test"] = round(float(gap), 3)
    if gap > 0.1:
        errors.append("Train/test class ratios differ by more than 10 points (stratification issue)")

    # Transformed matrix must be finite and consistent
    prep = model.named_steps["prep"]
    Xt_train, Xt_test = prep.transform(train[config.FEATURES]), prep.transform(test[config.FEATURES])
    checks["transformed_train_shape"] = list(Xt_train.shape)
    checks["transformed_test_shape"] = list(Xt_test.shape)
    if not (np.isfinite(Xt_train).all() and np.isfinite(Xt_test).all()):
        errors.append("Non-finite values after preprocessing")
    if Xt_train.shape[1] != Xt_test.shape[1]:
        errors.append("Train/test feature-count mismatch after preprocessing")

    # Model sanity
    proba = model.predict_proba(test[config.FEATURES])[:, 1]
    if not ((proba >= 0) & (proba <= 1)).all():
        errors.append("Predicted probabilities outside [0, 1]")

    # Expected output files
    expected = [config.OUTPUTS_DIR / n for n in
                ["cv_model_comparison.csv", "test_results.csv", "metrics.json",
                 "classification_report.txt", "false_negatives.csv", "false_positives.csv",
                 "sample_prediction.json", "training_summary.json"]]
    expected += [config.ARTIFACTS_DIR / n for n in
                 ["confusion_matrix.png", "roc_curve.png", "feature_importance.png",
                  "cv_model_comparison.png", "eda_target_distribution.png"]]
    missing = [str(p.relative_to(config.PROJECT_ROOT)) for p in expected if not p.exists()]
    if missing:
        errors.append(f"Missing output files: {missing}")

    report = {"validation_status": "PASSED" if not errors else "FAILED",
              "checks": checks, "errors": errors}
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(config.ARTIFACTS_DIR / "output_validation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    if errors:
        print("[ERROR] Output validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False
    print("[SUCCESS] Output validation PASSED.")
    return True


if __name__ == "__main__":
    sys.exit(0 if validate_outputs() else 1)
