"""Step 1 - validate the RAW survey data before any processing.

Plain pandas checks (no extra dependency). Writes artifacts/data_validation_report.json
and exits with code 1 if a hard check fails, so the pipeline stops early.
"""
import json
import sys

import pandas as pd

import config


def validate_raw_data(df: pd.DataFrame) -> dict:
    errors, warnings = [], []

    missing_cols = [c for c in config.RAW_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in config.RAW_COLUMNS]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    if extra_cols:
        errors.append(f"Unexpected columns: {extra_cols}")

    if not missing_cols:
        yes_no_cols = ["Marital status", "Do you have Depression?", "Do you have Anxiety?",
                       "Do you have Panic attack?", "Did you seek any specialist for a treatment?"]
        for col in yes_no_cols:
            bad = sorted(set(df[col].astype(str).str.strip()) - {"Yes", "No"})
            if bad:
                errors.append(f"'{col}' has values other than Yes/No: {bad}")

        bad = sorted(set(df["Choose your gender"].astype(str).str.strip()) - {"Male", "Female"})
        if bad:
            errors.append(f"Unexpected gender values: {bad}")

        cgpa_bad = sorted(set(df["What is your CGPA?"].astype(str).str.strip()) - set(config.CGPA_ORDER))
        if cgpa_bad:
            errors.append(f"Unknown CGPA bands: {cgpa_bad}")

        year_vals = df["Your current year of Study"].astype(str).str.strip().str.lower()
        if not year_vals.str.fullmatch(r"year [1-9]").all():
            errors.append("Unexpected 'year of study' format")

        age = df["Age"].dropna()
        if ((age < 15) | (age > 60)).any():
            errors.append("Age values outside the plausible range 15-60")

        # Non-fatal findings
        n_missing_age = int(df["Age"].isna().sum())
        if n_missing_age:
            warnings.append(f"{n_missing_age} missing Age value(s) - will be median-imputed")
        if df.duplicated().sum():
            warnings.append(f"{int(df.duplicated().sum())} duplicate row(s)")
        raw_courses = df["What is your course?"].nunique()
        clean_courses = (df["What is your course?"].astype(str).str.strip().str.lower()
                         .replace(config.COURSE_ALIASES).nunique())
        warnings.append(f"Course spellings: {raw_courses} raw -> {clean_courses} after cleaning")

    return {
        "validation_status": "PASSED" if not errors else "FAILED",
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v},
        "errors": errors,
        "warnings": warnings,
    }


def main() -> bool:
    print("[INFO] Validating raw data...")
    if not config.RAW_DATA_PATH.exists():
        print(f"[ERROR] Dataset not found: {config.RAW_DATA_PATH}")
        return False
    report = validate_raw_data(pd.read_csv(config.RAW_DATA_PATH))

    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(config.ARTIFACTS_DIR / "data_validation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    for w in report["warnings"]:
        print(f"[WARN] {w}")
    if report["errors"]:
        print("[ERROR] Data validation FAILED:")
        for e in report["errors"]:
            print(f"  - {e}")
        return False
    print(f"[SUCCESS] Data validation PASSED ({report['rows']} rows, {report['columns']} columns).")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
