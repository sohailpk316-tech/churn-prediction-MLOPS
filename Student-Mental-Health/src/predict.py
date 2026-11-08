"""Step 6 - predict depression risk for a NEW student with the saved model.

Usage:
  python src/predict.py                      # built-in sample student
  python src/predict.py --json '{"gender": "Male", "age": 21, "course": "Law",
        "year_of_study": "year 3", "cgpa": "2.50 - 2.99", "marital_status": "No",
        "anxiety": "Yes", "panic_attack": "No", "sought_treatment": "No"}'

Raw survey-style answers go in; the same cleaning rules used in training are applied.
"""
import argparse
import json

import joblib
import pandas as pd

import config
from features import prepare_dataframe

SAMPLE_STUDENT = {          # same hypothetical student as the notebook (Section 11)
    "gender": "Female", "age": 20, "course": "Engineering", "year_of_study": "year 2",
    "cgpa": "3.00 - 3.49", "marital_status": "No", "anxiety": "Yes",
    "panic_attack": "Yes", "sought_treatment": "No",
}


def predict_student(student: dict, model=None) -> dict:
    model = model or joblib.load(config.BEST_MODEL_PATH)
    df = prepare_dataframe(pd.DataFrame([student]))
    if df[config.FEATURES].drop(columns=["age"]).isna().any().any():
        raise ValueError("Invalid input: check gender/yes-no answers, year_of_study "
                         "(e.g. 'year 2') and cgpa band (e.g. '3.00 - 3.49').")
    X = df[config.FEATURES]
    proba = float(model.predict_proba(X)[0, 1])
    label = int(model.predict(X)[0])
    return {"input": student,
            "predicted_class": "Depression" if label == 1 else "No Depression",
            "probability_of_depression": round(proba, 4)}


def main():
    parser = argparse.ArgumentParser(description="Predict depression risk for a new student.")
    parser.add_argument("--json", help="Student profile as a JSON string")
    args = parser.parse_args()
    student = json.loads(args.json) if args.json else SAMPLE_STUDENT

    result = predict_student(student)
    print(f"[INFO] Predicted class: {result['predicted_class']}")
    print(f"[INFO] Probability of depression: {result['probability_of_depression']:.2%}")

    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(config.OUTPUTS_DIR / "sample_prediction.json", "w") as f:
        json.dump(result, f, indent=4)
    print("[SUCCESS] Saved outputs/sample_prediction.json")


if __name__ == "__main__":
    main()
