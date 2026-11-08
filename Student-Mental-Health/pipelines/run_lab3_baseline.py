r"""Lab 3 - Baseline models.

Run from the project root:   python .\pipelines\run_lab3_baseline.py

Trains two simple baselines on the training split and scores them on the
held-out test split:
  1. DummyClassifier (most frequent class)  - the "do nothing" floor
  2. Logistic Regression with default settings (full preprocessing pipeline)
Any later model has to beat these to be worth keeping.
Saves outputs/lab3_baseline_metrics.json and models/baseline_model.pkl.
"""
import json

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from _common import ROOT, config, load_train_test, score
from features import build_preprocessor


def main():
    print("[INFO] ===== Lab 3: Baseline models =====")
    X_train, y_train, X_test, y_test = load_train_test()
    print(f"[INFO] Train rows: {len(X_train)} | Test rows: {len(X_test)} | "
          f"Depression rate (train): {y_train.mean():.2%}")

    baselines = {
        "Dummy (most frequent)": Pipeline([
            ("prep", build_preprocessor()),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression (default)": Pipeline([
            ("prep", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
        ]),
    }

    rows = []
    for name, pipe in baselines.items():
        pipe.fit(X_train, y_train)
        rows.append({"model": name, **score(pipe, X_test, y_test)})

    results = pd.DataFrame(rows)
    print("\n--- Baseline results (test set) ---")
    print(results.round(3).to_string(index=False))

    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    (config.OUTPUTS_DIR / "lab3_baseline_metrics.json").write_text(
        json.dumps(results.round(4).to_dict(orient="records"), indent=4))
    joblib.dump(baselines["Logistic Regression (default)"],
                config.MODELS_DIR / "baseline_model.pkl")

    print("\n[SUCCESS] Lab 3 complete.")
    print(f"[INFO] Saved {config.OUTPUTS_DIR.relative_to(ROOT) / 'lab3_baseline_metrics.json'} "
          f"and {config.MODELS_DIR.relative_to(ROOT) / 'baseline_model.pkl'}")


if __name__ == "__main__":
    main()
