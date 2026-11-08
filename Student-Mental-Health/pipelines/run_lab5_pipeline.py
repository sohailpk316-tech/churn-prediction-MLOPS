r"""Lab 5 - End-to-end ML pipeline.

Run from the project root:   python .\pipelines\run_lab5_pipeline.py

Builds ONE reusable scikit-learn Pipeline (preprocessing + model), tunes it,
saves it, and proves the saved pipeline works:

  Stage 1  validate raw data        (src/validate_data.py)
  Stage 2  preprocess + split       (src/preprocess.py, only if train/test are missing)
  Stage 3  build the Pipeline       (imputer -> scaler -> one-hot -> Logistic Regression)
  Stage 4  tune with GridSearchCV   (5-fold stratified CV, scored by F1, train data only)
  Stage 5  evaluate on the test set
  Stage 6  save pipeline, reload it and check predictions are identical
  Stage 7  predict for a raw, un-processed new student

Outputs: models/lab5_pipeline.pkl, outputs/lab5_pipeline_report.json
"""
import json
import subprocess
import sys
import time
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from _common import ROOT, config, ensure_data, load_train_test, score
from features import build_preprocessor, prepare_dataframe
from predict import SAMPLE_STUDENT

PIPELINE_PATH = config.MODELS_DIR / "lab5_pipeline.pkl"
REPORT_PATH = config.OUTPUTS_DIR / "lab5_pipeline_report.json"
timings = {}


def stage(title):
    print(f"\n[INFO] ---> {title}")
    return time.perf_counter()


def done(name, start):
    timings[name] = round(time.perf_counter() - start, 2)


def main():
    print(f"[INFO] ===== Lab 5: End-to-end pipeline - {datetime.now():%Y-%m-%d %H:%M:%S} =====")

    # ---- Stage 1: validate raw data ---------------------------------------
    t = stage("Stage 1: validate raw data")
    proc = subprocess.run([sys.executable, str(ROOT / "src" / "validate_data.py")],
                          cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout.rstrip())
    if proc.returncode != 0:
        print(proc.stderr.rstrip())
        sys.exit("[ERROR] Raw data validation failed - pipeline stopped.")
    done("validate_data", t)

    # ---- Stage 2: preprocess + split --------------------------------------
    t = stage("Stage 2: preprocess and train/test split")
    ensure_data()
    X_train, y_train, X_test, y_test = load_train_test()
    print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")
    done("preprocess", t)

    # ---- Stage 3: build the pipeline --------------------------------------
    t = stage("Stage 3: build the sklearn Pipeline")
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
    ])
    for name, step in pipe.steps:
        print(f"  - {name}: {type(step).__name__}")
    done("build_pipeline", t)

    # ---- Stage 4: tune ----------------------------------------------------
    t = stage("Stage 4: hyper-parameter tuning (GridSearchCV, CV on train only)")
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    grid = {"model__C": [0.01, 0.1, 1, 10, 100], "model__class_weight": [None, "balanced"]}
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=1)
    search.fit(X_train, y_train)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print(f"[INFO] Best CV F1 = {search.best_score_:.3f} | params = {best_params}")
    done("tuning", t)

    # ---- Stage 5: evaluate ------------------------------------------------
    t = stage("Stage 5: evaluate on the held-out test set")
    best = search.best_estimator_
    test_metrics = score(best, X_test, y_test)
    for k, v in test_metrics.items():
        print(f"  {k:10s}: {v:.3f}")
    done("evaluate", t)

    # ---- Stage 6: save, reload, verify ------------------------------------
    t = stage("Stage 6: save the pipeline, reload it and verify")
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best, PIPELINE_PATH)
    reloaded = joblib.load(PIPELINE_PATH)
    same = bool((reloaded.predict_proba(X_test) == best.predict_proba(X_test)).all())
    print(f"[INFO] Saved {PIPELINE_PATH.relative_to(ROOT)} | reloaded predictions identical: {same}")
    if not same:
        sys.exit("[ERROR] Reloaded pipeline gives different predictions.")
    done("save_and_verify", t)

    # ---- Stage 7: predict from raw input ----------------------------------
    t = stage("Stage 7: predict for a raw new student")
    X_new = prepare_dataframe(pd.DataFrame([SAMPLE_STUDENT]))[config.FEATURES]
    proba = float(reloaded.predict_proba(X_new)[0, 1])
    label = "Depression" if int(reloaded.predict(X_new)[0]) == 1 else "No Depression"
    print(f"[INFO] Sample student -> {label} (probability {proba:.2%})")
    done("predict", t)

    # ---- report -----------------------------------------------------------
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline_steps": [f"{n}: {type(s).__name__}" for n, s in best.steps],
        "best_params": best_params,
        "cv_f1_train": round(float(search.best_score_), 4),
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "reload_check_passed": same,
        "sample_prediction": {"class": label, "probability": round(proba, 4)},
        "stage_seconds": timings,
    }, indent=4))
    print(f"\n[SUCCESS] Lab 5 complete. Saved {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
