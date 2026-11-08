"""Step 5 - one-time evaluation on the held-out TEST set.

Outputs:
  outputs/test_results.csv               all models (baseline + tuned) on the test set
  outputs/metrics.json                   final model metrics + confusion matrix
  outputs/classification_report.txt
  outputs/false_negatives.csv / false_positives.csv   error analysis (like the reference project)
  artifacts/confusion_matrix.png, roc_curve.png, feature_importance.png
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

import config

READABLE_COLS = ["gender", "age", "course", "year_of_study", "cgpa",
                 "marital_status", "anxiety", "panic_attack", "sought_treatment"]


def score_model(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, proba),
    }


def run_evaluation():
    warnings.filterwarnings("ignore")
    print("[INFO] Starting evaluation on the held-out test set...")
    test_df = pd.read_csv(config.TEST_PATH)
    X_test, y_test = test_df[config.FEATURES], test_df["target"]

    best_model = joblib.load(config.BEST_MODEL_PATH)
    all_models = joblib.load(config.MODELS_DIR / "all_models.pkl")
    with open(config.OUTPUTS_DIR / "training_summary.json") as f:
        training = json.load(f)
    best_name = training["selected_model"]

    # ---- all models on the test set -----------------------------------------
    rows = [{"Model": n, **score_model(m, X_test, y_test)} for n, m in all_models.items()]
    results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    results.to_csv(config.OUTPUTS_DIR / "test_results.csv", index=False)
    print("\n--- Test-set results (all models) ---")
    print(results.round(3).to_string(index=False))

    # ---- final (CV-selected) model ------------------------------------------
    pred = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {"selected_model": best_name,
               "cv_f1_train": training["tuning"][best_name.replace(" (tuned)", "")]["best_cv_f1"],
               "test_metrics": {k: round(float(v), 4) for k, v in
                                score_model(best_model, X_test, y_test).items()},
               "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
               "n_test_rows": int(len(y_test)),
               "note": "Test set has only 21 students; metrics are noisy - use as a rough guide."}
    with open(config.OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(y_test, pred, target_names=["No Depression", "Depression"])
    (config.OUTPUTS_DIR / "classification_report.txt").write_text(
        f"Model: {best_name}\n\n{report}")
    print(f"\n--- Final model: {best_name} ---")
    print(report)

    # ---- plots --------------------------------------------------------------
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No Depression", "Depression"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.savefig(config.ARTIFACTS_DIR / "confusion_matrix.png", bbox_inches="tight", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, name=best_name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
    ax.legend()
    ax.set_title("ROC Curve - final model")
    fig.savefig(config.ARTIFACTS_DIR / "roc_curve.png", bbox_inches="tight", dpi=120)
    plt.close(fig)

    # feature importance / coefficients
    prep, clf = best_model.named_steps["prep"], best_model.named_steps["model"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    if hasattr(clf, "feature_importances_"):
        values, label = clf.feature_importances_, "Importance"
    elif hasattr(clf, "coef_"):
        values, label = clf.coef_[0], "Coefficient (standardised features)"
    else:
        values = None
    if values is not None:
        s = pd.Series(values, index=names)
        top = s.reindex(s.abs().sort_values(ascending=False).head(12).index).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5.5))
        top.plot(kind="barh", color=["#c0392b" if v < 0 else "teal" for v in top], ax=ax)
        ax.set_title(f"Top 12 features - {best_name}")
        ax.set_xlabel(label)
        fig.savefig(config.ARTIFACTS_DIR / "feature_importance.png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        print("--- Top features ---")
        print(top.iloc[::-1].round(3).to_string())

    # ---- error analysis -----------------------------------------------------
    errors = test_df[READABLE_COLS].copy()
    errors["Actual_Depression"] = y_test.values
    errors["Predicted_Depression"] = pred
    errors["Predicted_Probability"] = proba.round(3)
    errors[(errors.Actual_Depression == 1) & (errors.Predicted_Depression == 0)] \
        .to_csv(config.OUTPUTS_DIR / "false_negatives.csv", index=False)
    errors[(errors.Actual_Depression == 0) & (errors.Predicted_Depression == 1)] \
        .to_csv(config.OUTPUTS_DIR / "false_positives.csv", index=False)
    print(f"\n[INFO] False negatives: {fn} | False positives: {fp} (saved to outputs/)")
    print("[SUCCESS] Evaluation complete.")


if __name__ == "__main__":
    run_evaluation()
