"""Step 4 - model comparison, tuning and selection (training data only).

1. Compare 7 algorithms with 5-fold stratified CV (F1) on the TRAIN set.
2. Tune the top-2 with GridSearchCV.
3. Pick the final model by best cross-validated F1 (NOT by test score, so the
   test set stays untouched until evaluate.py).

Every model is a full sklearn Pipeline (preprocessor + estimator), so imputation,
scaling and one-hot encoding are re-fit inside each CV fold -> no leakage.
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config
from features import build_preprocessor

RS = config.RANDOM_STATE

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RS),
    "Decision Tree": DecisionTreeClassifier(random_state=RS),
    "Random Forest": RandomForestClassifier(random_state=RS),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SVM (RBF kernel)": SVC(probability=True, random_state=RS),
    "Naive Bayes": GaussianNB(),
}

# Hyper-parameter grids ("model__" = step name inside the Pipeline)
PARAM_GRIDS = {
    "Random Forest": {"model__n_estimators": [100, 200, 300],
                      "model__max_depth": [None, 3, 5, 8],
                      "model__min_samples_leaf": [1, 2, 4]},
    "Gradient Boosting": {"model__n_estimators": [50, 100, 200],
                          "model__learning_rate": [0.01, 0.1, 0.2],
                          "model__max_depth": [2, 3, 4]},
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
    "SVM (RBF kernel)": {"model__C": [0.1, 1, 10], "model__gamma": ["scale", "auto"]},
    "K-Nearest Neighbors": {"model__n_neighbors": [3, 5, 7, 9, 11],
                            "model__weights": ["uniform", "distance"]},
    "Decision Tree": {"model__max_depth": [2, 3, 4, 5, None],
                      "model__min_samples_leaf": [1, 2, 4]},
    "Naive Bayes": {},
}


def make_pipeline(estimator):
    return Pipeline([("prep", build_preprocessor()), ("model", clone(estimator))])


def run_training():
    warnings.filterwarnings("ignore")
    print("[INFO] Starting model training...")
    train_df = pd.read_csv(config.TRAIN_PATH)
    X_train, y_train = train_df[config.FEATURES], train_df["target"]
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=RS)

    # ---- 1. Compare all models with cross-validation ------------------------
    print(f"\n[INFO] {config.CV_FOLDS}-fold stratified CV (F1) on {len(X_train)} training rows")
    cv_scores, baseline_fitted = {}, {}
    for name, est in MODELS.items():
        pipe = make_pipeline(est)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        cv_scores[name] = scores
        baseline_fitted[name] = pipe.fit(X_train, y_train)
        print(f"  {name:22s} mean F1 = {scores.mean():.3f}  std = {scores.std():.3f}")

    cv_table = (pd.DataFrame({"Model": list(cv_scores),
                              "CV_F1_mean": [s.mean() for s in cv_scores.values()],
                              "CV_F1_std": [s.std() for s in cv_scores.values()]})
                .sort_values("CV_F1_mean", ascending=False).reset_index(drop=True))
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    cv_table.to_csv(config.OUTPUTS_DIR / "cv_model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([cv_scores[m] for m in cv_table["Model"]], tick_labels=cv_table["Model"])
    ax.set_ylabel("F1 (5-fold CV)")
    ax.set_title("Model comparison - cross-validated F1")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.savefig(config.ARTIFACTS_DIR / "cv_model_comparison.png", bbox_inches="tight", dpi=120)
    plt.close(fig)

    # ---- 2. Tune the two best models ----------------------------------------
    top_2 = cv_table["Model"].head(2).tolist()
    print(f"\n[INFO] Shortlisted for tuning: {top_2}")
    tuned, tuning_info = {}, {}
    for name in top_2:
        gs = GridSearchCV(make_pipeline(MODELS[name]), PARAM_GRIDS[name],
                          cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train)
        tuned[name] = gs.best_estimator_
        best_params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        tuning_info[name] = {"best_cv_f1": float(gs.best_score_), "best_params": best_params}
        print(f"  {name}: best CV F1 = {gs.best_score_:.3f}  params = {best_params}")

    # ---- 3. Select the final model by CV F1 (never by test score) -----------
    best_name = max(tuning_info, key=lambda n: tuning_info[n]["best_cv_f1"])
    print(f"\n[INFO] Selected model: {best_name} (tuned)")

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(tuned[best_name], config.BEST_MODEL_PATH)
    all_models = dict(baseline_fitted)
    all_models.update({f"{n} (tuned)": m for n, m in tuned.items()})
    joblib.dump(all_models, config.MODELS_DIR / "all_models.pkl")

    with open(config.OUTPUTS_DIR / "training_summary.json", "w") as f:
        json.dump({"selected_model": best_name + " (tuned)",
                   "selection_criterion": "highest cross-validated F1 on training data",
                   "tuning": tuning_info,
                   "cv_folds": config.CV_FOLDS,
                   "n_train_rows": int(len(X_train))}, f, indent=4)
    print("[SUCCESS] Training complete. Model saved to models/best_model.pkl")


if __name__ == "__main__":
    run_training()
