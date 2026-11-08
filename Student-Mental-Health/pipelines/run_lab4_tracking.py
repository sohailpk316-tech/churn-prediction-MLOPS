r"""Lab 4 - Experiment tracking with MLflow.

Run from the project root:   python .\pipelines\run_lab4_tracking.py

Trains several models and logs every run to a local MLflow store
(mlflow.db + mlartifacts/): parameters, CV F1, test metrics, the fitted
model and a confusion-matrix plot. Afterwards run `mlflow ui` (see README)
and open http://127.0.0.1:5000 to compare runs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from _common import (EXPERIMENT_NAME, TRACKING_URI, config, load_train_test,
                     log_sklearn_model, score, setup_mlflow)
from features import build_preprocessor

RS = config.RANDOM_STATE

# (run name, estimator, params to log)
EXPERIMENTS = [
    ("logreg_C0.1", LogisticRegression(C=0.1, max_iter=1000, random_state=RS)),
    ("logreg_C1", LogisticRegression(C=1, max_iter=1000, random_state=RS)),
    ("logreg_C10", LogisticRegression(C=10, max_iter=1000, random_state=RS)),
    ("svm_rbf_C1", SVC(C=1, probability=True, random_state=RS)),
    ("decision_tree_d3", DecisionTreeClassifier(max_depth=3, random_state=RS)),
    ("random_forest_100", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RS)),
]


def main():
    print("[INFO] ===== Lab 4: Experiment tracking (MLflow) =====")
    mlflow = setup_mlflow()

    X_train, y_train, X_test, y_test = load_train_test()
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=RS)

    for run_name, estimator in EXPERIMENTS:
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("lab", "lab4_tracking")
            mlflow.log_param("model_type", type(estimator).__name__)
            mlflow.log_params({f"model__{k}": v for k, v in estimator.get_params().items()})
            mlflow.log_param("cv_folds", config.CV_FOLDS)
            mlflow.log_param("n_train_rows", len(X_train))

            cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
            mlflow.log_metric("cv_f1_mean", float(cv_f1.mean()))
            mlflow.log_metric("cv_f1_std", float(cv_f1.std()))

            pipe.fit(X_train, y_train)
            metrics = score(pipe, X_test, y_test)
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

            fig, ax = plt.subplots(figsize=(5, 4.5))
            ConfusionMatrixDisplay.from_estimator(
                pipe, X_test, y_test, display_labels=["No Depression", "Depression"],
                cmap="Blues", ax=ax)
            ax.set_title(run_name)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            log_sklearn_model(mlflow, pipe, input_example=X_train.head(3))

            print(f"  {run_name:20s} cv_f1={cv_f1.mean():.3f}  "
                  f"test_f1={metrics['f1']:.3f}  test_auc={metrics['roc_auc']:.3f}")

    print(f"\n[SUCCESS] Lab 4 complete. Logged {len(EXPERIMENTS)} runs to experiment "
          f"'{EXPERIMENT_NAME}'.")
    print(f"[INFO] Tracking store: {TRACKING_URI}")
    print("[INFO] View them with:  mlflow ui --backend-store-uri "
          f"{TRACKING_URI}   ->  http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
