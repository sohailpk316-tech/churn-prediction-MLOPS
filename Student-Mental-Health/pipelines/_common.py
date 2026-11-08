"""Shared helpers for the lab pipelines (Lab 3, Lab 4, Lab 6).

Makes `src/` importable, makes sure the processed train/test CSVs exist
(runs preprocessing automatically if they are missing) and provides the
data loader + metric function used by all three lab scripts.
"""
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                    # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, precision_score,   # noqa: E402
                             recall_score, roc_auc_score)

import config                                          # noqa: E402

warnings.filterwarnings("ignore")

# MLflow backend: a local SQLite file + a local artifact folder (no server needed).
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT / "mlartifacts"
TRACKING_URI = "sqlite:///" + MLFLOW_DB.as_posix()
EXPERIMENT_NAME = "student_depression_prediction"
REGISTERED_MODEL_NAME = "student_depression_classifier"


def ensure_data():
    """Create data/processed/train.csv + test.csv if they do not exist yet."""
    if config.TRAIN_PATH.exists() and config.TEST_PATH.exists():
        return
    print("[INFO] Processed data not found - running src/preprocess.py first...")
    subprocess.run([sys.executable, str(ROOT / "src" / "preprocess.py")],
                   cwd=ROOT, check=True)


def load_train_test():
    ensure_data()
    train = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)
    return (train[config.FEATURES], train["target"],
            test[config.FEATURES], test["target"])


def score(model, X, y):
    """Accuracy / precision / recall / F1 / ROC-AUC for a fitted classifier."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def require_mlflow():
    try:
        import mlflow  # noqa: F401
        return mlflow
    except ImportError:
        sys.exit("[ERROR] mlflow is not installed. Run:  pip install -r requirements.txt")


def setup_mlflow():
    """Point MLflow at the local SQLite DB and (create/select) the experiment."""
    mlflow = require_mlflow()
    MLFLOW_ARTIFACTS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME,
                                 artifact_location=MLFLOW_ARTIFACTS.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


def log_sklearn_model(mlflow, model, input_example=None):
    """Log sklearn model with explicitly trusted skops types."""
    import mlflow.sklearn as mlflow_sklearn

    trusted_types = ["numpy.dtype", "sklearn.tree._tree.Tree"]
    try:
        return mlflow_sklearn.log_model(
            model,
            name="model",
            input_example=input_example,
            skops_trusted_types=trusted_types
        )
    except TypeError:
        return mlflow_sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example,
            skops_trusted_types=trusted_types
        )