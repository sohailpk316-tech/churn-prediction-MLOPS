r"""Lab 6 - MLflow Model Registry.

Run from the project root:   python .\pipelines\run_lab6_registry.py

1. Finds the best Lab 4 run (highest cross-validated F1) - runs Lab 4 first
   if no runs exist yet.
2. Registers that run's model in the MLflow Model Registry.
3. Gives the new version the alias "champion" (and a description + tags).
4. Loads the model back FROM THE REGISTRY and predicts for a sample student
   to prove the registered model works.
"""
import json
import subprocess
import sys

import pandas as pd

from _common import (EXPERIMENT_NAME, REGISTERED_MODEL_NAME, ROOT, TRACKING_URI,
                     config, setup_mlflow)
from features import prepare_dataframe
from predict import SAMPLE_STUDENT


def best_run(mlflow):
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return None
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                              filter_string="tags.lab = 'lab4_tracking'",
                              order_by=["metrics.cv_f1_mean DESC"], max_results=1)
    return None if runs.empty else runs.iloc[0]


def main():
    print("[INFO] ===== Lab 6: Model registry (MLflow) =====")
    mlflow = setup_mlflow()
    from mlflow.tracking import MlflowClient
    import mlflow.sklearn as mlflow_sklearn
    client = MlflowClient()

    run = best_run(mlflow)
    if run is None:
        print("[INFO] No Lab 4 runs found - running Lab 4 first...")
        subprocess.run([sys.executable, str(ROOT / "pipelines" / "run_lab4_tracking.py")],
                       cwd=ROOT, check=True)
        run = best_run(mlflow)
        if run is None:
            sys.exit("[ERROR] Could not find any tracked runs.")

    run_id = run["run_id"]
    print(f"[INFO] Best run: {run['tags.mlflow.runName']}  (run_id={run_id[:8]}...)  "
          f"cv_f1={run['metrics.cv_f1_mean']:.3f}  test_f1={run['metrics.test_f1']:.3f}")

    # ---- register ----------------------------------------------------------
    version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    client.update_registered_model(
        REGISTERED_MODEL_NAME,
        description="Predicts whether a university student reports depression "
                    "(binary classifier: preprocessing + sklearn model).")
    client.update_model_version(
        REGISTERED_MODEL_NAME, version.version,
        description=f"Best cross-validated F1 from experiment '{EXPERIMENT_NAME}' "
                    f"(source run: {run['tags.mlflow.runName']}).")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "validated", "true")
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version.version)
    print(f"[INFO] Registered '{REGISTERED_MODEL_NAME}' version {version.version} "
          f"with alias 'champion'")

    # ---- load back from the registry and predict ----------------------------
    model = mlflow_sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")
    X = prepare_dataframe(pd.DataFrame([SAMPLE_STUDENT]))[config.FEATURES]
    proba = float(model.predict_proba(X)[0, 1])
    label = "Depression" if int(model.predict(X)[0]) == 1 else "No Depression"
    print(f"[INFO] Prediction from registry model: {label} (probability {proba:.2%})")

    summary = {"registered_model": REGISTERED_MODEL_NAME, "version": int(version.version),
               "alias": "champion", "source_run_id": run_id,
               "source_run_name": run["tags.mlflow.runName"],
               "cv_f1_mean": float(run["metrics.cv_f1_mean"]),
               "sample_prediction": {"class": label, "probability": round(proba, 4)}}
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    (config.OUTPUTS_DIR / "lab6_registry_summary.json").write_text(json.dumps(summary, indent=4))

    print("\n[SUCCESS] Lab 6 complete. Saved outputs/lab6_registry_summary.json")
    print(f"[INFO] Registry store: {TRACKING_URI}")


if __name__ == "__main__":
    main()
