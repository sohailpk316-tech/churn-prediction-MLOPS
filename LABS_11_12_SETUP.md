# Labs 11 and 12 - Churn Prediction
The professor's Lab 11 and Lab 12 scripts and their mapped supporting files are integrated into this project.

## Lab 11
- Deployment preparation: `python pipelines/run_lab11_prepare_deployment.py`
- API source: `api/`; Dockerfile: project root; Kubernetes manifests: `k8s/`.
- Build/run the container from the project root: `docker build -t churn-prediction-api:lab11 .`
- The API needs the deployment bundle produced by the Lab 11 preparation step. Use the existing Docker/Kubernetes practical instructions in `lab11_complete_flow.md` and `lab11_commands/`.

## Lab 12
- Individual monitoring/drift/retraining scripts are in `src/mlops_pipeline/12_*.py`.
- Start with the dataset/drift scripts and follow each script's prerequisites. Some scripts expect inference logs, processed arrays, or an MLflow Production model created by earlier labs.
- The convenience launcher `pipelines/run_lab12_monitoring.py` runs a selected sequence and stops at the first missing prerequisite; run individual scripts when following the lab handout.

Existing student files were preserved; any destination collision was left untouched. This integration does not claim the practical execution is complete.
