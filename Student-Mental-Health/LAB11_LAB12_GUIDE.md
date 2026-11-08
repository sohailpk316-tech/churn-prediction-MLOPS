# Student Mental Health - Labs 11 and 12

These two labs are purpose-built for this Student Mental Health project. They do not reuse the Churn Prediction model or its customer fields.

## Lab 11 - Model deployment and containerization
1. Ensure Labs 1-10 have produced `models/best_model.pkl`.
2. `python pipelines/run_lab11_containerization.py` verifies/stages the model and prints Docker commands.
3. From the project root: `docker build -t student-mental-health-api:lab11 .`
4. `docker run --rm -p 8000:8000 student-mental-health-api:lab11`
5. Open `http://localhost:8000/docs`; test `GET /health` and `POST /predict`.
6. For Kubernetes (after building the image locally): `kubectl apply -f k8s/`; verify with `kubectl get pods` and `kubectl get service student-mental-health-service`.

## Lab 12 - Monitoring and drift detection
1. Run `python pipelines/run_lab12_monitoring.py` to create a synthetic drift batch, calculate PSI, evaluate held-out model performance, and create a retraining recommendation.
2. To monitor API requests, send a few requests to `/predict`, then run `python src/mlops_pipeline/12_api_performance_monitoring.py`.
3. Optional retraining: only when the monitoring report recommends it, run `python src/mlops_pipeline/12_retrain_model.py`. This uses the existing training pipeline and may take several minutes.

## Important limitations
This project uses a small student survey dataset. Drift data is synthetic and is only for demonstrating monitoring mechanics. The model output is an educational coursework prediction, not a diagnosis, screening instrument, or basis for real student decisions. The API logs request metadata and results only, not submitted student answers.
