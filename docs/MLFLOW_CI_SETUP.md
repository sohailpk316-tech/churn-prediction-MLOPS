# MLflow / CI Setup

Install dependencies with `python -m pip install -r requirements.txt`. CI smoke tests do not require a remote tracking server. For local tracking, use `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
