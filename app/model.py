# Import Libraries
import os
import time
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv

load_dotenv()

_ensemble  = None

def load_model():
    global _ensemble
    if _ensemble is not None:
        return _ensemble
    
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = MlflowClient()

    # Retry until MLflow is ready
    experiment = None
    for attempt in range(10):
        try:
            experiment = client.get_experiment_by_name(
                os.getenv("MLFLOW_EXPERIMENT_NAME", "Churn_Prediction")
            )
            if experiment is not None:
                break
        except Exception as e:
            print(f"Retry {attempt+1}/10... {e}")
            time.sleep(3)
    
    if experiment is None:
        print("No experiment found. Train a model first then call /reload-model")
        return None

    # Get the latest run
    runs = client.search_runs(
        experiment_ids = [experiment.experiment_id],
        filter_string = "status = 'FINISHED'",
        order_by = ["start_time DESC"],
        max_results = 1
    )
    if not runs:
        print("No finished runs found. Train a model first then call /reload-model")
        return None
    
    run_id = runs[0].info.run_id
    lgb_model = mlflow.sklearn.load_model(f"runs:/{run_id}/lgb_model")
    xgb_model = mlflow.sklearn.load_model(f"runs:/{run_id}/xgb_model")
    cat_model = mlflow.sklearn.load_model(f"runs:/{run_id}/cat_model")
    print("Models loaded!")

    # Get weighted model weights
    metrics = runs[0].data.metrics
    w_lgb = metrics.get('w_lgb')
    w_xgb = metrics.get('w_xgb')
    w_cat = metrics.get('w_cat')
    
    if None in (w_lgb, w_xgb, w_cat):
        raise ValueError("Model weights not found in MLflow run")

    _ensemble = {
        'lgb_model': lgb_model,
        'xgb_model': xgb_model,
        'cat_model': cat_model,
        'w_lgb': w_lgb,
        'w_xgb': w_xgb,
        'w_cat': w_cat
    }
    return _ensemble