# Import Libraries
import os
import json
import mlflow
import joblib
from datetime import datetime
from dotenv import load_dotenv

# Import Custom Function
from mlflow_launcher import start_mlflow_server

# Load Environment Variables
load_dotenv()
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Churn_Prediction")
CHAMPION_FOLDER = os.getenv("CHAMPION_FOLDER", "models/champion")
CHALLENGER_FOLDER = os.getenv("CHALLENGER_FOLDER", "models/challenger")

def get_experiment_runs():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")
    
    runs = client.search_runs(
        experiment_ids = [experiment.experiment_id],
        order_by = ["metrics.oof_auc_ensemble DESC"]
    )
    return client, runs

def download_models_and_weights(client, run_id, folder):
    os.makedirs(folder, exist_ok = True)
    for model_name in ["lgb_model", "xgb_model", "cat_model"]:
        local_path = mlflow.artifacts.download_artifacts(
            run_id = run_id,
            artifact_path = model_name,
        )
        model = mlflow.sklearn.load_model(local_path)
        joblib.dump(model, os.path.join(folder, f"{model_name}.pkl"))
        print(f"{model_name.upper()}.pkl saved to {folder}")
        
    # Fetch run data
    run = client.get_run(run_id)

    # Save model info
    model_info = {
        "run_id": run_id,
        "trained_at": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
        "role": os.path.basename(folder),
        "weights": {
            "w_lgb": run.data.metrics['w_lgb'],
            "w_xgb": run.data.metrics['w_xgb'],
            "w_cat": run.data.metrics['w_cat']
        },
        "oof_roc_auc_scores": {
            "roc_auc_lgb": run.data.metrics['oof_auc_lgb'],
            "roc_auc_xgb": run.data.metrics['oof_auc_xgb'],
            "roc_auc_cat": run.data.metrics['oof_auc_cat'],
            "roc_auc_ensemble": run.data.metrics['oof_auc_ensemble']
        },
        "parameters": {
            "lgb": {
                "n_estimators": run.data.params['lgb_n_estimators'],
                "max_depth": run.data.params['lgb_max_depth'],
                "learning_rate": run.data.params['lgb_learning_rate'],
                "subsample": run.data.params['lgb_subsample'],
                "colsample_bytree": run.data.params['lgb_colsample_bytree'],
                "min_child_samples": run.data.params['lgb_min_child_samples'],
                "num_leaves": run.data.params['lgb_num_leaves']
            },
            "xgb": {
                "n_estimators": run.data.params['xgb_n_estimators'],
                "max_depth": run.data.params['xgb_max_depth'],
                "learning_rate": run.data.params['xgb_learning_rate'],
                "subsample": run.data.params['xgb_subsample'],
                "colsample_bytree": run.data.params['xgb_colsample_bytree'],
                "min_child_weight": run.data.params['xgb_min_child_weight'],
                "gamma": run.data.params['xgb_gamma']
            },
            "cat": {
                "iterations": run.data.params['cat_iterations'],
                "depth": run.data.params['cat_depth'],
                "learning_rate": run.data.params['cat_learning_rate'],
                "l2_leaf_reg": run.data.params['cat_l2_leaf_reg'],
                "subsample": run.data.params['cat_subsample']
            }
        }
    }
    with open(os.path.join(folder, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent = 4)
    print(f"Model info saved to {folder}/model_info.json")

def main():
    
    # Start MLflow server
    mlflow_process = start_mlflow_server(TRACKING_URI)
    mlflow.set_tracking_uri(TRACKING_URI)

    client, runs = get_experiment_runs()
    if not runs:
        print("No runs found in MLflow.")
        return

    champion_exists = os.path.exists(CHAMPION_FOLDER) and \
        len(os.listdir(CHAMPION_FOLDER)) > 0
    if not champion_exists:
        print("No champion found — saving latest run to champion.")
        download_models_and_weights(client, runs[0].info.run_id, CHAMPION_FOLDER)
        print("Champion models saved!")
        return

    # Champion exists — ask user
    print("\nChampion already exists. What do you want to do?")
    print("  1. Save best run (by ROC AUC) to challenger/")
    print("  2. Save latest run to challenger/")
    print("  3. Exit")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        run_id = runs[0].info.run_id
        print(f"\nSaving best run ({run_id[:8]}...) to challenger folder.")
        download_models_and_weights(client, run_id, CHALLENGER_FOLDER)
        print("Challenger models saved!")

    elif choice == "2":
        latest_runs = client.search_runs(
            experiment_ids = [runs[0].info.experiment_id],
            order_by = ["start_time DESC"]
        )
        run_id = latest_runs[0].info.run_id
        print(f"\nSaving latest run ({run_id[:8]}...) to challenger folder.")
        download_models_and_weights(client, run_id, CHALLENGER_FOLDER)
        print("Challenger models saved!")

    elif choice == "3":
        print("Exiting.")
    else:
        print("Invalid choice.")
    
    if mlflow_process:
        mlflow_process.terminate()
        print("MLflow stopped.")

if __name__ == "__main__":
    main()