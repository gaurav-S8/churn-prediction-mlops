# Import Libraries
import os
import json
import mlflow
import joblib
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
        order_by = ["start_time DESC"]
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
    # Extract ensemble weights
    weights = {
        "w_lgb": run.data.metrics["w_lgb"],
        "w_xgb": run.data.metrics["w_xgb"],
        "w_cat": run.data.metrics["w_cat"]
    }
    # Save weights.json
    with open(os.path.join(folder, "weights.json"), "w") as f:
        json.dump(weights, f, indent = 4)
    print(f"Weights saved to {folder}/weights.json")

def main():
    
    # Start MLflow server
    mlflow_process = start_mlflow_server()
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
    print("  1. Save latest run to challenger/")
    print("  2. Save best run (by ROC AUC) to challenger/")
    print("  3. Exit")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        run_id = runs[0].info.run_id
        print(f"\nSaving latest run ({run_id[:8]}...) to challenger folder.")
        download_models_and_weights(client, run_id, CHALLENGER_FOLDER)
        print("Challenger models saved!")

    elif choice == "2":
        best_runs = client.search_runs(
            experiment_ids = [runs[0].info.experiment_id],
            order_by = ["metrics.oof_auc_ensemble DESC"]
        )
        run_id = best_runs[0].info.run_id
        print(f"\nSaving best run ({run_id[:8]}...) to challenger folder.")
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