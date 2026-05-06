# End-to-End Churn Prediction MLOps System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat&logo=mlflow&logoColor=white)
![MinIO](https://img.shields.io/badge/MinIO-S3--Storage-C72E49?style=flat&logo=minio&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-5A0FC8?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

## Overview
An end-to-end MLOps pipeline for predicting customer churn using a weighted ensemble of LightGBM, XGBoost, and CatBoost models.

The system integrates model training, hyperparameter tuning, and experiment tracking with MLflow, while storing artifacts in MinIO (S3-compatible storage). Predictions are served via a FastAPI REST API, with all components orchestrated using Docker for reproducibility.

---

## Kaggle Benchmark

**Competition**: [Playground Series S6E3 — Predict Customer Churn](https://www.kaggle.com/competitions/playground-series-s6e3)  
**Final Rank**: **1160 / 4142 teams (Top 28%)**

This project extends the competition solution into a full end-to-end MLOps pipeline, incorporating experiment tracking, model ensembling, and production-style API deployment.


### Model Performance

- **Validation ROC-AUC**: 0.91538  
- **Top Leaderboard Score**: 0.91856  
- **Gap to Top**: ~0.003  

Achieved near top-tier performance (~0.003 from the best score) using a weighted ensemble of LightGBM, XGBoost, and CatBoost models with Optuna-based hyperparameter tuning.

---

## Architecture Overview
```
    TRAIN                          SERVE
┌────────────────────────────────────────────────────────────────┐
│  train_model.py                         Client                 │ 
│       │                                   │                    │ 
│       │  fit + tune (Optuna)              │  POST /predict     │
│       ▼                                   ▼                    │ 
│  ┌──────────┐              ┌──────────────────────────────┐    │
│  │  MLflow  │──────────────│           FastAPI            │    │
│  │  :5000   │              │           :8000              │    │
│  └──────────┘              │                              │    │
│       │                    │  1. Validate (Pydantic)      │    │
│       │  store artifacts   │  2. Preprocess + Engineer    │    │
│       ▼                    │  3. LGB · XGB · CAT ensemble │    │
│  ┌──────────┐              │  4. Weighted avg → response  │    │
│  │  MinIO   │ ◄────────────│                              │    │
│  │  :9000   │              └──────────────────────────────┘    │
│  └──────────┘                                                  │
│       ▲                                                        │
│       └──────── FastAPI loads model at startup                 │
└────────────────────────────────────────────────────────────────┘
```

| Service  | Role                                      | Port(s)        |
|----------|-------------------------------------------|----------------|
| FastAPI  | Serves prediction endpoints               | `8000`         |
| MLflow   | Experiment tracking & artifact serving    | `5000`         |
| MinIO    | S3-compatible artifact storage backend    | `9000`, `9001` |

---

## Machine Learning Pipeline

The pipeline trains three gradient boosting models — **LightGBM, XGBoost, and CatBoost** — on a shared preprocessing and feature engineering foundation, and combines them into a weighted ensemble optimized for ROC-AUC.

- **Preprocessing**: Schema validation, categorical encoding, and handling of domain-specific values (e.g., "No internet service")  
- **Feature Engineering**: Derived features such as `TENURE × MONTHLY_CHARGES`, spend ratios, service counts, and binary payment indicators  
- **Validation**: Stratified K-Fold cross-validation across all models  
- **Hyperparameter Tuning**: Optuna used to optimize learning rate, tree depth, and subsampling  
- **Ensemble**: Out-of-Fold predictions used to optimize ensemble weights via Optuna, combined using weighted averaging  
- **Experiment Tracking**: All runs logged in MLflow (parameters, ROC-AUC metrics), with artifacts stored in MinIO

### Model Details

The API uses a **weighted ensemble** of three gradient boosting models:

- **LightGBM** (`lgb`)
- **XGBoost** (`xgb`)
- **CatBoost** (`cat`)

Models and their weights are loaded at startup from the MLflow tracking server. The final churn probability is computed as:

```
final_probability = (w_lgb × p_lgb) + (w_xgb × p_xgb) + (w_cat × p_cat)
```

A probability above `0.5` is classified as **churn**.

---

## API Endpoints

### `GET /`

```json
{ "message": "Churn Prediction API is running" }
```

### `GET /health`
Health check for the API.

```json
{ "status": "Ok" }
```

### `POST /predict`
Predicts churn probability for a given customer.

**Request body** (all fields required):

| Field              | Type    | Description                                                                                          |
|--------------------|---------|------------------------------------------------------------------------------------------------------|
| `CustomerID`       | string  | Unique customer identifier                                                                           |
| `Gender`           | string  | `"Male"` or `"Female"`                                                                               |
| `SeniorCitizen`    | int     | `0` or `1`                                                                                           |
| `Partner`          | string  | `"Yes"` or `"No"`                                                                                    |
| `Dependents`       | string  | `"Yes"` or `"No"`                                                                                    |
| `Tenure`           | int     | Months with the company (0–3000)                                                                     |
| `PhoneService`     | string  | `"Yes"` or `"No"`                                                                                    |
| `MultipleLines`    | string  | `"Yes"`, `"No"`, or `"No phone service"`                                                             |
| `InternetService`  | string  | `"No"`, `"DSL"`, or `"Fiber optic"`                                                                  |
| `OnlineSecurity`   | string  | `"Yes"`, `"No"`, or `"No internet service"`                                                          |
| `OnlineBackup`     | string  | `"Yes"`, `"No"`, or `"No internet service"`                                                          |
| `DeviceProtection` | string  | `"Yes"`, `"No"`, or `"No internet service"`                                                          |
| `TechSupport`      | string  | `"Yes"`, `"No"`, or `"No internet service"`                                                          |
| `StreamingTV`      | string  | `"Yes"`, `"No"`, or `"No internet service"`                                                          |
| `StreamingMovies`  | string  | `"Yes"`, `"No"`, or `"No internet service"`                                                          |
| `Contract`         | string  | `"Month-to-month"`, `"One year"`, or `"Two year"`                                                    |
| `PaperlessBilling` | string  | `"Yes"` or `"No"`                                                                                    |
| `PaymentMethod`    | string  | `"Bank transfer (automatic)"`, `"Credit card (automatic)"`, `"Electronic check"`, `"Mailed check"`   |
| `MonthlyCharges`   | float   | Monthly charge amount (≥ 0)                                                                          |
| `TotalCharges`     | float   | Total charges to date (≥ 0)                                                                          |

**Example request (curl):**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CustomerID": "7590-VHVEG",
    "Gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure": 12,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 358.20
  }'
```

**Example request (Python):**

```python
import requests

payload = {
    "CustomerID": "7590-VHVEG",
    "Gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure": 12,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 358.20
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
```

**Example response:**

```json
{
  "churn_probability": 0.7312,
  "churn_prediction": "Yes",
  "model_predictions": {
    "lgb": 0.7401,
    "xgb": 0.7189,
    "cat": 0.7344
  }
}
```

### `POST /reload-model`

Reloads the latest model from MLflow **without restarting the server**. Use this after retraining (`train/train_model.py`) to hot-swap the model into the running API.

```bash
curl -X POST http://localhost:8000/reload-model
```

```json
{ "status": "Latest model loaded!!" }
```

---

## How To Run

### Prerequisites

- [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)
- Python **3.10+**

### Installation
Clone the repository:

```bash
git clone https://github.com/gaurav-s8/churn-prediction-mlops.git
cd churn-prediction-mlops
```

(Optional) Install dependencies for local scripts:
```bash
pip install -r requirements.txt
pip install -e .
```

### Environment Variables
The project uses a `.env` file for local configuration. Create one at the root of the project:

A minimal `.env` looks like this:

```env
DATA_PATH=data
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_EXPERIMENT_NAME=Churn_Prediction
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
```

> **Note:** When running inside Docker, these values are already set via `docker-compose.yml`. The `.env` file is only needed for running scripts (like `create_bucket.py`) directly on the host machine.

### Run the stack

**Step 1 — Build and start all services**
```bash
docker compose up --build
```

Wait until MLflow and MinIO are fully up before proceeding to the next step. You can verify by visiting http://localhost:5000 (MLflow) and http://localhost:9001 (MinIO UI).

**Step 2 — Create the MinIO bucket for MLflow artifacts**

```bash
docker exec -it fastapi python src/create_bucket.py
```

This creates the `mlflow-artifacts` S3 bucket that MLflow uses to store models and artifacts. This only needs to be run once.

**Step 3 — Train the models**

```bash
docker exec -it fastapi python train/train_model.py
```

This runs the training pipeline inside the FastAPI container, logs metrics and models to MLflow, and stores artifacts in MinIO.
> **Note:** Replace `fastapi` with your container name (check using `docker ps`)

Once all steps complete, the services are available at:

| Service       | URL                                               |
|---------------|---------------------------------------------------|
| API           | http://localhost:8000                             |
| API Docs      | http://localhost:8000/docs                        |
| MLflow UI     | http://localhost:5000                             |
| MinIO Console | http://localhost:9001 (`minioadmin`/`minioadmin`) |

---

## Configuration

Configuration is managed via environment variables (defined in `docker-compose.yml` or `.env`).

| Variable                 | Description                         |
|--------------------------|-------------------------------------|
| `MLFLOW_TRACKING_URI`    | MLflow server address               |
| `MLFLOW_EXPERIMENT_NAME` | Experiment name                     |
| `MLFLOW_S3_ENDPOINT_URL` | MinIO endpoint                      |
| `AWS_ACCESS_KEY_ID`      | MinIO access key                    |
| `AWS_SECRET_ACCESS_KEY`  | MinIO secret key                    |
| `MINIO_ROOT_USER`        | MinIO admin username                |
| `MINIO_ROOT_PASSWORD`    | MinIO admin password                |

---

## Project Structure

```
.
├── app/
│   ├── main.py                          # FastAPI application & endpoints
│   └── model.py                         # Loads trained models from MLflow
├── data/                                # Raw datasets (not included — Download from Kaggle)
├── kaggle_submission/                   # Generated submission files
├── notebook/                            # Exploratory data analysis notebooks
├── src/
│   ├── create_bucket.py                 # MinIO bucket setup script (run once)
│   ├── optuna_hp_optimization.py        # Hyperparameter tuning
│   └── preprocess.py                    # Feature preprocessing & engineering
├── tests/                               # Unit and integration tests
├── train/
│   └── train_model.py                   # Model training & MLflow logging
├── .dockerignore
├── .env                                 # Local environment variables (not committed)
├── .gitignore
├── docker-compose.yml                   # Multi-service orchestration
├── Dockerfile                           # Container definition for FastAPI
├── requirements.txt                     # Python dependencies
└── setup.py                             # Package setup
```
> **Note:** This represents the repository structure; not all directories are included in the Docker image.

---

## Future Improvements

- **Model Versioning** — Integrate MLflow Model Registry for managing and promoting model versions  
- **Logging & Monitoring** — Add structured logging and basic request/response monitoring  
- **Cloud Deployment** — Deploy the system on AWS/GCP with managed storage and scalable services  
- **CI/CD Pipeline** — Automate build, testing, and deployment workflows  

---

## Author

**Gaurav Singariya**  
MSc Data Science  
[GitHub](https://github.com/gaurav-s8)

---

## License

This project is licensed under the [MIT License](LICENSE).