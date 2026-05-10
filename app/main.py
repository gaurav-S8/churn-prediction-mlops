# Import Libraries
import os
import time
import uuid

from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import Security
from fastapi.responses import HTMLResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request

from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler

# Import Custom Modules
from app.schemas import CustomerData
from app.load_model import load_model
from app.drift import get_drift_report
from app.monitoring import get_recent_inputs
from app.registry import sync_model_registry
from app.db import init_db, init_pool, execute_query
from app.logging import log_prediction, log_raw_input
from app.benchmark import get_benchmark_metrics, get_ab_metrics
from app.predict import choose_model, prepare_input, run_ensemble, run_shap_explainability

# Load env
load_dotenv()

# Cache ensemble globally
champion_ensemble = None
challenger_ensemble = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check if database tables exists!
    # Create them if doesn't
    init_pool()
    init_db()
    sync_model_registry()

    # Load model once at startup
    global champion_ensemble, challenger_ensemble
    champion_ensemble = load_model("champion", os.getenv("CHAMPION_FOLDER", "models/champion"))
    challenger_ensemble = load_model("challenger", os.getenv("CHALLENGER_FOLDER", "models/challenger"))

    yield
    # Cleanup on shutdown (optional)

app = FastAPI(
    title = "Churn Prediction API",
    description = "Predicts Customer Churn probability",
    version = "1.0.0",
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        os.getenv("HF_SPACE_URL"),
        os.getenv("LOCALHOST", "http://localhost:8000"),
        os.getenv("STREAMLIT_LOCALHOST", "http://localhost:8501"),
    ],
    allow_methods = ["GET", "POST"],
    allow_headers = ["*"]
)

limiter = Limiter(key_func = get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

api_key_header = APIKeyHeader(name = "X-API-Key", auto_error = True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code = 403, detail = "Invalid API key")

@app.get("/")
async def read_root():
    return {"message": "Churn Prediction API is running"}

@app.get("/health")
async def check_api_health():
    return {"status": "Ok"}

@app.get("/benchmark")
async def benchmark(api_key: str = Security(verify_api_key)):
    metrics = get_benchmark_metrics()
    return metrics

@app.get("/ab-report")
async def ab_report(api_key: str = Security(verify_api_key)):
    ab_metrics = get_ab_metrics()
    return ab_metrics

@app.get("/model-registry")
async def model_info(api_key: str = Security(verify_api_key)):
    return None

@app.get("/drift")
@limiter.limit("2/minute")
async def drift(request: Request, limit: int = 100, api_key: str = Security(verify_api_key)):
    _, summary = get_drift_report()
    return summary

@app.get("/drift/report", response_class = HTMLResponse)
@limiter.limit("2/minute")
async def drift_report(request: Request, limit: int = 100, api_key: str = Security(verify_api_key)):
    report, _ = get_drift_report()
    if report is None:
        return "<h1>Not enough data</h1>"
    report.save_html("drift_report.html")
    with open("drift_report.html", "r", encoding = "utf-8") as f:
        return f.read()

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(
    request: Request, customer: CustomerData, background_tasks: BackgroundTasks, api_key: str = Security(verify_api_key)
) -> dict:
    start = time.time()
    ensemble, model_role = choose_model(champion_ensemble, challenger_ensemble)
    if ensemble is None:
        raise HTTPException(status_code = 503, detail = "Models not loaded")

    customer_id, processed_df = prepare_input(customer)
    result, infer_times = run_ensemble(ensemble, processed_df)

    # Generated Prediction ID
    prediction_id = str(uuid.uuid4())

    latency = time.time() - start

    # Log to Postgre DB - Predictions & Raw Inputs
    background_tasks.add_task(
        log_prediction,
        prediction_id = prediction_id,
        customer_id = customer_id.values[0],
        model_role = model_role,
        model_run_id = ensemble['run_id'],
        result = result,
        infer_times = infer_times,
        latency = latency
    )
    background_tasks.add_task(
        log_raw_input,
        prediction_id = prediction_id,
        customer_id = customer_id.values[0],
        customer = customer.model_dump()
    )
    return result

@app.post("/explain")
@limiter.limit("10/minute")
async def explain(request: Request, customer: CustomerData, api_key: str = Security(verify_api_key)) -> dict:
    ensemble, model_version = choose_model(champion_ensemble, challenger_ensemble)
    if ensemble is None:
        raise HTTPException(status_code = 503, detail = "Models not loaded")

    customer_id, processed_df = prepare_input(customer)
    result = run_shap_explainability(ensemble, processed_df)
    return result