# Import Libraries
import pandas as pd
from typing import Literal
from fastapi import FastAPI

from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator

# Import Custom Modules
from app.model import load_model
from src.preprocess import preprocess_and_engineer_feature

# Cache ensemble globally
ensemble = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup
    global ensemble
    ensemble = load_model()
    yield
    # Cleanup on shutdown (optional)

# Input schema — matches your raw data columns
class CustomerData(BaseModel):
    # To forbit extra features
    model_config = {
        "extra": "forbid"
    }
    
    CustomerID: str
    Gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    Tenure: int = Field(ge = 0, le = 3000)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["No", "DSL", "Fiber optic"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["One year", "Two year", "Month-to-month"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"
    ]
    MonthlyCharges: float = Field(ge = 0)
    TotalCharges: float = Field(ge = 0)

    @field_validator("Gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", mode = "before")
    def normalize_basic_fields(cls, v):
        if isinstance(v, str):
            return v.strip().title()
        return v

app = FastAPI(
    title = "Churn Prediction API",
    description = "Predicts Customer Churn probability",
    version = "1.0.0",
    lifespan = lifespan
)

@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running"}

@app.get("/health")
def check_api_health():
    return {"status": "Ok"}

@app.post("/predict")
def predict(customer: CustomerData) -> dict:
    
    global ensemble
    if ensemble is None:
        ensemble = load_model()
        if ensemble is None:
            raise ValueError("No experiment logged in MLflow")
    
    # Convert input to dataframe
    input_df = pd.DataFrame([customer.model_dump()])
    processed_df = preprocess_and_engineer_feature(input_df)

    customer_id = processed_df['CUSTOMERID']
    processed_df = processed_df.drop(columns = ['CUSTOMERID'])

    lgb_model = ensemble['lgb_model']
    xgb_model = ensemble['xgb_model']
    cat_model = ensemble['cat_model']
    w_lgb = ensemble['w_lgb']
    w_xgb = ensemble['w_xgb']
    w_cat = ensemble['w_cat']
    
    pred_lgb = lgb_model.predict_proba(processed_df)[:, 1][0]
    pred_xgb = xgb_model.predict_proba(processed_df)[:, 1][0]
    pred_cat = cat_model.predict_proba(processed_df)[:, 1][0]

    final_pred = pred_lgb * w_lgb + pred_xgb * w_xgb + pred_cat * w_cat
    return {
        'churn_probability': round(float(final_pred), 4),
        'churn_prediction': 'Yes' if final_pred > 0.5 else 'No',
        'model_predictions': {
            'lgb': round(float(pred_lgb), 4),
            'xgb': round(float(pred_xgb), 4),
            'cat': round(float(pred_cat), 4)
        }
    }

@app.post("/reload-model")
def reload_model():
    global ensemble
    ensemble = load_model()
    return {"status": "Latest model loaded!!"}