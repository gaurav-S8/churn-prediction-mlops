# Import Libraries
from typing import Literal
from pydantic import BaseModel, Field, field_validator

# Input schema — Matches raw data columns
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