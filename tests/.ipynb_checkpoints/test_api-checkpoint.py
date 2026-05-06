# Import Libraries
import os
import sys
import json
import pytest
import requests
from fastapi.testclient import TestClient

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

# Import Custom Mmdules
from app.main import app

@pytest.fixture(scope = "session")
def client():
    with TestClient(app) as c:
        yield c

def valid_payload():
    return {
        "CustomerID": "C123",
        "Gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0
    }

# Number of Tests
# Test it with a valid payload.
# Check minimum number of features the json should have.
# Check if all the correct keys are present
# Check if any column is missing - Do this for all columns
# Check if there's an extra column
# Check the input after preprocessing - Should have 30 columns.
# Check with invalid values - Do this for all the columns.

def test_valid_payload_should_pass(client):
    response = client.post("/predict", json = valid_payload())
    assert response.status_code == 200

def test_single_missing_field_should_fail(client):
    payload = valid_payload()
    del payload["Gender"]
    response = client.post("/predict", json = payload)
    response_json = response.json()
    assert response.status_code == 422
    assert response_json["detail"][0]["msg"] == "Field required"

def test_multiple_missing_fields_should_fail(client):
    payload = valid_payload()
    del payload["Gender"]
    del payload["Tenure"]
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_extra_field_should_fail(client):
    payload = valid_payload()
    payload['Extra'] = 0
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_invalid_value_gender_should_fail(client):
    payload = valid_payload()
    payload['Gender'] = 'Lame'
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_uppercase_gender_should_pass(client):
    payload = valid_payload()
    payload['Gender'] = 'MALE'
    response = client.post("/predict", json = payload)
    assert response.status_code == 200

def test_lowercase_gender_should_pass(client):
    payload = valid_payload()
    payload["Gender"] = "male"
    response = client.post("/predict", json = payload)
    assert response.status_code == 200

def test_empty_string_gender_should_fail(client):
    payload = valid_payload()
    payload["Gender"] = ""
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_invalid_value_senior_citizen_should_fail(client):
    payload = valid_payload()
    payload['SeniorCitizen'] = -1
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_invalid_value_partner_should_fail(client):
    payload = valid_payload()
    payload['Partner'] = 'Ye'
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_invalid_value_dependents_should_fail(client):
    payload = valid_payload()
    payload['Dependents'] = 'Nooo'
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_wrong_type_tenure_should_fail(client):
    payload = valid_payload()
    payload["Tenure"] = "ten"
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_negative_tenure_should_fail(client):
    payload = valid_payload()
    payload['Tenure'] = -1
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_higher_than_maximum_tenure_should_fail(client):
    payload = valid_payload()
    payload['Tenure'] = 3001
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_min_tenure_should_pass(client):
    payload = valid_payload()
    payload["Tenure"] = 0
    response = client.post("/predict", json = payload)
    assert response.status_code == 200

def test_max_tenure_should_pass(client):
    payload = valid_payload()
    payload["Tenure"] = 3000
    response = client.post("/predict", json = payload)
    assert response.status_code == 200

def test_invalid_value_phone_service_should_fail(client):
    payload = valid_payload()
    payload['PhoneService'] = 'No phone'
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_valid_value_internet_service_should_pass(client):
    payload = valid_payload()
    payload['InternetService'] = 'DSL'
    response = client.post("/predict", json = payload)
    assert response.status_code == 200

def test_wrong_type_monthly_charges_should_fail(client):
    payload = valid_payload()
    payload["MonthlyCharges"] = "cheap"
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_zero_monthly_charges_should_pass(client):
    payload = valid_payload()
    payload["MonthlyCharges"] = 0
    response = client.post("/predict", json = payload)
    assert response.status_code == 200

def test_invalid_value_monthly_charges_should_fail(client):
    payload = valid_payload()
    payload['MonthlyCharges'] = -0.5
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_invalid_value_total_charges_should_fail(client):
    payload = valid_payload()
    payload['TotalCharges'] = -1
    response = client.post("/predict", json = payload)
    assert response.status_code == 422

def test_response_structure_should_be_correct(client):
    response = client.post("/predict", json=valid_payload())
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "model_predictions" in data
    assert "lgb" in data["model_predictions"]
    assert "xgb" in data["model_predictions"]
    assert "cat" in data["model_predictions"]

def test_final_churn_probability_should_be_between_0_and_1_should_pass(client):
    payload = valid_payload()
    response = client.post("/predict", json = payload)
    response_json = response.json()
    assert response.status_code == 200
    assert (response_json["churn_probability"] >= 0 and response_json["churn_probability"] <= 1)

def test_churn_prediction_value(client):
    payload = valid_payload()
    response = client.post("/predict", json = payload)
    response_json = response.json()
    assert response_json["churn_prediction"] in ["Yes", "No"]

def test_model_predictions_should_be_between_0_and_1_should_pass(client):
    payload = valid_payload()
    response = client.post("/predict", json = payload)
    response_json = response.json()
    assert response.status_code == 200
    assert (response_json["model_predictions"]["lgb"] >= 0 and response_json["model_predictions"]["lgb"] <= 1)
    assert (response_json["model_predictions"]["xgb"] >= 0 and response_json["model_predictions"]["xgb"] <= 1)
    assert (response_json["model_predictions"]["cat"] >= 0 and response_json["model_predictions"]["cat"] <= 1)

def test_same_input_should_give_same_output(client):
    payload = valid_payload()
    res1 = client.post("/predict", json = payload).json()
    res2 = client.post("/predict", json = payload).json()
    assert res1["churn_probability"] == res2["churn_probability"]

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "Ok"}

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200