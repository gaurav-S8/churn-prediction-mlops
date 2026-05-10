import time
import requests
import pandas as pd

df = pd.read_csv('data/train.csv')
df = df.rename(
    columns = {
        'id': 'CustomerID',
        'gender': 'Gender',
        'tenure': 'Tenure'
    }
)
sample = df.sample(100, random_state = 99)

print(sample)

URL = "http://localhost:8000/predict"

for i, row in sample.iterrows():
    payload = {
        "CustomerID": str(row['CustomerID']),
        "Gender": row['Gender'],
        "SeniorCitizen": int(row['SeniorCitizen']),
        "Partner": row['Partner'],
        "Dependents": row['Dependents'],
        "Tenure": int(row['Tenure']),
        "PhoneService": row['PhoneService'],
        "MultipleLines": row['MultipleLines'],
        "InternetService": row['InternetService'],
        "OnlineSecurity": row['OnlineSecurity'],
        "OnlineBackup": row['OnlineBackup'],
        "DeviceProtection": row['DeviceProtection'],
        "TechSupport": row['TechSupport'],
        "StreamingTV": row['StreamingTV'],
        "StreamingMovies": row['StreamingMovies'],
        "Contract": row['Contract'],
        "PaperlessBilling": row['PaperlessBilling'],
        "PaymentMethod": row['PaymentMethod'],
        "MonthlyCharges": float(row['MonthlyCharges']),
        "TotalCharges": float(row['TotalCharges'])
    }
    response = requests.post(URL, json = payload)
    print(f"Request {i+1}: {response.status_code}")
    # 2 seconds between requests
    time.sleep(2)