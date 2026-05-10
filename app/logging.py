# Import Custom Modules
from app.db import execute_query

def log_prediction(prediction_id, customer_id, model_role, model_run_id, result: dict, infer_times: dict, latency: float):
    query = """
        INSERT INTO predictions (
            PREDICTION_ID, CUSTOMER_ID, MODEL_ROLE, MODEL_RUN_ID,
            CHURN_PREDICTION, CHURN_PROBABILITY, LGB_PREDICTION, XGB_PREDICTION, CAT_PREDICTION,
            LGB_INFER_TIME, XGB_INFER_TIME, CAT_INFER_TIME, ENSEMBLE_INFER_TIME, END_TO_END_LATENCY
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    execute_query(
        query, (
            prediction_id,
            customer_id,
            model_role,
            model_run_id,
            result['churn_prediction'],
            result['churn_probability'],
            result['model_predictions']['lgb'],
            result['model_predictions']['xgb'],
            result['model_predictions']['cat'],
            infer_times['lgb_infer_time'],
            infer_times['xgb_infer_time'],
            infer_times['cat_infer_time'],
            infer_times['ensemble_infer_time'],
            latency
        )
    )

def log_raw_input(prediction_id, customer_id, customer: dict):
    query = """
        INSERT INTO raw_inputs (
            PREDICTION_ID, CUSTOMER_ID, GENDER, SENIOR_CITIZEN,
            PARTNER, DEPENDENTS, TENURE, PHONE_SERVICE, MULTIPLE_LINES,
            INTERNET_SERVICE, ONLINE_SECURITY, ONLINE_BACKUP, DEVICE_PROTECTION,
            TECH_SUPPORT, STREAMING_TV, STREAMING_MOVIES, CONTRACT,
            PAPERLESS_BILLING, PAYMENT_METHOD, MONTHLY_CHARGES, TOTAL_CHARGES
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        );
    """
    execute_query(
        query, (
            prediction_id,
            customer_id,
            customer['Gender'],
            customer['SeniorCitizen'],
            customer['Partner'],
            customer['Dependents'],
            customer['Tenure'],
            customer['PhoneService'],
            customer['MultipleLines'],
            customer['InternetService'],
            customer['OnlineSecurity'],
            customer['OnlineBackup'],
            customer['DeviceProtection'],
            customer['TechSupport'],
            customer['StreamingTV'],
            customer['StreamingMovies'],
            customer['Contract'],
            customer['PaperlessBilling'],
            customer['PaymentMethod'],
            customer['MonthlyCharges'],
            customer['TotalCharges']
        )
    )