# Import Libraries
import pandas as pd

# Import Custom Modules
from app.db import execute_query

def get_recent_inputs(limit = 100):
    rows = execute_query(
        """
            SELECT
                GENDER, SENIOR_CITIZEN, PARTNER, DEPENDENTS, TENURE,
                PHONE_SERVICE, MULTIPLE_LINES, INTERNET_SERVICE,
                ONLINE_SECURITY, ONLINE_BACKUP, DEVICE_PROTECTION,
                TECH_SUPPORT, STREAMING_TV, STREAMING_MOVIES, CONTRACT,
                PAPERLESS_BILLING, PAYMENT_METHOD, MONTHLY_CHARGES, TOTAL_CHARGES
            FROM raw_inputs
            ORDER BY timestamp DESC
            LIMIT %s;
        """,
        (limit, )
    )
    columns = [
        'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    return pd.DataFrame(rows, columns = columns)