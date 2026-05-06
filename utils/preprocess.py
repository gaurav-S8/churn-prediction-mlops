# Import Libraries
import pandas as pd
from typing import Tuple

def preprocess_and_engineer_feature(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    
    # Make a copy of dataframe
    dataframe = dataframe.copy()

    # Standardize columns
    dataframe.columns = [
        col.upper() for col in dataframe.columns
    ]

    column_change_dict = {
        "SENIORCITIZEN": "SENIOR_CITIZEN",
        "PHONESERVICE": "PHONE_SERVICE",
        "MULTIPLELINES": "MULTIPLE_LINES",
        "INTERNETSERVICE": "INTERNET_SERVICE",
        "ONLINESECURITY": "ONLINE_SECURITY",
        "ONLINEBACKUP": "ONLINE_BACKUP",
        "DEVICEPROTECTION": "DEVICE_PROTECTION",
        "TECHSUPPORT": "TECH_SUPPORT",
        "STREAMINGTV": "STREAMING_TV",
        "STREAMINGMOVIES": "STREAMING_MOVIES",
        "PAPERLESSBILLING": "PAPERLESS_BILLING",
        "PAYMENTMETHOD": "PAYMENT_METHOD",
        "MONTHLYCHARGES": "MONTHLY_CHARGES",
        "TOTALCHARGES": "TOTAL_CHARGES"
    }
    
    dataframe.rename(
        columns = column_change_dict, inplace = True
    )

    # Drop Columns
    columns_to_drop = [
        'ID', 'PAYMENT_METHOD'
    ]

    # Encoding
    dataframe['GENDER'] = dataframe['GENDER'].map({'Female': 0, 'Male': 1})
    dataframe['PARTNER'] = dataframe['PARTNER'].map({'No': 0, 'Yes': 1})
    dataframe['DEPENDENTS'] = dataframe['DEPENDENTS'].map({'No': 0, 'Yes': 1})
    dataframe['PHONE_SERVICE'] = dataframe['PHONE_SERVICE'].map({'No': 0, 'Yes': 1})
    dataframe['MULTIPLE_LINES'] = dataframe['MULTIPLE_LINES'].map({'No': 0, 'No phone service': -1, 'Yes': 1})
    dataframe['INTERNET_SERVICE'] = dataframe['INTERNET_SERVICE'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
    dataframe['ONLINE_SECURITY'] = dataframe['ONLINE_SECURITY'].map({'No': 0, 'No internet service': -1, 'Yes': 1})
    dataframe['ONLINE_BACKUP'] = dataframe['ONLINE_BACKUP'].map({'No': 0, 'No internet service': -1, 'Yes': 1})
    dataframe['DEVICE_PROTECTION'] = dataframe['DEVICE_PROTECTION'].map({'No': 0, 'No internet service': -1, 'Yes': 1})
    dataframe['TECH_SUPPORT'] = dataframe['TECH_SUPPORT'].map({'No': 0, 'No internet service': -1, 'Yes': 1})
    dataframe['STREAMING_TV'] = dataframe['STREAMING_TV'].map({'No': 0, 'No internet service': -1, 'Yes': 1})
    dataframe['STREAMING_MOVIES'] = dataframe['STREAMING_MOVIES'].map({'No': 0, 'No internet service': -1, 'Yes': 1})
    dataframe['CONTRACT'] = dataframe['CONTRACT'].map({'Month-to-month': -1, 'One year': 0, 'Two year': 1})
    dataframe['PAPERLESS_BILLING'] = dataframe['PAPERLESS_BILLING'].map({'No': 0, 'Yes': 1})

    # --------------------------------------- Feature Engineering ---------------------------------------
    # Actual Total Charges incured
    dataframe['ACT_TOTAL_CHARGES'] = dataframe['TENURE'] * dataframe['MONTHLY_CHARGES']

    # TC_RATIO - How much they actually paid vs expected
    dataframe['CHARGES_PAID_RATIO'] = dataframe['TOTAL_CHARGES'] / dataframe['ACT_TOTAL_CHARGES']

    # Average monthly spend over entire tenure
    dataframe['AVG_MONTLY_CHARGES'] = dataframe['TOTAL_CHARGES'] / dataframe['TENURE']

    # Number of Streaming Services
    dataframe['IS_STREAMING_ANYTHING'] = (
        (dataframe['STREAMING_MOVIES'] == 1) | (dataframe['STREAMING_TV'] == 1)
    ).astype(int)
    
    dataframe['NUM_STREAMING_SERVICES'] = (
        (dataframe[['STREAMING_MOVIES', 'STREAMING_TV']] == 1)
    ).sum(axis = 1)

    # Number of Additional Services
    dataframe['NUM_ADDITIONAL_SERVICES'] = (
        (dataframe[['ONLINE_SECURITY', 'ONLINE_BACKUP', 'DEVICE_PROTECTION', 'TECH_SUPPORT']] == 1)
    ).sum(axis = 1)

    # Total Number of Services
    dataframe['NUM_TOTAL_SERVICES'] = dataframe['NUM_STREAMING_SERVICES'] + dataframe['NUM_ADDITIONAL_SERVICES']

    # Payment Method
    dataframe['PAYMENT_METHOD'] = dataframe['PAYMENT_METHOD'].map(
        {'Bank transfer (automatic)': 'BT', 'Credit card (automatic)': 'CC', 'Electronic check': 'EC', 'Mailed check': 'MC'}
    )

    # Payment Categories
    payment_categories = [
        'BT', 'CC', 'EC', 'MC'
    ]
    for method in payment_categories:
        dataframe[f'PAYMENT_{method}'] = (dataframe['PAYMENT_METHOD'] == method).astype(int)

    # Is Payment Automatic
    dataframe['IS_PAYMENT_AUTOMATIC'] = dataframe['PAYMENT_METHOD'].isin(
        ['BT', 'CC']
    ).astype(int)

    # Check if 'CHURN' is dataframe
    if('CHURN' in dataframe.columns):
        dataframe['CHURN'] = dataframe['CHURN'].map({'No': 0, 'Yes': 1})

    # Drop columns
    dropped_columns = [
        col for col in columns_to_drop if col in dataframe.columns
    ]
    
    dataframe = dataframe.drop(columns = dropped_columns)
    return dataframe


def split_features_and_target(
    dataframe: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    
    if('CHURN' not in dataframe.columns):
        raise ValueError("'CHURN' column not found in dataframe.")
    target = dataframe['CHURN']
    features = dataframe.drop(columns = ['CHURN'])
    return features, target