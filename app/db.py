# Import Libraries
import json
import psycopg2
import pandas as pd
from psycopg2 import pool

# Import Custom Modules
from app.config.settings import DATABASE_URL

connection_pool = None
def init_pool():
    global connection_pool
    connection_pool = pool.SimpleConnectionPool(1, 10, DATABASE_URL)

def get_connection():
    return psycopg2.connect(DATABASE_URL)

def execute_query(query, params = None):
    conn = connection_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        try:
            result = cursor.fetchall()
        except:
            result = None
        conn.commit()
        cursor.close()
        return result
    finally:
        connection_pool.putconn(conn)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    # Create prediction table if it doesn't exists!!
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS predictions (
                ID SERIAL PRIMARY KEY,
                PREDICTION_ID VARCHAR(255),
                CUSTOMER_ID VARCHAR(255),
                MODEL_ROLE VARCHAR(20),
                MODEL_RUN_ID VARCHAR(500),
                CHURN_PREDICTION VARCHAR(10),
                CHURN_PROBABILITY FLOAT,
                LGB_PREDICTION FLOAT,
                XGB_PREDICTION FLOAT,
                CAT_PREDICTION FLOAT,
                LGB_INFER_TIME FLOAT,
                XGB_INFER_TIME FLOAT,
                CAT_INFER_TIME FLOAT,
                ENSEMBLE_INFER_TIME FLOAT,
                END_TO_END_LATENCY FLOAT,
                TIMESTAMP TIMESTAMP DEFAULT NOW()
            );
        """
    )
    # Create raw_input table if it doesn't exists!!
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS raw_inputs (
                ID SERIAL PRIMARY KEY,
                PREDICTION_ID VARCHAR(255),
                CUSTOMER_ID VARCHAR(255),
                GENDER VARCHAR(10),
                SENIOR_CITIZEN INTEGER,
                PARTNER VARCHAR(5),
                DEPENDENTS VARCHAR(5),
                TENURE INTEGER,
                PHONE_SERVICE VARCHAR(5),
                MULTIPLE_LINES VARCHAR(20),
                INTERNET_SERVICE VARCHAR(20),
                ONLINE_SECURITY VARCHAR(20),
                ONLINE_BACKUP VARCHAR(20),
                DEVICE_PROTECTION VARCHAR(20),
                TECH_SUPPORT VARCHAR(20),
                STREAMING_TV VARCHAR(20),
                STREAMING_MOVIES VARCHAR(20),
                CONTRACT VARCHAR(20),
                PAPERLESS_BILLING VARCHAR(5),
                PAYMENT_METHOD VARCHAR(40),
                MONTHLY_CHARGES FLOAT,
                TOTAL_CHARGES FLOAT,
                TIMESTAMP TIMESTAMP DEFAULT NOW()
            );
        """
    )
    # Create model_version table if it doesn't exists!!
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS model_versions (
                RUN_ID VARCHAR(500) PRIMARY KEY,
                ROLE VARCHAR(20),
                TRAINED_AT TIMESTAMP,
                PROMOTED_AT TIMESTAMP DEFAULT NOW(),
                RETIRED_AT TIMESTAMP,
                IS_ACTIVE INT,
                MODEL_WEIGHTS JSONB,
                ROC_AUC_SCORES JSONB,
                LGB_PARAMETERS JSONB,
                XGB_PARAMETERS JSONB,
                CAT_PARAMETERS JSONB,
                NUM_API_REQ_SERVED INT,
                AVG_LGB_INFER_TIME FLOAT,
                AVG_XGB_INFER_TIME FLOAT,
                AVG_CAT_INFER_TIME FLOAT,
                AVG_ENSEMBLE_INFER_TIME FLOAT,
                AVG_END_TO_END_LATENCY FLOAT
            );
        """
    )

    conn.commit()
    cursor.close()
    conn.close()