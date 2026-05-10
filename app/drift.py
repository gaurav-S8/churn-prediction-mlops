# Import Libraries
import pandas as pd
from evidently.presets import *
from fastapi.responses import HTMLResponse
from evidently import Report, Dataset, DataDefinition

# Import Custom Modules
from app.monitoring import get_recent_inputs

def get_drift_report():
    reference_df = pd.read_csv("data/reference.csv")
    current_df = get_recent_inputs(limit = 100)

    if len(current_df) < 10:
        return None, {"message": "Not enough data. Need at least 10 requests"}
    
    numerical_columns = reference_df.select_dtypes(include = ['int64', 'float64']).columns.tolist()
    categorical_columns = [col for col in reference_df.columns if col not in numerical_columns]

    schema = DataDefinition(
        numerical_columns = numerical_columns,
        categorical_columns = categorical_columns
    )

    eval_data_1 = Dataset.from_pandas(
        current_df,
        data_definition = schema
    )

    eval_data_2 = Dataset.from_pandas(
        reference_df,
        data_definition = schema
    )

    report = Report([
        DataDriftPreset() 
    ])

    my_evaluation = report.run(eval_data_1, eval_data_2)

    result = my_evaluation.dict()
    metrics = result['metrics']

    # First metric is the overall drift count
    drift_count = metrics[0]['value']['count']
    drift_share = metrics[0]['value']['share']

    # Rest are per-feature drift values
    feature_drift = {}
    for metric in metrics[1:]:
        column = metric['config']['column']
        p_value = float(metric['value'])
        feature_drift[column] = {
            'p_value': round(p_value, 4),
            'drift_detected': p_value < 0.05
        }

    summary = {
        'drift_detected': drift_count > 0,
        'drifted_features_count': int(drift_count),
        'drifted_features_share': round(drift_share, 4),
        'total_features': len(feature_drift),
        'feature_drift': feature_drift
    }
    return my_evaluation, summary