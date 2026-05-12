# Import Libraries
import time
import shap
import random
import pandas as pd

# Import Custom Modules
from app.schemas import CustomerData
from utils.preprocess import preprocess_and_engineer_feature

def infer_model(model, dataframe):
    start_time = time.time()
    pred = model.predict_proba(dataframe)[:, 1][0]
    return pred, time.time() - start_time

def choose_model(champion_ensemble, challenger_ensemble):
    if challenger_ensemble and random.random() > 0.8:
        return challenger_ensemble, 'challenger'
    return champion_ensemble, 'champion'

def prepare_input(customer: CustomerData):
    input_df = pd.DataFrame([customer.model_dump()])
    processed_df = preprocess_and_engineer_feature(input_df)
    customer_id = processed_df['CUSTOMERID']
    processed_df = processed_df.drop(columns = ['CUSTOMERID'])
    return customer_id, processed_df

def run_ensemble(ensemble, processed_df):
    start_time = time.time()
    lgb_model = ensemble['lgb_model']
    xgb_model = ensemble['xgb_model']
    cat_model = ensemble['cat_model']
    w_lgb = ensemble['w_lgb']
    w_xgb = ensemble['w_xgb']
    w_cat = ensemble['w_cat']
    
    pred_lgb, lgb_infer_time = infer_model(lgb_model, processed_df)
    pred_xgb, xgb_infer_time = infer_model(xgb_model, processed_df)
    pred_cat, cat_infer_time = infer_model(cat_model, processed_df)

    final_pred = pred_lgb * w_lgb + pred_xgb * w_xgb + pred_cat * w_cat
    return {
        'churn_probability': round(float(final_pred), 4),
        'churn_prediction': 'Yes' if final_pred > 0.5 else 'No',
        'model_predictions': {
            'lgb': round(float(pred_lgb), 4),
            'xgb': round(float(pred_xgb), 4),
            'cat': round(float(pred_cat), 4)
        },
    }, {
        'lgb_infer_time': lgb_infer_time,
        'xgb_infer_time': xgb_infer_time,
        'cat_infer_time': cat_infer_time,
        'ensemble_infer_time': time.time() - start_time
    }

def run_shap_explainability(ensemble, processed_df):
    lgb_model = ensemble['lgb_model']
    cat_model = ensemble['cat_model']
    w_lgb = ensemble['w_lgb']
    w_cat = ensemble['w_cat']

    # SHAP explainers for each model
    explainer_lgb = shap.TreeExplainer(lgb_model.named_steps['model'])
    explainer_cat = shap.TreeExplainer(cat_model.named_steps['model'])
    
    # SHAP values for churn class
    shap_lgb = explainer_lgb.shap_values(processed_df)[0]
    shap_cat = explainer_cat.shap_values(processed_df)[0]

    # Weighted average — Same weights as ensemble
    total_weight = w_lgb + w_cat
    shap_ensemble = (w_lgb/total_weight) * shap_lgb + (w_cat/total_weight) * shap_cat

    feature_importance = dict(
        zip(
            processed_df.columns.tolist(),
            shap_ensemble.tolist()
        )
    )

    # Sort by absolute importance
    feature_importance = dict(
        sorted(
            feature_importance.items(),
            key = lambda x: abs(x[1]),
            reverse = True
        )
    )
    return {'shap_values': feature_importance}