# Import Libraries
import os
import sys
import mlflow
import pandas as pd
from dotenv import load_dotenv

# Import ML Libraries
import optuna
import xgboost as xgb
import catboost as cat
import lightgbm as lgb
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Import Custom Modules
from src.optuna_hp_optimization import run_study, optimize_ensemble_weights
from src.preprocess import preprocess_and_engineer_feature, split_features_and_target

# Load Environment Variables
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Churn_Prediction")

# Load Dataset
train_df = pd.read_csv(f'{DATA_PATH}/train.csv')
print(f"Length of Train dataframe: {len(train_df)}")

# Preprocess dataset
train_df = preprocess_and_engineer_feature(train_df)

# Split train dataframe in (Features, Target)
X_full, y_full = split_features_and_target(train_df)

# Initialize Models
lgb_model = Pipeline(
    [
        (
            'model', lgb.LGBMClassifier(
                verbose = 0
            )
        )
    ]
)

xgb_model = Pipeline(
    [
        (
            'model', xgb.XGBClassifier(
                verbose = 0
            )
        )
    ]
)

cat_model = Pipeline(
    [
        (
            'model', cat.CatBoostClassifier(
                verbose = 0,
                allow_writing_files = False
            )
        )
    ]
)

# Model Dictionary
models = {
    'lgb': lgb_model,
    'xgb': xgb_model,
    'cat': cat_model
}

# ------------------------------------- MLFLOW Experiment -------------------------------------
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
seeds = [42, 123, 456, 789, 2024]
try:
    with mlflow.start_run(run_name = "ensemble_weighted"):
        # Hyper-parameter optimization
        best_params = {}
        best_scores = {}
        for name, model in models.items():
            study = run_study(base_model = model, model_name = name, X = X_full, y = y_full, n_trials = 1, cv = 5, seed = 42)
            best_params[name] = study.best_params
            best_scores[name] = study.best_value
            mlflow.log_params({f'{name}_{k}': v for k, v in best_params[name].items()})
            mlflow.log_metric(f'cv_auc_{name}', best_scores[name])

        print(best_params)
        
        # Create Best Models
        lgb_best = clone(models['lgb'])
        lgb_best.set_params(**{f'model__{k}': v for k, v in best_params['lgb'].items()})
    
        xgb_best = clone(models['xgb'])
        xgb_best.set_params(**{f'model__{k}': v for k, v in best_params['xgb'].items()})
        
        cat_best = clone(models['cat'])
        cat_best.set_params(**{f'model__{k}': v for k, v in best_params['cat'].items()})
    
        # OOF Prediction
        oof_lgb = cross_val_predict(lgb_best, X_full, y_full, cv = skf, method = 'predict_proba')[:, 1]
        oof_xgb = cross_val_predict(xgb_best, X_full, y_full, cv = skf, method = 'predict_proba')[:, 1]
        oof_cat = cross_val_predict(cat_best, X_full, y_full, cv = skf, method = 'predict_proba')[:, 1]
        
        # Optimize Weights
        weight_study = optuna.create_study(direction = 'maximize')
        weight_study.optimize(optimize_ensemble_weights(oof_lgb, oof_xgb, oof_cat, y_full), n_trials = 500, show_progress_bar = True)
    
        w_lgb = weight_study.best_params['w_lgb']
        w_xgb = weight_study.best_params['w_xgb']
        w_cat  = 1 - w_lgb - w_xgb
        oof_ensemble = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
    
        # Fit the final model
        lgb_best.fit(X_full, y_full)
        xgb_best.fit(X_full, y_full)
        cat_best.fit(X_full, y_full)

        # Log all metrics and models
        mlflow.log_metrics(
            {
                'w_lgb': w_lgb,
                'w_xgb': w_xgb,
                'w_cat': w_cat,
                'oof_auc_lgb': roc_auc_score(y_full, oof_lgb),
                'oof_auc_xgb': roc_auc_score(y_full, oof_xgb),
                'oof_auc_cat': roc_auc_score(y_full, oof_cat),
                'oof_auc_ensemble': roc_auc_score(y_full, oof_ensemble)
            }
        )
        mlflow.log_params({'seeds': str(seeds)})
        
        mlflow.sklearn.log_model(
            sk_model = lgb_best,
            name = "lgb_model"
        )
        mlflow.sklearn.log_model(
            sk_model = xgb_best,
            name = "xgb_model"
        )
        mlflow.sklearn.log_model(
            sk_model = cat_best,
            name = "cat_model"
        )
        print("Training DONE!!!")
        
except Exception as e:
    print(f"Training failed: {e}")
    mlflow.end_run(status = "FAILED")
    raise e