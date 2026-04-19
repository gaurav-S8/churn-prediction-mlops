# Import Libraries
import optuna
from sklearn.base import clone
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

def get_params(trial, model_name):
    if(model_name == 'lgb'):
        return {
            'model__n_estimators': trial.suggest_categorical('n_estimators', [300, 500, 700, 900]),
            'model__max_depth': trial.suggest_int('max_depth', 3, 7),
            'model__learning_rate': trial.suggest_categorical('learning_rate', [0.05, 0.1, 0.2, 0.3, 0.5]),
            'model__subsample': trial.suggest_float('subsample', 0.6, 1, step = 0.1),
            'model__colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step = 0.1),
            'model__min_child_samples': trial.suggest_categorical('min_child_samples', [5, 10, 20, 30, 50]),
            'model__num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63, 127]),
        }
    elif(model_name == 'xgb'):
        return {
            'model__n_estimators': trial.suggest_categorical('n_estimators', [300, 500, 700, 900]),
            'model__max_depth': trial.suggest_int('max_depth', 3, 7),
            'model__learning_rate': trial.suggest_categorical('learning_rate', [0.05, 0.1, 0.2, 0.3, 0.5]),
            'model__subsample': trial.suggest_float('subsample', 0.6, 1.0, step = 0.1),
            'model__colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step = 0.1),
            'model__min_child_weight': trial.suggest_categorical('min_child_weight', [1, 3, 5, 7]),
            'model__gamma': trial.suggest_categorical('gamma', [0, 0.1, 0.2, 0.3]),
        }
    elif(model_name == 'cat'):
        return {
            'model__iterations': trial.suggest_categorical('iterations', [300, 500, 700, 900]),
            'model__depth': trial.suggest_int('depth', 3, 7),
            'model__learning_rate': trial.suggest_categorical('learning_rate', [0.05, 0.1, 0.2, 0.3, 0.5]),
            'model__l2_leaf_reg': trial.suggest_categorical('l2_leaf_reg', [1, 3, 5, 7, 9]),
            'model__subsample': trial.suggest_float('subsample', 0.6, 1.0, step = 0.1),
            'model__colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0, step = 0.1), 
        }
    else:
        raise ValueError(f"{model_name}: Model Type not supported!!")

def make_objective(base_model, model_name, X, y, cv = 3):
    def objective(trial):
        params = get_params(trial, model_name)
        model = clone(base_model)
        model.set_params(**params)
        skf = StratifiedKFold(n_splits = cv, shuffle = True, random_state = 42)
        scores = cross_val_score(model, X, y, cv = skf, scoring = 'roc_auc', n_jobs = 1)
        return scores.mean()
    return objective

def run_study(base_model, model_name, X, y, n_trials = 50, cv = 3, seed = 42):
    study = optuna.create_study(
        direction = 'maximize',
        sampler = TPESampler(seed = seed),
        study_name = model_name
    )
    study.optimize(
        make_objective(base_model, model_name, X, y, cv),
        n_trials = n_trials,
        n_jobs = -1,
        show_progress_bar = True
    )
    return study

def optimize_ensemble_weights(oof_lgb, oof_xgb, oof_cat, y):
    def objective(trial):
        w_lgb = trial.suggest_float('w_lgb', 0.1, 0.8)
        w_xgb = trial.suggest_float('w_xgb', 0.1, 0.8)
        w_cat = 1 - w_lgb - w_xgb
    
        combined = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
        return roc_auc_score(y, combined)
    return objective