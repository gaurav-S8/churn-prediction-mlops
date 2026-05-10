# Import Libraries
import os
import time
import json
import joblib
from dotenv import load_dotenv

# Import Custom Modules
from app.registry import load_model_weights

_cache = {}

def load_model(model_role, path):
    global _cache
    if path in _cache:
        return _cache[path]
    try:
        # Load models from the appropriate folder   
        lgb_model = joblib.load(os.path.join(path, 'lgb_model.pkl'))
        xgb_model = joblib.load(os.path.join(path, 'xgb_model.pkl'))
        cat_model = joblib.load(os.path.join(path, 'cat_model.pkl'))
        
        # Get model weights
        weights, run_id = load_model_weights(model_role)

        _cache[path] = {
            'lgb_model': lgb_model,
            'xgb_model': xgb_model,
            'cat_model': cat_model,
            'w_lgb': weights.get('w_lgb'),
            'w_xgb': weights.get('w_xgb'),
            'w_cat': weights.get('w_cat'),
            'run_id': run_id
        }
        return _cache[path]
    except:
        return None