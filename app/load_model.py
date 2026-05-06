# Import Libraries
import os
import time
import json
import joblib
from dotenv import load_dotenv

_cache = {}

def load_model(path):
    global _cache
    if path in _cache:
        return _cache[path]

    try:
        # Get weighted model weights
        with open(os.path.join(path, 'weights.json')) as f:
            weights = json.load(f)
        
        lgb_model = joblib.load(os.path.join(path, 'lgb_model.pkl'))
        xgb_model = joblib.load(os.path.join(path, 'xgb_model.pkl'))
        cat_model = joblib.load(os.path.join(path, 'cat_model.pkl'))

        _cache[path] = {
            'lgb_model': lgb_model,
            'xgb_model': xgb_model,
            'cat_model': cat_model,
            'w_lgb': weights.get('w_lgb'),
            'w_xgb': weights.get('w_xgb'),
            'w_cat': weights.get('w_cat')
        }
        return _cache[path]
    except:
        return None