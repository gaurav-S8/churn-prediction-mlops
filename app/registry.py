# Import Libraries
import os
import json

# Import Custom Modules
from app.db import execute_query

def load_model_weights(model_role):
    rows = execute_query(
        """
            SELECT MODEL_WEIGHTS, RUN_ID
            FROM model_versions
            WHERE UPPER(ROLE) = %s
            AND IS_ACTIVE = 1;
        """,
        (model_role.upper(), )
    )
    if not rows:
        return None, None
    
    result = rows[0]
    return result[0], result[1]

def sync_model_registry():
    for role in ["champion", "challenger"]:
        model_info_path = f"models/{role}/model_info.json"
        
        if not os.path.exists(model_info_path):
            continue
        
        with open(model_info_path) as f:
            model_info = json.load(f)
        
        run_id = model_info['run_id']
        
        # Check if this run_id already exists
        existing = execute_query(
            "SELECT RUN_ID FROM model_versions WHERE UPPER(ROLE) = %s AND IS_ACTIVE = 1;",
            (role.upper(), )
        )

        if existing:
            # Already registered, skip
            continue
        
        # Retire current active model for this role
        execute_query(
            """
                UPDATE model_versions
                SET IS_ACTIVE = 0, RETIRED_AT = NOW()
                WHERE UPPER(ROLE) = %s AND IS_ACTIVE = 1;
            """,
            (role.upper(), )
        )
        
        # Insert new model
        execute_query(
            """
                INSERT INTO model_versions (
                    RUN_ID, ROLE, TRAINED_AT, IS_ACTIVE,
                    MODEL_WEIGHTS, ROC_AUC_SCORES,
                    LGB_PARAMETERS, XGB_PARAMETERS, CAT_PARAMETERS
                ) VALUES (%s, %s, %s, 1, %s, %s, %s, %s, %s);
            """,
            (
                run_id,
                role,
                model_info['trained_at'],
                json.dumps(model_info['weights']),
                json.dumps(model_info['oof_roc_auc_scores']),
                json.dumps(model_info['parameters']['lgb']),
                json.dumps(model_info['parameters']['xgb']),
                json.dumps(model_info['parameters']['cat'])
            )
        )
        print(f"Registered {role} model with run_id: {run_id[:8]}...")