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

def get_current_run_id_by_role(role):
    # Get run_id for this model
    existing = execute_query(
        """
            SELECT
                RUN_ID
            FROM model_versions
            WHERE UPPER(ROLE) = %s AND 
                IS_ACTIVE = 1;
        """,
        (role.upper(), )
    )
    return existing


def update_model_registry_info(run_id, role):
    query = execute_query(
        """
            UPDATE model_versions
            SET
            (
                NUM_API_REQ_SERVED,
                AVG_LGB_INFER_TIME,
                AVG_XGB_INFER_TIME,
                AVG_CAT_INFER_TIME,
                AVG_ENSEMBLE_INFER_TIME,
                AVG_END_TO_END_LATENCY
            ) =
            (
                SELECT
                    COUNT(*),
                    AVG(LGB_INFER_TIME),
                    AVG(XGB_INFER_TIME),
                    AVG(CAT_INFER_TIME),
                    AVG(ENSEMBLE_INFER_TIME),
                    AVG(END_TO_END_LATENCY)
                FROM predictions
                WHERE MODEL_RUN_ID = %s
                AND UPPER(MODEL_ROLE) = %s
            )
            WHERE RUN_ID = %s
            AND UPPER(ROLE) = %s
            AND IS_ACTIVE = 1;
        """,
        (run_id, role.upper(), run_id, role.upper(), )
    )

def sync_model_registry():
    for role in ["champion", "challenger"]:
        model_info_path = f"models/{role}/model_info.json"
        
        if not os.path.exists(model_info_path):
            continue
        
        with open(model_info_path) as f:
            model_info = json.load(f)
        
        run_id = model_info['run_id']

        existing = get_current_run_id_by_role(role)

        current_run_id = None
        if existing:
            current_run_id = existing[0][0]
            # Check if run_id already exists
            if current_run_id == run_id:
                # Already registered, skip
                continue
        
        # Update metrics for this model
        if current_run_id:
            update_model_registry_info(current_run_id, role)
        
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
        # print(f"Registered {role} model with run_id: {run_id[:8]}...")
    
def get_model_version_info():
    for role in ["champion", "challenger"]:
        existing = get_current_run_id_by_role(role)

        current_run_id = None
        if existing:
            current_run_id = existing[0][0]
            if current_run_id:
                update_model_registry_info(current_run_id, role)
    
    rows = execute_query("""
        SELECT
            DISTINCT RUN_ID,
            ROLE,
            TRAINED_AT,
            PROMOTED_AT,
            RETIRED_AT,
            IS_ACTIVE,
            NUM_API_REQ_SERVED,
            AVG_LGB_INFER_TIME,
            AVG_XGB_INFER_TIME,
            AVG_CAT_INFER_TIME,
            AVG_ENSEMBLE_INFER_TIME,
            AVG_END_TO_END_LATENCY
        FROM model_versions
        ORDER BY
            IS_ACTIVE ASC,
            PROMOTED_AT DESC;
    """)

    models = []
    for row in rows:
        models.append({
            "run_id": row[0],
            "role": row[1],
            "trained_at": row[2],
            "promoted_at": row[3],
            "retired_at": row[4],
            "is_active": bool(row[5]),
            "num_api_req_served": row[6],
            "latency": {
                "lgb_ms": row[7],
                "xgb_ms": row[8],
                "cat_ms": row[9],
                "ensemble_ms": row[10],
                "end_to_end_ms": row[11]
            }
        })
    return {"models": models}