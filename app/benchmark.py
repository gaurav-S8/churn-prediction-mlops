# Import Custom Modules
from app.db import execute_query

def parse_row(row):
    return {
        "total_requests": row[1],
        "lgb": {
            "min_ms": row[2], "avg_ms": row[3], "max_ms": row[4], "p95_ms": row[5]
        },
        "xgb": {
            "min_ms": row[6], "avg_ms": row[7], "max_ms": row[8], "p95_ms": row[9]
        },
        "cat": {
            "min_ms": row[10], "avg_ms": row[11], "max_ms": row[12], "p95_ms": row[13]
        },
        "ensemble": {
            "min_ms": row[14], "avg_ms": row[15], "max_ms": row[16], "p95_ms": row[17]
        },
        "end_to_end": {
            "min_ms": row[18], "avg_ms": row[19], "max_ms": row[20], "p95_ms": row[21]
        },
        "avg_overhead_ms": row[22]
    }

def get_benchmark_metrics():
    result = {}

    rows = execute_query(
        """
            SELECT
                MODEL_ROLE,
                COUNT(*) AS TOTAL_REQUESTS,
                ROUND(CAST(MIN(LGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS MIN_LGB_MS,
                ROUND(CAST(AVG(LGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS AVG_LGB_MS,
                ROUND(CAST(MAX(LGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS MAX_LGB_MS,
                ROUND(CAST(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY LGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS P95_LGB_MS,
                ROUND(CAST(MIN(XGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS MIN_XGB_MS,
                ROUND(CAST(AVG(XGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS AVG_XGB_MS,
                ROUND(CAST(MAX(XGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS MAX_XGB_MS,
                ROUND(CAST(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY XGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS P95_XGB_MS,
                ROUND(CAST(MIN(CAT_INFER_TIME) * 1000 AS NUMERIC), 2) AS MIN_CAT_MS,
                ROUND(CAST(AVG(CAT_INFER_TIME) * 1000 AS NUMERIC), 2) AS AVG_CAT_MS,
                ROUND(CAST(MAX(CAT_INFER_TIME) * 1000 AS NUMERIC), 2) AS MAX_CAT_MS,
                ROUND(CAST(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY CAT_INFER_TIME) * 1000 AS NUMERIC), 2) AS P95_CAT_MS,
                ROUND(CAST(MIN(ENSEMBLE_INFER_TIME) * 1000 AS NUMERIC), 2) AS MIN_ENSEMBLE_MS,
                ROUND(CAST(AVG(ENSEMBLE_INFER_TIME) * 1000 AS NUMERIC), 2) AS AVG_ENSEMBLE_MS,
                ROUND(CAST(MAX(ENSEMBLE_INFER_TIME) * 1000 AS NUMERIC), 2) AS MAX_ENSEMBLE_MS,
                ROUND(CAST(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ENSEMBLE_INFER_TIME) * 1000 AS NUMERIC), 2) AS P95_ENSEMBLE_MS,
                ROUND(CAST(MIN(END_TO_END_LATENCY) * 1000 AS NUMERIC), 2) AS MIN_E2E_MS,
                ROUND(CAST(AVG(END_TO_END_LATENCY) * 1000 AS NUMERIC), 2) AS AVG_E2E_MS,
                ROUND(CAST(MAX(END_TO_END_LATENCY) * 1000 AS NUMERIC), 2) AS MAX_E2E_MS,
                ROUND(CAST(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY END_TO_END_LATENCY) * 1000 AS NUMERIC), 2) AS P95_E2E_MS,
                ROUND(CAST(AVG(END_TO_END_LATENCY - ENSEMBLE_INFER_TIME) * 1000 AS NUMERIC), 2) AS AVG_OVERHEAD_MS
            FROM predictions
            WHERE MODEL_RUN_ID IN (
                SELECT RUN_ID FROM model_versions WHERE IS_ACTIVE = 1
            )
            GROUP BY MODEL_ROLE
            ORDER BY MODEL_ROLE DESC
        """,
        ()
    )
    
    for row in rows:
        result[row[0]] = parse_row(row)
    
    total_row = execute_query(
        """
            SELECT
                'TOTAL' AS MODEL_ROLE,
                COUNT(*) AS TOTAL_REQUESTS,
                ROUND(CAST(MIN(LGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS MIN_LGB_MS,
                ROUND(CAST(AVG(LGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS AVG_LGB_MS,
                ROUND(CAST(MAX(LGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS MAX_LGB_MS,
                ROUND(CAST(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY LGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS P95_LGB_MS,
                ROUND(CAST(MIN(XGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS MIN_XGB_MS,
                ROUND(CAST(AVG(XGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS AVG_XGB_MS,
                ROUND(CAST(MAX(XGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS MAX_XGB_MS,
                ROUND(CAST(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY XGB_INFER_TIME) * 1000 AS NUMERIC), 2) AS P95_XGB_MS,
                ROUND(CAST(MIN(CAT_INFER_TIME) * 1000 AS NUMERIC), 2) AS MIN_CAT_MS,
                ROUND(CAST(AVG(CAT_INFER_TIME) * 1000 AS NUMERIC), 2) AS AVG_CAT_MS,
                ROUND(CAST(MAX(CAT_INFER_TIME) * 1000 AS NUMERIC), 2) AS MAX_CAT_MS,
                ROUND(CAST(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY CAT_INFER_TIME) * 1000 AS NUMERIC), 2) AS P95_CAT_MS,
                ROUND(CAST(MIN(ENSEMBLE_INFER_TIME) * 1000 AS NUMERIC), 2) AS MIN_ENSEMBLE_MS,
                ROUND(CAST(AVG(ENSEMBLE_INFER_TIME) * 1000 AS NUMERIC), 2) AS AVG_ENSEMBLE_MS,
                ROUND(CAST(MAX(ENSEMBLE_INFER_TIME) * 1000 AS NUMERIC), 2) AS MAX_ENSEMBLE_MS,
                ROUND(CAST(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ENSEMBLE_INFER_TIME) * 1000 AS NUMERIC), 2) AS P95_ENSEMBLE_MS,
                ROUND(CAST(MIN(END_TO_END_LATENCY) * 1000 AS NUMERIC), 2) AS MIN_E2E_MS,
                ROUND(CAST(AVG(END_TO_END_LATENCY) * 1000 AS NUMERIC), 2) AS AVG_E2E_MS,
                ROUND(CAST(MAX(END_TO_END_LATENCY) * 1000 AS NUMERIC), 2) AS MAX_E2E_MS,
                ROUND(CAST(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY END_TO_END_LATENCY) * 1000 AS NUMERIC), 2) AS P95_E2E_MS,
                ROUND(CAST(AVG(END_TO_END_LATENCY - ENSEMBLE_INFER_TIME) * 1000 AS NUMERIC), 2) AS AVG_OVERHEAD_MS
            FROM predictions
            WHERE MODEL_RUN_ID IN (
                SELECT RUN_ID FROM model_versions WHERE IS_ACTIVE = 1
            )
        """,
        ()
    )

    if total_row:
        t = total_row[0]
        result['total'] = parse_row(t)
    
    for role in ['champion', 'challenger', 'total']:
        if role not in result:
            result[role] = {
                "total_requests": 0,
                "lgb": {"min_ms": None, "avg_ms": None, "max_ms": None, "p95_ms": None},
                "xgb": {"min_ms": None, "avg_ms": None, "max_ms": None, "p95_ms": None},
                "cat": {"min_ms": None, "avg_ms": None, "max_ms": None, "p95_ms": None},
                "ensemble": {"min_ms": None, "avg_ms": None, "max_ms": None, "p95_ms": None},
                "end_to_end": {"min_ms": None, "avg_ms": None, "max_ms": None, "p95_ms": None},
                "avg_overhead_ms": None
            }
    return result

def get_ab_metrics():
    rows = execute_query(
        """
            SELECT
                MODEL_ROLE,
                COUNT(*) as total_requests,
                ROUND(CAST(AVG(CHURN_PROBABILITY) AS NUMERIC), 4) as avg_churn_probability,
                ROUND(CAST(SUM(CASE WHEN CHURN_PREDICTION = 'Yes' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS NUMERIC), 4) as churn_rate
            FROM predictions
            WHERE MODEL_RUN_ID IN (
                SELECT RUN_ID FROM model_versions WHERE IS_ACTIVE = 1
            )
            GROUP BY MODEL_ROLE
        """,
        ()
    )

    if not rows:
        return {"message": "No requests yet"}

    result = {}
    for row in rows:
        result[row[0]] = {
            "total_requests": row[1],
            "avg_churn_probability": row[2],
            "churn_rate": row[3]
        }

    result["note"] = "Statistical significance testing requires ground truth labels. Planned for future implementation."
    return result