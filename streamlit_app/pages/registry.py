# Import Libraries
import pandas as pd
import streamlit as st

# Import Custom Modules
from utils.api import cached_get

def render():

    st.markdown("""
        <div class="page-heading">
            <div class="page-title">Model Registry</div>
            <div class="page-desc">
                <span class="badge badge-get">GET</span>
                /model-registry — Active and retired model versions
            </div>
        </div>
    """, unsafe_allow_html = True)

    data, status = cached_get("/model-registry")

    if status != 200 or not data:
        st.markdown("""
            <div class="card">
                <div class="card-label">Status</div>
                <div style="color: rgba(255,255,255,0.4); font-size: 14px; margin-top: 8px;">
                    Model registry endpoint is live.
                    No data available yet.
                </div>
            </div>
        """, unsafe_allow_html = True)
        return

    models = data.get("models", [])
    if not models:
        st.warning("No registered models found.")
        return
    rows = []

    for model in models:
        latency = model.get("latency", {})
        rows.append(
            {
                "Role": model.get("role", "").upper(),
                "Status": "ACTIVE" if model.get("is_active") else "RETIRED",
                "Run ID": model.get("run_id", "")[:12] + "...",
                "Requests": model.get("num_api_req_served", 0),
                "Promoted": model.get("promoted_at"),
                "Retired": model.get("retired_at"),
                "LGB (ms)": round((latency.get("lgb_ms") or 0) * 1000, 3),
                "XGB (ms)": round((latency.get("xgb_ms") or 0) * 1000, 3),
                "CAT (ms)": round((latency.get("cat_ms") or 0) * 1000, 3),
                "Ensemble (ms)": round((latency.get("ensemble_ms") or 0) * 1000, 3),
                "End-to-End (ms)": round((latency.get("end_to_end_ms") or 0) * 1000, 3),
            }
        )

    df = pd.DataFrame(rows)
    st.markdown("""
        <div class="section-heading">
            <div class="section-title">
                Registry Overview
            </div>
            <div class = "section-subtitle">
                Deployment history, active versions, and inference latency metrics
            </div>
        </div>
    """, unsafe_allow_html = True)

    st.dataframe(
        df, use_container_width = True, hide_index = True
    )

    active_models = df[df["Status"] == "ACTIVE"]
    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html = True)
    st.markdown("""
        <div class="section-heading">
            <div class="section-title">
                Active Deployments
            </div>
            <div class="section-subtitle">
                Currently serving production traffic
            </div>
        </div>
    """, unsafe_allow_html = True)

    cols = st.columns(len(active_models))
    for idx, (_, row) in enumerate(active_models.iterrows()):
        with cols[idx]:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="card-label">
                        {row["Role"]}
                    </div>
                    <div style="font-size: 24px; font-weight: 700; margin-top: 10px; color: #22c55e;">
                        ACTIVE
                    </div>
                    <div style="margin-top: 14px; font-size: 12px; color: rgba(255,255,255,0.45); font-family: 'DM Mono', monospace;">
                        {row["Run ID"]}
                    </div>
                    <div style="margin-top: 16px; font-size: 13px; color: rgba(255,255,255,0.65);">
                        Requests Served:
                        <b>{row["Requests"]}</b>
                    </div>
                </div>
            """, unsafe_allow_html = True)