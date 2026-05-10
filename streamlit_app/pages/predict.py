# Import Libraries
import streamlit as st
import plotly.graph_objects as go

# Import Custom Modules
from components.customer_form import customer_form
from utils.api import api_post
from utils.plots import base_layout, GRID_CLR

def render():
    st.markdown("""
    <div class="page-heading">
        <div class="page-title">Predict Churn</div>
        <div class="page-desc"><span class="badge badge-post">POST</span>/predict — Real-time ensemble inference</div>
    </div>
    """, unsafe_allow_html = True)

    payload = customer_form("predict", mode = "predict")

    if payload:
        with st.spinner("Running inference..."):
            data, status = api_post("/predict", payload)

        if status == 200:
            prob = data.get("churn_probability", 0)
            pred = data.get("churn_prediction", "")
            ver = data.get("model_role", "—")
            preds = data.get("model_predictions", {})
            val_cls = "churn" if pred == "Yes" else "safe"

            st.markdown(f"""
                <div class="section-heading">
                    <div class="section-title">
                        Inference Results
                    </div>
                    <div class="section-subtitle">
                        Ensemble prediction and churn risk analysis
                    </div>
                </div>
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="label">Prediction</div>
                        <div class="value {val_cls}">{pred}</div>
                    </div>
                    <div class="metric-item">
                        <div class="label">Churn Probability</div>
                        <div class="value">{prob:.1%}</div>
                    </div>
                    <div class="metric-item">
                        <div class="label">Model Version</div>
                        <div class="value" style="font-size:18px;font-family:'DM Mono',monospace">{ver}</div>
                    </div>
                </div>
            """, unsafe_allow_html = True)

            col_gauge, col_models = st.columns(2)

            with col_gauge:
                fig = go.Figure(
                    go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        number = {"suffix": "%", "font": {"color": "#fff", "family": "DM Sans", "size": 36}},
                        gauge = {
                            "axis": {"range": [0, 100], "tickfont": {"size": 11}},
                            "bar": {"color": "#ef4444" if prob > 0.5 else "#22c55e", "thickness": 0.3},
                            "bgcolor": "rgba(255, 255, 255, 0.03)",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 50], "color": "rgba(34, 197, 94, 0.05)"},
                                {"range": [50, 100], "color": "rgba(239, 68, 68, 0.05)"}
                            ],
                            "threshold": {
                                "line": {"color": "rgba(255, 255, 255, 0.2)", "width": 1},
                                "value": 50
                            }
                        }
                    )
                )
                fig.update_layout(**base_layout(300))
                st.plotly_chart(fig, use_container_width = True)

            with col_models:
                if preds:
                    models = list(preds.keys())
                    values = list(preds.values())
                    colors = ["#ef4444" if v > 0.5 else "#22c55e" for v in values]
                    fig2 = go.Figure(
                        go.Bar(
                            x = [m.upper() for m in models],
                            y = values,
                            marker = dict(color = colors, cornerradius = 6),
                            text = [f"{v:.4f}" for v in values],
                            textposition = "outside",
                            textfont = dict(color = "rgba(255, 255, 255, 0.6)", size = 12)
                        )
                    )
                    fig2.add_hline(
                        y = 0.5,
                        line_dash = "dot",
                        line_color = "rgba(255, 255, 255, 0.15)",
                        annotation_text = "threshold",
                        annotation_font_color = "rgba(255, 255, 255, 0.3)"
                    )
                    fig2.update_layout(
                        **base_layout(300),
                        title = dict(
                            text = "Per-Model Churn Scores",
                            x = 0,
                            xanchor = "left",
                            font = dict(
                                size = 16,
                                color = "#ffffff",
                                family = "DM Sans"
                            )
                        ),
                        yaxis = dict(range = [0, 1], gridcolor = GRID_CLR)
                    )
                    st.plotly_chart(fig2, use_container_width = True)
        else:
            st.error(f"Error {status}: {data.get('detail', data)}")