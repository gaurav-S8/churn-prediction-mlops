import streamlit as st
import plotly.graph_objects as go
from components.customer_form import customer_form
from utils.api import api_post
from utils.plots import base_layout, GRID_CLR

def render():
    st.markdown("""
    <div class="page-heading">
        <div class="page-title">Explain Prediction</div>
        <div class="page-desc"><span class="badge badge-post">POST</span>/explain — SHAP feature importance</div>
    </div>
    """, unsafe_allow_html = True)

    payload = customer_form("explain")

    if payload:
        with st.spinner("Computing SHAP values..."):
            data, status = api_post("/explain", payload)

        if status == 200:
            shap = data.get("shap_values", {})
            if shap:
                feats = list(shap.keys())
                vals = list(shap.values())
                colors = ["#ef4444" if v > 0 else "#22c55e" for v in vals]

                fig = go.Figure(
                    go.Bar(
                        x = vals,
                        y = feats,
                        orientation = "h",
                        marker = dict(
                            color = colors,
                            cornerradius = 4
                        ),
                    )
                )
                fig.update_layout(
                    **base_layout(520),
                    xaxis = dict(title = "SHAP Value", gridcolor = GRID_CLR),
                    yaxis = dict(gridcolor = GRID_CLR, tickfont = dict(size = 11)),
                    annotations = [
                        dict(
                            x = 0.5,
                            y = -0.08,
                            xref = "paper",
                            yref = "paper",
                            text = "Red = increases churn risk   |   Green = decreases churn risk",
                            showarrow = False,
                            font = dict(size = 11, color = "rgba(255, 255, 255, 0.3)")
                        )
                    ]
                )
                st.plotly_chart(fig, use_container_width = True)
        else:
            st.error(f"Error {status}: {data.get('detail', data)}")