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
        <div class="page-title">Explain Prediction</div>
        <div class="page-desc"><span class="badge badge-post">POST</span>/explain — SHAP feature importance</div>
    </div>
    """, unsafe_allow_html = True)

    payload = customer_form("explain", mode = "explain")

    if payload:
        with st.spinner("Computing SHAP values..."):
            data, status = api_post("/explain", payload)

        if status == 200:
            st.markdown("""
                <div class = "section-heading">
                    <div class = "section-title">
                        SHAP Feature Contributions
                    </div>
                    <div class = "section-subtitle">
                        Feature-level impact on churn prediction
                    </div>
                </div>
            """, unsafe_allow_html = True)

            st.info(
                """
                Red features increased churn risk.
                Green features reduced churn risk.
                """
            )

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
                    **base_layout(600),
                    xaxis = dict(title = "SHAP Value", gridcolor = GRID_CLR),
                    yaxis = dict(gridcolor = GRID_CLR, tickfont = dict(size = 11))
                )
                st.plotly_chart(fig, use_container_width = True)
        else:
            st.error(f"Error {status}: {data.get('detail', data)}")