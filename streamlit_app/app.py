# Import Libraries
import streamlit as st

# Import Custom Modules
from styles.global_styles import inject_styles
from components.navbar import render_navbar
from pages import predict, explain, benchmark, ab_report, drift, registry

st.set_page_config(
    page_title = "Churn Prediction",
    page_icon = "◈",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

inject_styles()
render_navbar()

page = st.session_state.get("page", "Predict")

if page == "Predict":
    predict.render()
elif page == "Explain":
    explain.render()
elif page == "Benchmark":
    benchmark.render()
elif page == "A/B Report":
    ab_report.render()
elif page == "Drift":
    drift.render()
elif page == "Registry":
    registry.render()