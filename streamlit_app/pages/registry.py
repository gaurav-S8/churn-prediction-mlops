import streamlit as st
from utils.api import cached_get

def render():
    st.markdown("""
    <div class="page-heading">
        <div class="page-title">Model Registry</div>
        <div class="page-desc"><span class="badge badge-get">GET</span>/model-registry — Active model versions</div>
    </div>
    """, unsafe_allow_html = True)

    data, status = cached_get("/model-registry")

    if status == 200 and data:
        st.json(data)
    else:
        st.markdown("""
        <div class="card">
            <div class="card-label">Status</div>
            <div style="color:rgba(255,255,255,0.4);font-size:14px;margin-top:8px;">
                Model registry endpoint is live. No data available yet.
            </div>
        </div>
        """, unsafe_allow_html = True)