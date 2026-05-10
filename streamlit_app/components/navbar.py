import streamlit as st
from config.settings import PAGES

def render_navbar():
    if "page" not in st.session_state:
        st.session_state.page = "Predict"

    cols = st.columns([2, 4])

    with cols[0]:
        st.markdown("""
            <div class = "nav-brand">Churn <span>Prediction API</span></div>
            <div class="nav-status"><div class="status-dot"></div>Live</div>
        """, unsafe_allow_html = True
        )

    with cols[1]:
        tab_cols = st.columns(len(PAGES))
        for i, p in enumerate(PAGES):
            with tab_cols[i]:
                if st.button(p, key = f"nav_{p}", use_container_width = True):
                    st.session_state.page = p
                    st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html = True)