# Import Libraries
import streamlit as st

# Import Custom Modules
from utils.api import cached_get

def render():
    st.markdown("""
        <div class="page-heading">
            <div class="page-title">
                A/B Report
            </div>
            <div class="page-desc">
                <span class="badge badge-get">GET</span>
                /ab-report — Champion vs Challenger
            </div>
        </div>
    """, unsafe_allow_html = True)

    data, status = cached_get("/ab-report")
    if status != 200:
        st.error(f"Error {status}: {data}")
        return

    note = data.pop("note", None)
    champion = data.get("champion", {})
    challenger = data.get("challenger", {})

    # HEADERS
    left, center, right = st.columns([1, 4, 1])
    with center:
        h1, h2, h3 = st.columns([1.5, 1, 1])
        with h1:
            st.markdown(
                '<div class="benchmark-header benchmark-header-left">Metric</div>',
                unsafe_allow_html = True
            )

        with h2:
            st.markdown(
                '<div class="benchmark-header">Champion</div>',
                unsafe_allow_html = True
            )

        with h3:
            st.markdown(
                '<div class="benchmark-header">Challenger</div>',
                unsafe_allow_html = True
            )
        
        st.markdown(
            "<div style='margin-bottom:10px'></div>",
            unsafe_allow_html = True
        )

    # ROWS
    rows = [
        (
            "Requests",
            champion.get("total_requests", 0),
            challenger.get("total_requests", 0)
        ),
        (
            "Avg Churn Probability",
            (
                f"{champion.get('avg_churn_probability', 0):.4f}"
                if champion.get("avg_churn_probability") is not None
                else "—"
            ),
            (
                f"{challenger.get('avg_churn_probability', 0):.4f}"
                if challenger.get("avg_churn_probability") is not None
                else "—"
            )
        ),
        (
            "Churn Rate",
            (
                f"{float(champion.get('churn_rate', 0)):.1%}"
                if champion.get("churn_rate") is not None
                else "—"
            ),
            (
                f"{float(challenger.get('churn_rate', 0)):.1%}"
                if challenger.get("churn_rate") is not None
                else "—"
            )
        )
    ]

    for metric, champ, chall in rows:
        with center:
            c1, c2, c3 = st.columns([1.5, 1, 1])
            with c1:
                st.markdown(f"""
                    <div class="benchmark-metric-name">
                        {metric.upper()}
                    </div>
                """, unsafe_allow_html = True)
            with c2:
                st.markdown(f"""
                    <div class="ab-value">
                        {champ}
                    </div>
                """, unsafe_allow_html = True)
            with c3:
                if chall in [0, "0", "0.0000", "0.0%", None]:
                    st.markdown("""
                        <div class="benchmark-empty">
                            No traffic
                        </div>
                    """, unsafe_allow_html = True)
                else:
                    st.markdown(f"""
                        <div class="ab-value">
                            {chall}
                        </div>
                    """, unsafe_allow_html = True)
              
            st.markdown(
                "<div style='margin-bottom:8px'></div>",
                unsafe_allow_html = True
            )

    if note:
        st.markdown(
            "<div style='margin-top:24px'></div>",
            unsafe_allow_html = True
        )
        st.info(note)