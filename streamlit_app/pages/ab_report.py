import streamlit as st
import plotly.graph_objects as go
from utils.api import cached_get
from utils.plots import base_layout

def render():
    st.markdown("""
    <div class="page-heading">
        <div class="page-title">A/B Report</div>
        <div class="page-desc"><span class="badge badge-get">GET</span>/ab-report — Champion vs Challenger</div>
    </div>
    """, unsafe_allow_html = True)

    data, status = cached_get("/ab-report")

    if status == 200:
        note = data.pop("note", None)
        roles = list(data.keys())

        if roles:
            cols = st.columns(len(roles))
            for i, role in enumerate(roles):
                m = data[role]
                with cols[i]:
                    st.markdown(f'<div class="card-label">{role.upper()}</div>', unsafe_allow_html = True)
                    st.metric("Requests", m.get("total_requests", 0))
                    st.metric("Avg Churn Probability", f"{m.get('avg_churn_probability', 0):.4f}")
                    st.metric("Churn Rate", f"{float(m.get('churn_rate', 0)):.1%}")

            if len(roles) >= 2:
                st.markdown('<hr class="divider">', unsafe_allow_html = True)
                r1, r2 = roles[0], roles[1]
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        name = r1.capitalize(),
                        x = ["Avg Churn Prob", "Churn Rate"],
                        y = [data[r1].get("avg_churn_probability", 0), float(data[r1].get("churn_rate", 0))],
                        marker = dict(color = "#6366f1", cornerradius = 6)
                    )
                )
                fig.add_trace(
                    go.Bar(
                        name = r2.capitalize(),
                        x = ["Avg Churn Prob", "Churn Rate"],
                        y = [data[r2].get("avg_churn_probability", 0), float(data[r2].get("churn_rate", 0))],
                        marker = dict(color = "#22c55e", cornerradius = 6)
                    )
                )
                fig.update_layout(**base_layout(280), barmode = "group")
                st.plotly_chart(fig, use_container_width = True)

        if note:
            st.info(note)
    else:
        st.error(f"Error {status}: {data}")