# Import Libraries
import streamlit as st
import plotly.graph_objects as go

# Import Custom Modules
from utils.api import api_get
from utils.plots import base_layout, GRID_CLR
from config.settings import API_URL

def render():
    st.markdown("""
    <div class="page-heading">
        <div class="page-title">Data Drift</div>
        <div class="page-desc"><span class="badge badge-get">GET</span>/drift — Evidently AI drift detection</div>
    </div>
    """, unsafe_allow_html = True)

    c1, _, c2 = st.columns([4, 0.2, 1])
    with c1:
        limit = st.slider(
            "Requests to analyze", 10, 500, 100, 5
        )

    with c2:
        st.markdown(
            "<div style='height:28px'></div>",
            unsafe_allow_html = True
        )
        analyze_clicked = st.button(
            "Analyze Drift",
            use_container_width = True
        )
    
    if analyze_clicked:
        with st.spinner("Analyzing..."):
            data, status = api_get(f"/drift?limit = {limit}")

        if status == 200:
            if "message" in data:
                st.warning(data["message"])
            else:
                detected = data.get("drift_detected", False)
                count = data.get("drifted_features_count", 0)
                total = data.get("total_features", 0)
                share = data.get("drifted_features_share", 0)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Drift Detected", "Yes" if detected else "No")
                c2.metric("Drifted Features", count)
                c3.metric("Total Features", total)
                c4.metric("Drift Share", f"{share:.1%}")

                fd = data.get("feature_drift", {})
                if fd:
                    feats = list(fd.keys())
                    pvals = [v["p_value"] for v in fd.values()]
                    colors = ["#ef4444" if v["drift_detected"] else "#22c55e" for v in fd.values()]

                    fig = go.Figure(
                        go.Bar(
                            x = feats, y = pvals,
                            marker = dict(color = colors, cornerradius = 4),
                        )
                    )
                    fig.add_hline(
                        y = 0.05,
                        line_dash = "dot",
                        line_color = "rgba(255,255,255,0.25)",
                        annotation_text = "p = 0.05",
                        annotation_font_color = "rgba(255,255,255,0.4)"
                    )
                    fig.update_layout(
                        **base_layout(320),
                        xaxis = dict(tickangle = -40, gridcolor = GRID_CLR),
                        yaxis = dict(title = "p-value", gridcolor = GRID_CLR),
                    )
                    st.plotly_chart(fig, use_container_width = True)

                st.markdown('<hr class = "divider">', unsafe_allow_html = True)
                st.markdown(f"""
                    <a href = "{API_URL}/drift/report"
                    target = "_blank"
                    class = "evidently-link">
                        View Full Evidently Report →
                    </a>
                """, unsafe_allow_html = True
                )
        else:
            st.error(f"Error {status}: {data}")