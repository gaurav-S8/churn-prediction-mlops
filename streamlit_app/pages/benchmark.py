# Import Libraries
import streamlit as st

# Import Custom Modules
from utils.api import cached_get

def metric_block(data):
    if not data or data.get("avg_ms") is None:
        return """
        <div class="benchmark-empty">
            No traffic
        </div>
        """

    return f"""
        <div class="benchmark-cell">
            <div class="benchmark-main">
                {data.get("avg_ms", "—")} ms
            </div>
            <div class="benchmark-stats">
                <span>
                    avg: {data.get("avg_ms", "—")} ms |
                </span>
                <span>
                    p95: {data.get("p95_ms", "—")} ms |
                </span>
                <span>
                    min: {data.get("min_ms", "—")} ms |
                </span>
                <span>
                    max: {data.get("max_ms", "—")} ms
                </span>
            </div>
        </div>
    """


def render():

    # PAGE HEADER
    st.markdown("""
    <div class="page-heading">
        <div class="page-title">
            Benchmark
        </div>
        <div class="page-desc">
            <span class="badge badge-get">GET</span>
            /benchmark — Per-model inference performance
        </div>
    </div>
    """, unsafe_allow_html = True)

    # FETCH DATA
    with st.spinner("Loading benchmark metrics..."):
        data, status = cached_get("/benchmark")

    if status != 200:
        st.error(f"Error {status}: {data}")
        return

    champion = data.get("champion", {})
    challenger = data.get("challenger", {})
    total = data.get("total", {})

    # REQUEST SUMMARY
    st.markdown(f"""
        <div class="benchmark-summary">
            <div class="summary-card">
                <div class="summary-label">
                    Champion Requests
                </div>
                <div class="summary-value">
                    {champion.get("total_requests", 0)}
                </div>
            </div>
            <div class="summary-card">
                <div class="summary-label">
                    Challenger Requests
                </div>
                <div class="summary-value">
                    {challenger.get("total_requests", 0)}
                </div>
            </div>
            <div class="summary-card">
                <div class="summary-label">
                    Total Requests
                </div>
                <div class="summary-value">
                    {total.get("total_requests", 0)}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='margin-top:40px'></div>",
        unsafe_allow_html=True
    )

    # TABLE HEADER
    h1, h2, h3, h4 = st.columns([0.5, 1.5, 1.5, 1.5])

    with h1:
        st.markdown("""
            <div class="benchmark-header benchmark-header-left">
                Component
            </div>
        """, unsafe_allow_html=True)

    with h2:
        st.markdown("""
            <div class="benchmark-header">
                Champion
            </div>
        """, unsafe_allow_html=True)

    with h3:
        st.markdown("""
            <div class="benchmark-header">
                Challenger
            </div>
        """, unsafe_allow_html=True)

    with h4:
        st.markdown("""
            <div class="benchmark-header">
                Total
            </div>
        """, unsafe_allow_html=True)

    st.markdown(
        "<div style='margin-bottom:18px'></div>",
        unsafe_allow_html=True
    )

    # MODEL ROWS
    metrics = [
        ("lgb", "LGB"),
        ("xgb", "XGB"),
        ("cat", "CAT"),
        ("ensemble", "ENSEMBLE"),
        ("end_to_end", "END_TO_END")
    ]

    for key, label in metrics:
        c1, c2, c3, c4 = st.columns([0.5, 1.5, 1.5, 1.5])

        # METRIC LABEL
        with c1:
            st.markdown(f"""
                <div class="benchmark-metric-name">
                    {label}
                </div>
            """, unsafe_allow_html=True)

        # CHAMPION
        with c2:
            st.markdown(
                metric_block(champion.get(key, {})),
                unsafe_allow_html=True
            )

        # CHALLENGER
        with c3:
            st.markdown(
                metric_block(challenger.get(key, {})),
                unsafe_allow_html=True
            )

        # TOTAL
        with c4:
            st.markdown(
                metric_block(total.get(key, {})),
                unsafe_allow_html=True
            )

        st.markdown(
            "<div style='margin-bottom:18px'></div>",
            unsafe_allow_html=True
        )

    # OVERHEAD
    c1, c2, c3, c4 = st.columns([0.5, 1.5, 1.5, 1.5])
    with c1:
        st.markdown("""
            <div class="benchmark-metric-name">
                OVERHEAD
            </div>
        """, unsafe_allow_html=True)

    for idx, role in enumerate([champion, challenger, total]):
        overhead = role.get("avg_overhead_ms")
        html = f"""
            <div class="benchmark-cell">
                <div class="benchmark-main">
                    {f"{overhead} ms" if overhead is not None else "—"}
                </div>
            </div>
        """

        with [c2, c3, c4][idx]:
            st.markdown(
                html,
                unsafe_allow_html=True
            )