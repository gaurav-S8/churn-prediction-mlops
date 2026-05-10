import streamlit as st
from utils.api import cached_get

def render():
    st.markdown("""
    <div class="page-heading">
        <div class="page-title">Benchmark</div>
        <div class="page-desc"><span class="badge badge-get">GET</span>/benchmark — Per-model inference performance</div>
    </div>
    """, unsafe_allow_html = True)

    data, status = cached_get("/benchmark")

    if status == 200:
        for role in ["champion", "challenger", "total"]:
            r = data.get(role, {})
            st.markdown(f'<div class="card-label">{role.upper()}</div>', unsafe_allow_html = True)

            models = ["lgb", "xgb", "cat", "ensemble"]
            cols = st.columns(len(models) + 1)

            for i, m in enumerate(models):
                md = r.get(m, {})
                avg = md.get("avg_ms")
                p95 = md.get("p95_ms")
                with cols[i]:
                    st.metric(
                        m.upper(),
                        f"{avg} ms" if avg is not None else "—",
                        f"p95: {p95} ms" if p95 is not None else "—",
                        delta_color = "off"
                    )

            overhead = r.get("avg_overhead_ms")
            with cols[-1]:
                st.metric(
                    "Overhead",
                    f"{overhead} ms" if overhead is not None else "—",
                    delta_color = "off"
                )

            e2e = r.get("end_to_end", {})
            st.caption(f"End-to-End — avg: {e2e.get('avg_ms', '—')} ms | p95: {e2e.get('p95_ms', '—')} ms | max: {e2e.get('max_ms', '—')} ms")
            st.markdown('<hr class="divider">', unsafe_allow_html = True)
    else:
        st.error(f"Error {status}: {data}")