import streamlit as st

def inject_styles():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0c0c0e; }

#MainMenu, { visibility: hidden; }

.page-heading { margin-bottom: 32px; }
.page-title { font-size: 26px; font-weight: 600; color: #ffffff; margin-bottom: 6px; letter-spacing: -0.02em; }
.page-desc { font-size: 14px; color: rgba(255,255,255,0.35); font-family: 'DM Mono', monospace; }

.badge { display: inline-block; padding: 3px 10px; border-radius: 5px; font-size: 10px; font-weight: 600; letter-spacing: 0.08em; font-family: 'DM Mono', monospace; margin-right: 8px; }
.badge-post { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.2); }
.badge-get  { background: rgba(34,197,94,0.1);  color: #86efac; border: 1px solid rgba(34,197,94,0.15); }

.card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 24px; margin-bottom: 16px; }
.card-label { font-size: 11px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: rgba(255,255,255,0.3); margin-bottom: 10px; }

.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
.metric-item { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 18px 20px; }
.metric-item .label { font-size: 11px; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.metric-item .value { font-size: 28px; font-weight: 600; color: #ffffff; letter-spacing: -0.02em; line-height: 1; }
.metric-item .value.churn { color: #ef4444; }
.metric-item .value.safe  { color: #22c55e; }

.nav-status { display: flex; align-items: center; gap: 8px; font-size: 12px; color: rgba(255,255,255,0.3); font-family: 'DM Mono', monospace; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: #22c55e; }
.nav-brand { font-family: 'DM Sans', sans-serif; font-size: 15px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #ffffff; }
.nav-brand span { color: rgba(255,255,255,0.3); font-weight: 300; }

.divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 28px 0; }

/* Streamlit overrides */
div[data-testid="stForm"] { background: transparent; border: none; padding: 0; }

.stSelectbox > label, .stNumberInput > label, .stTextInput > label {
    font-size: 12px !important; font-weight: 500 !important;
    color: rgba(255,255,255,0.5) !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}

.stButton > button {
    background: #ffffff !important; color: #0c0c0e !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 14px !important;
    padding: 10px 24px !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stSlider > label { font-size: 12px !important; color: rgba(255,255,255,0.5) !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
div[data-testid="stAlert"] { border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.08) !important; }
section[data-testid="stSidebar"] { display: none; }

.block-container {
    padding-top: 5rem !important;
}

</style>
""", unsafe_allow_html = True)
