import requests
import streamlit as st
from config.settings import API_URL

def api_get(path):
    try:
        r = requests.get(f"{API_URL}{path}", timeout = 30)
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def api_post(path, payload):
    try:
        r = requests.post(f"{API_URL}{path}", json = payload, timeout = 60)
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500

@st.cache_data(ttl = 30)
def cached_get(path):
    return api_get(path)