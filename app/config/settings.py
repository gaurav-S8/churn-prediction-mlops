import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
API_URL = os.getenv("API_URL", "http://localhost:8000")

DATABASE_URL = os.getenv("DATABASE_URL")

CHAMPION_FOLDER = os.getenv("CHAMPION_FOLDER", "models/champion")
CHALLENGER_FOLDER = os.getenv("CHALLENGER_FOLDER", "models/challenger")

HF_SPACE_URL = os.getenv("HF_SPACE_URL")
LOCALHOST_URL = os.getenv("LOCALHOST", "http://localhost:8000")
STREAMLIT_LOCALHOST = os.getenv("STREAMLIT_LOCALHOST", "http://localhost:8501")