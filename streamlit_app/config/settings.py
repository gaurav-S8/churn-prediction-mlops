# Import Libraries
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY", "")
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Pages
PAGES = ["Predict", "Explain", "Drift", "A/B Report", "Registry", "Benchmark"]