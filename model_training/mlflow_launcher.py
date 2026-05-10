# Import Libraries
import time
import subprocess
import requests

def start_mlflow_server(TRACKING_URI):
    try:
        requests.get(TRACKING_URI, timeout = 5)
        print("MLflow already running.")
        return None
    except requests.exceptions.RequestException:
        process = subprocess.Popen(
            ["mlflow", "ui", "--port", "5000"],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )

        time.sleep(2)
        print("MLflow started.")
        return process