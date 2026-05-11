# Get Python docker image
FROM python:3.10.19-slim

# Create a Working directory
WORKDIR /app

# Install needed libraries
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy folders
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]