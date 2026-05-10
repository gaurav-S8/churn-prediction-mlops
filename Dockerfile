# Get Python docker image
FROM python:3.10.19-slim

# Create a Working directory
WORKDIR /app

# Install system dependencies
# For boosting methods
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# For psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install needed libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only needed folders
COPY app/ ./app/
COPY data/reference.csv ./data/reference.csv
COPY models/ ./models/
COPY utils/ ./utils/
COPY setup.py .

# Install project as package
RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]