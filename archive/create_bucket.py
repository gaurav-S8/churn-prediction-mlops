import os
import boto3
from dotenv import load_dotenv
from botocore.client import Config

load_dotenv()

def create_bucket():
    s3 = boto3.client(
        "s3",
        endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        config = Config(signature_version = "s3v4")
    )

    bucket_name = os.getenv("MLFLOW_BUCKET_NAME", "mlflow-artifacts")
    existing = [b["Name"] for b in s3.list_buckets()["Buckets"]]

    if bucket_name not in existing:
        s3.create_bucket(Bucket = bucket_name)
        print(f"Bucket '{bucket_name}' created!")
    else:
        print(f"Bucket '{bucket_name}' already exists!")

if __name__ == "__main__":
    create_bucket()