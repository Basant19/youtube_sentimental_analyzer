from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(override=True)

print("MLFLOW URI:", os.getenv("MLFLOW_SERVER_TRACKING_URI_EC2"))
print("AWS_ACCESS_KEY_ID:", os.getenv("AWS_ACCESS_KEY_ID"))
print("AWS_SECRET_ACCESS_KEY:", os.getenv("AWS_SECRET_ACCESS_KEY"))
import mlflow
print("Tracking URI used for registration:", mlflow.get_tracking_uri())
