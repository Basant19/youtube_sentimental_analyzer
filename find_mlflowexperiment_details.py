import os
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# Load environment variables
load_dotenv(override=True)

# Set tracking URI
mlflow_tracking_uri = os.getenv("MLFLOW_SERVER_TRACKING_URI_EC2")
client = MlflowClient(tracking_uri=mlflow_tracking_uri)

# List and print all experiments with their artifact locations
experiments = client.search_experiments()

for experiment in experiments:
    print(f"Name: {experiment.name}")
    print(f"ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print("-" * 60)
