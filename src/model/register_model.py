# register model

import json
import mlflow
import logging
import os
import argparse

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# Load environment variables from .env
load_dotenv(override=True)

# Set the MLflow tracking URI
mlflow_tracking_uri = os.getenv("MLFLOW_SERVER_TRACKING_URI_EC2")
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient()

# Set up logging
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers if script runs multiple times
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('model_registration_errors.log')
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug(f"Registering model from URI: {model_uri}")

        model_version = mlflow.register_model(model_uri, model_name)

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.debug(f"Model '{model_name}' version {model_version.version} registered and transitioned to Staging.")

    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    parser = argparse.ArgumentParser(description="Register an ML model in MLflow.")
    parser.add_argument('--model-info-path', default='experiment_info.json', help='Path to the JSON file with run_id and model_path')
    parser.add_argument('--model-name', default='my_model', help='Name to register the model under in MLflow Registry')

    args = parser.parse_args()

    try:
        model_info = load_model_info(args.model_info_path)
        register_model(args.model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
