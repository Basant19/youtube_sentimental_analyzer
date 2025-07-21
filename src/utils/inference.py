"""
inference.py

This module handles loading a model from the MLflow Model Registry and making sentiment predictions
on a list of YouTube comments using a trained ML pipeline model.

It includes detailed logging for debugging and error tracking.
"""

import os
import logging
import mlflow.pyfunc
from dotenv import load_dotenv
from src.utils.preprocessing import preprocess_comments_list

# Load environment variables
load_dotenv(override=True)

# Logging setup
logger = logging.getLogger("inference")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("inference_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_model_from_registry(model_name: str, stage: str = "Staging"):
    """
    Load the model from MLflow model registry using the provided model name and stage.
    """
    try:
        tracking_uri = os.getenv("MLFLOW_SERVER_TRACKING_URI_EC2")
        mlflow.set_tracking_uri(tracking_uri)

        model_uri = f"models:/{model_name}/{stage}"
        logger.debug(f"Loading model from: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully from registry")
        return model

    except Exception as e:
        logger.error("Failed to load model from MLflow registry: %s", str(e))
        raise

def predict_sentiment(comments: list, model):
    """
    Preprocess and predict sentiment for a list of comments using the MLflow pipeline model.
    """
    try:
        logger.debug(f"Received {len(comments)} comments for sentiment prediction")

        preprocessed = preprocess_comments_list(comments)
        logger.debug(f"Preprocessed comments: {preprocessed}")

        predictions = model.predict(preprocessed)
        logger.info("Sentiment prediction successful")

        return list(predictions)

    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise
