#src/utils inference.py (Fully Updated)

import os
import logging
import pickle
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from dotenv import load_dotenv
from src.utils.preprocessing import preprocess_comments_list

# Load .env
load_dotenv(override=True)

# Logger setup
logger = logging.getLogger("inference")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("inference_errors.log",encoding="utf-8")

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_model_and_artifacts(model_name: str, stage: str = "Staging"):
    try:
        tracking_uri = os.getenv("MLFLOW_SERVER_TRACKING_URI_EC2")
        mlflow.set_tracking_uri(tracking_uri)

        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(model.metadata.run_id)

        vectorizer_path = None
        for pattern in ["vectorizer.pkl", "tfidf_vectorizer.pkl"]:
            vectorizer_path = next((a.path for a in artifacts if pattern in a.path), None)
            if vectorizer_path:
                break

        if not vectorizer_path:
            raise ValueError("Vectorizer artifact not found.")

        logger.info(f"Loading vectorizer from: {vectorizer_path}")
        temp_path = mlflow.artifacts.download_artifacts(
            run_id=model.metadata.run_id,
            artifact_path=vectorizer_path
        )

        with open(temp_path, 'rb') as f:
            vectorizer = pickle.load(f)

        return model, vectorizer

    except Exception as e:
        logger.error(f"Model/vectorizer loading failed: {e}")
        raise

def predict_sentiment(comments: list, model, vectorizer):
    try:
        logger.debug(f"Raw input comments: {comments}")
        cleaned = preprocess_comments_list(comments)
        logger.debug(f"Preprocessed comments: {cleaned}")

        vectorized = vectorizer.transform(cleaned)
        if isinstance(vectorized, csr_matrix):
            vectorized = vectorized.toarray()

        logger.debug(f"Vectorized input shape: {vectorized.shape}")
        logger.debug(f"Vectorized input type: {type(vectorized)}")

        # Handle schema enforcement
        if hasattr(model.metadata, "signature") and model.metadata.signature:
            input_schema = model.metadata.signature.inputs
            col_names = [col.name for col in input_schema.inputs]
            logger.debug(f"Model expects columns: {col_names}")

            if len(col_names) != vectorized.shape[1]:
                raise ValueError(
                    f"Feature mismatch: Model expects {len(col_names)} columns, "
                    f"but got {vectorized.shape[1]}"
                )

            input_df = pd.DataFrame(vectorized, columns=col_names)
        else:
            input_df = pd.DataFrame(vectorized)

        predictions = model.predict(input_df)
        return list(zip(comments, predictions))

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
