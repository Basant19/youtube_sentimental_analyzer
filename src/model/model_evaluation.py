import os
import json
import yaml
import pickle
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Load environment variables
load_dotenv(override=True)

# Logging setup
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

if not logger.handlers:
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("model_evaluation_errors.log")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_csv(path):
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    return df

def ensure_experiment(name: str) -> str:
    """Ensure experiment exists or create a new one. Returns experiment_id."""
    client = MlflowClient()
    experiments = {
        e.name: e for e in client.search_experiments(view_type=mlflow.entities.ViewType.ALL)
    }
    if name in experiments:
        exp = experiments[name]
        if exp.lifecycle_stage == "deleted":
            raise MlflowException(f"Experiment '{name}' is deleted. Restore it or use a new name.")
        logger.info(f"Using existing experiment: {name} (id: {exp.experiment_id})")
        return exp.experiment_id
    else:
        exp_id = client.create_experiment(name)
        logger.info(f"Created new experiment: {name} (id: {exp_id})")
        return exp_id

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    logger.debug("Model evaluation completed")
    return report, cm

def log_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    path = f"confusion_matrix_{dataset_name}.png"
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()

def save_model_info(run_id: str, model_path: str, artifact_uri: str, file_path: str = "experiment_info.json"):
    info = {
        "run_id": run_id,
        "model_path": model_path,
        "artifact_uri": artifact_uri
    }
    with open(file_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info(f"Model info saved to {file_path}")

def main():
    try:
        # Root path
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

        # Set tracking URI from .env
        tracking_uri = os.getenv("MLFLOW_SERVER_TRACKING_URI_EC2")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Tracking URI: {tracking_uri}")

        experiment_name = "dvc-pipeline-current"
        experiment_id = ensure_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        # Load config
        params = load_yaml(os.path.join(root, "params.yaml"))
        model_path = os.path.join(root, "lgbm_model.pkl")
        vectorizer_path = os.path.join(root, "tfidf_vectorizer.pkl")
        test_path = os.path.join(root, "data/interim/test_processed.csv")

        with mlflow.start_run(experiment_id=experiment_id) as run:
            # Log parameters
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Load model, vectorizer, test data
            model = load_pickle(model_path)
            vectorizer = load_pickle(vectorizer_path)
            df = load_csv(test_path)

            X_test = vectorizer.transform(df["clean_comment"])
            y_test = df["category"].values

            input_example = pd.DataFrame(X_test.toarray()[:5], columns=vectorizer.get_feature_names_out())
            signature = infer_signature(input_example, model.predict(X_test[:5]))

            # Log model to MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="lgbm_model",
                signature=signature,
                input_example=input_example
            )

            # Log vectorizer as artifact
            mlflow.log_artifact(vectorizer_path)

            # Evaluate
            report, cm = evaluate_model(model, X_test, y_test)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, score in metrics.items():
                        mlflow.log_metric(f"{label}_{metric_name}", score)

            log_confusion_matrix(cm, "test")

            # Save model info for registration
            artifact_uri = mlflow.get_artifact_uri()
            save_model_info(run.info.run_id, "lgbm_model", artifact_uri)

            # Tag the run
            mlflow.set_tags({
                "model_type": "LightGBM",
                "dataset": "YouTube Comments",
                "stage": "evaluation complete"
            })

            logger.info("Model evaluation and logging complete.")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
