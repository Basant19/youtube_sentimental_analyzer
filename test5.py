import sys
import os
import logging
import pickle

# Update path for module resolution
root_dir = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(root_dir, "src")
sys.path.append(src_path)

from utils.inference import load_model_from_registry, predict_sentiment

# Logging setup
logger = logging.getLogger("test_inference")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("test5_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_vectorizer():
    """Load the vectorizer saved during training."""
    try:
        vectorizer_path = os.path.join(root_dir, 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error("Failed to load vectorizer: %s", e)
        raise

def main():
    try:
        model_name = "my_model"
        stage = "Staging"

        logger.info("Loading model from MLflow registry...")
        model = load_model_from_registry(model_name, stage)

        logger.info("Loading vectorizer from local pickle file...")
        vectorizer = load_vectorizer()

        comments = [
            "I love this video, it was so helpful!",
            "This is the worst video I've ever seen.",
            "Meh, it was okay. Could be better.",
            "Fantastic content! Keep it coming!"
        ]

        logger.info("Running prediction on sample comments...")
        results = predict_sentiment(comments, model, vectorizer)

        print("\n📊 Prediction Results:")
        for comment, sentiment in results:
            print(f"🗨️ Comment: {comment}\n✅ Predicted Sentiment: {sentiment}\n")

    except Exception as e:
        logger.error("An error occurred during inference test: %s", e)
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
