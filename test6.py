# test6.py

import sys
import logging

# Ensure proper path to inference module
sys.path.append(r"D:\bybappy\youtube_sentimental_analyzer\src\utils")

from inference import load_model_and_artifacts, predict_sentiment

# Setup test logger
logger = logging.getLogger("test6")
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console)

if __name__ == "__main__":
    try:
        model_name = "my_model"
        model, vectorizer = load_model_and_artifacts(model_name)

        test_comments = [
            "I absolutely love this video!",
            "Worst experience ever, disliked it.",
            "",
            "The editing was okay, not great."
        ]

        predictions = predict_sentiment(test_comments, model, vectorizer)

        for comment, sentiment in predictions:
            print(f"Comment: {comment} → Sentiment: {sentiment}")

    except Exception as e:
        logger.error(f"❌ ERROR during test: {e}")
