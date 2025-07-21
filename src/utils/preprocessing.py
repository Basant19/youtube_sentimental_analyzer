# utils/preprocessing.py

import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Logging configuration
logger = logging.getLogger("preprocessing")
logger.setLevel(logging.DEBUG)

# Avoid adding multiple handlers if this module is imported multiple times
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("preprocessing.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Download NLTK data if not found
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# Stopwords and lemmatizer setup
stop_words = set(stopwords.words("english")) - {"not", "no", "but", "however", "yet"}
lemmatizer = WordNetLemmatizer()

def preprocess_comment(comment: str) -> str:
    """
    Cleans and normalizes a single comment by removing emojis,
    punctuation, stopwords, and applying lemmatization.
    """
    try:
        logger.debug(f"Original: {comment}")
        comment = comment.lower()
        comment = comment.strip()
        comment = comment.encode("ascii", "ignore").decode("ascii")  # Removes emojis & symbols
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)
        comment = " ".join([word for word in comment.split() if word not in stop_words])
        comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])
        logger.debug(f"Cleaned: {comment}")
        return comment
    except Exception as e:
        logger.error(f"Failed to preprocess comment: {comment}. Error: {e}")
        return comment

def preprocess_comments_list(comments: list) -> list:
    """
    Takes a list of raw comments and returns a list of cleaned comments.
    """
    try:
        logger.info(f"Preprocessing {len(comments)} comment(s)")
        return [preprocess_comment(c) for c in comments]
    except Exception as e:
        logger.error(f"Failed to preprocess comment list. Error: {e}")
        return comments
