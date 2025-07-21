# test4.py
import sys
import os

# Add src/ to the module search path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))





import logging
from utils.scraper import get_youtube_comments
from utils.preprocessing import preprocess_comments_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        video_url = input("Enter a YouTube video URL or ID: ").strip()
        comments = get_youtube_comments(video_url, max_comments=10)
        
        if not comments:
            logger.warning("No comments fetched. Exiting.")
        else:
            logger.info(f"✅ Raw Comments Fetched: {len(comments)}")
            for i, c in enumerate(comments, 1):
                print(f"Raw {i}: {c}")

            cleaned = preprocess_comments_list(comments)

            print("\n✅ Cleaned Comments:")
            for i, c in enumerate(cleaned, 1):
                print(f"Cleaned {i}: {c}")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
