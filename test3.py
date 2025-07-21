# test3.py

import sys
import os
import logging

# Add the 'src' directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.scraper import get_youtube_comments, extract_video_id

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Test case: Replace this with a valid YouTube video link or ID
    test_input = input("Enter a YouTube video URL or video ID: ").strip()

    # Test extract_video_id
    video_id = extract_video_id(test_input)
    if video_id:
        logger.info(f"✅ Extracted Video ID: {video_id}")
    else:
        logger.error("❌ Failed to extract a valid Video ID.")
        return

    # Test get_youtube_comments
    comments = get_youtube_comments(test_input, max_comments=5)
    if comments:
        logger.info(f"✅ Successfully fetched {len(comments)} comment(s).")
        for idx, comment in enumerate(comments, 1):
            print(f"{idx}. {comment}")
    else:
        logger.warning("⚠️ No comments returned. Check if the video is public or has comments enabled.")

if __name__ == "__main__":
    main()
