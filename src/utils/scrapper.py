# utils/scrapper.py

import os
import re
import logging
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_video_id(youtube_url_or_id):
    """
    Extracts the video ID from a YouTube URL or returns it directly if it's already an ID.
    
    Args:
        youtube_url_or_id (str): A full YouTube URL or just the video ID.
    
    Returns:
        str or None: The extracted video ID, or None if invalid.
    """
    # If it's already 11 characters (most video IDs are), return it directly
    if re.match(r"^[a-zA-Z0-9_-]{11}$", youtube_url_or_id):
        return youtube_url_or_id

    # Parse the YouTube URL
    try:
        parsed_url = urlparse(youtube_url_or_id)
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            query = parse_qs(parsed_url.query)
            return query.get("v", [None])[0]
        elif parsed_url.hostname == 'youtu.be':
            return parsed_url.path.lstrip("/")
        else:
            return None
    except Exception as e:
        logger.exception("Failed to extract video ID.")
        return None


def get_youtube_comments(youtube_url_or_id, max_comments=100):
    """
    Fetch top-level comments from a YouTube video.

    Args:
        youtube_url_or_id (str): YouTube video URL or video ID.
        max_comments (int): Maximum number of comments to fetch.

    Returns:
        list: A list of comment strings.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.error("YOUTUBE_API_KEY not found in environment variables.")
        return []

    video_id = extract_video_id(youtube_url_or_id)
    if not video_id:
        logger.error("Invalid YouTube video URL or ID.")
        return []

    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        comments = []
        next_page_token = None

        while len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        logger.info(f"Fetched {len(comments)} comments for video ID: {video_id}")
        return comments

    except Exception as e:
        logger.exception("Error occurred while fetching YouTube comments.")
        return []

# Example usage
if __name__ == "__main__":
    sample_input = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    comments = get_youtube_comments(sample_input)
    for idx, comment in enumerate(comments[:5], 1):
        print(f"{idx}. {comment}")
