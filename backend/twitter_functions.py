"""
Simple Twitter posting functions for MCP integration
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional
from twitter_poster import TwitterPoster
import tweepy
from dotenv import load_dotenv

load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

def post_video_to_twitter(video_path: str, caption: str = "") -> Dict:
    """
    Post a video to Twitter

    Args:
        video_path (str): Path to the video file (.mp4 or .mov)
        caption (str): Tweet caption/text (optional)

    Returns:
        Dict: {"success": bool, "tweet_id": str, "tweet_url": str, "error": str}
    """
    try:
        poster = TwitterPoster()
        result = poster.post_video_with_retry(video_path, caption)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def post_text_to_twitter(caption: str) -> Dict:
    """
    Post a text-only tweet

    Args:
        caption (str): Tweet text content

    Returns:
        Dict: {"success": bool, "tweet_id": str, "tweet_url": str, "error": str}
    """
    try:
        client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
        )

        tweet = client.create_tweet(text=caption)

        return {
            "success": True,
            "tweet_id": tweet.data['id'],
            "tweet_url": f"https://twitter.com/user/status/{tweet.data['id']}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def test_twitter_connection() -> Dict:
    """
    Test if Twitter credentials are working

    Returns:
        Dict: {"success": bool, "username": str, "user_id": str, "error": str}
    """
    try:
        client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
        )

        me = client.get_me()

        return {
            "success": True,
            "username": me.data.username,
            "user_id": str(me.data.id),
            "message": "Twitter connection successful"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def validate_video_file(video_path: str) -> Dict:
    """
    Validate if video file meets Twitter requirements

    Args:
        video_path (str): Path to the video file

    Returns:
        Dict: {"valid": bool, "file_size": int, "format": str, "error": str}
    """
    try:
        if not os.path.exists(video_path):
            return {
                "valid": False,
                "error": f"File not found: {video_path}"
            }

        file_size = os.path.getsize(video_path)
        file_ext = Path(video_path).suffix.lower()

        max_size = 512 * 1024 * 1024  # 512MB
        valid_formats = ['.mp4', '.mov']

        if file_ext not in valid_formats:
            return {
                "valid": False,
                "file_size": file_size,
                "format": file_ext,
                "error": f"Invalid format. Supported: {valid_formats}"
            }

        if file_size > max_size:
            return {
                "valid": False,
                "file_size": file_size,
                "format": file_ext,
                "error": f"File too large: {file_size} bytes (max: {max_size} bytes)"
            }

        return {
            "valid": True,
            "file_size": file_size,
            "format": file_ext,
            "message": "Video file is valid for Twitter"
        }

    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

# Example usage functions for testing
def example_usage():
    """Example of how to use these functions"""

    # Test connection
    print("Testing Twitter connection...")
    result = test_twitter_connection()
    print(f"Connection test: {result}")

    # Validate a video file
    video_path = "capy_4.mp4"
    print(f"\nValidating video: {video_path}")
    validation = validate_video_file(video_path)
    print(f"Validation result: {validation}")

    # Post a text tweet
    print("\nPosting text tweet...")
    text_result = post_text_to_twitter("Hello from my Twitter function! #test")
    print(f"Text tweet result: {text_result}")

    # Post a video (if validation passed)
    if validation.get("valid"):
        print(f"\nPosting video: {video_path}")
        video_result = post_video_to_twitter(video_path, "Check out this video! #automation")
        print(f"Video post result: {video_result}")

if __name__ == "__main__":
    example_usage()