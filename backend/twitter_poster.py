import os
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
import tweepy

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterPoster:
    def __init__(self):
        self.setup_credentials()
        self.setup_client()

    def setup_credentials(self):
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

        if not all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            raise ValueError("Missing Twitter API credentials. Please check your .env file.")

    def setup_client(self):
        try:
            # For posting tweets (v2)
            self.client = tweepy.Client(
                bearer_token=self.bearer_token,
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret,
                wait_on_rate_limit=True
            )

            # For media upload (v1.1)
            auth = tweepy.OAuth1UserHandler(
                self.api_key,
                self.api_secret,
                self.access_token,
                self.access_token_secret
            )
            self.api_v1 = tweepy.API(auth, wait_on_rate_limit=True)

            logger.info("Twitter client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")
            raise

    def validate_video(self, video_path):
        """Validate video meets Twitter requirements"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        file_size = os.path.getsize(video_path)
        max_size = 512 * 1024 * 1024  # 512MB

        if file_size > max_size:
            raise ValueError(f"Video file too large: {file_size} bytes (max: {max_size} bytes)")

        valid_extensions = ['.mp4', '.mov']
        file_ext = Path(video_path).suffix.lower()
        if file_ext not in valid_extensions:
            raise ValueError(f"Invalid video format: {file_ext}. Supported: {valid_extensions}")

        return True

    def post_video(self, video_path, caption=""):
        """Post video to Twitter"""
        try:
            self.validate_video(video_path)

            logger.info(f"Uploading video: {video_path}")

            # Upload media using v1.1 API
            media = self.api_v1.media_upload(video_path)
            logger.info(f"Video uploaded successfully. Media ID: {media.media_id}")

            # Post tweet with media using v2 API
            tweet = self.client.create_tweet(
                text=caption,
                media_ids=[media.media_id]
            )

            logger.info(f"Tweet posted successfully! Tweet ID: {tweet.data['id']}")

            return {
                "success": True,
                "tweet_id": tweet.data['id'],
                "tweet_url": f"https://twitter.com/user/status/{tweet.data['id']}"
            }

        except Exception as e:
            logger.error(f"Failed to post video: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def post_video_with_retry(self, video_path, caption="", max_retries=3):
        """Post video with retry logic"""
        for attempt in range(max_retries):
            try:
                result = self.post_video(video_path, caption)
                if result["success"]:
                    return result

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30  # 30, 60, 90 seconds
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(30 * (attempt + 1))

        return {"success": False, "error": "All retry attempts failed"}

def main():
    # Example usage
    poster = TwitterPoster()

    video_path = "/path/to/your/video.mp4"  # Change this to your video path
    caption = "Check out this awesome video! 🎥 #automation #python"

    if os.path.exists(video_path):
        result = poster.post_video_with_retry(video_path, caption)
        print(f"Result: {result}")
    else:
        print(f"Video file not found: {video_path}")

if __name__ == "__main__":
    main()