import os
import sys
import time
import logging
from utils.downloader import handle_youtube_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_status_callback(status, progress, message):
    """Mock status callback function"""
    logger.info(f"Status: {status}, Progress: {progress}%, Message: {message}")

def main():
    """Test the downloader fixes"""
    # Create test directories
    upload_dir = "test_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Test YouTube URL (a short video for testing)
    test_url = "https://www.youtube.com/watch?v=9bZkp7q19f0"  # PSY - GANGNAM STYLE
    
    logger.info("Starting YouTube download test...")
    
    try:
        # Test the download function
        video_path, video_id = handle_youtube_download(
            yt_url=test_url,
            upload_dir=upload_dir,
            video_id="test_video_001",
            status_callback=test_status_callback
        )
        
        logger.info(f"Download successful! Video saved to: {video_path}")
        logger.info(f"Video ID: {video_id}")
        
        # Check if file exists
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            logger.info(f"Downloaded file size: {file_size} bytes")
        else:
            logger.error("Downloaded file not found!")
            return False
            
        # Clean up test files
        if os.path.exists(video_path):
            os.remove(video_path)
            logger.info("Cleaned up test file")
            
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Test completed successfully!")
        sys.exit(0)
    else:
        logger.error("Test failed!")
        sys.exit(1)