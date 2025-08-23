from typing import Dict, Any
import os


def load_config() -> Dict[str, Any]:
    """Return application configuration with environment overrides."""
    return {
        "UPLOAD_DIR": os.environ.get("UPLOAD_DIR", "uploads"),
        "PROCESSED_DIR": os.environ.get("PROCESSED_DIR", "processed"),
        "MAX_CONTENT_LENGTH": int(os.environ.get("MAX_CONTENT_LENGTH", str(100 * 1024 * 1024))),
        "MAX_VIDEO_DURATION": int(os.environ.get("MAX_VIDEO_DURATION", str(40 * 60))),
        "SUPPORTED_EXTENSIONS": {ext.strip() for ext in os.environ.get("SUPPORTED_EXTENSIONS", "mp4,avi,mov,mkv,webm").split(',') if ext.strip()},
        "DEFAULT_LANGUAGE": os.environ.get("DEFAULT_LANGUAGE", "en"),
        "DEFAULT_VOICE": os.environ.get("DEFAULT_VOICE", "en-US-Standard-C"),
        "DEFAULT_RESOLUTION": os.environ.get("DEFAULT_RESOLUTION", "480p"),
        "ENABLE_DARK_MODE": os.environ.get("ENABLE_DARK_MODE", "true").lower() == "true",
        "SECRET_KEY": os.environ.get("FLASK_SECRET_KEY", "dev-key-change-in-production"),
        "PARALLEL_DOWNLOADS": int(os.environ.get("PARALLEL_DOWNLOADS", "5")),
    }



