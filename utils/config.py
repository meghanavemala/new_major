import os
import logging
import subprocess
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# All major Indian languages supported by Whisper
SUPPORTED_LANGUAGES = {
    # Major Indian Languages
    'hindi': 'hi',           # हिन्दी - Most widely spoken
    'bengali': 'bn',         # বাংলা - 2nd most spoken
    'telugu': 'te',          # తెలుగు - Andhra Pradesh, Telangana
    'marathi': 'mr',         # मराठी - Maharashtra
    'tamil': 'ta',           # தமிழ் - Tamil Nadu
    'gujarati': 'gu',        # ગુજરાતી - Gujarat
    'urdu': 'ur',           # اردو - Multiple states
    'kannada': 'kn',        # ಕನ್ನಡ - Karnataka
    'odia': 'or',           # ଓଡ଼ିଆ - Odisha
    'malayalam': 'ml',      # മലയാളം - Kerala
    'punjabi': 'pa',        # ਪੰਜਾਬੀ - Punjab
    'assamese': 'as',       # অসমীয়া - Assam
    'sanskrit': 'sa',       # संस्कृतम् - Classical
    
    # International Languages
    'english': 'en',        # English
    'arabic': 'ar',         # العربية
    'chinese': 'zh',        # 中文
    'spanish': 'es',        # Español
    'french': 'fr',         # Français
    'german': 'de',         # Deutsch
    'japanese': 'ja',       # 日本語
    'korean': 'ko',         # 한국어
    'russian': 'ru'         # Русский
}

# Recommended model sizes by language for optimal performance
MODEL_SIZE_BY_LANGUAGE = {
    # Well-supported languages - use large-v3 for best accuracy
    'hi': 'large-v3',    # Hindi - excellent support
    'bn': 'large-v3',    # Bengali - excellent support
    'ta': 'large-v3',    # Tamil - excellent support
    'mr': 'large-v3',    # Marathi - excellent support
    'te': 'large-v3',    # Telugu - excellent support
    'ml': 'large-v3',    # Malayalam - excellent support
    'kn': 'large-v3',    # Kannada - excellent support
    'pa': 'large-v3',    # Punjabi - excellent support
    'en': 'large-v3',    # English - excellent support
    
    # Moderately supported languages - use medium for balance
    'or': 'medium',      # Odia - medium model
    'as': 'medium',      # Assamese - medium model
    'sa': 'medium',      # Sanskrit - medium model
    
    # International languages
    'ar': 'medium',      # Arabic - good support
    'zh': 'large-v3',    # Chinese - excellent support
    'es': 'large-v3',    # Spanish - excellent support
    'fr': 'large-v3',    # French - excellent support
    'de': 'large-v3',    # German - excellent support
    'ja': 'large-v3',    # Japanese - excellent support
    'ko': 'large-v3',    # Korean - excellent support
    'ru': 'medium'       # Russian - good support
}

# Model size selection based on video length and quality preference
VIDEO_LENGTH_MODEL_SIZES = {
    'fast': {
        'short_video': 'small',     # < 5 minutes
        'medium_video': 'medium',   # 5-15 minutes
        'long_video': 'medium'      # > 15 minutes
    },
    'balanced': {
        'short_video': 'medium',    # < 5 minutes
        'medium_video': 'large',    # 5-15 minutes
        'long_video': 'large-v3'    # > 15 minutes
    },
    'accurate': {
        'short_video': 'medium',    # < 5 minutes
        'medium_video': 'large',    # 5-15 minutes
        'long_video': 'large-v3'    # > 15 minutes
    }
}

def get_smart_model_size(
    video_path: str,
    language: str = 'en',
    quality_preference: str = 'balanced'
) -> str:
    """Smart model selection based on video length and quality preference."""
    try:
        # Get video duration using ffprobe
        cmd = [
            'ffprobe', 
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        duration = float(subprocess.run(cmd, capture_output=True, text=True).stdout)
        
        # Determine video length category
        if duration < 300:  # 5 minutes
            length_category = 'short_video'
        elif duration < 900:  # 15 minutes
            length_category = 'medium_video'
        else:
            length_category = 'long_video'
        
        # Get model size based on video length and quality preference
        model_size = VIDEO_LENGTH_MODEL_SIZES[quality_preference][length_category]
        
        # Check if language has a specific recommended model size
        if language in MODEL_SIZE_BY_LANGUAGE:
            # Use language-specific model if it's larger than the length-based one
            lang_model = MODEL_SIZE_BY_LANGUAGE[language]
            model_sizes = ['small', 'medium', 'large', 'large-v3']
            if model_sizes.index(lang_model) > model_sizes.index(model_size):
                model_size = lang_model
        
        return model_size
        
    except Exception as e:
        logger.warning(f"Error determining smart model size: {e}. Using medium model.")
        return 'medium'
