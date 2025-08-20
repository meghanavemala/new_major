import os
import json
import logging
import whisper
import torch
from typing import Dict, List, Tuple, Optional
from pydub import AudioSegment
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# All major Indian languages supported by Whisper
# Based on the 22 official languages of India as per the Constitution
SUPPORTED_LANGUAGES = {
    # Major Indian Languages
    'hindi': 'hi',           # हिन्दी - Most widely spoken
    'bengali': 'bn',         # বাংলা - 2nd most spoken
    'telugu': 'te',          # తెలుగు - Andhra Pradesh, Telangana
    'marathi': 'mr',         # मराठी - Maharashtra
    'tamil': 'ta',           # தமிழ் - Tamil Nadu
    'gujarati': 'gu',        # ગુજરાતી - Gujarat
    'urdu': 'ur',            # اردو - Widely understood
    'kannada': 'kn',         # ಕನ್ನಡ - Karnataka
    'odia': 'or',            # ଓଡ଼ିଆ - Odisha
    'malayalam': 'ml',       # മലയാളം - Kerala
    'punjabi': 'pa',         # ਪੰਜਾਬੀ - Punjab
    'assamese': 'as',        # অসমীয়া - Assam
    'maithili': 'mai',       # मैथिली - Bihar, Nepal
    'santali': 'sat',        # ᱥᱟᱱᱛᱟᱲᱤ - Jharkhand, West Bengal
    'nepali': 'ne',          # नेपाली - Sikkim, West Bengal
    'kashmiri': 'ks',        # कॉशुर / کٲشُر - Kashmir
    'konkani': 'gom',        # कोंकणी - Goa
    'sindhi': 'sd',          # سندھی / सिन्धी - 
    'dogri': 'doi',          # डोगरी - Jammu & Kashmir
    'manipuri': 'mni',       # মৈইতৈইলোন্ - Manipur
    'bodo': 'brx',           # बर'/बड़ो - Assam
    'sanskrit': 'sa',        # संस्कृत - Classical language
    
    # International languages for comparison/translation
    'english': 'en',         # English - Widely used
    'arabic': 'ar',          # العربية - For Islamic content
    'chinese': 'zh',         # 中文 - For international content
    'spanish': 'es',         # Español - International
    'french': 'fr',          # Français - International
    'german': 'de',          # Deutsch - International
    'japanese': 'ja',        # 日本語 - International
    'korean': 'ko',          # 한국어 - International
    'russian': 'ru',         # Русский - International
    'portuguese': 'pt',      # Português - International
}

# For backward compatibility
LANGUAGE_MAP = SUPPORTED_LANGUAGES

# Model size mapping for different languages
# Larger models are more accurate but slower and require more memory
# Recommendations based on language complexity and available training data
MODEL_SIZES = {
    # Well-supported languages with large datasets
    'en': 'base',       # English - excellent support
    'hi': 'medium',     # Hindi - good support
    'bn': 'medium',     # Bengali - good support
    'ta': 'medium',     # Tamil - good support
    'te': 'medium',     # Telugu - good support
    'mr': 'medium',     # Marathi - good support
    'gu': 'medium',     # Gujarati - good support
    'ur': 'medium',     # Urdu - good support
    'kn': 'medium',     # Kannada - good support
    'ml': 'medium',     # Malayalam - good support
    'pa': 'medium',     # Punjabi - good support
    
    # Languages with moderate support - use larger model for better accuracy
    'or': 'large',      # Odia - moderate support
    'as': 'large',      # Assamese - moderate support
    'mai': 'large',     # Maithili - limited support
    'ne': 'large',      # Nepali - moderate support
    'sa': 'large',      # Sanskrit - specialized
    
    # Languages with limited support - use largest model available
    'sat': 'large',     # Santali - limited support
    'ks': 'large',      # Kashmiri - limited support
    'gom': 'large',     # Konkani - limited support
    'sd': 'large',      # Sindhi - limited support
    'doi': 'large',     # Dogri - very limited support
    'mni': 'large',     # Manipuri - limited support
    'brx': 'large',     # Bodo - limited support
    
    # International languages
    'ar': 'medium',     # Arabic - good support
    'zh': 'medium',     # Chinese - good support
    'es': 'base',       # Spanish - excellent support
    'fr': 'base',       # French - excellent support
    'de': 'base',       # German - excellent support
    'ja': 'medium',     # Japanese - good support
    'ko': 'medium',     # Korean - good support
    'ru': 'medium',     # Russian - good support
    'pt': 'base',       # Portuguese - good support
}

def extract_audio(
    video_path: str, 
    output_path: str, 
    sample_rate: int = 16000
) -> bool:
    """Extract audio from video using ffmpeg.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save extracted audio
        sample_rate: Sample rate for output audio (Hz)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', video_path,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono audio
            '-f', 'wav',
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600  # 10 minutes safety timeout
        )
        logger.debug(f"FFmpeg audio extraction output: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e.stderr}")
        return False
    except subprocess.TimeoutExpired as e:
        logger.error(f"FFmpeg audio extraction timed out: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in extract_audio: {str(e)}", exc_info=True)
        return False

def transcribe_video(
    video_path: str, 
    processed_dir: str, 
    video_id: str, 
    language: str = 'english',
    model_size: Optional[str] = None
) -> Tuple[Optional[str], List[Dict]]:
    """Transcribe video to text using Whisper.
    
    Args:
        video_path: Path to input video file
        processed_dir: Directory to save output files
        video_id: Unique ID for the video
        language: Language of the video ('english', 'hindi', or 'kannada')
        model_size: Override the default model size (tiny, base, small, medium, large)
        
    Returns:
        Tuple containing:
        - Path to segments JSON file (or None if failed)
        - List of segment dictionaries
    """
    try:
        # Validate language
        language = language.lower()
        if language not in LANGUAGE_MAP:
            logger.warning(f"Unsupported language: {language}. Defaulting to English.")
            language = 'english'
            
        lang_code = LANGUAGE_MAP[language]
        
        # Set model size if not provided
        if model_size is None:
            model_size = MODEL_SIZES.get(lang_code, 'base')
        
        logger.info(f"Transcribing video in {language} using Whisper {model_size} model...")
        
        # Create output directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)
        
        # Paths for intermediate and output files
        audio_path = os.path.join(processed_dir, f"{video_id}_audio.wav")
        seg_path = os.path.join(processed_dir, f"{video_id}_segments.json")
        
        # Extract audio from video
        if not extract_audio(video_path, audio_path):
            logger.error("Failed to extract audio from video")
            return None, []
        
        # Check if GPU is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load Whisper model
        model = whisper.load_model(model_size, device=device)
        
        # Transcribe audio
        result = model.transcribe(
            audio_path,
            language=lang_code,
            verbose=False,
            fp16=(device == 'cuda')  # Use mixed precision on GPU
        )
        
        # Get segments
        segments = result.get('segments', [])
        
        # Save segments to file
        with open(seg_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Transcription complete. Saved to {seg_path}")
        return seg_path, segments
        
    except Exception as e:
        logger.error(f"Error in transcribe_video: {str(e)}", exc_info=True)
        return None, []
    finally:
        # Clean up temporary audio file if it exists
        if 'audio_path' in locals() and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary audio file: {e}")
              