import os
import json
import logging
import whisper
import torch
from typing import Dict, List, Tuple, Optional, Any
from pydub import AudioSegment
import subprocess
import numpy as np
from faster_whisper import WhisperModel
import librosa
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.WARNING)
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
    # Well-supported languages - use medium models for good balance
    'en': 'medium',     # English - medium model for good accuracy/speed balance
    'hi': 'medium',     # Hindi - medium model for Indian language support
    'bn': 'medium',     # Bengali - medium model for good balance
    'ta': 'medium',     # Tamil - medium model for good balance
    'te': 'medium',     # Telugu - medium model for good balance
    'mr': 'medium',     # Marathi - medium model for good balance
    'gu': 'medium',     # Gujarati - medium model for good balance
    'ur': 'medium',     # Urdu - medium model for good balance
    'kn': 'medium',     # Kannada - medium model for good balance
    'ml': 'medium',     # Malayalam - medium model for good balance
    'pa': 'medium',     # Punjabi - medium model for good balance
    
    # Languages with moderate support - use medium for balance
    'or': 'medium',     # Odia - medium model
    'as': 'medium',     # Assamese - medium model
    'mai': 'medium',    # Maithili - medium model
    'ne': 'medium',     # Nepali - medium model
    'sa': 'medium',     # Sanskrit - medium model
    
    # Languages with limited support - use medium for balance
    'sat': 'medium',    # Santali - medium model
    'ks': 'medium',     # Kashmiri - medium model
    'gom': 'medium',    # Konkani - medium model
    'sd': 'medium',     # Sindhi - medium model
    'doi': 'medium',    # Dogri - medium model
    'mni': 'medium',    # Manipuri - medium model
    'brx': 'medium',    # Bodo - medium model
    
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

# Smart model selection strategy - balances accuracy with speed
MODEL_SELECTION_STRATEGY = {
    'fast': {
        'short_video': 'tiny',      # < 5 minutes
        'medium_video': 'base',     # 5-15 minutes
        'long_video': 'small'       # > 15 minutes
    },
    'balanced': {
        'short_video': 'base',      # < 5 minutes
        'medium_video': 'medium',   # 5-15 minutes
        'long_video': 'medium'      # > 15 minutes
    },
    'accurate': {
        'short_video': 'medium',    # < 5 minutes
        'medium_video': 'large',    # 5-15 minutes
        'long_video': 'large-v3'    # > 15 minutes
    }
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

def get_smart_model_size(
    video_path: str,
    language: str = 'en',
    quality_preference: str = 'balanced'
) -> str:
    """Smart model selection based on video length and quality preference."""
    try:
        # Get video duration using ffprobe (faster than loading video)
        import subprocess
        import json
        
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        
        # Determine video length category
        if duration < 300:  # < 5 minutes
            video_category = 'short_video'
        elif duration < 900:  # < 15 minutes
            video_category = 'medium_video'
        else:  # > 15 minutes
            video_category = 'long_video'
        
        # Get model size from strategy
        strategy = MODEL_SELECTION_STRATEGY.get(quality_preference, 'balanced')
        model_size = strategy.get(video_category, 'medium')
        
        logger.info(f"Video duration: {duration:.1f}s, category: {video_category}, selected model: {model_size}")
        return model_size
        
    except Exception as e:
        logger.warning(f"Error in smart model selection: {e}, using default medium model")
        return 'medium'

def transcribe_video(
    video_path: str, 
    processed_dir: str, 
    video_id: str, 
    language: str = 'english',
    model_size: Optional[str] = None,
    quality_preference: str = 'balanced'
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
        
        # Set model size if not provided - use smart selection
        if model_size is None:
            model_size = get_smart_model_size(video_path, lang_code, quality_preference)
        else:
            # Use provided model size but log the choice
            logger.info(f"Using manually specified model size: {model_size}")
        
        logger.info(f"Transcribing video in {language} using Whisper {model_size} model...")
        
        # Log model selection reasoning
        logger.info(f"Model selection: {model_size} for {language} language")
        logger.info(f"Quality preference: {quality_preference}")
        
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

# Alternative transcription models for better accuracy
TRANSCRIPTION_MODELS = {
    'whisper': 'whisper',           # OpenAI Whisper (default)
    'faster_whisper': 'faster_whisper',  # Faster Whisper (faster, better accuracy)
    'whisperx': 'whisperx',         # WhisperX (best accuracy, slower)
    'stable_whisper': 'stable_whisper'   # Stable Whisper (good balance)
}

def transcribe_with_faster_whisper(
    audio_path: str,
    language: str = 'en',
    model_size: str = 'large-v3',
    device: str = 'auto'
) -> Tuple[Optional[Dict], List[Dict]]:
    """Transcribe audio using Faster Whisper for better accuracy and speed."""
    try:
        # Use Faster Whisper for better performance
        model = WhisperModel(
            model_size,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
            download_root=None,
            local_files_only=False
        )
        
        # Transcribe with better parameters
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,  # Better beam search
            best_of=5,    # Keep best 5 candidates
            temperature=0.0,  # Deterministic output
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            initial_prompt=None
        )
        
        # Convert to standard format
        result = {
            'text': '',
            'segments': [],
            'language': info.language,
            'language_probability': info.language_probability
        }
        
        for segment in segments:
            result['segments'].append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'words': getattr(segment, 'words', [])
            })
            result['text'] += segment.text.strip() + ' '
        
        result['text'] = result['text'].strip()
        return result, result['segments']
        
    except Exception as e:
        logger.error(f"Error in transcribe_with_faster_whisper: {str(e)}", exc_info=True)
        return None, []

def enhance_audio_quality(audio_path: str, enhanced_path: str) -> bool:
    """Enhance audio quality for better transcription accuracy."""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Apply noise reduction
        y_reduced = librosa.effects.preemphasis(y, coef=0.97)
        
        # Normalize audio
        y_normalized = librosa.util.normalize(y_reduced)
        
        # Save enhanced audio
        sf.write(enhanced_path, y_normalized, sr)
        
        return True
        
    except Exception as e:
        logger.error(f"Error enhancing audio: {str(e)}", exc_info=True)
        return False

def get_model_recommendations(video_path: str = None, language: str = 'en') -> Dict[str, Any]:
    """Get model recommendations based on video characteristics and language."""
    recommendations = {
        'fast': {
            'description': 'Fastest processing, lower accuracy',
            'models': ['tiny', 'base'],
            'best_for': 'Quick previews, testing, short videos',
            'speed': 'Very Fast',
            'accuracy': 'Good'
        },
        'balanced': {
            'description': 'Good balance of speed and accuracy',
            'models': ['base', 'small', 'medium'],
            'best_for': 'Most use cases, production videos',
            'speed': 'Fast',
            'accuracy': 'Very Good'
        },
        'accurate': {
            'description': 'Highest accuracy, slower processing',
            'models': ['medium', 'large', 'large-v3'],
            'best_for': 'Important content, final production',
            'speed': 'Slow',
            'accuracy': 'Excellent'
        }
    }
    
    if video_path:
        try:
            # Get video duration for specific recommendations
            import subprocess
            import json
            
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'json',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            
            # Add video-specific recommendations
            if duration < 300:  # < 5 minutes
                recommendations['recommended'] = 'fast'
                recommendations['reason'] = f'Short video ({duration:.1f}s) - use fast models'
            elif duration < 900:  # < 15 minutes
                recommendations['recommended'] = 'balanced'
                recommendations['reason'] = f'Medium video ({duration:.1f}s) - use balanced models'
            else:  # > 15 minutes
                recommendations['recommended'] = 'balanced'
                recommendations['reason'] = f'Long video ({duration:.1f}s) - use balanced models for efficiency'
                
        except Exception as e:
            logger.warning(f"Could not analyze video for recommendations: {e}")
    
    # Add language-specific recommendations
    if language in ['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'ur', 'kn', 'ml', 'pa']:
        recommendations['language_note'] = f'Indian language ({language}) - medium models recommended for best balance'
    elif language in ['en', 'es', 'fr', 'de']:
        recommendations['language_note'] = f'Well-supported language ({language}) - any model size works well'
    else:
        recommendations['language_note'] = f'Language ({language}) - medium models recommended for compatibility'
    
    return recommendations