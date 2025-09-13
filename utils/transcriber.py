import os
import json
import logging
import subprocess
import torch
from typing import Dict, List, Tuple, Optional, Any
import librosa
import soundfile as sf
from pydub import AudioSegment
from faster_whisper import WhisperModel
from .gpu_config import get_device, clear_gpu_memory
from .config import SUPPORTED_LANGUAGES, get_smart_model_size
from .audio_utils import extract_audio, enhance_audio_quality

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
    """
    Smart model selection based on video length and quality preference.
    """
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

def _transcribe_with_model(audio_path: str, model, language: str) -> List[Dict]:
    """Helper function to transcribe audio with a model."""
    segments_fw, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True
    )
    
    segments = []
    if segments_fw and len(segments_fw) > 0:
        for s in segments_fw:
            segments.append({
                'start': s.start,
                'end': s.end,
                'text': s.text.strip(),
                'words': getattr(s, 'words', [])
            })
        return segments
    else:
        raise Exception("No speech segments detected")

def _transcribe_with_whisper(audio_path: str, model_size: str, device: str, lang_code: str) -> List[Dict]:
    """Transcribe using Faster-Whisper with fallback from GPU to CPU."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []

    segments = []
    logger.info(f"Transcribing with {model_size} model on {device}")
    
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type="float16" if device == "cuda" else "int8"
        )
        segments_fw, info = model.transcribe(
            audio_path,
            language=lang_code,
            beam_size=5,
            vad_filter=True
        )
        
        if segments_fw and len(segments_fw) > 0:
            for s in segments_fw:
                segments.append({
                    'start': s.start,
                    'end': s.end,
                    'text': s.text.strip(),
                    'words': getattr(s, 'words', [])
                })
            return segments
    except Exception as e:
        logger.warning(f"Transcription failed: {e}")
        return []
    
    return segments

def transcribe_video(
    video_path: str,
    processed_dir: str,
    video_id: str,
    language: str = 'english',
    model_size: Optional[str] = None,
    quality_preference: str = 'balanced'
) -> Tuple[Optional[str], List[Dict]]:
    """Transcribe video to text using Faster-Whisper.
    
    This function takes a video file and transcribes its audio content using
    Faster-Whisper. It supports both GPU and CPU processing, with automatic
    fallback to CPU if GPU memory is insufficient.
    
    Args:
        video_path (str): Path to the input video file
        processed_dir (str): Directory to save processed files
        video_id (str): Unique identifier for the video
        language (str, optional): Language of the video. Defaults to 'english'
        model_size (Optional[str], optional): Size of Whisper model. Defaults to None
        quality_preference (str, optional): Quality preference. Defaults to 'balanced'
        
    Returns:
        Tuple[Optional[str], List[Dict]]: A tuple containing:
            - str or None: Path to the segments JSON file, or None if transcription failed
            - List[Dict]: List of transcribed segments, each containing start time,
                       end time, text, and word-level timing information
    """
    segments = []
    audio_path = None
    seg_path = None
    
    # Setup paths and extract audio
    try:
        os.makedirs(processed_dir, exist_ok=True)
        audio_path = os.path.join(processed_dir, f"{video_id}_audio.wav")
        seg_path = os.path.join(processed_dir, f"{video_id}_segments.json")
        
        if not extract_audio(video_path, audio_path):
            logger.error("Failed to extract audio from video")
            return None, []
        
        # Configure model and device
        lang_code = SUPPORTED_LANGUAGES.get(language.lower(), language.lower())
        device = get_device()
        if not model_size:
            model_size = get_smart_model_size(video_path, lang_code, quality_preference)
        
        # Check GPU memory and adjust model size if needed
        if device == 'cuda':
            total_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            if total_mem <= 4096:
                logger.warning("GPU has 4GB or less. Using 'small' model.")
                model_size = 'small'
                
        logger.info(f"Transcribing with {model_size} model on {device}")
        
        try:
            if device == 'cuda':
                try:
                    torch.cuda.empty_cache()
                    model = WhisperModel(model_size, device='cuda', compute_type="float16")
                    segments_fw, info = model.transcribe(
                        audio_path,
                        language=lang_code,
                        beam_size=5,
                        vad_filter=True
                    )
                    
                    if segments_fw:
                        for s in segments_fw:
                            segments.append({
                                'start': s.start,
                                'end': s.end,
                                'text': s.text.strip(),
                                'words': getattr(s, 'words', [])
                            })
                        logger.info(f"GPU transcription successful: {len(segments)} segments")
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        logger.warning("CUDA OOM error, falling back to CPU. Clearing GPU memory.")
                        device = 'cpu'
                        model_size = 'small'  # Fallback to a smaller model for CPU
                        # Explicitly delete the model and clear memory
                        if 'model' in locals():
                            del model
                        clear_gpu_memory()
                    else:
                        raise
            
            # Use CPU if GPU failed or wasn't available
            if device == 'cpu' or not segments:
                model = WhisperModel(model_size, device='cpu', compute_type="int8")
                segments_fw, info = model.transcribe(
                    audio_path,
                    language=lang_code,
                    beam_size=5,
                    vad_filter=True
                )
                
                if segments_fw:
                    segments.clear()  # Clear any failed GPU attempt results
                    for s in segments_fw:
                        segments.append({
                            'start': s.start,
                            'end': s.end,
                            'text': s.text.strip(),
                            'words': getattr(s, 'words', [])
                        })
                    logger.info(f"CPU transcription successful: {len(segments)} segments")
            
        except ImportError as e:
            logger.error(f"Failed to import Faster-Whisper: {e}")
            return None, []
        
        # Save results if we got any segments
        if segments:
            with open(seg_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
            return seg_path, segments
        
        logger.error("No speech segments detected")
        return None, []
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return None, []
    
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary audio file: {e}")
        
        # Aggressively clear GPU memory
        clear_gpu_memory()

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
    """Transcribe audio using Faster Whisper for better accuracy and speed"""
    try:
        # Get optimized device and settings
        if device == 'auto':
            device = get_device()
        
        optimizations = optimize_for_model(f"faster-whisper-{model_size}")
        
        logger.info(f"Using Faster Whisper on device: {device}")
        log_gpu_status()
        
        # Use Faster Whisper for better performance
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=None,
            local_files_only=False
        )
        
        # Transcribe with optimized parameters
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=optimizations.get('beam_size', 5),
            best_of=optimizations.get('best_of', 5),
            temperature=optimizations.get('temperature', 0.0),
            compression_ratio_threshold=optimizations.get('compression_ratio_threshold', 2.4),
            log_prob_threshold=optimizations.get('log_prob_threshold', -1.0),
            no_speech_threshold=optimizations.get('no_speech_threshold', 0.6),
            condition_on_previous_text=optimizations.get('condition_on_previous_text', True),
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
        
        # Clear GPU memory after transcription
        clear_gpu_memory()
        
        return result, result['segments']
        
    except Exception as e:
        logger.error(f"Error in transcribe_with_faster_whisper: {str(e)}", exc_info=True)
        return None, []
    finally:
        # Always clear GPU memory in finally block
        clear_gpu_memory()

def enhance_audio_quality(audio_path: str, enhanced_path: str) -> bool:
    """
    Enhance audio quality for better transcription accuracy.
    
    Args:
        audio_path: Path to the input audio file
        enhanced_path: Path where the enhanced audio will be saved
        
    Returns:
        bool: True if enhancement was successful, False otherwise
    """
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
    """
    Get model recommendations based on video characteristics and language.
    
    Args:
        video_path: Optional path to the video file for duration-based recommendations
        language: Language code for language-specific recommendations
        
    Returns:
        Dict[str, Any]: Dictionary containing model recommendations and explanations
    """
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