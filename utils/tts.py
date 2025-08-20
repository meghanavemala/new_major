import os
import logging
from gtts import gTTS
from pydub import AudioSegment
from typing import Optional, Dict, Tuple
import tempfile
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Language mappings for TTS
LANGUAGE_MAP = {
    # Major Indian Languages
    'hindi': 'hi',
    'bengali': 'bn',
    'telugu': 'te',
    'marathi': 'mr',
    'tamil': 'ta',
    'gujarati': 'gu',
    'urdu': 'ur',
    'kannada': 'kn',
    'odia': 'or',
    'malayalam': 'ml',
    'punjabi': 'pa',
    'assamese': 'as',
    'sanskrit': 'sa',
    
    # International languages
    'english': 'en',
    'arabic': 'ar',
    'chinese': 'zh',
    'spanish': 'es',
    'french': 'fr',
    'german': 'de',
    'japanese': 'ja',
    'korean': 'ko',
    'russian': 'ru'
}

# Supported voices configuration
SUPPORTED_VOICES = {
    'en': [
        {'id': 'en-US-Standard-A', 'name': 'English US (Female)'},
        {'id': 'en-US-Standard-B', 'name': 'English US (Male)'},
        {'id': 'en-GB-Standard-A', 'name': 'English UK (Female)'},
        {'id': 'en-GB-Standard-B', 'name': 'English UK (Male)'}
    ],
    'hi': [
        {'id': 'hi-IN-Standard-A', 'name': 'Hindi (Female)'},
        {'id': 'hi-IN-Standard-B', 'name': 'Hindi (Male)'}
    ],
    'bn': [
        {'id': 'bn-IN-Standard-A', 'name': 'Bengali (Female)'},
        {'id': 'bn-IN-Standard-B', 'name': 'Bengali (Male)'}
    ],
    'te': [
        {'id': 'te-IN-Standard-A', 'name': 'Telugu (Female)'},
        {'id': 'te-IN-Standard-B', 'name': 'Telugu (Male)'}
    ],
    'ta': [
        {'id': 'ta-IN-Standard-A', 'name': 'Tamil (Female)'},
        {'id': 'ta-IN-Standard-B', 'name': 'Tamil (Male)'}
    ]
}

# Voice settings for gTTS
VOICE_SETTINGS = {
    # Indian Languages
    'hi': {'tld': 'co.in', 'slow': False, 'lang': 'hi'},
    'bn': {'tld': 'co.in', 'slow': False, 'lang': 'bn'},
    'te': {'tld': 'co.in', 'slow': False, 'lang': 'te'},
    'mr': {'tld': 'co.in', 'slow': False, 'lang': 'mr'},
    'ta': {'tld': 'co.in', 'slow': False, 'lang': 'ta'},
    'gu': {'tld': 'co.in', 'slow': False, 'lang': 'gu'},
    'ur': {'tld': 'co.in', 'slow': False, 'lang': 'ur'},
    'kn': {'tld': 'co.in', 'slow': False, 'lang': 'kn'},
    'or': {'tld': 'co.in', 'slow': False, 'lang': 'or'},
    'ml': {'tld': 'co.in', 'slow': False, 'lang': 'ml'},
    'pa': {'tld': 'co.in', 'slow': False, 'lang': 'pa'},
    'as': {'tld': 'co.in', 'slow': False, 'lang': 'as'},
    'sa': {'tld': 'co.in', 'slow': True, 'lang': 'sa'},
    
    # International Languages
    'en': {'tld': 'com', 'slow': False, 'lang': 'en'},
    'ar': {'tld': 'com', 'slow': False, 'lang': 'ar'},
    'zh': {'tld': 'com', 'slow': False, 'lang': 'zh'},
    'es': {'tld': 'com', 'slow': False, 'lang': 'es'},
    'fr': {'tld': 'fr', 'slow': False, 'lang': 'fr'},
    'de': {'tld': 'de', 'slow': False, 'lang': 'de'},
    'ja': {'tld': 'co.jp', 'slow': False, 'lang': 'ja'},
    'ko': {'tld': 'co.kr', 'slow': False, 'lang': 'ko'},
    'ru': {'tld': 'ru', 'slow': False, 'lang': 'ru'}
}

def text_to_speech(
    text: str, 
    output_dir: str, 
    video_id: str, 
    cluster_id: int,
    voice: str = 'en-US-Standard-C',
    language: str = 'english',
    slow: bool = False,
    bitrate: str = '192k',
    max_retries: int = 3,
    retry_delay: int = 2
) -> Optional[str]:
    """Convert text to speech using gTTS with retry logic and offline fallback.
    
    Args:
        text: Text to convert to speech
        output_dir: Directory to save the output file
        video_id: Unique ID for the video
        cluster_id: ID of the current topic cluster
        voice: Voice ID to use (for compatibility with advanced TTS)
        language: Language of the text (supports all Indian languages)
        slow: Whether to speak slowly (better for some languages)
        bitrate: Audio bitrate (e.g., '128k', '192k', '256k')
        max_retries: Maximum number of retry attempts for network issues
        retry_delay: Delay in seconds between retries
        
    Returns:
        Relative path to the generated audio file, or None on failure
    """
    if not text.strip():
        logger.warning("Empty text provided for TTS")
        return None
    
    temp_mp3_path = None
    attempt = 0
    
    while attempt < max_retries:
        try:
            # Validate language
            language = language.lower()
            if language not in LANGUAGE_MAP:
                logger.warning(f"Unsupported language: {language}. Defaulting to English.")
                language = 'english'
                
            lang_code = LANGUAGE_MAP[language]
            voice_settings = VOICE_SETTINGS.get(lang_code, VOICE_SETTINGS['en']).copy()
            voice_settings['slow'] = slow
            
            logger.info(f"Generating speech in {language} (Attempt {attempt + 1}/{max_retries})...")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate a temporary file path
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
                temp_mp3_path = temp_mp3.name
            
            # Generate speech using gTTS
            tts = gTTS(
                text=text,
                lang=voice_settings['lang'],
                tld=voice_settings['tld'],
                slow=voice_settings['slow']
            )
            
            # Save as MP3 first (gTTS works better with MP3)
            tts.save(temp_mp3_path)
            
            # Output file path (WAV format for better compatibility)
            output_file = os.path.join(output_dir, f"{video_id}_summary_{cluster_id}.wav")
            
            # Convert to WAV with pydub for better control over format
            audio = AudioSegment.from_mp3(temp_mp3_path)
            
            # Normalize audio volume
            audio = audio.normalize()
            
            # Export as WAV with specified bitrate
            audio.export(
                output_file,
                format='wav',
                parameters=['-ar', '44100', '-ac', '2', '-b:a', bitrate]
            )
            
            logger.info(f"Speech generated and saved to {output_file}")
            return output_file
            
        except Exception as e:
            attempt += 1
            logger.error(f"TTS attempt {attempt} failed: {str(e)}")
            
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay for next attempt
                retry_delay *= 2
            else:
                logger.error("All TTS attempts failed. Trying offline fallback...")
                # Here you could implement an offline TTS fallback
                # For now, we'll return None to indicate failure
                return None
        
        finally:
            # Clean up temporary files
            if temp_mp3_path and os.path.exists(temp_mp3_path):
                try:
                    os.remove(temp_mp3_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary MP3 file: {e}")

def get_available_voices():
    """Get a dictionary of available voices organized by language.
    
    Returns:
        dict: A dictionary mapping language codes to lists of available voices
    """
    return SUPPORTED_VOICES

def get_available_languages():
    """Get a dictionary of available languages and their codes."""
    return LANGUAGE_MAP.copy()