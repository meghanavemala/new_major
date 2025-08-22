import os
import logging
from gtts import gTTS
from pydub import AudioSegment
from typing import Optional, Dict, Tuple
import tempfile
import time
from pathlib import Path
import asyncio

try:
    import edge_tts  # Microsoft Edge neural TTS
    EDGE_TTS_AVAILABLE = True
except Exception:
    EDGE_TTS_AVAILABLE = False

try:
    from elevenlabs import generate as eleven_generate
    from elevenlabs import save as eleven_save
    from elevenlabs.client import ElevenLabs
    ELEVEN_AVAILABLE = True
except Exception:
    ELEVEN_AVAILABLE = False

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

# Supported voices configuration (generic; actual provider used is selected at runtime)
SUPPORTED_VOICES = {
    'en': [
        {'id': 'en-US-Standard-A', 'name': 'English US (Female)'},
        {'id': 'en-US-Standard-B', 'name': 'English US (Male)'},
        {'id': 'en-GB-Standard-A', 'name': 'English UK (Female)'},
        {'id': 'en-GB-Standard-B', 'name': 'English UK (Male)'},
        # Edge Neural examples
        {'id': 'en-US-JennyNeural', 'name': 'English US Jenny (Neural)'},
        {'id': 'en-US-GuyNeural', 'name': 'English US Guy (Neural)'}
    ],
    'hi': [
        {'id': 'hi-IN-Standard-A', 'name': 'Hindi (Female)'},
        {'id': 'hi-IN-Standard-B', 'name': 'Hindi (Male)'},
        {'id': 'hi-IN-SwaraNeural', 'name': 'Hindi Swara (Neural)'},
        {'id': 'hi-IN-MadhurNeural', 'name': 'Hindi Madhur (Neural)'}
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

# Minimal mapping to more realistic Edge neural voices when provider is 'edge'
EDGE_DEFAULT_VOICE_BY_LANG = {
    'en': 'en-US-JennyNeural',
    'hi': 'hi-IN-SwaraNeural',
    'bn': 'bn-IN-TanishaaNeural',
    'te': 'te-IN-ShrutiNeural',
    'mr': 'mr-IN-AarohiNeural',
    'ta': 'ta-IN-PallaviNeural',
    'gu': 'gu-IN-DhwaniNeural',
    'ur': 'ur-PK-UzmaNeural',
    'kn': 'kn-IN-SapnaNeural',
    'or': 'or-IN-TapasNeural',
    'ml': 'ml-IN-SobhanaNeural',
    'pa': 'pa-IN-GaganNeural',
    'as': 'as-IN-DhwaniNeural',
    'sa': 'sa-IN-AbhijitNeural',
}


def _normalize_voice_for_provider(voice: str, lang_code: str, provider: str) -> str:
    """Map generic voice ids to provider-specific ones when possible."""
    if provider == 'edge':
        # If user passed an Edge voice, keep it; else pick a default neural for language
        if voice and 'Neural' in voice:
            return voice
        return EDGE_DEFAULT_VOICE_BY_LANG.get(lang_code, EDGE_DEFAULT_VOICE_BY_LANG.get('en', 'en-US-JennyNeural'))
    return voice or ''


async def _edge_tts_speak_to_mp3(text: str, voice: str, out_mp3_path: str, rate: str = '+0%', pitch: str = '+0Hz') -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(out_mp3_path)


def _eleven_tts_to_mp3(text: str, voice: str, out_mp3_path: str, model: str = 'eleven_turbo_v2') -> None:
    api_key = os.environ.get('ELEVENLABS_API_KEY')
    if not api_key:
        raise RuntimeError('ELEVENLABS_API_KEY missing')
    client = ElevenLabs(api_key=api_key)
    # If user didn't specify a known eleven voice id, fallback to a default
    voice_id = voice if voice and len(voice) > 10 else os.environ.get('ELEVENLABS_VOICE_ID', '')
    audio = client.generate(text=text, voice=voice_id or 'Rachel', model=model)
    eleven_save(audio, out_mp3_path)

def text_to_speech(
    text: str, 
    output_dir: str, 
    video_id: str, 
    cluster_id: int,
    voice: str = 'en-US-JennyNeural',
    language: str = 'english',
    slow: bool = False,
    bitrate: str = '192k',
    max_retries: int = 3,
    retry_delay: int = 2,
    provider: Optional[str] = None,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
    eleven_voice_id: Optional[str] = None,
    eleven_model: Optional[str] = None
) -> Optional[str]:
    """Convert text to speech using a realistic provider (Edge neural by default) with fallback to gTTS.
    
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
    provider = (provider or os.environ.get('TTS_PROVIDER', 'eleven')).lower()
    edge_rate = (rate or os.environ.get('TTS_RATE', '+0%'))
    edge_pitch = (pitch or os.environ.get('TTS_PITCH', '+0Hz'))
    eleven_model = (eleven_model or os.environ.get('ELEVENLABS_MODEL', 'eleven_turbo_v2'))
    
    while attempt < max_retries:
        try:
            # Validate language
            language = language.lower()
            if language not in LANGUAGE_MAP:
                logger.warning(f"Unsupported language: {language}. Defaulting to English.")
                language = 'english'
                
            lang_code = LANGUAGE_MAP[language]
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Generate a temporary file path
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
                temp_mp3_path = temp_mp3.name

            used_provider = provider
            # First choice: ElevenLabs, if configured
            if provider == 'eleven' and ELEVEN_AVAILABLE and os.environ.get('ELEVENLABS_API_KEY'):
                try:
                    logger.info("[TTS] Using ElevenLabs for ultra-realistic voice")
                    _eleven_tts_to_mp3(text, eleven_voice_id or voice, temp_mp3_path, model=eleven_model)
                except Exception as e11:
                    logger.warning(f"ElevenLabs TTS failed, falling back to Edge: {e11}")
                    used_provider = 'edge'
            # Second choice: Edge neural voices if available
            if used_provider == 'edge' and EDGE_TTS_AVAILABLE:
                edge_voice = _normalize_voice_for_provider(voice, lang_code, 'edge')
                logger.info(f"[TTS] Using Edge neural voice: {edge_voice}")
                try:
                    asyncio.run(_edge_tts_speak_to_mp3(text, edge_voice, temp_mp3_path, rate=edge_rate, pitch=edge_pitch))
                except Exception as edge_err:
                    logger.warning(f"Edge TTS failed, falling back to gTTS: {edge_err}")
                    used_provider = 'gtts'
            else:
                used_provider = 'gtts'

            if used_provider == 'gtts':
                voice_settings = VOICE_SETTINGS.get(lang_code, VOICE_SETTINGS['en']).copy()
                voice_settings['slow'] = slow
                logger.info(f"[TTS] Using gTTS for language {language}")
                tts = gTTS(
                    text=text,
                    lang=voice_settings['lang'],
                    tld=voice_settings['tld'],
                    slow=voice_settings['slow']
                )
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