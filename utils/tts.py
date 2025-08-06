import os
import logging
from gtts import gTTS
from pydub import AudioSegment
from typing import Optional, Dict, Tuple
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Comprehensive language code mapping for gTTS
# Supports all major Indian languages and international languages
LANGUAGE_MAP = {
    # Major Indian Languages
    'hindi': 'hi',           # हिन्दी - Most widely spoken
    'bengali': 'bn',         # বাংলা - 2nd most spoken
    'telugu': 'te',          # తెలుగు - Andhra Pradesh, Telangana
    'marathi': 'mr',         # मराठी - Maharashtra
    'tamil': 'ta',           # தமிழ் - Tamil Nadu
    'gujarati': 'gu',        # ગુજરાતી - Gujarat
    'urdu': 'ur',            # اردو - Widely understood
    'kannada': 'kn',         # ಕನ್ನಡ - Karnataka
    'malayalam': 'ml',       # മലയാളം - Kerala
    'punjabi': 'pa',         # ਪੰਜਾਬੀ - Punjab
    'nepali': 'ne',          # नेपाली - Sikkim, West Bengal
    'sanskrit': 'sa',        # संस्कृत - Classical language
    'sindhi': 'sd',          # सिन्धी / سندھی
    
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
    'italian': 'it',         # Italiano - International
    'dutch': 'nl',           # Nederlands - International
    'turkish': 'tr',         # Türkçe - International
    'polish': 'pl',          # Polski - International
    'thai': 'th',            # ไทย - International
    'vietnamese': 'vi',      # Tiếng Việt - International
    'indonesian': 'id',      # Bahasa Indonesia - International
    'malay': 'ms',           # Bahasa Melayu - International
}

# Supported voices for each language
# Using gTTS which provides natural-sounding voices for many languages
SUPPORTED_VOICES = {
    # Indian Languages
    'hi': [
        {'id': 'hi-IN-Standard-A', 'name': 'Hindi (Female)'},
        {'id': 'hi-IN-Standard-B', 'name': 'Hindi (Male)'},
        {'id': 'hi-IN-Wavenet-A', 'name': 'Hindi (Female - Natural)'},
        {'id': 'hi-IN-Wavenet-B', 'name': 'Hindi (Male - Natural)'}
    ],
    'bn': [
        {'id': 'bn-IN-Standard-A', 'name': 'Bengali (Female)'},
        {'id': 'bn-IN-Standard-B', 'name': 'Bengali (Male)'}
    ],
    'te': [
        {'id': 'te-IN-Standard-A', 'name': 'Telugu (Female)'},
        {'id': 'te-IN-Standard-B', 'name': 'Telugu (Male)'}
    ],
    'mr': [
        {'id': 'mr-IN-Standard-A', 'name': 'Marathi (Female)'},
        {'id': 'mr-IN-Standard-B', 'name': 'Marathi (Male)'}
    ],
    'ta': [
        {'id': 'ta-IN-Standard-A', 'name': 'Tamil (Female)'},
        {'id': 'ta-IN-Standard-B', 'name': 'Tamil (Male)'}
    ],
    'gu': [
        {'id': 'gu-IN-Standard-A', 'name': 'Gujarati (Female)'},
        {'id': 'gu-IN-Standard-B', 'name': 'Gujarati (Male)'}
    ],
    'ur': [
        {'id': 'ur-IN-Standard-A', 'name': 'Urdu (Female)'},
        {'id': 'ur-IN-Standard-B', 'name': 'Urdu (Male)'}
    ],
    'kn': [
        {'id': 'kn-IN-Standard-A', 'name': 'Kannada (Female)'},
        {'id': 'kn-IN-Standard-B', 'name': 'Kannada (Male)'}
    ],
    'ml': [
        {'id': 'ml-IN-Standard-A', 'name': 'Malayalam (Female)'},
        {'id': 'ml-IN-Standard-B', 'name': 'Malayalam (Male)'}
    ],
    'pa': [
        {'id': 'pa-IN-Standard-A', 'name': 'Punjabi (Female)'},
        {'id': 'pa-IN-Standard-B', 'name': 'Punjabi (Male)'}
    ],
    'ne': [
        {'id': 'ne-NP-Standard-A', 'name': 'Nepali (Female)'},
        {'id': 'ne-NP-Standard-B', 'name': 'Nepali (Male)'}
    ],
    'sa': [
        {'id': 'sa-IN-Standard-A', 'name': 'Sanskrit (Female)'},
        {'id': 'sa-IN-Standard-B', 'name': 'Sanskrit (Male)'}
    ],
    
    # International Languages
    'en': [
        {'id': 'en-US-Standard-A', 'name': 'US English (Male)'},
        {'id': 'en-US-Standard-B', 'name': 'US English (Female)'},
        {'id': 'en-US-Standard-C', 'name': 'US English (Female 2)'},
        {'id': 'en-US-Standard-D', 'name': 'US English (Male 2)'},
        {'id': 'en-US-Wavenet-A', 'name': 'US English (Male - Natural)'},
        {'id': 'en-US-Wavenet-C', 'name': 'US English (Female - Natural)'},
        {'id': 'en-GB-Standard-A', 'name': 'British English (Female)'},
        {'id': 'en-GB-Standard-B', 'name': 'British English (Male)'},
        {'id': 'en-AU-Standard-A', 'name': 'Australian English (Female)'},
        {'id': 'en-AU-Standard-B', 'name': 'Australian English (Male)'}
    ],
    'ar': [
        {'id': 'ar-XA-Standard-A', 'name': 'Arabic (Female)'},
        {'id': 'ar-XA-Standard-B', 'name': 'Arabic (Male)'},
        {'id': 'ar-XA-Wavenet-A', 'name': 'Arabic (Female - Natural)'}
    ],
    'zh': [
        {'id': 'cmn-CN-Standard-A', 'name': 'Chinese Mandarin (Female)'},
        {'id': 'cmn-CN-Standard-B', 'name': 'Chinese Mandarin (Male)'},
        {'id': 'cmn-CN-Wavenet-A', 'name': 'Chinese Mandarin (Female - Natural)'}
    ],
    'es': [
        {'id': 'es-ES-Standard-A', 'name': 'Spanish (Female)'},
        {'id': 'es-ES-Standard-B', 'name': 'Spanish (Male)'},
        {'id': 'es-US-Standard-A', 'name': 'Spanish US (Female)'},
        {'id': 'es-US-Standard-B', 'name': 'Spanish US (Male)'}
    ],
    'fr': [
        {'id': 'fr-FR-Standard-A', 'name': 'French (Female)'},
        {'id': 'fr-FR-Standard-B', 'name': 'French (Male)'},
        {'id': 'fr-FR-Wavenet-A', 'name': 'French (Female - Natural)'}
    ],
    'de': [
        {'id': 'de-DE-Standard-A', 'name': 'German (Female)'},
        {'id': 'de-DE-Standard-B', 'name': 'German (Male)'},
        {'id': 'de-DE-Wavenet-A', 'name': 'German (Female - Natural)'}
    ],
    'ja': [
        {'id': 'ja-JP-Standard-A', 'name': 'Japanese (Female)'},
        {'id': 'ja-JP-Standard-B', 'name': 'Japanese (Female 2)'},
        {'id': 'ja-JP-Standard-C', 'name': 'Japanese (Male)'},
        {'id': 'ja-JP-Wavenet-A', 'name': 'Japanese (Female - Natural)'}
    ],
    'ko': [
        {'id': 'ko-KR-Standard-A', 'name': 'Korean (Female)'},
        {'id': 'ko-KR-Standard-B', 'name': 'Korean (Female 2)'},
        {'id': 'ko-KR-Standard-C', 'name': 'Korean (Male)'},
        {'id': 'ko-KR-Wavenet-A', 'name': 'Korean (Female - Natural)'}
    ],
    'ru': [
        {'id': 'ru-RU-Standard-A', 'name': 'Russian (Female)'},
        {'id': 'ru-RU-Standard-B', 'name': 'Russian (Male)'},
        {'id': 'ru-RU-Wavenet-A', 'name': 'Russian (Female - Natural)'}
    ],
    'pt': [
        {'id': 'pt-BR-Standard-A', 'name': 'Portuguese Brazil (Female)'},
        {'id': 'pt-BR-Standard-B', 'name': 'Portuguese Brazil (Male)'},
        {'id': 'pt-PT-Standard-A', 'name': 'Portuguese (Female)'},
        {'id': 'pt-PT-Standard-B', 'name': 'Portuguese (Male)'}
    ],
    'it': [
        {'id': 'it-IT-Standard-A', 'name': 'Italian (Female)'},
        {'id': 'it-IT-Standard-B', 'name': 'Italian (Female 2)'},
        {'id': 'it-IT-Standard-C', 'name': 'Italian (Male)'},
        {'id': 'it-IT-Wavenet-A', 'name': 'Italian (Female - Natural)'}
    ],
    'tr': [
        {'id': 'tr-TR-Standard-A', 'name': 'Turkish (Female)'},
        {'id': 'tr-TR-Standard-B', 'name': 'Turkish (Male)'},
        {'id': 'tr-TR-Wavenet-A', 'name': 'Turkish (Female - Natural)'}
    ],
    'th': [
        {'id': 'th-TH-Standard-A', 'name': 'Thai (Female)'},
        {'id': 'th-TH-Standard-B', 'name': 'Thai (Male)'}
    ],
    'vi': [
        {'id': 'vi-VN-Standard-A', 'name': 'Vietnamese (Female)'},
        {'id': 'vi-VN-Standard-B', 'name': 'Vietnamese (Male)'},
        {'id': 'vi-VN-Wavenet-A', 'name': 'Vietnamese (Female - Natural)'}
    ]
}

# Voice settings for each language with gTTS optimization
VOICE_SETTINGS = {
    # Indian Languages
    'hi': {'tld': 'co.in', 'slow': False, 'lang': 'hi'},       # Hindi
    'bn': {'tld': 'co.in', 'slow': False, 'lang': 'bn'},       # Bengali
    'te': {'tld': 'co.in', 'slow': False, 'lang': 'te'},       # Telugu
    'mr': {'tld': 'co.in', 'slow': False, 'lang': 'mr'},       # Marathi
    'ta': {'tld': 'co.in', 'slow': False, 'lang': 'ta'},       # Tamil
    'gu': {'tld': 'co.in', 'slow': False, 'lang': 'gu'},       # Gujarati
    'ur': {'tld': 'co.in', 'slow': False, 'lang': 'ur'},       # Urdu
    'kn': {'tld': 'co.in', 'slow': False, 'lang': 'kn'},       # Kannada
    'ml': {'tld': 'co.in', 'slow': False, 'lang': 'ml'},       # Malayalam
    'pa': {'tld': 'co.in', 'slow': False, 'lang': 'pa'},       # Punjabi
    'ne': {'tld': 'com.np', 'slow': False, 'lang': 'ne'},      # Nepali
    'sa': {'tld': 'co.in', 'slow': True, 'lang': 'sa'},        # Sanskrit (slower for clarity)
    'sd': {'tld': 'co.in', 'slow': False, 'lang': 'sd'},       # Sindhi
    
    # International Languages
    'en': {'tld': 'com', 'slow': False, 'lang': 'en'},         # English (US)
    'ar': {'tld': 'com.eg', 'slow': False, 'lang': 'ar'},      # Arabic
    'zh': {'tld': 'com', 'slow': False, 'lang': 'zh'},         # Chinese
    'es': {'tld': 'com', 'slow': False, 'lang': 'es'},         # Spanish
    'fr': {'tld': 'fr', 'slow': False, 'lang': 'fr'},          # French
    'de': {'tld': 'de', 'slow': False, 'lang': 'de'},          # German
    'ja': {'tld': 'co.jp', 'slow': False, 'lang': 'ja'},       # Japanese
    'ko': {'tld': 'co.kr', 'slow': False, 'lang': 'ko'},       # Korean
    'ru': {'tld': 'ru', 'slow': False, 'lang': 'ru'},          # Russian
    'pt': {'tld': 'com.br', 'slow': False, 'lang': 'pt'},      # Portuguese
    'it': {'tld': 'it', 'slow': False, 'lang': 'it'},          # Italian
    'nl': {'tld': 'nl', 'slow': False, 'lang': 'nl'},          # Dutch
    'tr': {'tld': 'com.tr', 'slow': False, 'lang': 'tr'},      # Turkish
    'pl': {'tld': 'pl', 'slow': False, 'lang': 'pl'},          # Polish
    'th': {'tld': 'co.th', 'slow': False, 'lang': 'th'},       # Thai
    'vi': {'tld': 'com.vn', 'slow': False, 'lang': 'vi'},      # Vietnamese
    'id': {'tld': 'co.id', 'slow': False, 'lang': 'id'},       # Indonesian
    'ms': {'tld': 'com.my', 'slow': False, 'lang': 'ms'},      # Malay
}

def text_to_speech(
    text: str, 
    output_dir: str, 
    video_id: str, 
    cluster_id: int,
    voice: str = 'en-US-Standard-C',
    language: str = 'english',
    slow: bool = False,
    bitrate: str = '192k'
) -> Optional[str]:
    """Convert text to speech using gTTS and save as WAV file.
    
    Args:
        text: Text to convert to speech
        output_dir: Directory to save the output file
        video_id: Unique ID for the video
        cluster_id: ID of the current topic cluster
        voice: Voice ID to use (for compatibility with advanced TTS)
        language: Language of the text (supports all Indian languages)
        slow: Whether to speak slowly (better for some languages)
        bitrate: Audio bitrate (e.g., '128k', '192k', '256k')
        
    Returns:
        Relative path to the generated audio file, or None on failure
    """
    if not text.strip():
        logger.warning("Empty text provided for TTS")
        return None
    
    try:
        # Validate language
        language = language.lower()
        if language not in LANGUAGE_MAP:
            logger.warning(f"Unsupported language: {language}. Defaulting to English.")
            language = 'english'
            
        lang_code = LANGUAGE_MAP[language]
        voice_settings = VOICE_SETTINGS.get(lang_code, VOICE_SETTINGS['en']).copy()
        voice_settings['slow'] = slow
        
        logger.info(f"Generating speech in {language}...")
        
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
        logger.error(f"Error in text_to_speech: {str(e)}", exc_info=True)
        return None
        
    finally:
        # Clean up temporary files
        if 'temp_mp3_path' in locals() and os.path.exists(temp_mp3_path):
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
