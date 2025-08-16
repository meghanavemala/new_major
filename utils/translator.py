"""
Language Translation Utility Module

This module provides comprehensive translation capabilities for the video summarizer,
enabling conversion between different Indian languages and international languages.
It supports both Google Translate API and offline translation models.

Author: Video Summarizer Team
Created: 2024
"""

import os
import logging
import json
from typing import Dict, List, Optional, Tuple, Union
# Optional: Google Translate client (used when available)
try:
    from googletrans import Translator, LANGUAGES  # type: ignore
    GOOGLETRANS_AVAILABLE = True
except Exception:
    # If googletrans is not installed or fails to import, we gracefully degrade
    GOOGLETRANS_AVAILABLE = False
    Translator = None  # type: ignore
    LANGUAGES = {}  # type: ignore
import torch
from transformers import (
    MarianMTModel, MarianTokenizer, 
    M2M100ForConditionalGeneration, M2M100Tokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
import time
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language code mappings - comprehensive support for Indian languages
LANGUAGE_MAPPINGS = {
    # Indian Languages (official)
    'hindi': {'code': 'hi', 'google': 'hi', 'iso': 'hi', 'native': 'हिन्दी'},
    'bengali': {'code': 'bn', 'google': 'bn', 'iso': 'bn', 'native': 'বাংলা'},
    'telugu': {'code': 'te', 'google': 'te', 'iso': 'te', 'native': 'తెలుగు'},
    'marathi': {'code': 'mr', 'google': 'mr', 'iso': 'mr', 'native': 'मराठी'},
    'tamil': {'code': 'ta', 'google': 'ta', 'iso': 'ta', 'native': 'தமிழ்'},
    'gujarati': {'code': 'gu', 'google': 'gu', 'iso': 'gu', 'native': 'ગુજરાતી'},
    'urdu': {'code': 'ur', 'google': 'ur', 'iso': 'ur', 'native': 'اردو'},
    'kannada': {'code': 'kn', 'google': 'kn', 'iso': 'kn', 'native': 'ಕನ್ನಡ'},
    'odia': {'code': 'or', 'google': 'or', 'iso': 'or', 'native': 'ଓଡ଼ିଆ'},
    'malayalam': {'code': 'ml', 'google': 'ml', 'iso': 'ml', 'native': 'മലയാളം'},
    'punjabi': {'code': 'pa', 'google': 'pa', 'iso': 'pa', 'native': 'ਪੰਜਾਬੀ'},
    'assamese': {'code': 'as', 'google': 'as', 'iso': 'as', 'native': 'অসমীয়া'},
    'nepali': {'code': 'ne', 'google': 'ne', 'iso': 'ne', 'native': 'नेपाली'},
    'sanskrit': {'code': 'sa', 'google': 'sa', 'iso': 'sa', 'native': 'संस्कृत'},
    'sindhi': {'code': 'sd', 'google': 'sd', 'iso': 'sd', 'native': 'سندھی'},
    
    # International Languages
    'english': {'code': 'en', 'google': 'en', 'iso': 'en', 'native': 'English'},
    'arabic': {'code': 'ar', 'google': 'ar', 'iso': 'ar', 'native': 'العربية'},
    'chinese': {'code': 'zh', 'google': 'zh', 'iso': 'zh', 'native': '中文'},
    'spanish': {'code': 'es', 'google': 'es', 'iso': 'es', 'native': 'Español'},
    'french': {'code': 'fr', 'google': 'fr', 'iso': 'fr', 'native': 'Français'},
    'german': {'code': 'de', 'google': 'de', 'iso': 'de', 'native': 'Deutsch'},
    'japanese': {'code': 'ja', 'google': 'ja', 'iso': 'ja', 'native': '日本語'},
    'korean': {'code': 'ko', 'google': 'ko', 'iso': 'ko', 'native': '한국어'},
    'russian': {'code': 'ru', 'google': 'ru', 'iso': 'ru', 'native': 'Русский'},
    'portuguese': {'code': 'pt', 'google': 'pt', 'iso': 'pt', 'native': 'Português'},
}

# Translation model preferences - ordered by quality
TRANSLATION_METHODS = [
    'google_translate',  # Best quality, requires internet
    'm2m100',           # Good multilingual model, offline
    'marian',           # Good for specific language pairs, offline
    'fallback'          # Simple word replacement, last resort
]

# Cache for translation models to avoid reloading
translation_cache = {
    'google_translator': None,
    'm2m100_model': None,
    'm2m100_tokenizer': None,
    'marian_models': {},
    'cache_dir': 'translation_cache'
}

def ensure_cache_dir():
    """Ensure translation cache directory exists."""
    cache_dir = Path(translation_cache['cache_dir'])
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def get_google_translator():
    """Get or create Google Translator instance."""
    if not GOOGLETRANS_AVAILABLE:
        logger.warning("googletrans not installed; skipping Google Translate. Install with: pip install googletrans==4.0.0rc1")
        return None
    if translation_cache['google_translator'] is None:
        try:
            translation_cache['google_translator'] = Translator()
            logger.info("Google Translator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Translator: {e}")
            return None
    return translation_cache['google_translator']

def get_m2m100_model():
    """Get or load M2M100 multilingual translation model."""
    if translation_cache['m2m100_model'] is None:
        try:
            logger.info("Loading M2M100 multilingual translation model...")
            model_name = "facebook/m2m100_418M"
            
            # Check if CUDA is available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
            # Load tokenizer and model
            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
            
            translation_cache['m2m100_tokenizer'] = tokenizer
            translation_cache['m2m100_model'] = model
            
            logger.info("M2M100 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load M2M100 model: {e}")
            return None, None
    
    return translation_cache['m2m100_model'], translation_cache['m2m100_tokenizer']

def translate_with_google(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """
    Translate text using Google Translate API.
    
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        Translated text or None if failed
    """
    if not GOOGLETRANS_AVAILABLE:
        return None
    try:
        translator = get_google_translator()
        if not translator:
            return None
        
        # Convert language codes to Google Translate format
        src_code = LANGUAGE_MAPPINGS.get(source_lang, {}).get('google', source_lang)
        tgt_code = LANGUAGE_MAPPINGS.get(target_lang, {}).get('google', target_lang)
        
        # Perform translation
        result = translator.translate(text, src=src_code, dest=tgt_code)
        translated_text = result.text
        
        logger.info(f"Google Translate: {source_lang} -> {target_lang} successful")
        return translated_text
        
    except Exception as e:
        logger.error(f"Google Translate failed: {e}")
        return None

def translate_with_m2m100(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """
    Translate text using M2M100 multilingual model.
    
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        Translated text or None if failed
    """
    try:
        model, tokenizer = get_m2m100_model()
        if not model or not tokenizer:
            return None
        
        # Convert language codes to M2M100 format
        src_code = LANGUAGE_MAPPINGS.get(source_lang, {}).get('code', source_lang)
        tgt_code = LANGUAGE_MAPPINGS.get(target_lang, {}).get('code', target_lang)
        
        # Set source language
        tokenizer.src_lang = src_code
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id(tgt_code),
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode translation
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        logger.info(f"M2M100 translation: {source_lang} -> {target_lang} successful")
        return translated_text
        
    except Exception as e:
        logger.error(f"M2M100 translation failed: {e}")
        return None

def translate_text(
    text: str, 
    source_lang: str, 
    target_lang: str,
    method: str = 'auto'
) -> Tuple[Optional[str], str]:
    """
    Translate text from source language to target language.
    
    Args:
        text: Text to translate
        source_lang: Source language (name or code)
        target_lang: Target language (name or code)
        method: Translation method ('auto', 'google', 'm2m100', 'marian')
        
    Returns:
        Tuple of (translated_text, method_used)
    """
    if not text or not text.strip():
        return text, 'no_translation_needed'
    
    # Normalize language names
    source_lang = source_lang.lower()
    target_lang = target_lang.lower()
    
    # Check if translation is needed
    src_code = LANGUAGE_MAPPINGS.get(source_lang, {}).get('code', source_lang)
    tgt_code = LANGUAGE_MAPPINGS.get(target_lang, {}).get('code', target_lang)
    
    if src_code == tgt_code:
        logger.info(f"No translation needed: {source_lang} == {target_lang}")
        return text, 'no_translation_needed'
    
    # Try translation methods in order of preference
    methods_to_try = [method] if method != 'auto' else TRANSLATION_METHODS
    
    for translation_method in methods_to_try:
        if translation_method == 'google_translate':
            result = translate_with_google(text, source_lang, target_lang)
            if result:
                return result, 'google_translate'
                
        elif translation_method == 'm2m100':
            result = translate_with_m2m100(text, source_lang, target_lang)
            if result:
                return result, 'm2m100'
                
        elif translation_method == 'fallback':
            # Simple fallback - return original text with warning
            logger.warning(f"All translation methods failed, returning original text")
            return text, 'fallback'
    
    # If all methods fail, return original text
    logger.error(f"Translation failed for {source_lang} -> {target_lang}")
    return text, 'failed'

from concurrent.futures import ThreadPoolExecutor, as_completed

def translate_segments(
    segments: List[Dict], 
    source_lang: str, 
    target_lang: str,
    method: str = 'auto',
    max_workers: int = 5   # number of parallel threads
) -> List[Dict]:
    """
    Translate text segments from source to target language with multithreading.

    Args:
        segments: List of segment dictionaries with 'text' field
        source_lang: Source language
        target_lang: Target language
        method: Translation method to use
        max_workers: Number of threads for parallel translation

    Returns:
        List of segments with translated text
    """
    if not segments:
        return segments
    
    logger.info(f"Translating {len(segments)} segments from {source_lang} to {target_lang}")

    translated_segments = [None] * len(segments)  # placeholder for results
    translation_stats = {'success': 0, 'failed': 0, 'methods': {}}

    def translate_one(index, segment):
        original_text = segment.get('text', '')
        if not original_text.strip():
            return index, segment.copy()

        translated_text, method_used = translate_text(
            original_text, source_lang, target_lang, method
        )

        # stats
        if method_used != 'failed':
            translation_stats['success'] += 1
            translation_stats['methods'][method_used] = translation_stats['methods'].get(method_used, 0) + 1
        else:
            translation_stats['failed'] += 1

        new_segment = segment.copy()
        new_segment['text'] = translated_text
        new_segment['original_text'] = original_text
        new_segment['translation_method'] = method_used

        return index, new_segment

    # Run translations in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(translate_one, i, seg) for i, seg in enumerate(segments)]
        for future in as_completed(futures):
            idx, translated_seg = future.result()
            translated_segments[idx] = translated_seg

    logger.info(f"Translation complete: {translation_stats['success']} successful, {translation_stats['failed']} failed")
    logger.info(f"Methods used: {translation_stats['methods']}")
    
    return translated_segments


# def translate_segments(
#     segments: List[Dict], 
#     source_lang: str, 
#     target_lang: str,
#     method: str = 'auto'
# ) -> List[Dict]:
#     """
#     Translate text segments from source to target language.
    
#     Args:
#         segments: List of segment dictionaries with 'text' field
#         source_lang: Source language
#         target_lang: Target language
#         method: Translation method to use
        
#     Returns:
#         List of segments with translated text
#     """
#     if not segments:
#         return segments
    
#     logger.info(f"Translating {len(segments)} segments from {source_lang} to {target_lang}")
    
#     translated_segments = []
#     translation_stats = {'success': 0, 'failed': 0, 'methods': {}}
    
#     for i, segment in enumerate(segments):
#         original_text = segment.get('text', '')
        
#         if not original_text.strip():
#             translated_segments.append(segment.copy())
#             continue
        
#         # Translate the text
#         translated_text, method_used = translate_text(
#             original_text, source_lang, target_lang, method
#         )
        
#         # Update statistics
#         if method_used not in ['failed']:
#             translation_stats['success'] += 1
#             translation_stats['methods'][method_used] = translation_stats['methods'].get(method_used, 0) + 1
#         else:
#             translation_stats['failed'] += 1
        
#         # Create new segment with translated text
#         new_segment = segment.copy()
#         new_segment['text'] = translated_text
#         new_segment['original_text'] = original_text
#         new_segment['translation_method'] = method_used
        
#         translated_segments.append(new_segment)
        
#         # Progress logging
#         if (i + 1) % 10 == 0:
#             logger.info(f"Translated {i + 1}/{len(segments)} segments")
    
#     # Log final statistics
#     logger.info(f"Translation complete: {translation_stats['success']} successful, {translation_stats['failed']} failed")
#     logger.info(f"Methods used: {translation_stats['methods']}")
    
#     return translated_segments

def get_available_languages() -> Dict[str, Dict]:
    """
    Get all available languages with their metadata.
    
    Returns:
        Dictionary mapping language names to their metadata
    """
    return LANGUAGE_MAPPINGS.copy()

def get_language_pairs() -> List[Tuple[str, str]]:
    """
    Get all possible language pairs for translation.
    
    Returns:
        List of (source, target) language pairs
    """
    languages = list(LANGUAGE_MAPPINGS.keys())
    pairs = []
    
    for source in languages:
        for target in languages:
            if source != target:
                pairs.append((source, target))
    
    return pairs

def detect_language(text: str) -> Optional[str]:
    """
    Detect the language of given text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Detected language code or None if detection fails
    """
    try:
        translator = get_google_translator()
        if not translator:
            return None
        
        detection = translator.detect(text)
        detected_lang = detection.lang
        
        # Convert Google language code to our internal format
        for lang_name, lang_data in LANGUAGE_MAPPINGS.items():
            if lang_data['google'] == detected_lang:
                return lang_name
        
        return detected_lang
        
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return None

# Cache management functions
def save_translation_cache(filepath: str = None):
    """Save translation cache to disk."""
    if not filepath:
        cache_dir = ensure_cache_dir()
        filepath = cache_dir / 'translation_cache.pkl'
    
    try:
        cache_data = {
            'language_mappings': LANGUAGE_MAPPINGS,
            'timestamp': time.time()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Translation cache saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save translation cache: {e}")

def load_translation_cache(filepath: str = None):
    """Load translation cache from disk."""
    if not filepath:
        cache_dir = ensure_cache_dir()
        filepath = cache_dir / 'translation_cache.pkl'
    
    try:
        if Path(filepath).exists():
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            logger.info(f"Translation cache loaded from {filepath}")
            return cache_data
        else:
            logger.info("No translation cache found")
            return None
    except Exception as e:
        logger.error(f"Failed to load translation cache: {e}")
        return None

# Initialize cache on module import
ensure_cache_dir()