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
from .gpu_config import get_device, is_gpu_available, optimize_for_model, clear_gpu_memory, log_gpu_status

# Configure logging
logging.basicConfig(level=logging.WARNING)
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
            
            # Get optimized device and settings
            device = get_device()
            optimizations = optimize_for_model("m2m100")
            
            logger.info(f"Using device: {device}")
            log_gpu_status()
            
            # Load tokenizer and model
            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
            
            translation_cache['m2m100_tokenizer'] = tokenizer
            translation_cache['m2m100_model'] = model
            
            logger.info("M2M100 model loaded successfully on GPU" if device == 'cuda' else "M2M100 model loaded successfully on CPU")
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
        
        # Get optimization settings
        optimizations = optimize_for_model("m2m100")
        
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
        
        # Generate translation with optimized settings
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id(tgt_code),
                max_length=optimizations.get('max_length', 512),
                num_beams=optimizations.get('num_beams', 5),
                early_stopping=optimizations.get('early_stopping', True)
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
    max_workers: int = 5,   # number of parallel threads
    fallback_to_google: bool = True  # Whether to fall back to Google Translate if default fails
) -> List[Dict]:
    """
    Translate text segments from source to target language with multithreading.
    Supports bidirectional translation between all specified languages.
    Defaults to M2M100 for offline use with Google Translate as fallback.

    Args:
        segments: List of segment dictionaries with 'text' field
        source_lang: Source language code or name (e.g., 'hi', 'hindi')
        target_lang: Target language code or name (e.g., 'en', 'english')
        method: Translation method to use ('auto', 'google', 'm2m100', 'marian')
        max_workers: Number of threads for parallel translation
        fallback_to_google: Whether to fall back to Google Translate if default fails

    Returns:
        List of segments with translated text and metadata
    """
    if not segments:
        return []

    # Normalize language codes to ISO 639-1
    def normalize_lang(lang: str) -> str:
        """Convert language name to ISO code if needed."""
        if not lang:
            return 'en'  # Default to English
        
        # Check if it's already an ISO code
        if len(lang) == 2 and any(lang.lower() == lc for lc in [v['code'] for v in LANGUAGE_MAPPINGS.values()]):
            return lang.lower()
            
        # Try to find matching language
        lang_lower = lang.lower()
        for name, data in LANGUAGE_MAPPINGS.items():
            if (lang_lower == name.lower() or 
                lang_lower == data['code'].lower() or 
                lang_lower == data.get('google', '').lower() or
                lang_lower == data.get('native', '').lower()):
                return data['code']
        
        logger.warning(f"Unrecognized language: {lang}, defaulting to English")
        return 'en'

    source_lang = normalize_lang(source_lang)
    target_lang = normalize_lang(target_lang)

    # If source and target languages are the same, return as is
    if source_lang == target_lang:
        return segments

    # Method resolution - prefer M2M100 as default offline option
    if method == 'auto':
        methods = ['m2m100']  # Default to M2M100 first
        if fallback_to_google and GOOGLETRANS_AVAILABLE:
            methods.append('google_translate')
        methods.append('marian')  # Fallback to Marian if available
    else:
        methods = [method]
        if fallback_to_google and 'google_translate' not in methods and GOOGLETRANS_AVAILABLE:
            methods.append('google_translate')

    # Prepare segments for translation
    segments_to_translate = []
    for seg in segments:
        text = seg.get('text', '').strip()
        if text:
            segments_to_translate.append({
                **seg,
                '_original_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'translation_method': None,
                'translation_error': None
            })

    # Skip if no segments to translate
    if not segments_to_translate:
        return segments

    # Try each method until one succeeds
    translated_segments = None
    used_method = None
    
    for method_try in methods:
        try:
            logger.info(f"Attempting translation with {method_try} ({source_lang}->{target_lang})")
            
            if method_try == 'google_translate' and GOOGLETRANS_AVAILABLE:
                translated = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            translate_with_google, 
                            seg['_original_text'], 
                            source_lang, 
                            target_lang
                        )
                        for seg in segments_to_translate
                    ]
                    
                    for future, seg in zip(as_completed(futures), segments_to_translate):
                        translated_text = future.result()
                        if translated_text:
                            translated.append({
                                **seg,
                                'text': translated_text,
                                'translation_method': 'google_translate',
                                'translation_error': None
                            })
                        else:
                            translated.append({
                                **seg,
                                'translation_error': 'Google Translate failed',
                                'translation_method': None
                            })
                
                # Only consider successful if we got at least one translation
                if any(seg.get('translation_method') == 'google_translate' for seg in translated):
                    translated_segments = translated
                    used_method = 'google_translate'
                    break

            elif method_try == 'm2m100':
                translated = []
                model, tokenizer = get_m2m100_model()
                if not model or not tokenizer:
                    logger.warning("M2M100 model not available")
                    continue
                
                # Get language codes for M2M100
                src_code = LANGUAGE_MAPPINGS.get(source_lang, {}).get('code', source_lang)
                tgt_code = LANGUAGE_MAPPINGS.get(target_lang, {}).get('code', target_lang)
                
                # Process in batches for efficiency
                batch_size = min(8, max(1, max_workers))
                for i in range(0, len(segments_to_translate), batch_size):
                    batch = segments_to_translate[i:i+batch_size]
                    texts = [seg['_original_text'] for seg in batch]
                    
                    try:
                        # Set source language
                        tokenizer.src_lang = src_code
                        
                        # Tokenize input
                        inputs = tokenizer(
                            texts, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=512
                        )
                        
                        # Move to same device as model
                        device = next(model.parameters()).device
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Get optimization settings
                        optimizations = optimize_for_model("m2m100")
                        
                        # Generate translation with optimized settings
                        with torch.no_grad():
                            generated_tokens = model.generate(
                                **inputs,
                                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
                                max_length=optimizations.get('max_length', 512),
                                num_beams=optimizations.get('num_beams', 5),
                                early_stopping=optimizations.get('early_stopping', True)
                            )
                        
                        # Decode the generated tokens
                        translated_texts = tokenizer.batch_decode(
                            generated_tokens, 
                            skip_special_tokens=True
                        )
                        
                        # Update segments with translations
                        for seg, trans_text in zip(batch, translated_texts):
                            translated.append({
                                **seg,
                                'text': trans_text,
                                'translation_method': 'm2m100',
                                'translation_error': None
                            })
                            
                    except Exception as e:
                        logger.error(f"M2M100 batch translation failed: {e}")
                        # Add untranslated segments with error
                        translated.extend([
                            {**seg, 'translation_error': f'M2M100: {str(e)}'}
                            for seg in batch
                        ])
                
                # Only consider successful if we got translations
                if any(seg.get('translation_method') == 'm2m100' for seg in translated):
                    translated_segments = translated
                    used_method = 'm2m100'
                    break

        except Exception as e:
            logger.error(f"Translation with {method_try} failed: {e}", exc_info=True)
            continue

    # If all methods failed, return original segments with error flags
    if translated_segments is None:
        logger.error(f"All translation methods failed for {source_lang}->{target_lang}")
        return [
            {
                **seg, 
                'translation_error': 'All translation methods failed',
                'translation_method': None
            }
            for seg in segments
        ]

    # Log successful translation
    success_count = sum(1 for seg in translated_segments if seg.get('translation_method'))
    if used_method and success_count > 0:
        logger.info(
            f"Successfully translated {success_count}/{len(segments_to_translate)} "
            f"segments using {used_method} ({source_lang}->{target_lang})"
        )
    
    # Clear GPU memory after translation
    clear_gpu_memory()
    
    return translated_segments
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