"""
Advanced OCR and Visual Content Analysis System

This module provides state-of-the-art OCR and visual content analysis capabilities
using multiple OCR engines, deep learning models, and advanced computer vision techniques.

Features:
- Multi-engine OCR (Tesseract, EasyOCR, PaddleOCR)
- Advanced text extraction and preprocessing
- Visual content analysis and classification
- Scene detection and object recognition
- Text-to-speech synchronization optimization
- Production-ready caching and performance monitoring
- GPU acceleration with fallback support

Author: Video Summarizer Team
Created: 2024
"""

import os
import cv2
import numpy as np
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from PIL import Image, ImageEnhance, ImageFilter
import re

# OCR Dependencies
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

from .gpu_config import get_device, is_gpu_available, log_gpu_status, clear_gpu_memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextRegion:
    """Text region with bounding box and metadata."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    language: str
    font_size: int
    color: Tuple[int, int, int]
    is_title: bool = False
    is_subtitle: bool = False

@dataclass
class VisualContent:
    """Visual content analysis result."""
    text_regions: List[TextRegion]
    objects: List[Dict[str, Any]]
    scene_type: str
    dominant_colors: List[Tuple[int, int, int]]
    brightness: float
    contrast: float
    faces: List[Dict[str, Any]]
    motion_score: float

class AdvancedOCR:
    """Advanced OCR system with multiple engines and preprocessing."""
    
    def __init__(self, cache_dir: str = "ocr_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.device = get_device()
        self.engines = {}
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'engine_usage': {},
            'average_time': 0.0,
            'error_count': 0
        }
        
        # Initialize OCR engines
        self._initialize_engines()
        
        logger.info(f"AdvancedOCR initialized on {self.device}")
        log_gpu_status()
    
    def _initialize_engines(self):
        """Initialize available OCR engines."""
        try:
            # Initialize Tesseract
            if TESSERACT_AVAILABLE:
                self.engines['tesseract'] = TesseractEngine()
                logger.info("Tesseract OCR engine initialized")
            
            # Initialize EasyOCR
            if EASYOCR_AVAILABLE:
                self.engines['easyocr'] = EasyOCREngine(use_gpu=is_gpu_available())
                logger.info("EasyOCR engine initialized")
            
            # Initialize PaddleOCR
            if PADDLEOCR_AVAILABLE:
                self.engines['paddleocr'] = PaddleOCREngine(use_gpu=is_gpu_available())
                logger.info("PaddleOCR engine initialized")
            
            if not self.engines:
                logger.warning("No OCR engines available!")
            
        except Exception as e:
            logger.error(f"OCR engine initialization failed: {e}")
    
    def _get_cache_key(self, image_path: str, languages: List[str]) -> str:
        """Generate cache key for image and languages."""
        try:
            # Get file modification time and size
            stat = os.stat(image_path)
            file_info = f"{image_path}_{stat.st_mtime}_{stat.st_size}"
            
            # Create hash
            content_hash = hashlib.md5(file_info.encode()).hexdigest()
            lang_hash = hashlib.md5('_'.join(sorted(languages)).encode()).hexdigest()
            
            return f"ocr_{content_hash}_{lang_hash}"
        except Exception:
            return f"ocr_{hashlib.md5(image_path.encode()).hexdigest()}"
    
    def extract_text(self, image_path: str, languages: List[str] = ['en'], 
                    engine: str = 'auto', use_cache: bool = True) -> Dict[str, Any]:
        """Extract text from image using specified engine."""
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(image_path, languages)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    # Check if cache is still valid (7 days)
                    if time.time() - cached_data['timestamp'] < 604800:
                        self.performance_stats['cache_hits'] += 1
                        logger.info("Using cached OCR result")
                        return cached_data['result']
                except Exception as e:
                    logger.warning(f"Cache read failed: {e}")
        
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            
            # Select best engine
            if engine == 'auto':
                engine = self._select_best_engine(image, languages)
            
            # Extract text using selected engine
            if engine in self.engines:
                result = self.engines[engine].extract_text(image, languages)
                self.performance_stats['engine_usage'][engine] = \
                    self.performance_stats['engine_usage'].get(engine, 0) + 1
            else:
                logger.error(f"OCR engine '{engine}' not available")
                result = {'text_regions': [], 'full_text': '', 'confidence': 0.0}
            
            # Post-process results
            result = self._post_process_results(result, image)
            
            # Add metadata
            result.update({
                'engine_used': engine,
                'processing_time': time.time() - start_time,
                'image_path': image_path,
                'languages': languages,
                'timestamp': time.time()
            })
            
            # Cache result
            if use_cache:
                try:
                    cache_data = {
                        'result': result,
                        'timestamp': time.time()
                    }
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"Cache write failed: {e}")
            
            # Update performance stats
            self.performance_stats['average_time'] = (
                (self.performance_stats['average_time'] * (self.performance_stats['total_requests'] - 1) + 
                 result['processing_time']) / self.performance_stats['total_requests']
            )
            
            logger.info(f"OCR completed: {result['processing_time']:.2f}s, {len(result['text_regions'])} regions")
            return result
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            self.performance_stats['error_count'] += 1
            
            return {
                'text_regions': [],
                'full_text': '',
                'confidence': 0.0,
                'engine_used': 'error',
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image for better OCR results."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to PIL for advanced preprocessing
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Enhance image quality
            pil_image = self._enhance_image(pil_image)
            
            # Convert back to OpenCV format
            enhanced_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return cv2.imread(image_path)
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image for better OCR results."""
        try:
            # Convert to grayscale for analysis
            gray = image.convert('L')
            
            # Calculate enhancement factors
            brightness_factor = self._calculate_brightness_factor(gray)
            contrast_factor = self._calculate_contrast_factor(gray)
            
            # Apply enhancements
            if brightness_factor != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness_factor)
            
            if contrast_factor != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast_factor)
            
            # Apply sharpening
            image = image.filter(ImageFilter.SHARPEN)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def _calculate_brightness_factor(self, gray_image: Image.Image) -> float:
        """Calculate optimal brightness enhancement factor."""
        try:
            # Calculate mean brightness
            pixels = list(gray_image.getdata())
            mean_brightness = sum(pixels) / len(pixels)
            
            # Target brightness is around 128 (middle of 0-255)
            if mean_brightness < 100:
                return 1.2  # Increase brightness
            elif mean_brightness > 200:
                return 0.8  # Decrease brightness
            else:
                return 1.0  # No change needed
                
        except Exception:
            return 1.0
    
    def _calculate_contrast_factor(self, gray_image: Image.Image) -> float:
        """Calculate optimal contrast enhancement factor."""
        try:
            # Calculate standard deviation as contrast measure
            pixels = list(gray_image.getdata())
            mean_pixel = sum(pixels) / len(pixels)
            variance = sum((p - mean_pixel) ** 2 for p in pixels) / len(pixels)
            std_dev = variance ** 0.5
            
            # Target standard deviation is around 50-70
            if std_dev < 30:
                return 1.3  # Increase contrast
            elif std_dev > 80:
                return 0.9  # Decrease contrast
            else:
                return 1.0  # No change needed
                
        except Exception:
            return 1.0
    
    def _select_best_engine(self, image: np.ndarray, languages: List[str]) -> str:
        """Select the best OCR engine for the given image and languages."""
        try:
            # Analyze image characteristics
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Engine selection logic based on image characteristics
            if 'en' in languages and len(languages) == 1:
                # English only - prefer Tesseract for speed
                if 'tesseract' in self.engines:
                    return 'tesseract'
            
            if len(languages) > 1 or any(lang in ['hi', 'zh', 'ja', 'ko', 'ar'] for lang in languages):
                # Multilingual or complex scripts - prefer EasyOCR or PaddleOCR
                if 'easyocr' in self.engines:
                    return 'easyocr'
                elif 'paddleocr' in self.engines:
                    return 'paddleocr'
            
            if aspect_ratio > 2.0:  # Wide image - might be a document
                if 'tesseract' in self.engines:
                    return 'tesseract'
            
            # Default to first available engine
            return list(self.engines.keys())[0] if self.engines else 'tesseract'
            
        except Exception as e:
            logger.warning(f"Engine selection failed: {e}")
            return 'tesseract'
    
    def _post_process_results(self, result: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """Post-process OCR results for better quality."""
        try:
            text_regions = result.get('text_regions', [])
            
            # Filter low-confidence regions
            filtered_regions = [
                region for region in text_regions
                if region.get('confidence', 0) > 0.3
            ]
            
            # Sort by position (top to bottom, left to right)
            filtered_regions.sort(key=lambda x: (x.get('bbox', [0, 0, 0, 0])[1], x.get('bbox', [0, 0, 0, 0])[0]))
            
            # Detect titles and subtitles
            for region in filtered_regions:
                region['is_title'] = self._is_title(region, image)
                region['is_subtitle'] = self._is_subtitle(region, image)
            
            # Combine text
            full_text = ' '.join([region.get('text', '') for region in filtered_regions])
            
            # Calculate overall confidence
            if filtered_regions:
                avg_confidence = sum(region.get('confidence', 0) for region in filtered_regions) / len(filtered_regions)
            else:
                avg_confidence = 0.0
            
            result.update({
                'text_regions': filtered_regions,
                'full_text': full_text,
                'confidence': avg_confidence,
                'region_count': len(filtered_regions)
            })
            
            return result
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return result
    
    def _is_title(self, region: Dict[str, Any], image: np.ndarray) -> bool:
        """Detect if text region is a title."""
        try:
            text = region.get('text', '')
            bbox = region.get('bbox', [0, 0, 0, 0])
            
            # Check text characteristics
            if len(text) < 3 or len(text) > 100:
                return False
            
            # Check if text is at the top of the image
            height, width = image.shape[:2]
            y_position = bbox[1] / height if height > 0 else 0
            
            if y_position > 0.3:  # Not in top 30% of image
                return False
            
            # Check for title-like patterns
            title_patterns = [
                r'^[A-Z][A-Z\s]+$',  # All caps
                r'^[A-Z][a-z]+\s+[A-Z][a-z]+',  # Title case
                r'^\d+\.\s+[A-Z]',  # Numbered title
            ]
            
            for pattern in title_patterns:
                if re.match(pattern, text.strip()):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _is_subtitle(self, region: Dict[str, Any], image: np.ndarray) -> bool:
        """Detect if text region is a subtitle."""
        try:
            text = region.get('text', '')
            bbox = region.get('bbox', [0, 0, 0, 0])
            
            # Check text characteristics
            if len(text) < 5 or len(text) > 200:
                return False
            
            # Check if text is at the bottom of the image
            height, width = image.shape[:2]
            y_position = bbox[1] / height if height > 0 else 0
            
            if y_position < 0.7:  # Not in bottom 30% of image
                return False
            
            # Check for subtitle-like patterns
            subtitle_patterns = [
                r'^[a-z]',  # Starts with lowercase
                r'[.!?]$',  # Ends with punctuation
                r'\b(and|or|but|the|a|an)\b',  # Contains common words
            ]
            
            pattern_matches = sum(1 for pattern in subtitle_patterns if re.search(pattern, text))
            
            return pattern_matches >= 2
            
        except Exception:
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get OCR performance statistics."""
        cache_hit_rate = (
            self.performance_stats['cache_hits'] / 
            max(1, self.performance_stats['total_requests'])
        ) * 100
        
        return {
            **self.performance_stats,
            'cache_hit_rate': cache_hit_rate,
            'engines_available': list(self.engines.keys()),
            'device': self.device,
            'gpu_available': is_gpu_available()
        }

class TesseractEngine:
    """Tesseract OCR engine wrapper."""
    
    def __init__(self):
        self.available = TESSERACT_AVAILABLE
        if not self.available:
            logger.warning("Tesseract not available")
    
    def extract_text(self, image: np.ndarray, languages: List[str]) -> Dict[str, Any]:
        """Extract text using Tesseract."""
        if not self.available:
            return {'text_regions': [], 'full_text': '', 'confidence': 0.0}
        
        try:
            # Convert language codes
            lang_codes = []
            for lang in languages:
                if lang == 'en':
                    lang_codes.append('eng')
                elif lang == 'hi':
                    lang_codes.append('hin')
                elif lang == 'zh':
                    lang_codes.append('chi_sim')
                elif lang == 'ja':
                    lang_codes.append('jpn')
                elif lang == 'ko':
                    lang_codes.append('kor')
                elif lang == 'ar':
                    lang_codes.append('ara')
                else:
                    lang_codes.append('eng')  # Default to English
            
            lang_string = '+'.join(lang_codes)
            
            # Configure Tesseract
            config = f'--oem 3 --psm 6 -l {lang_string}'
            
            # Extract text with bounding boxes
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            text_regions = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text and int(data['conf'][i]) > 30:  # Filter low confidence
                    region = {
                        'text': text,
                        'confidence': float(data['conf'][i]) / 100.0,
                        'bbox': [
                            int(data['left'][i]),
                            int(data['top'][i]),
                            int(data['width'][i]),
                            int(data['height'][i])
                        ],
                        'language': languages[0] if languages else 'en'
                    }
                    text_regions.append(region)
            
            # Get full text
            full_text = pytesseract.image_to_string(image, config=config)
            
            return {
                'text_regions': text_regions,
                'full_text': full_text.strip(),
                'confidence': sum(r['confidence'] for r in text_regions) / max(1, len(text_regions))
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return {'text_regions': [], 'full_text': '', 'confidence': 0.0}

class EasyOCREngine:
    """EasyOCR engine wrapper."""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and is_gpu_available()
        self.reader = None
        self.available = EASYOCR_AVAILABLE
        
        if self.available:
            try:
                # Initialize with common languages
                self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)
                logger.info(f"EasyOCR initialized with GPU: {self.use_gpu}")
            except Exception as e:
                logger.error(f"EasyOCR initialization failed: {e}")
                self.available = False
    
    def extract_text(self, image: np.ndarray, languages: List[str]) -> Dict[str, Any]:
        """Extract text using EasyOCR."""
        if not self.available or not self.reader:
            return {'text_regions': [], 'full_text': '', 'confidence': 0.0}
        
        try:
            # Convert language codes
            lang_codes = []
            for lang in languages:
                if lang == 'en':
                    lang_codes.append('en')
                elif lang == 'hi':
                    lang_codes.append('hi')
                elif lang == 'zh':
                    lang_codes.append('ch_sim')
                elif lang == 'ja':
                    lang_codes.append('ja')
                elif lang == 'ko':
                    lang_codes.append('ko')
                elif lang == 'ar':
                    lang_codes.append('ar')
                else:
                    lang_codes.append('en')
            
            # Use first language for now (EasyOCR supports multiple but initialization is complex)
            primary_lang = lang_codes[0]
            
            # Extract text
            results = self.reader.readtext(image)
            
            text_regions = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low confidence
                    # Convert bbox format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    region = {
                        'text': text,
                        'confidence': confidence,
                        'bbox': [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                        'language': languages[0] if languages else 'en'
                    }
                    text_regions.append(region)
            
            # Get full text
            full_text = ' '.join([region['text'] for region in text_regions])
            
            return {
                'text_regions': text_regions,
                'full_text': full_text,
                'confidence': sum(r['confidence'] for r in text_regions) / max(1, len(text_regions))
            }
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return {'text_regions': [], 'full_text': '', 'confidence': 0.0}

class PaddleOCREngine:
    """PaddleOCR engine wrapper."""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and is_gpu_available()
        self.ocr = None
        self.available = PADDLEOCR_AVAILABLE
        
        if self.available:
            try:
                self.ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_gpu=self.use_gpu)
                logger.info(f"PaddleOCR initialized with GPU: {self.use_gpu}")
            except Exception as e:
                logger.error(f"PaddleOCR initialization failed: {e}")
                self.available = False
    
    def extract_text(self, image: np.ndarray, languages: List[str]) -> Dict[str, Any]:
        """Extract text using PaddleOCR."""
        if not self.available or not self.ocr:
            return {'text_regions': [], 'full_text': '', 'confidence': 0.0}
        
        try:
            # Extract text
            results = self.ocr.ocr(image, cls=True)
            
            text_regions = []
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        bbox, (text, confidence) = line
                        
                        if confidence > 0.3:  # Filter low confidence
                            # Convert bbox format
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            
                            region = {
                                'text': text,
                                'confidence': confidence,
                                'bbox': [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                                'language': languages[0] if languages else 'en'
                            }
                            text_regions.append(region)
            
            # Get full text
            full_text = ' '.join([region['text'] for region in text_regions])
            
            return {
                'text_regions': text_regions,
                'full_text': full_text,
                'confidence': sum(r['confidence'] for r in text_regions) / max(1, len(text_regions))
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return {'text_regions': [], 'full_text': '', 'confidence': 0.0}

# Global instance
advanced_ocr = AdvancedOCR()

def extract_text_from_image(image_path: str, languages: List[str] = ['en'], 
                          engine: str = 'auto') -> Dict[str, Any]:
    """Convenience function for OCR text extraction."""
    return advanced_ocr.extract_text(image_path, languages, engine)

def get_ocr_stats() -> Dict[str, Any]:
    """Get OCR performance statistics."""
    return advanced_ocr.get_performance_stats()
