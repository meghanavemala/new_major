"""
Enhanced Keyframe Extraction and OCR Analysis Module

This module provides comprehensive keyframe extraction with OCR capabilities
to extract text content from video frames for better context understanding.
It supports multiple OCR engines and text analysis features.

Author: Video Summarizer Team
Created: 2024
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import json
from pathlib import Path
import time

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
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Text processing
import re
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCR Configuration for different languages
OCR_LANGUAGES = {
    'english': 'eng',
    'hindi': 'hin',
    'bengali': 'ben',
    'telugu': 'tel',
    'marathi': 'mar',
    'tamil': 'tam',
    'gujarati': 'guj',
    'urdu': 'urd',
    'kannada': 'kan',
    'odia': 'ori',
    'malayalam': 'mal',
    'punjabi': 'pan',
    'assamese': 'asm',
    'nepali': 'nep',
    'sanskrit': 'san',
}

# EasyOCR language codes
EASYOCR_LANGUAGES = {
    'english': 'en',
    'hindi': 'hi',
    'bengali': 'bn',
    'tamil': 'ta',
    'korean': 'ko',
    'chinese': 'ch_sim',
    'japanese': 'ja',
    'arabic': 'ar',
}

class KeyframeOCR:
    """
    OCR engine for extracting text from keyframes.
    Supports multiple OCR backends with language-specific optimization.
    """
    
    def __init__(self, languages: List[str] = ['english'], ocr_engine: str = 'auto'):
        """
        Initialize OCR engine.
        
        Args:
            languages: List of languages to detect
            ocr_engine: OCR engine to use ('tesseract', 'easyocr', 'auto')
        """
        self.languages = languages
        self.ocr_engine = ocr_engine
        self.easyocr_reader = None
        
        # Initialize EasyOCR if available and requested
        if EASYOCR_AVAILABLE and ocr_engine in ['easyocr', 'auto']:
            try:
                # Convert language names to EasyOCR codes
                easyocr_langs = []
                for lang in languages:
                    if lang in EASYOCR_LANGUAGES:
                        easyocr_langs.append(EASYOCR_LANGUAGES[lang])
                
                if easyocr_langs:
                    self.easyocr_reader = easyocr.Reader(easyocr_langs)
                    logger.info(f"EasyOCR initialized with languages: {easyocr_langs}")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding for better text detection
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text using Tesseract OCR.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with extracted text and confidence scores
        """
        if not TESSERACT_AVAILABLE:
            return {'text': '', 'confidence': 0, 'method': 'tesseract_unavailable'}
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Build Tesseract language string
            tesseract_langs = []
            for lang in self.languages:
                if lang in OCR_LANGUAGES:
                    tesseract_langs.append(OCR_LANGUAGES[lang])
            
            lang_string = '+'.join(tesseract_langs) if tesseract_langs else 'eng'
            
            # Configure Tesseract
            config = r'--oem 3 --psm 6'
            
            # Extract text
            text = pytesseract.image_to_string(
                processed_image, 
                lang=lang_string, 
                config=config
            ).strip()
            
            # Get confidence scores
            data = pytesseract.image_to_data(processed_image, lang=lang_string, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': text,
                'confidence': avg_confidence,
                'method': 'tesseract',
                'word_count': len(text.split()) if text else 0
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {'text': '', 'confidence': 0, 'method': 'tesseract_error'}
    
    def extract_text_easyocr(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text using EasyOCR.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with extracted text and confidence scores
        """
        if not self.easyocr_reader:
            return {'text': '', 'confidence': 0, 'method': 'easyocr_unavailable'}
        
        try:
            # EasyOCR works better with original image
            results = self.easyocr_reader.readtext(image)
            
            # Combine all text and calculate average confidence
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low-confidence detections
                    text_parts.append(text)
                    confidences.append(confidence)
            
            combined_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': combined_text,
                'confidence': avg_confidence * 100,  # Convert to percentage
                'method': 'easyocr',
                'word_count': len(combined_text.split()) if combined_text else 0,
                'detections': len(results)
            }
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {'text': '', 'confidence': 0, 'method': 'easyocr_error'}
    
    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text using the best available OCR method.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with extracted text and metadata
        """
        results = []
        
        # Try different OCR methods
        if self.ocr_engine in ['tesseract', 'auto'] and TESSERACT_AVAILABLE:
            tesseract_result = self.extract_text_tesseract(image)
            results.append(tesseract_result)
        
        if self.ocr_engine in ['easyocr', 'auto'] and self.easyocr_reader:
            easyocr_result = self.extract_text_easyocr(image)
            results.append(easyocr_result)
        
        if not results:
            return {'text': '', 'confidence': 0, 'method': 'no_ocr_available'}
        
        # Select best result based on confidence and text quality
        best_result = max(results, key=lambda x: x['confidence'])
        
        # Add combined results
        all_text = ' '.join([r['text'] for r in results if r['text']])
        best_result['all_methods_text'] = all_text
        best_result['methods_tried'] = len(results)
        
        return best_result

def _are_frames_similar(frame1, frame2, threshold=0.8):
    """
    Check if two frames are similar using histogram comparison.
    
    Args:
        frame1: First frame
        frame2: Second frame
        threshold: Similarity threshold (0-1)
        
    Returns:
        True if frames are similar, False otherwise
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Compare histograms
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return correlation > threshold

def extract_keyframes(
    video_path: str, 
    processed_dir: str, 
    video_id: str,
    target_resolution: str = '480p',
    frame_interval: int = 60,
    similarity_threshold: float = 0.8,
    ocr_languages: List[str] = ['english'],
    enable_ocr: bool = True
) -> Optional[str]:
    """
    Extract keyframes from video with OCR text analysis.
    
    Args:
        video_path: Path to input video file
        processed_dir: Directory to save keyframes
        video_id: Unique identifier for the video
        target_resolution: Target resolution for keyframes ('480p', '720p', '1080p')
        frame_interval: Extract one frame every N frames
        similarity_threshold: Threshold for frame similarity (0-1)
        ocr_languages: Languages for OCR text extraction
        enable_ocr: Whether to perform OCR analysis on keyframes
        
    Returns:
        Path to keyframes directory or None if failed
    """
    try:
        logger.info(f"Extracting keyframes from {video_path}")
        
        # Create output directory
        keyframes_dir = os.path.join(processed_dir, f"{video_id}_keyframes")
        os.makedirs(keyframes_dir, exist_ok=True)
        
        # Initialize OCR if enabled
        ocr_engine = None
        if enable_ocr:
            ocr_engine = KeyframeOCR(languages=ocr_languages)
            logger.info(f"OCR enabled for languages: {ocr_languages}")
        
        # Resolution mapping
        resolution_map = {
            '480p': (854, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080)
        }
        target_width, target_height = resolution_map.get(target_resolution, (854, 480))
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video FPS: {fps}, Total frames: {total_frames}")
        
        keyframes = []
        keyframe_metadata = []
        prev_frame = None
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:  # Extract every Nth frame
                if prev_frame is None or not _are_frames_similar(frame, prev_frame, similarity_threshold):
                    # Resize frame to target resolution
                    resized_frame = cv2.resize(frame, (target_width, target_height))
                    
                    filename = f"keyframe_{saved_count:06d}.jpg"
                    filepath = os.path.join(keyframes_dir, filename)
                    cv2.imwrite(filepath, resized_frame)
                    
                    # Perform OCR if enabled
                    ocr_data = {}
                    if enable_ocr and ocr_engine:
                        ocr_result = ocr_engine.extract_text(resized_frame)
                        ocr_data = {
                            'text': ocr_result.get('text', ''),
                            'confidence': ocr_result.get('confidence', 0),
                            'method': ocr_result.get('method', 'none'),
                            'word_count': ocr_result.get('word_count', 0)
                        }
                    
                    # Store metadata
                    metadata = {
                        'filename': filename,
                        'filepath': filepath,
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'resolution': f"{target_width}x{target_height}",
                        'ocr_data': ocr_data
                    }
                    
                    keyframes.append(filepath)
                    keyframe_metadata.append(metadata)
                    prev_frame = resized_frame.copy()
                    saved_count += 1
            
            frame_count += 1
            
            # Progress logging
            if frame_count % 1000 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames, saved {saved_count} keyframes")
        
        cap.release()
        
        # Save metadata
        metadata_file = os.path.join(keyframes_dir, 'keyframes_metadata.json')
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'video_id': video_id,
                    'total_keyframes': len(keyframes),
                    'target_resolution': target_resolution,
                    'frame_interval': frame_interval,
                    'similarity_threshold': similarity_threshold,
                    'ocr_enabled': enable_ocr,
                    'ocr_languages': ocr_languages,
                    'keyframes': keyframe_metadata
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Keyframe metadata saved to {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save keyframe metadata: {e}")
        
        logger.info(f"Extracted {len(keyframes)} keyframes to {keyframes_dir}")
        
        # Extract and summarize OCR text
        if enable_ocr:
            all_ocr_text = []
            high_confidence_text = []
            
            for metadata in keyframe_metadata:
                ocr_data = metadata.get('ocr_data', {})
                text = ocr_data.get('text', '')
                confidence = ocr_data.get('confidence', 0)
                
                if text:
                    all_ocr_text.append(text)
                    if confidence > 70:  # High confidence threshold
                        high_confidence_text.append(text)
            
            # Save OCR summary
            ocr_summary = {
                'total_text_frames': len([t for t in all_ocr_text if t]),
                'high_confidence_frames': len(high_confidence_text),
                'all_text': ' '.join(all_ocr_text),
                'high_confidence_text': ' '.join(high_confidence_text),
                'word_frequency': dict(Counter(' '.join(all_ocr_text).split()).most_common(20))
            }
            
            ocr_summary_file = os.path.join(keyframes_dir, 'ocr_summary.json')
            try:
                with open(ocr_summary_file, 'w', encoding='utf-8') as f:
                    json.dump(ocr_summary, f, indent=2, ensure_ascii=False)
                logger.info(f"OCR summary saved to {ocr_summary_file}")
            except Exception as e:
                logger.error(f"Failed to save OCR summary: {e}")
        
        return keyframes_dir
        
    except Exception as e:
        logger.error(f"Error extracting keyframes: {e}")
        return None

def get_keyframe_text_summary(keyframes_dir: str) -> Dict[str, Any]:
    """
    Get OCR text summary for keyframes.
    
    Args:
        keyframes_dir: Directory containing keyframes
        
    Returns:
        Dictionary with OCR text summary
    """
    ocr_summary_file = os.path.join(keyframes_dir, 'ocr_summary.json')
    
    if os.path.exists(ocr_summary_file):
        try:
            with open(ocr_summary_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load OCR summary: {e}")
    
    return {
        'total_text_frames': 0,
        'high_confidence_frames': 0,
        'all_text': '',
        'high_confidence_text': '',
        'word_frequency': {}
    }