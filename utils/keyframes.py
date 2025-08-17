"""
Enhanced Keyframe Extraction and OCR Analysis Module

This module now extracts more keyframes (lower frame_interval, higher max_keyframes) and stores timestamps for each keyframe,
enabling per-topic keyframe selection for summary video generation.
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Dict, Any
import logging
import json
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    from PIL import Image  # noqa: F401  (kept for potential future use)
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

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

        # Denoising
        denoised = cv2.fastNlMeansDenoising(gray)

        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def extract_text_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text using Tesseract OCR.
        """
        if not TESSERACT_AVAILABLE:
            return {'text': '', 'confidence': 0, 'method': 'tesseract_unavailable'}

        try:
            processed_image = self.preprocess_image(image)

            # Build Tesseract language string
            tesseract_langs = []
            for lang in self.languages:
                if lang in OCR_LANGUAGES:
                    tesseract_langs.append(OCR_LANGUAGES[lang])

            lang_string = '+'.join(tesseract_langs) if tesseract_langs else 'eng'

            # Tesseract configuration
            config = r'--oem 3 --psm 6'

            # Extract text
            text = pytesseract.image_to_string(
                processed_image,
                lang=lang_string,
                config=config
            ).strip()

            # Confidence (from per-word data)
            data = pytesseract.image_to_data(
                processed_image, lang=lang_string, output_type=pytesseract.Output.DICT
            )
            confidences = []
            for conf in data.get('conf', []):
                try:
                    c = int(conf)
                    if c > 0:
                        confidences.append(c)
                except Exception:
                    continue
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
        """
        if not self.easyocr_reader:
            return {'text': '', 'confidence': 0, 'method': 'easyocr_unavailable'}

        try:
            # EasyOCR works better with the original image
            results = self.easyocr_reader.readtext(image)

            text_parts = []
            confidences = []

            for (_bbox, text, confidence) in results:
                if confidence > 0.5:  # filter low-confidence detections
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
        """
        results = []

        if self.ocr_engine in ['tesseract', 'auto'] and TESSERACT_AVAILABLE:
            results.append(self.extract_text_tesseract(image))

        if self.ocr_engine in ['easyocr', 'auto'] and self.easyocr_reader:
            results.append(self.extract_text_easyocr(image))

        if not results:
            return {'text': '', 'confidence': 0, 'method': 'no_ocr_available'}

        best_result = max(results, key=lambda x: x['confidence'])
        all_text = ' '.join([r['text'] for r in results if r.get('text')])
        best_result['all_methods_text'] = all_text
        best_result['methods_tried'] = len(results)
        return best_result


def _are_frames_similar(frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.8) -> bool:
    """
    Check if two frames are similar using histogram correlation (0..1).
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return corr > threshold


def extract_keyframes(
    video_path: str,
    processed_dir: str,
    video_id: str,
    target_resolution: str = '480p',
    frame_interval: int = 30,          # ~1 per second at 30fps
    similarity_threshold: float = 0.4,  # lower => catch more changes
    ocr_languages: List[str] = ['english'],
    enable_ocr: bool = True,
    max_keyframes: int = 100
) -> Optional[str]:
    """
    Extract keyframes from video with OCR text analysis.
    Returns the directory path containing keyframes and metadata, or None on failure.
    """
    keyframes_dir = None
    cap = None

    try:
        logger.info(f"Extracting keyframes from {video_path}")
        keyframes_dir = os.path.join(processed_dir, f"{video_id}_keyframes")
        os.makedirs(keyframes_dir, exist_ok=True)

        ocr_engine = KeyframeOCR(languages=ocr_languages) if enable_ocr else None
        if enable_ocr:
            logger.info(f"OCR enabled for languages: {ocr_languages}")

        resolution_map = {
            '480p': (854, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080),
        }
        target_width, target_height = resolution_map.get(target_resolution, (854, 480))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            logger.warning("FPS reported as 0; timestamps will use frame count only.")
        logger.info(f"Video FPS: {fps}, Total frames: {total_frames}")

        keyframes: List[str] = []
        keyframe_metadata: List[Dict[str, Any]] = []
        prev_frame = None
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                if saved_count >= max_keyframes:
                    logger.info(f"Reached maximum keyframe limit ({max_keyframes}), stopping extraction")
                    break

                if prev_frame is None or not _are_frames_similar(frame, prev_frame, similarity_threshold):
                    resized_frame = cv2.resize(frame, (target_width, target_height))
                    filename = f"keyframe_{saved_count:06d}.jpg"
                    filepath = os.path.join(keyframes_dir, filename)
                    cv2.imwrite(filepath, resized_frame)

                    ocr_data: Dict[str, Any] = {}
                    if enable_ocr and ocr_engine:
                        ocr_result = ocr_engine.extract_text(resized_frame)
                        ocr_data = {
                            'text': ocr_result.get('text', ''),
                            'confidence': ocr_result.get('confidence', 0),
                            'method': ocr_result.get('method', 'none'),
                            'word_count': ocr_result.get('word_count', 0)
                        }

                    metadata = {
                        'filename': filename,
                        'filepath': filepath,
                        'frame_number': frame_count,
                        'timestamp': (frame_count / fps) if fps > 0 else None,
                        'resolution': f"{target_width}x{target_height}",
                        'ocr_data': ocr_data
                    }

                    keyframes.append(filepath)
                    keyframe_metadata.append(metadata)
                    prev_frame = resized_frame.copy()
                    saved_count += 1

            frame_count += 1

            if total_frames > 0 and frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(
                    f"Progress: {progress:.1f}% - Processed {frame_count}/{total_frames} frames, "
                    f"saved {saved_count} keyframes"
                )
                # Early stop if we hit the configured cap
                if saved_count >= max_keyframes:
                    logger.info("Reached maximum keyframe limit, stopping extraction")
                    break

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
            all_ocr_text: List[str] = []
            high_confidence_text: List[str] = []
            for md in keyframe_metadata:
                ocr_data = md.get('ocr_data', {})
                text = ocr_data.get('text', '')
                confidence = float(ocr_data.get('confidence', 0) or 0)
                if text:
                    all_ocr_text.append(text)
                    if confidence > 70:
                        high_confidence_text.append(text)

            joined_text = ' '.join(all_ocr_text)
            ocr_summary = {
                'total_text_frames': sum(1 for t in all_ocr_text if t),
                'high_confidence_frames': len(high_confidence_text),
                'all_text': joined_text,
                'high_confidence_text': ' '.join(high_confidence_text),
                'word_frequency': dict(Counter(joined_text.split()).most_common(20))
            }

            ocr_summary_file = os.path.join(keyframes_dir, 'ocr_summary.json')
            try:
                with open(ocr_summary_file, 'w', encoding='utf-8') as f:
                    json.dump(ocr_summary, f, indent=2, ensure_ascii=False)
                logger.info(f"OCR summary saved to {ocr_summary_file}")
            except Exception as e:
                logger.error(f"Failed to save OCR summary: {e}")

    except Exception as e:
        logger.error(f"Error extracting keyframes: {e}")
        return None
    finally:
        if cap is not None:
            cap.release()
    
    return keyframes_dir


def extract_keyframes_for_time_range(video_path: str, start_time: float, end_time: float, max_keyframes: int = 30) -> List[Dict[str, Any]]:
    """
    Extract additional keyframes from a specific time range in the video.
    This ensures we have enough keyframes for smooth video generation.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if fps <= 0 or total_frames <= 0:
            logger.error("Invalid video properties (fps/total_frames).")
            return []

        # Calculate frame range for the time segment
        start_frame = int(max(0, start_time) * fps)
        end_frame = int(max(start_time, end_time) * fps)

        # Ensure we don't exceed video bounds
        start_frame = max(0, start_frame)
        end_frame = min(total_frames - 1, end_frame)

        if start_frame >= end_frame:
            return []

        # Calculate frame interval to get desired number of keyframes
        frame_interval = max(1, (end_frame - start_frame) // max(1, max_keyframes))

        keyframes: List[Dict[str, Any]] = []
        current_frame = start_frame

        while current_frame <= end_frame and len(keyframes) < max_keyframes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()

            if ret:
                # Calculate timestamp for this frame
                timestamp = current_frame / fps

                # Save frame
                frame_filename = f"additional_kf_{len(keyframes):04d}.jpg"
                frame_path = os.path.join(os.path.dirname(video_path), frame_filename)

                # Resize frame for consistency
                frame_resized = cv2.resize(frame, (854, 480))
                cv2.imwrite(frame_path, frame_resized)

                keyframes.append({
                    'filepath': frame_path,
                    'timestamp': timestamp,
                    'confidence': 0.8,  # Default confidence for additional frames
                    'text': '',         # No OCR text for additional frames
                    'type': 'additional'
                })

            current_frame += frame_interval

        logger.info(f"Extracted {len(keyframes)} additional keyframes for time range {start_time}-{end_time}")
        return keyframes

    except Exception as e:
        logger.error(f"Error extracting additional keyframes: {str(e)}")
        return []
    finally:
        if cap is not None:
            cap.release()


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
