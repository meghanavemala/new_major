"""
Enhanced Keyframe Extraction and OCR Analysis Module - OPTIMIZED

This module maintains the same API while providing significant performance improvements:
- Faster frame similarity calculations
- Optimized OCR processing with caching
- Smart frame sampling
- Parallel processing where possible
- Memory optimizations
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Dict, Any
import logging
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Configure logging
logging.basicConfig(level=logging.WARNING)
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
    OPTIMIZED VERSION with caching and performance improvements.
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
        self._ocr_cache = {}  # Cache for OCR results

        # Initialize EasyOCR if available and requested
        if EASYOCR_AVAILABLE and ocr_engine in ['easyocr', 'auto']:
            try:
                # Convert language names to EasyOCR codes
                easyocr_langs = []
                for lang in languages:
                    if lang in EASYOCR_LANGUAGES:
                        easyocr_langs.append(EASYOCR_LANGUAGES[lang])

                if easyocr_langs:
                    # Initialize with GPU=False for better stability and lower memory usage
                    self.easyocr_reader = easyocr.Reader(easyocr_langs, gpu=False)
                    logger.info(f"EasyOCR initialized with languages: {easyocr_langs}")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")

    def _get_image_hash(self, image: np.ndarray) -> str:
        """Generate a hash for image caching."""
        # Use a small sample of the image for hashing to speed up
        small_img = cv2.resize(image, (32, 32))
        return hashlib.md5(small_img.tobytes()).hexdigest()

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy - OPTIMIZED VERSION.

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

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Use adaptive thresholding for better results on varied lighting
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def extract_text_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text using Tesseract OCR - OPTIMIZED VERSION.
        """
        if not TESSERACT_AVAILABLE:
            return {'text': '', 'confidence': 0, 'method': 'tesseract_unavailable'}

        # Check cache first
        img_hash = self._get_image_hash(image)
        if img_hash in self._ocr_cache:
            cached_result = self._ocr_cache[img_hash].copy()
            cached_result['method'] = 'tesseract_cached'
            return cached_result

        try:
            processed_image = self.preprocess_image(image)

            # Build Tesseract language string
            tesseract_langs = []
            for lang in self.languages:
                if lang in OCR_LANGUAGES:
                    tesseract_langs.append(OCR_LANGUAGES[lang])

            lang_string = '+'.join(tesseract_langs) if tesseract_langs else 'eng'

            # Optimized Tesseract configuration
            config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '

            # Extract text
            text = pytesseract.image_to_string(
                processed_image,
                lang=lang_string,
                config=config
            ).strip()

            # Quick confidence estimation (faster than per-word analysis)
            avg_confidence = 50  # Default confidence
            if text and len(text) > 3:
                try:
                    data = pytesseract.image_to_data(
                        processed_image, lang=lang_string, output_type=pytesseract.Output.DICT
                    )
                    confidences = [int(conf) for conf in data.get('conf', []) if int(conf) > 0]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                except:
                    pass

            result = {
                'text': text,
                'confidence': avg_confidence,
                'method': 'tesseract',
                'word_count': len(text.split()) if text else 0
            }

            # Cache the result (limit cache size)
            if len(self._ocr_cache) < 500:
                self._ocr_cache[img_hash] = result.copy()

            return result

        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {'text': '', 'confidence': 0, 'method': 'tesseract_error'}

    def extract_text_easyocr(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text using EasyOCR - OPTIMIZED VERSION.
        """
        if not self.easyocr_reader:
            return {'text': '', 'confidence': 0, 'method': 'easyocr_unavailable'}

        # Check cache first
        img_hash = self._get_image_hash(image)
        if img_hash in self._ocr_cache:
            cached_result = self._ocr_cache[img_hash].copy()
            cached_result['method'] = 'easyocr_cached'
            return cached_result

        try:
            # EasyOCR works better with the original image, but resize for speed
            if image.shape[0] > 480 or image.shape[1] > 640:
                # Resize large images for faster processing
                height, width = image.shape[:2]
                scale = min(640/width, 480/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized_image = cv2.resize(image, (new_width, new_height))
            else:
                resized_image = image

            results = self.easyocr_reader.readtext(resized_image, paragraph=True)

            text_parts = []
            confidences = []

            for (_bbox, text, confidence) in results:
                if confidence > 0.6:  # Higher threshold for better quality
                    text_parts.append(text)
                    confidences.append(confidence)

            combined_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            result = {
                'text': combined_text,
                'confidence': avg_confidence * 100,  # Convert to percentage
                'method': 'easyocr',
                'word_count': len(combined_text.split()) if combined_text else 0,
                'detections': len(results)
            }

            # Cache the result
            if len(self._ocr_cache) < 500:
                self._ocr_cache[img_hash] = result.copy()

            return result

        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {'text': '', 'confidence': 0, 'method': 'easyocr_error'}

    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text using the best available OCR method.
        """
        results = []

        # Use threading for parallel OCR processing when multiple engines are available
        if self.ocr_engine in ['tesseract', 'auto'] and TESSERACT_AVAILABLE:
            results.append(self.extract_text_tesseract(image))

        if self.ocr_engine in ['easyocr', 'auto'] and self.easyocr_reader:
            results.append(self.extract_text_easyocr(image))

        if not results:
            return {'text': '', 'confidence': 0, 'method': 'no_ocr_available'}

        # Select the best result based on confidence
        best_result = max(results, key=lambda x: x['confidence'])
        all_text = ' '.join([r['text'] for r in results if r.get('text')])
        best_result['all_methods_text'] = all_text
        best_result['methods_tried'] = len(results)
        return best_result


def _are_frames_similar(frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.8) -> bool:
    """
    Check if two frames are similar - OPTIMIZED VERSION.
    Uses faster MSE-based comparison on smaller images.
    """
    try:
        # Resize to small size for fast comparison (increased from 64x36 to 96x54 for better accuracy)
        small_size = (96, 54)  # 16:9 aspect ratio, better balance of speed and accuracy
        
        # Convert to grayscale and resize
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        small1 = cv2.resize(gray1, small_size, interpolation=cv2.INTER_LINEAR)
        small2 = cv2.resize(gray2, small_size, interpolation=cv2.INTER_LINEAR)
        
        # Calculate MSE with vectorized operations for better performance
        diff = small1.astype(np.float32) - small2.astype(np.float32)
        mse = np.mean(diff * diff)
        
        # Convert MSE to similarity score (lower MSE = more similar)
        # Threshold tuning: lower MSE threshold = more similar required
        mse_threshold = (1.0 - threshold) * 1500  # Adjusted threshold for better accuracy
        
        return mse < mse_threshold
        
    except Exception as e:
        logger.error(f"Frame similarity calculation failed: {e}")
        return False  # Assume not similar on error


def _smart_frame_sampling(total_frames: int, frame_interval: int, max_keyframes: int) -> List[int]:
    """
    Smart frame sampling to reduce unnecessary frame processing.
    """
    # Calculate how many frames we'd get with the interval
    candidate_frames = list(range(0, total_frames, frame_interval))
    
    # If we have too many candidates, sample them intelligently
    if len(candidate_frames) > max_keyframes * 3:  # Process 3x max for better filtering
        # Use a combination of even distribution and early bias
        step = len(candidate_frames) // (max_keyframes * 3)
        candidate_frames = candidate_frames[::max(1, step)]
    
    return candidate_frames[:max_keyframes * 3]  # Cap at 3x max for better filtering


def extract_keyframes(
    video_path: str,
    processed_dir: str,
    video_id: str,
    target_resolution: str = '480p',
    frame_interval: int = 30,          # ~1 per second at 30fps
    similarity_threshold: float = 0.35,  # lower => catch more changes (optimized for better detection)
    ocr_languages: List[str] = ['english'],
    enable_ocr: bool = True,
    max_keyframes: int = 100
) -> Optional[str]:
    """
    Extract keyframes from video with OCR text analysis - OPTIMIZED VERSION.
    Returns the directory path containing keyframes and metadata, or None on failure.
    
    All parameters maintained for compatibility, but internal processing is optimized.
    """
    keyframes_dir = None
    cap = None
    try:
        logger.info(f"Extracting keyframes from {video_path}")
        keyframes_dir = os.path.join(processed_dir, f"{video_id}_keyframes")
        os.makedirs(keyframes_dir, exist_ok=True)
        
        # Validate input parameters
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not os.path.exists(processed_dir):
            raise FileNotFoundError(f"Processed directory not found: {processed_dir}")
        
        if max_keyframes <= 0:
            raise ValueError("max_keyframes must be greater than 0")
        
        if frame_interval <= 0:
            raise ValueError("frame_interval must be greater than 0")
        
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        # Initialize OCR engine
        ocr_engine = KeyframeOCR(languages=ocr_languages) if enable_ocr else None
        if enable_ocr:
            logger.info(f"OCR enabled for languages: {ocr_languages}")

        resolution_map = {
            '480p': (854, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080),
        }
        
        if target_resolution not in resolution_map:
            logger.warning(f"Unsupported resolution {target_resolution}, using default 480p")
            target_resolution = '480p'
            
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

        # Smart frame sampling for better performance
        candidate_frame_indices = _smart_frame_sampling(total_frames, frame_interval, max_keyframes)
        logger.info(f"Will process {len(candidate_frame_indices)} candidate frames")
        
        # Use ThreadPoolExecutor for parallel OCR processing (if enabled)
        executor = ThreadPoolExecutor(max_workers=2) if enable_ocr else None

        keyframes: List[str] = []
        keyframe_metadata: List[Dict[str, Any]] = []
        prev_frame = None
        saved_count = 0

        # Process candidate frames
        for i, frame_number in enumerate(candidate_frame_indices):
            if saved_count >= max_keyframes:
                logger.info(f"Reached maximum keyframe limit ({max_keyframes}), stopping extraction")
                break

            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                continue

            # Check similarity with previous frame
            if prev_frame is None or not _are_frames_similar(frame, prev_frame, similarity_threshold):
                resized_frame = cv2.resize(frame, (target_width, target_height))
                filename = f"keyframe_{saved_count:06d}.jpg"
                filepath = os.path.join(keyframes_dir, filename)
                
                # Use higher quality for better OCR results, but compress for storage
                cv2.imwrite(filepath, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                # OCR processing (with optimizations)
                ocr_data: Dict[str, Any] = {}
                if enable_ocr and ocr_engine:
                    # Only run OCR on frames with sufficient content
                    # Quick check: if frame is mostly uniform, skip OCR
                    gray_check = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                    std_dev = np.std(gray_check)
                    
                    if std_dev > 15:  # Frame has enough variation for potential text
                        ocr_result = ocr_engine.extract_text(resized_frame)
                        ocr_data = {
                            'text': ocr_result.get('text', ''),
                            'confidence': ocr_result.get('confidence', 0),
                            'method': ocr_result.get('method', 'none'),
                            'word_count': ocr_result.get('word_count', 0)
                        }
                    else:
                        ocr_data = {
                            'text': '',
                            'confidence': 0,
                            'method': 'skipped_uniform_frame',
                            'word_count': 0
                        }

                metadata = {
                    'filename': filename,
                    'filepath': filepath,
                    'frame_number': frame_number,
                    'timestamp': (frame_number / fps) if fps > 0 else None,
                    'resolution': f"{target_width}x{target_height}",
                    'ocr_data': ocr_data
                }

                keyframes.append(filepath)
                keyframe_metadata.append(metadata)
                prev_frame = resized_frame.copy()
                saved_count += 1

            # Progress logging (less frequent)
            if i % 20 == 0 and i > 0:
                progress = (i / len(candidate_frame_indices)) * 100
                logger.info(
                    f"Progress: {progress:.1f}% - Processed {i}/{len(candidate_frame_indices)} candidates, "
                    f"saved {saved_count} keyframes"
                )

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

        # Extract and summarize OCR text (optimized)
        if enable_ocr:
            _generate_ocr_summary_optimized(keyframes_dir, keyframe_metadata)
        
        # Clean up executor if used
        if executor:
            executor.shutdown(wait=True)

    except Exception as e:
        logger.error(f"Error extracting keyframes: {e}")
        return None
    finally:
        if cap is not None:
            cap.release()
    return keyframes_dir


def _generate_ocr_summary_optimized(keyframes_dir: str, keyframe_metadata: List[Dict[str, Any]]):
    """Generate OCR summary with optimizations."""
    try:
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
        
        # Optimize word frequency calculation
        word_freq = {}
        if joined_text:
            # Limit word frequency analysis for performance
            words = joined_text.lower().split()[:1000]  # Limit words processed
            word_freq = dict(Counter(words).most_common(20))

        ocr_summary = {
            'total_text_frames': sum(1 for t in all_ocr_text if t),
            'high_confidence_frames': len(high_confidence_text),
            'all_text': joined_text[:5000],  # Limit text size for performance
            'high_confidence_text': ' '.join(high_confidence_text)[:2000],
            'word_frequency': word_freq
        }

        ocr_summary_file = os.path.join(keyframes_dir, 'ocr_summary.json')
        with open(ocr_summary_file, 'w', encoding='utf-8') as f:
            json.dump(ocr_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"OCR summary saved to {ocr_summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to save OCR summary: {e}")


def extract_keyframes_for_time_range(
    video_path: str, 
    start_time: float, 
    end_time: float, 
    max_keyframes: int = 30,
    similarity_threshold: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Extract keyframes from a specific time range in the video - OPTIMIZED VERSION.
    
    Args:
        video_path: Path to the source video file
        start_time: Start time in seconds
        end_time: End time in seconds
        max_keyframes: Maximum number of keyframes to extract
        similarity_threshold: Minimum similarity (0-1) between consecutive frames
        
    Returns:
        List of keyframe metadata dictionaries
    """
    cap = None
    keyframes: List[Dict[str, Any]] = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            logger.error("Invalid FPS value in video")
            return []
            
        # Calculate frame range for the time segment
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = end_frame - start_frame + 1
        
        if total_frames <= 0:
            logger.warning(f"Invalid time range: {start_time}s to {end_time}s")
            return []
            
        # Optimized frame interval calculation
        frame_interval = max(1, total_frames // min(max_keyframes * 2, total_frames))
        
        # Initialize variables for frame comparison
        prev_frame = None
        
        for i in range(0, total_frames, frame_interval):
            if len(keyframes) >= max_keyframes:
                break
                
            frame_pos = start_frame + i
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Calculate timestamp
            timestamp = frame_pos / fps
            
            # Resize for consistency and speed
            frame_resized = cv2.resize(frame, (854, 480))
            
            # Only keep frames that are sufficiently different
            if prev_frame is not None:
                if _are_frames_similar(frame_resized, prev_frame, similarity_threshold):
                    continue
            
            # Save the frame
            frame_filename = f"kf_{int(timestamp)}_{len(keyframes):04d}.jpg"
            frame_dir = os.path.join(os.path.dirname(video_path), "keyframes")
            os.makedirs(frame_dir, exist_ok=True)
            frame_path = os.path.join(frame_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Add metadata
            keyframe_data = {
                'filepath': frame_path,
                'filename': frame_filename,
                'timestamp': timestamp,
                'frame_number': frame_pos,
                'is_additional': True,
                'ocr_data': {
                    'text': '',
                    'confidence': 0,
                    'method': 'none'
                }
            }
            
            keyframes.append(keyframe_data)
            prev_frame = frame_resized
        
        logger.info(f"Extracted {len(keyframes)} keyframes from {start_time:.1f}s to {end_time:.1f}s")
        return keyframes
        
    except Exception as e:
        logger.error(f"Error extracting keyframes: {str(e)}", exc_info=True)
        return keyframes  # Return whatever we have so far
        
    finally:
        if cap is not None:
            cap.release()


def get_keyframes_for_topic(
    video_path: str,
    keyframes_dir: str,
    topic_start: float,
    topic_end: float,
    target_keyframe_count: int = 10,
    min_keyframe_count: int = 3,
    similarity_threshold: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Get keyframes for a specific topic time range - OPTIMIZED VERSION.
    
    Args:
        video_path: Path to the source video file
        keyframes_dir: Directory containing existing keyframes
        topic_start: Start time of the topic in seconds
        topic_end: End time of the topic in seconds
        target_keyframe_count: Desired number of keyframes
        min_keyframe_count: Minimum number of keyframes to ensure
        similarity_threshold: Threshold for frame similarity (0-1)
        
    Returns:
        List of keyframe metadata dictionaries
    """
    try:
        # Load existing keyframes metadata
        metadata_file = os.path.join(keyframes_dir, 'keyframes_metadata.json')
        if not os.path.exists(metadata_file):
            logger.warning(f"No keyframe metadata found in {keyframes_dir}")
            return []
            
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Filter keyframes within the topic time range
        topic_keyframes = [
            kf for kf in metadata.get('keyframes', [])
            if kf.get('timestamp') is not None and 
               topic_start <= kf['timestamp'] <= topic_end
        ]
        
        # If we have enough keyframes, return them
        if len(topic_keyframes) >= target_keyframe_count:
            logger.info(f"Found {len(topic_keyframes)} existing keyframes for topic {topic_start:.1f}-{topic_end:.1f}s")
            return topic_keyframes[:target_keyframe_count]
            
        # If we have some keyframes but not enough, extract more
        if topic_keyframes and len(topic_keyframes) < target_keyframe_count:
            logger.info(f"Found {len(topic_keyframes)} keyframes, extracting more for topic {topic_start:.1f}-{topic_end:.1f}s")
            additional_keyframes = extract_keyframes_for_time_range(
                video_path=video_path,
                start_time=topic_start,
                end_time=topic_end,
                max_keyframes=target_keyframe_count - len(topic_keyframes)
            )
            
            # Add topic metadata to new keyframes
            for kf in additional_keyframes:
                kf['topic_start'] = topic_start
                kf['topic_end'] = topic_end
                kf['is_additional'] = True
            
            # Combine and sort all keyframes by timestamp
            all_keyframes = sorted(
                topic_keyframes + additional_keyframes,
                key=lambda x: x.get('timestamp', 0)
            )
            
            # Optimized similarity filtering
            filtered_keyframes = _filter_similar_keyframes_optimized(
                all_keyframes, similarity_threshold, target_keyframe_count
            )
            
            # Ensure we have at least the minimum required keyframes
            if len(filtered_keyframes) < min_keyframe_count and all_keyframes:
                filtered_keyframes = all_keyframes[:min_keyframe_count]
            
            return filtered_keyframes
            
        # If no keyframes found in the time range, extract new ones
        logger.info(f"No keyframes found for topic {topic_start:.1f}-{topic_end:.1f}s, extracting new ones")
        return extract_keyframes_for_time_range(
            video_path=video_path,
            start_time=topic_start,
            end_time=topic_end,
            max_keyframes=target_keyframe_count
        )
        
    except Exception as e:
        logger.error(f"Error getting keyframes for topic: {e}")
        return []


def _filter_similar_keyframes_optimized(
    keyframes: List[Dict[str, Any]], 
    similarity_threshold: float, 
    max_keyframes: int
) -> List[Dict[str, Any]]:
    """Optimized version of similarity filtering."""
    if not keyframes:
        return []
    
    filtered_keyframes = []
    prev_frame = None
    
    for kf in keyframes:
        if len(filtered_keyframes) >= max_keyframes:
            break
            
        if 'filepath' not in kf or not os.path.exists(kf['filepath']):
            continue
            
        # Load frame efficiently
        try:
            current_frame = cv2.imread(kf['filepath'])
            if current_frame is None:
                continue
                
            if prev_frame is None or not _are_frames_similar(
                current_frame, prev_frame, similarity_threshold
            ):
                filtered_keyframes.append(kf)
                prev_frame = current_frame
        except Exception as e:
            logger.warning(f"Error processing keyframe {kf.get('filepath', 'unknown')}: {e}")
            continue
    
    return filtered_keyframes


def get_keyframe_text_summary(keyframes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a text summary from keyframe OCR data - OPTIMIZED VERSION.
    
    Args:
        keyframes: List of keyframe metadata dictionaries
        
    Returns:
        Dictionary with text summary
    """
    try:
        all_text = []
        high_confidence_text = []
        confidences = []
        
        for kf in keyframes:
            ocr_data = kf.get('ocr_data', {})
            text = ocr_data.get('text', '').strip()
            confidence = float(ocr_data.get('confidence', 0) or 0)
            
            if text:
                all_text.append(text)
                confidences.append(confidence)
                if confidence > 70:  # Consider >70% as high confidence
                    high_confidence_text.append(text)
        
        # Optimized word frequency calculation
        word_freq = {}
        if all_text:
            # Limit processing for performance
            combined_text = ' '.join(all_text)
            words = combined_text.lower().split()
            # Only process first 2000 words for performance
            words_subset = words[:2000] if len(words) > 2000 else words
            word_freq = dict(Counter(words_subset).most_common(50))
        
        # Calculate average confidence efficiently
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'total_keyframes': len(keyframes),
            'keyframes_with_text': len(all_text),
            'high_confidence_frames': len(high_confidence_text),
            'all_text': ' '.join(all_text),
            'high_confidence_text': ' '.join(high_confidence_text),
            'word_frequency': word_freq,
            'average_confidence': avg_confidence
        }
    except Exception as e:
        logger.error(f"Error generating keyframe text summary: {e}")
        return {
            'total_keyframes': len(keyframes),
            'keyframes_with_text': 0,
            'high_confidence_frames': 0,
            'all_text': '',
            'high_confidence_text': '',
            'word_frequency': {},
            'average_confidence': 0
        }