"""
Advanced Keyframe-Audio Synchronization System

This module provides intelligent synchronization between keyframes and audio segments
to ensure perfect timing and visual-audio alignment in summary videos.

Features:
- Intelligent keyframe timing based on audio segments
- Visual content analysis for better keyframe selection
- Audio-visual correlation analysis
- Dynamic keyframe duration adjustment
- Smooth transition timing optimization
- Production-ready error handling and monitoring

Author: Video Summarizer Team
Created: 2024
"""

import os
import cv2
import numpy as np
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import librosa
import soundfile as sf
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
from .gpu_config import get_device, is_gpu_available, log_gpu_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioSegment:
    """Audio segment with timing and features."""
    start_time: float
    end_time: float
    duration: float
    text: str
    audio_features: Dict[str, Any]
    energy_level: float
    speech_rate: float
    pause_duration: float

@dataclass
class KeyframeSegment:
    """Keyframe segment with visual features."""
    timestamp: float
    filepath: str
    visual_features: Dict[str, Any]
    text_content: str
    importance_score: float
    transition_score: float

@dataclass
class SynchronizedSegment:
    """Synchronized audio-visual segment."""
    audio_segment: AudioSegment
    keyframes: List[KeyframeSegment]
    total_duration: float
    transition_timing: List[float]
    visual_flow_score: float

class AudioVisualSynchronizer:
    """Advanced audio-visual synchronization system."""
    
    def __init__(self):
        self.device = get_device()
        self.visual_analyzer = None
        self.audio_analyzer = None
        
        # Initialize analyzers
        self._initialize_analyzers()
        
        logger.info(f"AudioVisualSynchronizer initialized on {self.device}")
        log_gpu_status()
    
    def _initialize_analyzers(self):
        """Initialize visual and audio analysis components."""
        try:
            # Initialize visual analysis (can be extended with deep learning models)
            self.visual_analyzer = VisualContentAnalyzer()
            
            # Initialize audio analysis
            self.audio_analyzer = AudioContentAnalyzer()
            
            logger.info("Audio-visual analyzers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize analyzers: {e}")
    
    def analyze_audio_segments(self, audio_path: str, segments: List[Dict]) -> List[AudioSegment]:
        """Analyze audio segments for timing and features."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=16000)
            
            audio_segments = []
            for segment in segments:
                start_time = segment.get('start', 0)
                end_time = segment.get('end', start_time + 1)
                text = segment.get('text', '')
                
                # Extract audio features for this segment
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = y[start_sample:end_sample]
                
                # Calculate audio features
                features = self.audio_analyzer.extract_features(segment_audio, sr)
                
                audio_segment = AudioSegment(
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    text=text,
                    audio_features=features,
                    energy_level=features.get('energy', 0.5),
                    speech_rate=features.get('speech_rate', 1.0),
                    pause_duration=features.get('pause_duration', 0.1)
                )
                
                audio_segments.append(audio_segment)
            
            logger.info(f"Analyzed {len(audio_segments)} audio segments")
            return audio_segments
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return []
    
    def analyze_keyframes(self, keyframes_dir: str, keyframe_metadata: List[Dict]) -> List[KeyframeSegment]:
        """Analyze keyframes for visual features and importance."""
        try:
            keyframe_segments = []
            
            for metadata in keyframe_metadata:
                timestamp = metadata.get('timestamp', 0)
                filepath = metadata.get('filepath', '')
                
                if not os.path.exists(filepath):
                    continue
                
                # Load and analyze image
                image = cv2.imread(filepath)
                if image is None:
                    continue
                
                # Extract visual features
                visual_features = self.visual_analyzer.extract_features(image)
                
                # Calculate importance score
                importance_score = self._calculate_keyframe_importance(
                    visual_features, metadata.get('text_content', '')
                )
                
                # Calculate transition score
                transition_score = self._calculate_transition_score(visual_features)
                
                keyframe_segment = KeyframeSegment(
                    timestamp=timestamp,
                    filepath=filepath,
                    visual_features=visual_features,
                    text_content=metadata.get('text_content', ''),
                    importance_score=importance_score,
                    transition_score=transition_score
                )
                
                keyframe_segments.append(keyframe_segment)
            
            logger.info(f"Analyzed {len(keyframe_segments)} keyframes")
            return keyframe_segments
            
        except Exception as e:
            logger.error(f"Keyframe analysis failed: {e}")
            return []
    
    def synchronize_segments(self, audio_segments: List[AudioSegment], 
                           keyframe_segments: List[KeyframeSegment]) -> List[SynchronizedSegment]:
        """Synchronize audio and visual segments for optimal timing."""
        try:
            synchronized_segments = []
            
            for audio_seg in audio_segments:
                # Find keyframes within audio segment time range
                relevant_keyframes = [
                    kf for kf in keyframe_segments
                    if audio_seg.start_time <= kf.timestamp <= audio_seg.end_time
                ]
                
                if not relevant_keyframes:
                    # Find nearest keyframes if none in range
                    relevant_keyframes = self._find_nearest_keyframes(
                        audio_seg, keyframe_segments
                    )
                
                if not relevant_keyframes:
                    continue
                
                # Optimize keyframe selection and timing
                optimized_keyframes = self._optimize_keyframe_timing(
                    audio_seg, relevant_keyframes
                )
                
                # Calculate transition timing
                transition_timing = self._calculate_transition_timing(
                    audio_seg, optimized_keyframes
                )
                
                # Calculate visual flow score
                visual_flow_score = self._calculate_visual_flow_score(optimized_keyframes)
                
                synchronized_segment = SynchronizedSegment(
                    audio_segment=audio_seg,
                    keyframes=optimized_keyframes,
                    total_duration=audio_seg.duration,
                    transition_timing=transition_timing,
                    visual_flow_score=visual_flow_score
                )
                
                synchronized_segments.append(synchronized_segment)
            
            logger.info(f"Created {len(synchronized_segments)} synchronized segments")
            return synchronized_segments
            
        except Exception as e:
            logger.error(f"Synchronization failed: {e}")
            return []
    
    def _calculate_keyframe_importance(self, visual_features: Dict, text_content: str) -> float:
        """Calculate importance score for a keyframe."""
        try:
            # Visual importance factors
            visual_score = 0.0
            
            # Face detection score
            if visual_features.get('face_count', 0) > 0:
                visual_score += 0.3
            
            # Text content score
            if visual_features.get('text_regions', 0) > 0:
                visual_score += 0.2
            
            # Motion/activity score
            if visual_features.get('motion_score', 0) > 0.5:
                visual_score += 0.2
            
            # Color diversity score
            if visual_features.get('color_diversity', 0) > 0.6:
                visual_score += 0.1
            
            # Edge density score
            if visual_features.get('edge_density', 0) > 0.3:
                visual_score += 0.1
            
            # Text content importance
            text_score = 0.0
            if text_content:
                text_length = len(text_content.split())
                if text_length > 10:
                    text_score += 0.1
                if any(keyword in text_content.lower() for keyword in 
                      ['important', 'key', 'main', 'primary', 'essential']):
                    text_score += 0.1
            
            total_score = min(1.0, visual_score + text_score)
            return total_score
            
        except Exception as e:
            logger.warning(f"Importance calculation failed: {e}")
            return 0.5
    
    def _calculate_transition_score(self, visual_features: Dict) -> float:
        """Calculate how well a keyframe works for transitions."""
        try:
            # Factors that make good transitions
            transition_score = 0.0
            
            # Good color distribution
            if visual_features.get('color_diversity', 0) > 0.4:
                transition_score += 0.3
            
            # Moderate edge density (not too busy, not too plain)
            edge_density = visual_features.get('edge_density', 0)
            if 0.2 <= edge_density <= 0.6:
                transition_score += 0.3
            
            # Good contrast
            if visual_features.get('contrast_score', 0) > 0.5:
                transition_score += 0.2
            
            # Not too many faces (distracting)
            if visual_features.get('face_count', 0) <= 2:
                transition_score += 0.2
            
            return min(1.0, transition_score)
            
        except Exception as e:
            logger.warning(f"Transition score calculation failed: {e}")
            return 0.5
    
    def _find_nearest_keyframes(self, audio_segment: AudioSegment, 
                               keyframe_segments: List[KeyframeSegment]) -> List[KeyframeSegment]:
        """Find nearest keyframes when none exist in the time range."""
        try:
            # Find keyframes within 5 seconds of audio segment
            tolerance = 5.0
            audio_center = (audio_segment.start_time + audio_segment.end_time) / 2
            
            nearest_keyframes = []
            for kf in keyframe_segments:
                time_diff = abs(kf.timestamp - audio_center)
                if time_diff <= tolerance:
                    nearest_keyframes.append(kf)
            
            # Sort by proximity and importance
            nearest_keyframes.sort(key=lambda x: (
                abs(x.timestamp - audio_center),
                -x.importance_score
            ))
            
            return nearest_keyframes[:3]  # Return top 3 nearest
            
        except Exception as e:
            logger.warning(f"Nearest keyframe search failed: {e}")
            return []
    
    def _optimize_keyframe_timing(self, audio_segment: AudioSegment, 
                                 keyframes: List[KeyframeSegment]) -> List[KeyframeSegment]:
        """Optimize keyframe selection and timing for audio segment."""
        try:
            if not keyframes:
                return []
            
            # Sort by importance and timestamp
            sorted_keyframes = sorted(keyframes, key=lambda x: (
                -x.importance_score, x.timestamp
            ))
            
            # Select optimal number of keyframes based on audio duration
            optimal_count = self._calculate_optimal_keyframe_count(audio_segment.duration)
            
            # Select top keyframes
            selected_keyframes = sorted_keyframes[:optimal_count]
            
            # Adjust timing for better flow
            adjusted_keyframes = self._adjust_keyframe_timing(
                audio_segment, selected_keyframes
            )
            
            return adjusted_keyframes
            
        except Exception as e:
            logger.warning(f"Keyframe timing optimization failed: {e}")
            return keyframes[:3]  # Fallback to first 3
    
    def _calculate_optimal_keyframe_count(self, duration: float) -> int:
        """Calculate optimal number of keyframes for given duration."""
        # Base calculation: 1 keyframe per 2-3 seconds
        base_count = max(1, int(duration / 2.5))
        
        # Adjust based on duration
        if duration < 5:
            return min(2, base_count)
        elif duration < 15:
            return min(5, base_count)
        else:
            return min(8, base_count)
    
    def _adjust_keyframe_timing(self, audio_segment: AudioSegment, 
                               keyframes: List[KeyframeSegment]) -> List[KeyframeSegment]:
        """Adjust keyframe timing for better visual flow."""
        try:
            if len(keyframes) <= 1:
                return keyframes
            
            # Calculate even distribution within audio segment
            start_time = audio_segment.start_time
            end_time = audio_segment.end_time
            duration = end_time - start_time
            
            adjusted_keyframes = []
            for i, kf in enumerate(keyframes):
                # Calculate target timestamp
                if len(keyframes) == 1:
                    target_time = (start_time + end_time) / 2
                else:
                    target_time = start_time + (duration * i / (len(keyframes) - 1))
                
                # Create adjusted keyframe
                adjusted_kf = KeyframeSegment(
                    timestamp=target_time,
                    filepath=kf.filepath,
                    visual_features=kf.visual_features,
                    text_content=kf.text_content,
                    importance_score=kf.importance_score,
                    transition_score=kf.transition_score
                )
                
                adjusted_keyframes.append(adjusted_kf)
            
            return adjusted_keyframes
            
        except Exception as e:
            logger.warning(f"Keyframe timing adjustment failed: {e}")
            return keyframes
    
    def _calculate_transition_timing(self, audio_segment: AudioSegment, 
                                    keyframes: List[KeyframeSegment]) -> List[float]:
        """Calculate optimal transition timing between keyframes."""
        try:
            if len(keyframes) <= 1:
                return []
            
            transition_timing = []
            duration = audio_segment.duration
            
            # Calculate transition duration (10-20% of total duration)
            transition_duration = min(0.5, max(0.2, duration * 0.15))
            
            # Calculate transition points
            for i in range(len(keyframes) - 1):
                current_time = keyframes[i].timestamp
                next_time = keyframes[i + 1].timestamp
                
                # Transition starts slightly before next keyframe
                transition_start = next_time - transition_duration
                transition_timing.append(transition_start)
            
            return transition_timing
            
        except Exception as e:
            logger.warning(f"Transition timing calculation failed: {e}")
            return []
    
    def _calculate_visual_flow_score(self, keyframes: List[KeyframeSegment]) -> float:
        """Calculate visual flow score for keyframe sequence."""
        try:
            if len(keyframes) <= 1:
                return 0.5
            
            flow_score = 0.0
            
            # Check for good visual diversity
            colors = [kf.visual_features.get('dominant_color', [0, 0, 0]) for kf in keyframes]
            color_diversity = self._calculate_color_diversity(colors)
            flow_score += color_diversity * 0.3
            
            # Check for good importance distribution
            importance_scores = [kf.importance_score for kf in keyframes]
            importance_variance = np.var(importance_scores)
            if 0.1 <= importance_variance <= 0.4:  # Good variance
                flow_score += 0.3
            
            # Check for good transition scores
            transition_scores = [kf.transition_score for kf in keyframes]
            avg_transition_score = np.mean(transition_scores)
            flow_score += avg_transition_score * 0.4
            
            return min(1.0, flow_score)
            
        except Exception as e:
            logger.warning(f"Visual flow calculation failed: {e}")
            return 0.5
    
    def _calculate_color_diversity(self, colors: List[List[int]]) -> float:
        """Calculate color diversity score."""
        try:
            if len(colors) <= 1:
                return 0.0
            
            # Convert to numpy array
            color_array = np.array(colors)
            
            # Calculate pairwise distances
            distances = []
            for i in range(len(colors)):
                for j in range(i + 1, len(colors)):
                    dist = np.linalg.norm(color_array[i] - color_array[j])
                    distances.append(dist)
            
            # Higher average distance = more diversity
            avg_distance = np.mean(distances)
            diversity_score = min(1.0, avg_distance / 255.0)  # Normalize to 0-1
            
            return diversity_score
            
        except Exception as e:
            logger.warning(f"Color diversity calculation failed: {e}")
            return 0.0

class VisualContentAnalyzer:
    """Visual content analysis for keyframes."""
    
    def __init__(self):
        self.face_cascade = None
        self._initialize_face_detection()
    
    def _initialize_face_detection(self):
        """Initialize face detection cascade."""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception as e:
            logger.warning(f"Face detection initialization failed: {e}")
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract visual features from image."""
        try:
            features = {}
            
            # Basic image properties
            height, width = image.shape[:2]
            features['dimensions'] = (width, height)
            features['aspect_ratio'] = width / height
            
            # Color analysis
            features.update(self._analyze_colors(image))
            
            # Edge analysis
            features.update(self._analyze_edges(image))
            
            # Face detection
            features.update(self._detect_faces(image))
            
            # Text detection (basic)
            features.update(self._detect_text_regions(image))
            
            # Motion/activity estimation
            features.update(self._estimate_activity(image))
            
            return features
            
        except Exception as e:
            logger.error(f"Visual feature extraction failed: {e}")
            return {}
    
    def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze color properties of image."""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate color diversity
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Calculate entropy as diversity measure
            def entropy(hist):
                hist = hist.flatten()
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                return -np.sum(hist * np.log2(hist))
            
            color_diversity = (entropy(hist_h) + entropy(hist_s) + entropy(hist_v)) / 3
            color_diversity = min(1.0, color_diversity / 8.0)  # Normalize
            
            # Dominant color
            pixels = image.reshape(-1, 3)
            dominant_color = np.mean(pixels, axis=0).astype(int).tolist()
            
            # Brightness
            brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 255.0
            
            # Contrast
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray) / 255.0
            
            return {
                'color_diversity': color_diversity,
                'dominant_color': dominant_color,
                'brightness': brightness,
                'contrast_score': contrast
            }
            
        except Exception as e:
            logger.warning(f"Color analysis failed: {e}")
            return {}
    
    def _analyze_edges(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze edge properties of image."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Edge density
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Edge direction analysis
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
            avg_edge_strength = np.mean(edge_magnitude) / 255.0
            
            return {
                'edge_density': edge_density,
                'edge_strength': avg_edge_strength
            }
            
        except Exception as e:
            logger.warning(f"Edge analysis failed: {e}")
            return {}
    
    def _detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect faces in image."""
        try:
            if self.face_cascade is None:
                return {'face_count': 0, 'face_regions': []}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            face_regions = []
            for (x, y, w, h) in faces:
                face_regions.append({
                    'x': int(x), 'y': int(y), 
                    'width': int(w), 'height': int(h)
                })
            
            return {
                'face_count': len(faces),
                'face_regions': face_regions
            }
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return {'face_count': 0, 'face_regions': []}
    
    def _detect_text_regions(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect text regions in image."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Simple text detection using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(gray, kernel, iterations=1)
            
            # Find contours that might be text
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                area = w * h
                
                # Filter for text-like regions
                if (0.1 < aspect_ratio < 10 and 
                    area > 100 and 
                    h > 10 and w > 10):
                    text_regions += 1
            
            return {
                'text_regions': text_regions,
                'text_density': text_regions / (image.shape[0] * image.shape[1] / 10000)
            }
            
        except Exception as e:
            logger.warning(f"Text detection failed: {e}")
            return {'text_regions': 0, 'text_density': 0}
    
    def _estimate_activity(self, image: np.ndarray) -> Dict[str, Any]:
        """Estimate activity/motion level in image."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate local standard deviation as activity measure
            kernel = np.ones((5, 5), np.float32) / 25
            mean = cv2.filter2D(gray, -1, kernel)
            sqr_mean = cv2.filter2D(gray.astype(np.float32)**2, -1, kernel)
            variance = sqr_mean - mean**2
            std_dev = np.sqrt(variance)
            
            activity_score = np.mean(std_dev) / 255.0
            
            return {
                'motion_score': activity_score,
                'activity_level': 'high' if activity_score > 0.3 else 'medium' if activity_score > 0.15 else 'low'
            }
            
        except Exception as e:
            logger.warning(f"Activity estimation failed: {e}")
            return {'motion_score': 0.0, 'activity_level': 'low'}

class AudioContentAnalyzer:
    """Audio content analysis for segments."""
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract audio features from segment."""
        try:
            features = {}
            
            # Energy level
            energy = np.sum(audio**2) / len(audio)
            features['energy'] = min(1.0, energy * 1000)  # Normalize
            
            # Zero crossing rate (speech activity)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            features['zero_crossing_rate'] = zcr
            
            # Spectral centroid (brightness)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            features['spectral_centroid'] = spectral_centroid
            
            # Speech rate estimation
            speech_rate = self._estimate_speech_rate(audio, sr)
            features['speech_rate'] = speech_rate
            
            # Pause detection
            pause_duration = self._detect_pauses(audio, sr)
            features['pause_duration'] = pause_duration
            
            # Spectral rolloff
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
            features['spectral_rolloff'] = rolloff
            
            return features
            
        except Exception as e:
            logger.warning(f"Audio feature extraction failed: {e}")
            return {'energy': 0.5, 'speech_rate': 1.0, 'pause_duration': 0.1}
    
    def _estimate_speech_rate(self, audio: np.ndarray, sr: int) -> float:
        """Estimate speech rate in words per second."""
        try:
            # Simple estimation based on energy variations
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.01 * sr)     # 10ms hop
            
            # Calculate energy in frames
            energy_frames = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                energy = np.sum(frame**2)
                energy_frames.append(energy)
            
            # Find peaks (potential speech events)
            energy_array = np.array(energy_frames)
            threshold = np.mean(energy_array) + np.std(energy_array)
            peaks = np.where(energy_array > threshold)[0]
            
            # Estimate speech rate
            if len(peaks) > 0:
                speech_rate = len(peaks) / (len(audio) / sr)  # events per second
                return min(5.0, max(0.5, speech_rate))  # Clamp to reasonable range
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"Speech rate estimation failed: {e}")
            return 1.0
    
    def _detect_pauses(self, audio: np.ndarray, sr: int) -> float:
        """Detect pause duration in audio segment."""
        try:
            # Calculate energy in short frames
            frame_length = int(0.1 * sr)  # 100ms frames
            hop_length = int(0.05 * sr)   # 50ms hop
            
            energy_frames = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                energy = np.sum(frame**2)
                energy_frames.append(energy)
            
            # Find low energy frames (pauses)
            energy_array = np.array(energy_frames)
            threshold = np.mean(energy_array) * 0.1  # 10% of average energy
            
            pause_frames = np.where(energy_array < threshold)[0]
            pause_duration = len(pause_frames) * hop_length / sr
            
            return pause_duration
            
        except Exception as e:
            logger.warning(f"Pause detection failed: {e}")
            return 0.1

# Global instance
audio_visual_synchronizer = AudioVisualSynchronizer()

def synchronize_audio_visual(audio_path: str, segments: List[Dict], 
                           keyframes_dir: str, keyframe_metadata: List[Dict]) -> List[SynchronizedSegment]:
    """Convenience function for audio-visual synchronization."""
    audio_segments = audio_visual_synchronizer.analyze_audio_segments(audio_path, segments)
    keyframe_segments = audio_visual_synchronizer.analyze_keyframes(keyframes_dir, keyframe_metadata)
    return audio_visual_synchronizer.synchronize_segments(audio_segments, keyframe_segments)
