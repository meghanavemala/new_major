"""
Enhanced Video Maker Module

This module creates summary videos by combining:
- Topic-specific keyframes with proportional display duration
- Generated TTS audio
- Smooth transitions between keyframes
- Error handling and logging
"""
import os
import cv2
import numpy as np
from pydub import AudioSegment
import subprocess
import logging
import shutil
import json
import random
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path
from dataclasses import dataclass

# Supported video resolutions (width, height)
SUPPORTED_RESOLUTIONS = {
    '360p': (640, 360),
    '480p': (854, 480),  # Default for mobile compatibility
    '720p': (1280, 720),
    '1080p': (1920, 1080)
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Video generation settings
DEFAULT_SETTINGS = {
    'fps': 30,
    'transition_duration': 0.5,  # seconds
    'min_duration_per_frame': 1.0,  # seconds
    'max_duration_per_frame': 5.0,  # seconds
    'blur_radius': 15,  # For zoom/pan effect
    'zoom_factor': 1.1,  # 10% zoom for zoom effect
    'output_extension': '.mp4',
    'video_codec': 'libx264',
    'audio_codec': 'aac',
    'pixel_format': 'yuv420p',
    'crf': 23,  # Constant Rate Factor (18-28 is good, lower is better quality)
    'audio_bitrate': '192k',
    'temp_dir': 'temp_video_frames'
}

@dataclass
class VideoFrame:
    """Container for frame data and metadata."""
    image: np.ndarray
    timestamp: float = 0.0
    duration: float = 1.0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

def resize_frame(frame: np.ndarray, target_size: Tuple[int, int] = None, target_width: int = None) -> np.ndarray:
    """
    Resize frame to target dimensions while maintaining aspect ratio.
    
    Args:
        frame: Input frame as numpy array
        target_size: Tuple of (width, height)
        target_width: Target width (height will be calculated to maintain aspect ratio)
        
    Returns:
        Resized frame
    """
    if frame is None:
        raise ValueError("Input frame cannot be None")
        
    if target_size is None and target_width is None:
        target_width = SUPPORTED_RESOLUTIONS['480p'][0]
    
    if target_size:
        target_width, target_height = target_size
    else:
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        target_height = int(target_width / aspect_ratio)
    
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

def create_solid_frame(color: Tuple[int, int, int], size: Tuple[int, int]) -> np.ndarray:
    """Create a solid color frame."""
    return np.ones((size[1], size[0], 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)

def add_subtitle_to_frame(
    frame: np.ndarray, 
    text: str, 
    position: Tuple[int, int] = (50, 50),
    font_scale: float = 1.0, 
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2, 
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    background: bool = True,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    background_opacity: float = 0.7,
    max_width: int = None,
    line_spacing: float = 1.5,
    outline: bool = True,
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    outline_thickness: int = 2
) -> np.ndarray:
    """
    Add text to a video frame with optional background and word wrapping.
    
    Args:
        frame: Input frame
        text: Text to add
        position: (x, y) position of the text (top-left corner)
        font_scale: Font scale factor
        color: Text color (BGR)
        thickness: Text thickness
        font_face: Font type
        background: Whether to add background
        background_color: Background color (BGR)
        background_opacity: Background opacity (0-1)
        max_width: Maximum width before text wraps (None for no wrapping)
        line_spacing: Multiplier for line height
        outline: Whether to add outline to text
        outline_color: Outline color (BGR)
        outline_thickness: Outline thickness
        
    Returns:
        Frame with text added
    """
    if frame is None or not text:
        return frame
        
    frame = frame.copy()
    x, y = position
    
    # Get text size for a single line to calculate line height
    (_, text_height), _ = cv2.getTextSize("Test", font_face, font_scale, thickness)
    line_height = int(text_height * line_spacing)
    
    # Word wrap the text if max_width is specified
    if max_width is None:
        max_width = frame.shape[1] - x - 20  # Default to frame width with margin
    
    lines = []
    for line in str(text).split('\n'):
        words = line.split()
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            (width, _), _ = cv2.getTextSize(test_line, font_face, font_scale, thickness)
            
            if width <= max_width or not current_line:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
    
    # Calculate total text block size
    max_line_width = 0
    total_height = len(lines) * line_height
    
    for line in lines:
        (width, _), _ = cv2.getTextSize(line, font_face, font_scale, thickness)
        max_line_width = max(max_line_width, width)
    
    # Add padding
    padding = 10
    x0 = max(0, x - padding)
    y0 = max(0, y - text_height - padding)
    x1 = min(frame.shape[1], x + max_line_width + padding)
    y1 = min(frame.shape[0], y + (len(lines) - 1) * line_height + text_height + padding)
    
    # Draw semi-transparent background if needed
    if background and lines and background_opacity > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), background_color, -1)
        cv2.addWeighted(overlay, background_opacity, frame, 1 - background_opacity, 0, frame)
    
    # Draw the text line by line
    for i, line in enumerate(lines):
        y_pos = y + i * line_height
        
        # Draw outline if enabled
        if outline and outline_thickness > 0:
            cv2.putText(frame, line, (x, y_pos), font_face, font_scale, 
                       outline_color, thickness + outline_thickness, cv2.LINE_AA)
        
        # Draw main text
        cv2.putText(frame, line, (x, y_pos), font_face, font_scale, 
                   color, thickness, cv2.LINE_AA)
    
    return frame

def create_zoom_effect(frame: np.ndarray, zoom_factor: float = 1.1) -> List[np.ndarray]:
    """Create a smooth zoom effect on a frame."""
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    
    # Create zoomed-in version
    zoomed = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor, 
                       interpolation=cv2.INTER_LINEAR)
    
    # Calculate crop region (centered)
    crop_x = (zoomed.shape[1] - width) // 2
    crop_y = (zoomed.shape[0] - height) // 2
    
    # Return original and zoomed versions
    return [frame, zoomed[crop_y:crop_y+height, crop_x:crop_x+width]]

def create_slide_effect(frame: np.ndarray, direction: str = 'right', distance: float = 0.2) -> List[np.ndarray]:
    """Create a slide effect in the specified direction."""
    height, width = frame.shape[:2]
    slide_pixels = int(width * distance)
    
    if direction == 'right':
        # Slide right: start off-screen left, end at center
        start_pos = (-slide_pixels, 0)
        end_pos = (0, 0)
    elif direction == 'left':
        # Slide left: start off-screen right, end at center
        start_pos = (slide_pixels, 0)
        end_pos = (0, 0)
    elif direction == 'up':
        # Slide up: start below, end at center
        start_pos = (0, slide_pixels)
        end_pos = (0, 0)
    else:  # down
        # Slide down: start above, end at center
        start_pos = (0, -slide_pixels)
        end_pos = (0, 0)
    
    # Create start and end frames
    start_frame = np.zeros_like(frame)
    h, w = frame.shape[:2]
    
    # Calculate source and destination regions
    src_x = max(0, -start_pos[0])
    src_y = max(0, -start_pos[1])
    dst_x = max(0, start_pos[0])
    dst_y = max(0, start_pos[1])
    
    crop_w = min(w - src_x, w - dst_x)
    crop_h = min(h - src_y, h - dst_y)
    
    # Copy the visible region
    if crop_w > 0 and crop_h > 0:
        start_frame[dst_y:dst_y+crop_h, dst_x:dst_x+crop_w] = \
            frame[src_y:src_y+crop_h, src_x:src_x+crop_w]
    
    return [start_frame, frame]

def create_fade_effect(start_frame: np.ndarray, end_frame: np.ndarray, 
                      frames: int = 10) -> List[np.ndarray]:
    """Create a cross-fade transition between two frames."""
    result = []
    
    for i in range(frames):
        alpha = i / (frames - 1) if frames > 1 else 1.0
        blended = cv2.addWeighted(start_frame, 1 - alpha, end_frame, alpha, 0)
        result.append(blended)
    
    return result

def create_ken_burns_effect(
    frame: np.ndarray, 
    start_scale: float = 1.0,
    end_scale: float = 1.2,
    start_pos: Tuple[float, float] = (0.5, 0.5),
    end_pos: Tuple[float, float] = (0.6, 0.6),
    frames: int = 30
) -> List[np.ndarray]:
    """
    Create a Ken Burns effect on a frame.
    
    Args:
        frame: Input frame
        start_scale: Starting zoom level (1.0 = no zoom)
        end_scale: Ending zoom level (>1.0 = zoom in, <1.0 = zoom out)
        start_pos: Starting position as (x, y) in 0-1 range (from top-left)
        end_pos: Ending position as (x, y) in 0-1 range
        frames: Number of frames to generate
        
    Returns:
        List of frames with Ken Burns effect
    """
    height, width = frame.shape[:2]
    result = []
    
    for i in range(frames):
        # Calculate current progress (0 to 1)
        progress = i / (frames - 1) if frames > 1 else 1.0
        
        # Interpolate scale and position
        scale = start_scale + (end_scale - start_scale) * progress
        pos_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        pos_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize the frame
        resized = cv2.resize(frame, (new_width, new_height), 
                           interpolation=cv2.INTER_LINEAR)
        
        # Calculate crop region
        x = int((new_width - width) * pos_x)
        y = int((new_height - height) * pos_y)
        
        # Ensure we don't go out of bounds
        x = max(0, min(x, new_width - width))
        y = max(0, min(y, new_height - height))
        
        # Crop and add to result
        cropped = resized[y:y+height, x:x+width]
        result.append(cropped)
    
    return result

def select_keyframes_for_topic(keyframe_metadata: List[Dict], 
                             topic_start: float, 
                             topic_end: float) -> List[Dict]:
    """
    Select keyframes that fall within the specified topic time range.
    
    Args:
        keyframe_metadata: List of keyframe metadata dictionaries
        topic_start: Start time of the topic in seconds
        topic_end: End time of the topic in seconds
        
    Returns:
        Filtered list of keyframes within the time range
    """
    return [kf for kf in keyframe_metadata 
           if topic_start <= kf.get('timestamp', 0) <= topic_end]

def make_summary_video(
    output_path: str,
    keyframes: List[Dict],
    audio_path: Optional[str] = None,
    subtitles: Optional[List[Dict]] = None,
    resolution: Tuple[int, int] = (1280, 720),
    fps: int = 30,
    transition_type: str = 'fade',
    transition_duration: float = 0.5,
    min_frame_duration: float = 2.0,
    max_frame_duration: float = 8.0,
    watermark_path: Optional[str] = None,
    watermark_opacity: float = 0.7,
    watermark_position: str = 'bottom-right',
    progress_bar: bool = True,
    progress_bar_color: Tuple[int, int, int] = (0, 165, 255),  # Orange
    progress_bar_height: int = 5,
    show_timestamps: bool = True,
    timestamp_font_scale: float = 0.7,
    timestamp_color: Tuple[int, int, int] = (255, 255, 255),  # White
    timestamp_position: str = 'top-left',
    timestamp_format: str = '%H:%M:%S',
    background_color: Tuple[int, int, int] = (0, 0, 0),  # Black
    log_level: str = 'INFO'
) -> Dict:
    """
    Create a summary video from keyframes with smooth transitions and audio.
    
    Args:
        output_path: Path to save the output video
        keyframes: List of keyframe dictionaries with 'filepath' and 'timestamp' keys
        audio_path: Path to audio file (optional)
        subtitles: List of subtitle dictionaries with 'text', 'start_time', 'end_time' keys
        resolution: Output video resolution (width, height)
        fps: Frames per second for output video
        transition_type: Type of transition between keyframes ('fade', 'slide', 'zoom', 'ken_burns')
        transition_duration: Duration of transitions in seconds
        min_frame_duration: Minimum duration to show each keyframe (seconds)
        max_frame_duration: Maximum duration to show each keyframe (seconds)
        watermark_path: Path to watermark image (optional)
        watermark_opacity: Opacity of watermark (0.0 to 1.0)
        watermark_position: Position of watermark ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        progress_bar: Whether to show a progress bar
        progress_bar_color: Color of progress bar (BGR)
        progress_bar_height: Height of progress bar in pixels
        show_timestamps: Whether to show timestamps
        timestamp_font_scale: Font scale for timestamps
        timestamp_color: Color of timestamps (BGR)
        timestamp_position: Position of timestamps ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        timestamp_format: Format string for timestamps (strftime format)
        background_color: Background color for padding (BGR)
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        
    Returns:
        Dictionary with metadata about the created video
    """
    # Set up logging
    logging.basicConfig(level=getattr(logging, log_level.upper()),
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Validate inputs
    if not keyframes:
        raise ValueError("No keyframes provided")
    
    # Sort keyframes by timestamp
    keyframes = sorted(keyframes, key=lambda x: x.get('timestamp', 0))
    
    # Calculate total duration if audio is provided
    audio_duration = 0
    if audio_path and os.path.exists(audio_path):
        logger.info(f"Sorted {len(keyframes)} keyframes by timestamp")
    
    # Read the first image to get dimensions
    first_img = cv2.imread(keyframes[0]['filepath'])
    if first_img is None:
        logger.error(f"Failed to read image: {keyframes[0]['filepath']}")
        return None
    first_img = resize_frame(first_img, target_size=resolution)
    height, width = first_img.shape[:2]
    
    # Output paths
    temp_video = output_path + "_temp.avi"
    final_video = output_path + ".mp4"
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    if not out.isOpened():
        logger.error("Failed to create video writer with XVID codec.")
        return None
        
    # Calculate total frames based on audio duration or default
    if audio_duration > 0:
        total_frames = int(audio_duration * fps)
    else:
        # Default to 5 seconds per keyframe if no audio
        total_frames = len(keyframes) * (fps * 5)
    
    # Set maximum total frames to prevent excessive processing
    max_total_frames = 30 * 60 * fps  # 30 minutes max
    if total_frames > max_total_frames:
        logger.warning(f"Capping total frames from {total_frames} to {max_total_frames}")
        total_frames = max_total_frames
    
    # Calculate frame distribution for keyframes
    n_keyframes = len(keyframes)
    transition_frames = int(transition_duration * fps)
    
    # Ensure we have enough frames for transitions
    available_frames = total_frames - (transition_frames * (n_keyframes - 1))
    if available_frames <= 0:
        transition_frames = max(1, (total_frames // n_keyframes) // 4)
        available_frames = total_frames - (transition_frames * (n_keyframes - 1))
    
    # Distribute frames between keyframes
    base_frames_per_keyframe = max(1, available_frames // n_keyframes)
    remaining_frames = available_frames - (base_frames_per_keyframe * n_keyframes)
    
    keyframe_frames = []
    for i in range(n_keyframes):
        extra = 1 if i < remaining_frames else 0
        keyframe_frames.append(base_frames_per_keyframe + extra)
    
    logger.info(f"Audio duration: {audio_duration:.2f}s, Total frames: {total_frames}, "
                f"Frames per keyframe: ~{base_frames_per_keyframe}, "
                f"Transition frames: {transition_frames}")
    
    # Process keyframes and create video
    prev_img = None
    frames_written = 0
    
    try:
        for i, kf in enumerate(keyframes):
            # Load and process keyframe image
            img = cv2.imread(kf['filepath'])
            if img is None:
                logger.warning(f"Skipping corrupted image: {kf['filepath']}")
                continue
                
            # Resize image to target resolution
            img = resize_frame(img, resolution)
            frames_for_this_image = keyframe_frames[i]
            
            # Add transition from previous image if available
            if prev_img is not None and transition_frames > 0:
                for t in range(transition_frames):
                    alpha = t / transition_frames
                    blended = cv2.addWeighted(prev_img, 1 - alpha, img, alpha, 0)
                    out.write(blended)
                    frames_written += 1
            
            # Add the keyframe for its duration
            for _ in range(frames_for_this_image):
                out.write(img)
                frames_written += 1
                
            prev_img = img
            
            # Update progress
            progress = (i + 1) / len(keyframes) * 100
            logger.info(f"Processed {i+1}/{len(keyframes)} keyframes ({progress:.1f}%)")
    
    except Exception as e:
        logger.error(f"Error processing video frames: {e}")
        return None
    
    # Release the video writer
    out.release()
    logger.info(f"Finished writing {frames_written} frames to {temp_video}")
    
    # Check if the temp video was created successfully
    if not os.path.exists(temp_video) or os.path.getsize(temp_video) < 1000:
        logger.error(f"Temp video file missing or too small: {temp_video}")
        return None
    
    # Process audio if available
    temp_audio = None
    if audio_path and os.path.exists(audio_path):
        try:
            temp_audio = output_path + "_audio.wav"
            audio = AudioSegment.from_file(audio_path)
            audio.export(temp_audio, format="wav")
            
            if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) < 1000:
                logger.error(f"Temp audio file missing or too small: {temp_audio}")
                return None
            
            # Use FFmpeg to mux video and audio
            cmd = [
                'ffmpeg', '-y', '-i', temp_video, '-i', temp_audio,
                '-c:v', 'libx264', '-profile:v', 'main', '-preset', 'medium',
                '-crf', '23', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                '-shortest', '-movflags', '+faststart', final_video
            ]
            
            # Execute FFmpeg command
            import subprocess
            try:
                logger.info(f"Running FFmpeg: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, text=True, timeout=120)
                logger.debug(f"FFmpeg output: {result.stdout}")
                
                if not os.path.exists(final_video) or os.path.getsize(final_video) < 1000:
                    logger.error(f"Output video not created or too small: {final_video}")
                    return None
                    
                logger.info(f"Successfully created summary video: {final_video}")
                return final_video
                
            except subprocess.TimeoutExpired:
                logger.error("FFmpeg process timed out!")
                return None
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg error: {e.stderr}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error during FFmpeg processing: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Error in make_summary_video: {str(e)}", exc_info=True)
            return None
        finally:
            for temp_file in [temp_video, temp_audio]:
                try:
                    if temp_file and os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_file}: {e}")
    
    # Return the final video path if everything succeeded
    return final_video