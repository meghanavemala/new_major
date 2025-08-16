"""
This file handles the creation of summary videos by combining extracted keyframes, generated audio summaries, and optional subtitles.
It now supports per-topic video generation: for each topic, only the keyframes whose timestamps fall within the topic's segment times are used, and each keyframe is shown for a proportional duration to match the summary audio.
"""
import os
import cv2
import numpy as np
from pydub import AudioSegment
import subprocess
import logging
import shutil
from typing import Optional, Tuple, Dict, List
from pathlib import Path

# Supported video resolutions (width, height)
SUPPORTED_RESOLUTIONS = {
    '360p': (640, 360),
    '480p': (854, 480),
    '720p': (1280, 720),
    '1080p': (1920, 1080)
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resize_frame(frame: np.ndarray, target_width: int = 854) -> np.ndarray:
    """Resize frame to target width while maintaining aspect ratio."""
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    target_height = int(target_width / aspect_ratio)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

def add_subtitle_to_frame(frame: np.ndarray, text: str, position: Tuple[int, int] = (50, 50), 
                         font_scale: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255),
                         thickness: int = 2, font_face: int = cv2.FONT_HERSHEY_SIMPLEX) -> np.ndarray:
    """Add subtitle text to a video frame."""
    frame = frame.copy()  # Create a copy to avoid modifying the original
    
    # Split text into multiple lines if too long
    max_width = frame.shape[1] - 100  # Leave margins
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        (text_width, _), _ = cv2.getTextSize(' '.join(current_line), font_face, font_scale, thickness)
        if text_width > max_width:
            if len(current_line) > 1:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(' '.join(current_line))
                current_line = []
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw each line
    x, y = position
    line_height = int(text_height * 1.5)
    
    for i, line in enumerate(lines):
        y_pos = y + i * line_height
        (text_width, text_height), _ = cv2.getTextSize(line, font_face, font_scale, thickness)
        
        # Add black background for better text visibility
        cv2.rectangle(frame, 
                     (x-5, y_pos - text_height - 5),
                     (x + text_width + 5, y_pos + 5),
                     (0, 0, 0), -1)
        
        # Add text
        cv2.putText(frame, line, (x, y_pos), font_face, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame

def select_keyframes_for_topic(keyframe_metadata, topic_start, topic_end):
    """Select keyframes whose timestamps fall within the topic's segment times."""
    return [kf for kf in keyframe_metadata if topic_start <= kf['timestamp'] <= topic_end]

# Update make_summary_video to accept a list of keyframe filepaths and their timestamps
# Instead of reading all images from a directory, accept a list of filepaths (with timestamps)
def make_summary_video(
    keyframes: list,  # List of dicts with 'filepath' and 'timestamp'
    tts_audio_path: str,
    output_path: str,
    target_width: int = 854,
    fps: int = 30
) -> Optional[str]:
    """
    Create a summary video for a topic from selected keyframes and audio.
    Each keyframe is shown for a proportional duration to match the audio length.
    """
    temp_video = None
    temp_audio = None
    try:
        if not keyframes:
            logger.error("No keyframes provided for topic video.")
            return None
        # Read the first image to get dimensions
        first_img = cv2.imread(keyframes[0]['filepath'])
        if first_img is None:
            logger.error(f"Failed to read image: {keyframes[0]['filepath']}")
            return None
        first_img = resize_frame(first_img, target_width)
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
        # Load and process audio
        if not os.path.exists(tts_audio_path):
            logger.error(f"Audio file not found: {tts_audio_path}")
            return None
        audio = AudioSegment.from_file(tts_audio_path)
        audio_duration = len(audio) / 1000.0  # seconds
        n_keyframes = len(keyframes)
        frame_duration = audio_duration / n_keyframes if n_keyframes > 0 else 1.0
        frames_per_image = max(1, int(frame_duration * fps))
        
        # Ensure smooth transitions by limiting frames per image
        max_frames_per_image = max(1, int(audio_duration * fps / 30))  # Max 30 keyframes for smooth video
        frames_per_image = min(frames_per_image, max_frames_per_image)
        
        # Write frames
        for kf in keyframes:
            img = cv2.imread(kf['filepath'])
            if img is None:
                logger.warning(f"Skipping corrupted image: {kf['filepath']}")
                continue
            img = resize_frame(img, target_width)
            for _ in range(frames_per_image):
                out.write(img)
        out.release()
        # Save the trimmed audio
        temp_audio = output_path + "_audio.wav"
        audio.export(temp_audio, format="wav")
        # Use FFmpeg to mux .avi and .wav into .mp4
        cmd = [
            'ffmpeg', '-y', '-i', temp_video, '-i', temp_audio,
            '-c:v', 'libx264', '-profile:v', 'main', '-preset', 'medium',
            '-crf', '23', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
            '-shortest', '-movflags', '+faststart', final_video
        ]
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logger.debug(f"FFmpeg output: {result.stdout}")
            if not os.path.exists(final_video):
                logger.error(f"Output video not created: {final_video}")
                return None
            logger.info(f"Successfully created topic summary video: {final_video}")
            return final_video
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
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