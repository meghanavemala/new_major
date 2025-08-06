import os
import cv2
import numpy as np
from pydub import AudioSegment
import subprocess
import logging
import os
from typing import Optional, Tuple, Dict, List

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
    # Add black background for better text visibility
    (text_width, text_height), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x-5, y - text_height - 10), 
                 (x + text_width + 5, y + 5), (0, 0, 0), -1)
    
    # Add text
    cv2.putText(frame, text, (x, y), font_face, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def make_summary_video(
    keyframes_dir: str, 
    tts_audio_relpath: str, 
    processed_dir: str, 
    video_id: str, 
    cluster_id: int,
    subtitles: Optional[list] = None,
    target_width: int = 854,
    fps: int = 30
) -> Optional[str]:
    """
    Create a summary video from keyframes and audio.
    
    Args:
        keyframes_dir: Directory containing keyframe images
        tts_audio_relpath: Path to TTS audio file (relative to processed_dir)
        processed_dir: Directory to save output files
        video_id: Unique ID for the video
        cluster_id: ID of the current topic cluster
        subtitles: List of (start_time, end_time, text) for subtitles
        target_width: Target width of output video (height will be calculated)
        fps: Frames per second for output video
        
    Returns:
        Relative path to the generated video file, or None on failure
    """
    try:
        # Get all image files
        img_files = sorted(
            [f for f in os.listdir(keyframes_dir) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
        )
        
        if not img_files:
            logger.error(f"No image files found in {keyframes_dir}")
            return None
        
        # Read the first image to get dimensions
        first_img = cv2.imread(os.path.join(keyframes_dir, img_files[0]))
        if first_img is None:
            logger.error(f"Failed to read image: {img_files[0]}")
            return None
            
        # Resize to target dimensions while maintaining aspect ratio
        first_img = resize_frame(first_img, target_width)
        height, width = first_img.shape[:2]
        
        # Output paths
        temp_video = os.path.join(processed_dir, f"{video_id}_summary_{cluster_id}_temp.mp4")
        final_video = os.path.join(processed_dir, f"{video_id}_summary_{cluster_id}.mp4")
        
        # Create video writer with better quality settings
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Better compatibility than 'mp4v'
        out = cv2.VideoWriter(
            temp_video, 
            fourcc, 
            fps,  # Higher FPS for smoother playback
            (width, height)
        )
        
        # Calculate duration per frame in the output video
        audio_file = os.path.join(processed_dir, tts_audio_relpath)
        audio = AudioSegment.from_file(audio_file)
        audio_duration = len(audio) / 1000.0  # Convert to seconds
        
        # Calculate how many times to repeat each frame to match audio duration
        if len(img_files) > 0:
            frames_per_image = max(1, int((audio_duration * fps) / len(img_files)))
        else:
            frames_per_image = fps  # Fallback: 1 second per frame
        
        # Process each frame
        frame_times = []  # For subtitle timing
        current_time = 0
        time_per_frame = 1.0 / fps
        
        for img_file in img_files:
            img_path = os.path.join(keyframes_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                logger.warning(f"Skipping corrupted image: {img_file}")
                continue
                
            # Resize frame
            frame = resize_frame(frame, target_width)
            
            # Add this frame multiple times to match audio duration
            for _ in range(frames_per_image):
                # Add subtitles if available
                if subtitles:
                    # Find current subtitle
                    current_subtitle = next(
                        (sub for (start, end, text) in subtitles 
                         if start <= current_time < end), 
                        None
                    )
                    if current_subtitle:
                        frame = add_subtitle_to_frame(
                            frame.copy(), 
                            current_subtitle[2],  # Subtitle text
                            (50, height - 50)     # Position at bottom
                        )
                
                out.write(frame)
                frame_times.append(current_time)
                current_time += time_per_frame
        
        # Release the video writer
        out.release()
        
        # Trim audio to match video duration if needed
        video_duration = len(frame_times) * time_per_frame
        if audio_duration > video_duration:
            audio = audio[:int(video_duration * 1000)]
        
        # Save the trimmed audio
        temp_audio = os.path.join(processed_dir, f"temp_audio_{video_id}_{cluster_id}.wav")
        audio.export(temp_audio, format="wav")
        
        # Combine video and audio using ffmpeg with better quality settings
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', temp_video,  # Input video
            '-i', temp_audio,   # Input audio
            '-c:v', 'libx264',
            '-profile:v', 'main',
            '-preset', 'medium',
            '-crf', '23',  # Constant Rate Factor (lower = better quality, 23 is default)
            '-pix_fmt', 'yuv420p',  # Better compatibility
            '-c:a', 'aac',
            '-b:a', '192k',  # Higher audio bitrate
            '-shortest',
            '-movflags', '+faststart',  # Enable streaming
            final_video
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            logger.debug(f"FFmpeg output: {result.stdout}")
            
            if not os.path.exists(final_video):
                logger.error(f"Output video not created: {final_video}")
                return None
                
            logger.info(f"Successfully created summary video: {final_video}")
            return os.path.relpath(final_video, processed_dir)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Error in make_summary_video: {str(e)}", exc_info=True)
        return None
        
    finally:
        # Clean up temporary files
        for temp_file in [temp_video, temp_audio]:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")
