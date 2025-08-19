"""
Test script for video generation with transitions and effects.
"""
import os
import cv2
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine
import tempfile
import shutil

# Add parent directory to path to import utils
import sys
sys.path.append(str(Path(__file__).parent))
from utils.video_maker import make_summary_video

def create_test_keyframes(output_dir: str, num_keyframes: int = 5) -> List[Dict]:
    """Create test keyframes with sample images."""
    os.makedirs(output_dir, exist_ok=True)
    keyframes = []
    
    # Create sample images with different colors
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    for i in range(min(num_keyframes, len(colors))):
        # Create a solid color image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = colors[i]
        
        # Add text to identify the frame
        text = f"Frame {i+1}"
        cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Save the image
        img_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(img_path, img)
        
        # Add to keyframes list with timestamp
        keyframes.append({
            'filepath': img_path,
            'timestamp': i * 5.0,  # 5 seconds apart
            'text': f"This is frame {i+1} with some sample text."
        })
    
    return keyframes

def create_test_audio(output_path: str, duration_sec: int = 30) -> str:
    """Create a test audio file with beeps at regular intervals."""
    # Create a silent audio segment
    audio = AudioSegment.silent(duration=duration_sec * 1000)  # pydub works in milliseconds
    
    # Add beeps at regular intervals
    for i in range(duration_sec):
        if i % 5 == 0:  # Beep every 5 seconds
            # Generate a beep (440 Hz for 200ms)
            beep = Sine(440).to_audio_segment(duration=200, volume=-20)
            audio = audio.overlay(beep, position=i * 1000)
    
    # Export the audio file
    audio.export(output_path, format="wav")
    return output_path

def test_video_generation():
    """Test the video generation with different settings."""
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create test keyframes
        keyframes_dir = os.path.join(temp_dir, "keyframes")
        keyframes = create_test_keyframes(keyframes_dir, num_keyframes=5)
        
        # Create test audio
        audio_path = os.path.join(temp_dir, "test_audio.wav")
        create_test_audio(audio_path, duration_sec=30)
        
        # Create test subtitles
        subtitles = [
            {
                'text': "This is the first subtitle",
                'start_time': 0.0,
                'end_time': 5.0
            },
            {
                'text': "Now we're on the second frame",
                'start_time': 5.0,
                'end_time': 10.0
            },
            {
                'text': "Halfway through the test video",
                'start_time': 10.0,
                'end_time': 15.0
            },
            {
                'text': "Almost done with the test",
                'start_time': 15.0,
                'end_time': 20.0
            },
            {
                'text': "Final frame of the test",
                'start_time': 20.0,
                'end_time': 25.0
            }
        ]
        
        # Test 1: Basic video with fade transitions
        print("\nTesting basic video with fade transitions...")
        output_path = os.path.join(temp_dir, "test_fade.mp4")
        metadata = make_summary_video(
            output_path=output_path,
            keyframes=keyframes,
            audio_path=audio_path,
            subtitles=subtitles,
            resolution=(1280, 720),
            fps=30,
            transition_type='fade',
            transition_duration=0.5,
            min_frame_duration=3.0,
            max_frame_duration=8.0,
            progress_bar=True,
            show_timestamps=True,
            log_level='DEBUG'
        )
        print(f"Created video with fade transitions: {output_path}")
        print(f"Video metadata: {metadata}")
        
        # Test 2: Video with slide transitions
        print("\nTesting video with slide transitions...")
        output_path = os.path.join(temp_dir, "test_slide.mp4")
        metadata = make_summary_video(
            output_path=output_path,
            keyframes=keyframes,
            audio_path=audio_path,
            subtitles=subtitles,
            resolution=(1280, 720),
            fps=30,
            transition_type='slide',
            transition_duration=0.8,
            min_frame_duration=3.0,
            max_frame_duration=8.0,
            progress_bar=True,
            show_timestamps=True,
            log_level='DEBUG'
        )
        print(f"Created video with slide transitions: {output_path}")
        
        # Test 3: Video with Ken Burns effect
        print("\nTesting video with Ken Burns effect...")
        output_path = os.path.join(temp_dir, "test_ken_burns.mp4")
        metadata = make_summary_video(
            output_path=output_path,
            keyframes=keyframes,
            audio_path=audio_path,
            subtitles=subtitles,
            resolution=(1280, 720),
            fps=30,
            transition_type='ken_burns',
            transition_duration=1.0,
            min_frame_duration=4.0,
            max_frame_duration=10.0,
            progress_bar=True,
            show_timestamps=True,
            log_level='DEBUG'
        )
        print(f"Created video with Ken Burns effect: {output_path}")
        
        # Test 4: Video with zoom transitions and watermark
        print("\nTesting video with zoom transitions and watermark...")
        
        # Create a simple watermark
        watermark_path = os.path.join(temp_dir, "watermark.png")
        watermark = np.zeros((200, 400, 4), dtype=np.uint8)  # RGBA
        cv2.putText(watermark, "SAMPLE", (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255, 200), 3, cv2.LINE_AA)
        cv2.imwrite(watermark_path, watermark)
        
        output_path = os.path.join(temp_dir, "test_zoom.mp4")
        metadata = make_summary_video(
            output_path=output_path,
            keyframes=keyframes,
            audio_path=audio_path,
            subtitles=subtitles,
            resolution=(1280, 720),
            fps=30,
            transition_type='zoom',
            transition_duration=0.6,
            min_frame_duration=3.0,
            max_frame_duration=7.0,
            watermark_path=watermark_path,
            watermark_opacity=0.5,
            watermark_position='bottom-right',
            progress_bar=True,
            show_timestamps=True,
            timestamp_position='bottom-left',
            log_level='DEBUG'
        )
        print(f"Created video with zoom transitions and watermark: {output_path}")
        
        print("\nAll tests completed successfully!")
        print(f"Test files are available in: {temp_dir}")
        
        # Keep the temporary directory for inspection
        input("Press Enter to clean up temporary files...")

if __name__ == "__main__":
    test_video_generation()
