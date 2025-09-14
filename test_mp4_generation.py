#!/usr/bin/env python3

"""
Test video generation to debug MP4 conversion issues
"""

import sys
import os
import subprocess
import logging

# Add the MAJ directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.video_maker import make_summary_video, select_keyframes_for_topic

def test_video_generation():
    """Test video generation with existing keyframes and audio"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Find the latest processed video
    processed_dir = r"c:\Users\mythr\OneDrive\Desktop\MAJ\processed"
    folders = [f for f in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, f)) and f.isdigit()]
    
    if not folders:
        print("❌ No processed folders found")
        return False
    
    latest_folder = max(folders, key=int)
    video_id = latest_folder
    
    print(f"📁 Testing with video ID: {video_id}")
    
    # Check for keyframes
    keyframes_dir = os.path.join(processed_dir, f"{video_id}_keyframes")
    if not os.path.exists(keyframes_dir):
        print(f"❌ No keyframes directory found: {keyframes_dir}")
        return False
    
    # Load keyframes metadata
    import json
    keyframes_metadata_file = os.path.join(keyframes_dir, "keyframes_metadata.json")
    if not os.path.exists(keyframes_metadata_file):
        print(f"❌ No keyframes metadata found: {keyframes_metadata_file}")
        return False
    
    with open(keyframes_metadata_file, 'r') as f:
        keyframes_data = json.load(f)
    
    keyframes = keyframes_data.get('keyframes', [])
    print(f"📊 Found {len(keyframes)} keyframes")
    
    # Check for audio files
    audio_files = [f for f in os.listdir(processed_dir) if f.startswith(video_id) and f.endswith('.wav')]
    if not audio_files:
        print("❌ No audio files found")
        return False
    
    audio_file = os.path.join(processed_dir, audio_files[0])
    print(f"🔊 Using audio file: {audio_file}")
    
    # Test 1: Check FFmpeg availability
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        print("✅ FFmpeg is available")
        print(f"   Version: {result.stdout.split()[2] if len(result.stdout.split()) > 2 else 'Unknown'}")
    except Exception as e:
        print(f"❌ FFmpeg not available: {e}")
        return False
    
    # Test 2: Create a simple video with first few keyframes
    test_keyframes = keyframes[:5]  # Use first 5 keyframes for testing
    
    output_path = os.path.join(processed_dir, f"test_video_{video_id}")
    
    print(f"🎬 Creating test video with {len(test_keyframes)} keyframes...")
    
    try:
        result_path = make_summary_video(
            output_path=output_path,
            keyframes=test_keyframes,
            audio_path=audio_file,
            resolution=(854, 480),
            fps=30,
            log_level="INFO"
        )
        
        if result_path and os.path.exists(result_path):
            print(f"✅ Video created successfully: {result_path}")
            print(f"   File size: {os.path.getsize(result_path)} bytes")
            
            # Check file extension
            if result_path.endswith('.mp4'):
                print("✅ Output is MP4 format")
                return True
            else:
                print(f"⚠️  Output is not MP4: {result_path}")
                return False
        else:
            print(f"❌ Video creation failed. Result: {result_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error during video creation: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Testing video generation and MP4 conversion...")
    print("="*60)
    
    success = test_video_generation()
    
    if success:
        print("\n🎉 Video generation test passed!")
    else:
        print("\n❌ Video generation test failed!")
        
    sys.exit(0 if success else 1)