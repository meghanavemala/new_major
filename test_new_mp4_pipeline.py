#!/usr/bin/env python3

"""
Test generating a new topic video in MP4 format to verify the pipeline works
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.video_maker import make_summary_video
from pathlib import Path

def test_new_video_generation():
    """Test generating a new topic video in MP4 format"""
    
    processed_dir = Path(r"c:\Users\mythr\OneDrive\Desktop\MAJ\processed")
    
    # Find the latest video ID
    video_ids = [d.name for d in processed_dir.iterdir() 
                 if d.is_dir() and d.name.isdigit()]
    
    if not video_ids:
        print("❌ No processed video directories found")
        return False
    
    latest_video_id = max(video_ids, key=int)
    video_dir = processed_dir / latest_video_id
    
    print(f"📁 Using video ID: {latest_video_id}")
    
    # Check for keyframes
    keyframes_dir = processed_dir / f"{latest_video_id}_keyframes"
    if not keyframes_dir.exists():
        print(f"❌ No keyframes directory: {keyframes_dir}")
        return False
    
    # Load keyframes metadata
    keyframes_metadata_file = keyframes_dir / "keyframes_metadata.json"
    if not keyframes_metadata_file.exists():
        print(f"❌ No keyframes metadata: {keyframes_metadata_file}")
        return False
    
    with open(keyframes_metadata_file, 'r') as f:
        keyframes_data = json.load(f)
    
    keyframes = keyframes_data.get('keyframes', [])
    if not keyframes:
        print("❌ No keyframes found in metadata")
        return False
    
    print(f"📊 Found {len(keyframes)} keyframes")
    
    # Check for topic summaries
    topic_summaries_file = video_dir / "topic_summaries.json"
    if not topic_summaries_file.exists():
        print(f"❌ No topic summaries: {topic_summaries_file}")
        return False
    
    with open(topic_summaries_file, 'r') as f:
        topic_summaries = json.load(f)
    
    if not topic_summaries:
        print("❌ No topic summaries found")
        return False
    
    # Get first topic
    first_topic = topic_summaries[0]
    topic_start = first_topic.get('start_time', 0)
    topic_end = first_topic.get('end_time', 60)
    
    print(f"🎯 Testing with first topic: {topic_start:.1f}s - {topic_end:.1f}s")
    
    # Filter keyframes for this topic timeframe
    topic_keyframes = []
    for kf in keyframes:
        timestamp = kf.get('timestamp', 0)
        if topic_start <= timestamp <= topic_end:
            topic_keyframes.append(kf)
    
    if not topic_keyframes:
        # Use first few keyframes as fallback
        topic_keyframes = keyframes[:5]
        print(f"⚠️  No keyframes in time range, using first {len(topic_keyframes)} keyframes")
    
    print(f"🎬 Using {len(topic_keyframes)} keyframes for video generation")
    
    # Check for audio (use any available .wav file)
    audio_files = list(processed_dir.glob(f"{latest_video_id}_summary_*.wav"))
    if not audio_files:
        print("⚠️  No audio files found, creating video without audio")
        audio_path = None
    else:
        audio_path = str(audio_files[0])
        print(f"🔊 Using audio: {audio_files[0].name}")
    
    # Generate video
    output_path = processed_dir / f"test_new_video_{latest_video_id}"
    
    print(f"🎬 Generating MP4 video...")
    
    try:
        result_path = make_summary_video(
            output_path=str(output_path),
            keyframes=topic_keyframes,
            audio_path=audio_path,
            resolution=(854, 480),
            fps=30,
            log_level="INFO"
        )
        
        if result_path and os.path.exists(result_path):
            file_size = os.path.getsize(result_path)
            print(f"✅ Video generated successfully!")
            print(f"   📄 File: {os.path.basename(result_path)}")
            print(f"   💾 Size: {file_size:,} bytes")
            
            if result_path.endswith('.mp4'):
                print("✅ Output is in MP4 format - ready for frontend!")
                return True
            else:
                print(f"⚠️  Output format: {Path(result_path).suffix}")
                return False
        else:
            print(f"❌ Video generation failed: {result_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error during video generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 Testing new MP4 video generation...")
    print("="*60)
    
    success = test_new_video_generation()
    
    if success:
        print("\n🎉 New video generation test passed!")
        print("   The pipeline will now generate MP4 videos for the frontend.")
    else:
        print("\n❌ Video generation test failed!")
        
    sys.exit(0 if success else 1)