#!/usr/bin/env python3

"""
Convert existing .avi topic videos to .mp4 format for frontend compatibility
"""

import os
import subprocess
import logging
from pathlib import Path

def convert_avi_to_mp4():
    """Convert all .avi topic videos to .mp4 format"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    processed_dir = r"c:\Users\mythr\OneDrive\Desktop\MAJ\processed"
    
    # Find all .avi files in the processed directory
    avi_files = []
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith('.avi') and 'topic_' in file:
                avi_files.append(os.path.join(root, file))
    
    if not avi_files:
        print("No .avi topic videos found to convert")
        return True
    
    print(f"Found {len(avi_files)} .avi files to convert:")
    
    converted = 0
    failed = 0
    
    for avi_file in avi_files:
        mp4_file = avi_file.replace('.avi', '.mp4')
        
        # Skip if MP4 already exists and is newer
        if os.path.exists(mp4_file):
            if os.path.getmtime(mp4_file) > os.path.getmtime(avi_file):
                print(f"⏭️  Skipping {os.path.basename(avi_file)} (MP4 exists and is newer)")
                continue
        
        print(f"🔄 Converting {os.path.basename(avi_file)}...")
        
        try:
            # Use FFmpeg to convert AVI to MP4
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', avi_file,  # Input AVI
                '-c:v', 'libx264',  # Video codec
                '-c:a', 'aac',      # Audio codec
                '-crf', '23',       # Quality
                '-preset', 'medium', # Encoding speed
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-movflags', '+faststart',  # Web optimization
                mp4_file
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Verify the output file exists and is valid
            if os.path.exists(mp4_file) and os.path.getsize(mp4_file) > 1000:
                print(f"✅ Converted to {os.path.basename(mp4_file)}")
                
                # Optionally remove the original AVI file
                try:
                    os.remove(avi_file)
                    print(f"🗑️  Removed original {os.path.basename(avi_file)}")
                except Exception as e:
                    print(f"⚠️  Could not remove original file: {e}")
                
                converted += 1
            else:
                print(f"❌ Conversion failed - output file invalid")
                failed += 1
                
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e.stderr}")
            failed += 1
        except subprocess.TimeoutExpired:
            print(f"❌ Conversion timed out")
            failed += 1
        except Exception as e:
            print(f"❌ Conversion error: {e}")
            failed += 1
    
    print(f"\n📊 Conversion Summary:")
    print(f"   ✅ Converted: {converted}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 Total: {len(avi_files)}")
    
    return failed == 0

if __name__ == "__main__":
    print("🎬 Converting existing .avi topic videos to .mp4 format...")
    print("="*60)
    
    success = convert_avi_to_mp4()
    
    if success:
        print("\n🎉 All video conversions completed successfully!")
    else:
        print("\n⚠️  Some conversions failed. Check the logs above.")