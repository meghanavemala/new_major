import os
import re
import uuid
from urllib.parse import urlparse
import yt_dlp
import subprocess
def is_youtube_url(url: str) -> bool:
    """
    Check if the given string is a valid YouTube URL.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the URL is a valid YouTube URL, False otherwise
    """
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+/)?([\w-]{11})(?:\?[\w-]*=[\w-]*(?:&[\w-]*=[\w-]*)*)?$'
    )
    youtube_regex_match = re.match(youtube_regex, url)
    return youtube_regex_match is not None

def handle_video_upload_or_download(request, upload_dir):
    video_id = str(uuid.uuid4())
    if 'video_file' in request.files and request.files['video_file']:
        f = request.files['video_file']
        path = os.path.join(upload_dir, f"{video_id}_{f.filename}")
        file_content = f.read()
        
        # Save to disk manually from content
        with open(path, 'wb') as out_file:
            out_file.write(file_content)
        return path, video_id
    elif 'yt_url' in request.form and request.form['yt_url'].strip():
        yt_url = request.form['yt_url'].strip()
        temp_file = "temp_video.mp4"
        ydl_opts = {
            'outtmpl': temp_file,
            'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
            'merge_output_format': 'mp4'
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([yt_url])

        # Compress with ffmpeg for OCR-friendly video
        compressed_path = os.path.join(upload_dir, f"{video_id}.mp4")
        subprocess.run([
            "ffmpeg", "-i", temp_file,
            "-vcodec", "libx264", "-crf", "28",  # Higher CRF = more compression
            "-preset", "fast",
            "-acodec", "aac", "-b:a", "96k",
            compressed_path
        ])

        os.remove(temp_file)
        return compressed_path, video_id
    else:
        raise ValueError("No video file uploaded and no YouTube URL provided")
