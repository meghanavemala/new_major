import os
import re
import uuid
from urllib.parse import urlparse
from pytube import YouTube

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
    if 'video' in request.files and request.files['video']:
        f = request.files['video']
        path = os.path.join(upload_dir, f"{video_id}_{f.filename}")
        f.save(path)
        return path, video_id
    elif 'yt_url' in request.form and request.form['yt_url']:
        yt = YouTube(request.form['yt_url'])
        stream = yt.streams.filter(file_extension='mp4', progressive=True).first()
        path = stream.download(output_path=upload_dir, filename=f"{video_id}.mp4")
        return path, video_id
    else:
        raise ValueError("No upload or url provided")
