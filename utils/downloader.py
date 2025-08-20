import os
import re
import uuid
from urllib.parse import urlparse
import yt_dlp
import subprocess
from werkzeug.utils import secure_filename


def _get_max_upload_bytes(default: int = 100 * 1024 * 1024) -> int:
    """Best-effort retrieval of max upload size in bytes.
    Priority: Flask current_app.config[MAX_CONTENT_LENGTH] > env var > default.
    """
    try:
        # Avoid hard dependency if outside Flask context
        from flask import current_app  # type: ignore

        try:
            cfg_val = current_app.config.get("MAX_CONTENT_LENGTH")  # type: ignore[attr-defined]
            if cfg_val:
                return int(cfg_val)
        except Exception:
            pass
    except Exception:
        pass

    env_val = os.environ.get("MAX_CONTENT_LENGTH")
    if env_val:
        try:
            return int(env_val)
        except Exception:
            pass
    return default


def is_youtube_url(url: str) -> bool:
    """
    Permissive YouTube URL check. We rely on yt-dlp for actual parsing.
    Accepts any subdomain of youtube.com, youtu.be, or youtube-nocookie.com.
    """
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or '').lower()
        return (
            'youtube.com' in host
            or 'youtu.be' in host
            or 'youtube-nocookie.com' in host
        )
    except Exception:
        return False

    
def handle_video_upload_or_download(request, upload_dir):
    """Handle video upload or YouTube download with proper error handling."""
    import logging
    logger = logging.getLogger(__name__)

    try:
        # Ensure upload directory exists
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)
            logger.info(f"Created upload directory: {upload_dir}")
        if not os.access(upload_dir, os.W_OK):
            raise ValueError(f"Upload directory is not writable: {upload_dir}")

        video_id = str(uuid.uuid4())
        logger.info(f"Processing video with ID: {video_id}")

        # Check for uploaded video file
        if 'video_file' in request.files and request.files['video_file']:
            f = request.files['video_file']
            if not f or not hasattr(f, 'filename'):
                raise ValueError("Invalid file object received")
            if not f.filename:
                raise ValueError("Uploaded file has no filename")

            logger.info(f"File object valid: {f.filename}, content_type: {getattr(f, 'content_type', 'unknown')}")

            # Determine max allowed size from Flask config/env, fallback to 100MB
            max_bytes = _get_max_upload_bytes()
            max_mb = max_bytes / (1024 * 1024)

            # Pre-check using provided content_length if available
            file_size = getattr(f, 'content_length', None)
            if file_size is not None and int(file_size) > max_bytes:
                raise ValueError(f"File too large: {int(file_size) / (1024*1024):.1f}MB (max {max_mb:.0f}MB)")

            safe_name = secure_filename(f.filename)
            path = os.path.join(upload_dir, f"{video_id}_{safe_name}")

            # Stream to disk with enforced limit
            written = 0
            with open(path, 'wb') as out_file:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        out_file.close()
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                        raise ValueError(f"File too large: {(written)/(1024*1024):.1f}MB (max {max_mb:.0f}MB)")
                    out_file.write(chunk)

            actual_file_size = os.path.getsize(path)
            if actual_file_size > max_bytes:
                try:
                    os.remove(path)
                except Exception:
                    pass
                raise ValueError(f"File too large: {actual_file_size / (1024*1024):.1f}MB (max {max_mb:.0f}MB)")

            # Validate video with OpenCV
            try:
                import cv2
                cap = cv2.VideoCapture(path)
                if not cap.isOpened() or not cap.read()[0]:
                    raise ValueError("Invalid video file")
                cap.release()
            except ImportError:
                logger.warning("OpenCV not available, skipping video validation")

            return path, video_id

        # YouTube URL case
        elif 'yt_url' in request.form and request.form['yt_url'].strip():
            yt_url = request.form['yt_url'].strip()
            if not is_youtube_url(yt_url):
                raise ValueError("Invalid YouTube URL provided")

            temp_file = f"temp_video_{video_id}.mp4"
            ydl_opts = {
                'outtmpl': temp_file,
                'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
                'merge_output_format': 'mp4'
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([yt_url])

            compressed_path = os.path.join(upload_dir, f"{video_id}.mp4")
            try:
                result = subprocess.run([
                    "ffmpeg", "-i", temp_file,
                    "-vcodec", "libx264", "-crf", "28",
                    "-preset", "fast",
                    "-acodec", "aac", "-b:a", "96k",
                    compressed_path
                ], capture_output=True, text=True, timeout=900)

                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg compression failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                raise RuntimeError("FFmpeg compression timed out")
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            return compressed_path, video_id

        else:
            raise ValueError("No video file uploaded and no YouTube URL provided")

    except Exception as e:
        logger.error(f"Error in handle_video_upload_or_download: {str(e)}")
        raise


def handle_video_upload_or_download_from_data(video_file_data, yt_url_data, upload_dir):
    """Handle video upload or YouTube download using pre-copied data."""
    import logging
    logger = logging.getLogger(__name__)

    try:
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)

        if not os.access(upload_dir, os.W_OK):
            raise ValueError(f"Upload directory is not writable: {upload_dir}")

        video_id = str(uuid.uuid4())

        if video_file_data and video_file_data.get('content'):
            filename = video_file_data['filename']
            content = video_file_data['content']
            path = os.path.join(upload_dir, f"{video_id}_{filename}")

            with open(path, 'wb') as out_file:
                out_file.write(content)
                out_file.flush()
                os.fsync(out_file.fileno())

            if os.path.getsize(path) != len(content):
                raise ValueError("File size mismatch")

            return path, video_id

        elif yt_url_data and yt_url_data.strip():
            yt_url = yt_url_data.strip()
            if not is_youtube_url(yt_url):
                raise ValueError("Invalid YouTube URL provided")

            temp_file = f"temp_video_{video_id}.mp4"
            ydl_opts = {
                'outtmpl': temp_file,
                'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
                'merge_output_format': 'mp4'
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([yt_url])

            compressed_path = os.path.join(upload_dir, f"{video_id}.mp4")
            try:
                result = subprocess.run([
                    "ffmpeg", "-i", temp_file,
                    "-vcodec", "libx264", "-crf", "28",
                    "-preset", "fast",
                    "-acodec", "aac", "-b:a", "96k",
                    compressed_path
                ], capture_output=True, text=True, timeout=900)

                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg compression failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                raise RuntimeError("FFmpeg compression timed out")
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            return compressed_path, video_id

        else:
            raise ValueError("No video file data or YouTube URL provided")

    except Exception as e:
        logger.error(f"Error in handle_video_upload_or_download_from_data: {str(e)}")
        raise


def handle_youtube_download(yt_url, upload_dir, video_id=None, status_callback=None):
    """Handle YouTube download only with real-time progress updates."""
    import logging, traceback
    logger = logging.getLogger(__name__)

    try:
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)

        if not os.access(upload_dir, os.W_OK):
            raise ValueError(f"Upload directory is not writable: {upload_dir}")

        # Use provided video_id or generate new one
        if not video_id:
            video_id = str(uuid.uuid4())
        
        if not yt_url or not yt_url.strip():
            raise ValueError("No YouTube URL provided")

        yt_url = yt_url.strip()
        if not is_youtube_url(yt_url):
            logger.warning("YouTube URL did not match pattern; attempting download via yt-dlp anyway")

        temp_file = os.path.join(upload_dir, f"temp_video_{video_id}.mp4")
        
        # Progress hook function to update status in real-time
        def progress_hook(d):
            if status_callback and callable(status_callback):
                if d['status'] == 'downloading':
                    # Extract percentage from yt-dlp progress info
                    if '_percent_str' in d:
                        percent_str = d['_percent_str'].strip()
                        try:
                            # Convert "13.6%" to float
                            percent = float(percent_str.replace('%', ''))
                            # Map download progress to 5-15% range
                            progress = 5 + (percent * 0.1)  # 5% to 15%
                            status_callback('downloading', int(progress), f"Downloading video... {percent_str}")
                        except (ValueError, AttributeError):
                            # Fallback if percentage parsing fails
                            status_callback('downloading', 5, 'Downloading video...')
                    else:
                        status_callback('downloading', 5, 'Downloading video...')
                        
                elif d['status'] == 'finished':
                    status_callback('downloading', 15, 'Download complete, processing...')
                elif d['status'] == 'error':
                    status_callback('error', 0, f'Download failed: {d.get("error", "Unknown error")}')

        ydl_opts = {
            'outtmpl': temp_file,
            'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
            'merge_output_format': 'mp4',
            'socket_timeout': 60,
            'progress_hooks': [progress_hook]
        }

        logger.info(f"Starting YouTube download with progress hooks...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([yt_url])

        # Update status to show compression starting
        if status_callback:
            status_callback('downloading', 16, 'Compressing video...')

        compressed_path = os.path.join(upload_dir, f"{video_id}.mp4")
        try:
            result = subprocess.run([
                "ffmpeg", "-i", temp_file,
                "-vcodec", "libx264", "-crf", "28",
                "-preset", "fast",
                "-acodec", "aac", "-b:a", "96k",
                compressed_path
            ], capture_output=True, text=True, timeout=900)

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg compression failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            if status_callback:
                status_callback('error', 0, 'Compression timed out')
            raise
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        # Update status to show compression complete
        if status_callback:
            status_callback('downloading', 18, 'Video ready for processing')

        logger.info(f"YouTube video downloaded and compressed successfully: {compressed_path}")
        return compressed_path, video_id

    except Exception as e:
        logger.error(f"Error in handle_youtube_download: {str(e)}\n{traceback.format_exc()}")
        # Update status to show error
        if status_callback:
            status_callback('error', 0, f'Download failed: {str(e)}')
        raise
