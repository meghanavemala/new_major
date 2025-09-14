import os
import re
import uuid
from urllib.parse import urlparse
import yt_dlp
import subprocess
from werkzeug.utils import secure_filename
from .gpu_config import is_gpu_available, log_gpu_status


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
                # Use CPU encoding for maximum compatibility
                # GPU encoding (h264_nvenc) causes issues with older drivers
                cmd = [
                    "ffmpeg", "-i", temp_file,
                    "-vcodec", "libx264", "-crf", "28",
                    "-preset", "fast",
                    "-acodec", "aac", "-b:a", "96k",
                    compressed_path
                ]
                logger.info("Using CPU encoding for YouTube video compression")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

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
                # Use CPU encoding for maximum compatibility
                # GPU encoding (h264_nvenc) causes issues with older drivers
                cmd = [
                    "ffmpeg", "-i", temp_file,
                    "-vcodec", "libx264", "-crf", "28",
                    "-preset", "fast",
                    "-acodec", "aac", "-b:a", "96k",
                    compressed_path
                ]
                logger.info("Using CPU encoding for YouTube video compression")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

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
    import time
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

        # Use a more specific template to avoid conflicts and ensure proper cleanup
        temp_file_template = os.path.join(upload_dir, f"temp_video_{video_id}.%(ext)s")
        final_temp_file = os.path.join(upload_dir, f"temp_video_{video_id}.mp4")
        
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

        # Enhanced yt-dlp options with better error handling
        ydl_opts = {
            'outtmpl': temp_file_template,
            'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
            'merge_output_format': 'mp4',
            'socket_timeout': 60,
            'progress_hooks': [progress_hook],
            'nooverwrites': True,  # Don't overwrite existing files
            'retries': 3,  # Retry failed downloads
            'fragment_retries': 3,  # Retry failed fragments
            'file_access_retries': 3,  # Retry failed file access
            'concurrent_fragment_downloads': 5,  # Download multiple fragments concurrently
        }

        logger.info(f"Starting YouTube download with progress hooks...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([yt_url])

        # Wait a moment to ensure file operations are complete
        time.sleep(0.5)

        # Check if the file exists with a different extension and rename if needed
        if not os.path.exists(final_temp_file):
            import glob
            pattern = os.path.join(upload_dir, f"temp_video_{video_id}.*")
            files = glob.glob(pattern)
            if files:
                # Sort by modification time and take the most recent
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                most_recent = files[0]
                if most_recent != final_temp_file:
                    # Try to rename, but don't fail if it's already correct
                    try:
                        os.rename(most_recent, final_temp_file)
                    except OSError:
                        # If rename fails, use the existing file
                        final_temp_file = most_recent

        # Update status to show compression starting
        if status_callback:
            status_callback('downloading', 16, 'Compressing video...')

        compressed_path = os.path.join(upload_dir, f"{video_id}.mp4")
        
        # Retry mechanism for file operations
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if os.path.exists(final_temp_file):
                    # Use CPU encoding for maximum compatibility
                    # GPU encoding (h264_nvenc) causes issues with older drivers
                    cmd = [
                        "ffmpeg", "-i", final_temp_file,
                        "-vcodec", "libx264", "-crf", "28",
                        "-preset", "fast",
                        "-acodec", "aac", "-b:a", "96k",
                        compressed_path
                    ]
                    logger.info("Using CPU encoding for YouTube video compression")
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

                    if result.returncode != 0:
                        raise RuntimeError(f"FFmpeg compression failed: {result.stderr}")
                    break  # Success, exit retry loop
                else:
                    raise FileNotFoundError(f"Temporary file not found: {final_temp_file}")
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Final attempt failed
                    if isinstance(e, subprocess.TimeoutExpired):
                        if status_callback:
                            status_callback('error', 0, 'Compression timed out')
                        raise RuntimeError("FFmpeg compression timed out")
                    else:
                        raise
            finally:
                # Try to clean up temp file with retry mechanism
                if os.path.exists(final_temp_file):
                    for cleanup_attempt in range(3):
                        try:
                            os.remove(final_temp_file)
                            break
                        except OSError as e:
                            if cleanup_attempt < 2:
                                logger.warning(f"Failed to remove temp file (attempt {cleanup_attempt + 1}): {e}")
                                time.sleep(1)
                            else:
                                logger.warning(f"Failed to remove temp file after 3 attempts: {e}")

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
