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
    """Handle video upload or YouTube download with proper error handling."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure upload directory exists and is writable
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
            
            # Check if file is readable
            if not hasattr(f, 'read') or not callable(getattr(f, 'read', None)):
                raise ValueError("File object is not readable")
            
            # Get file size without seeking (to avoid closing the file)
            # Flask file objects have a content_length property
            file_size = f.content_length if hasattr(f, 'content_length') else 0
            
            # If content_length is not available, we'll check after saving
            if file_size == 0:
                logger.info(f"Processing uploaded file: {f.filename} (size unknown)")
            else:
                if file_size > 100 * 1024 * 1024:  # 100MB limit
                    raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB (max 100MB)")
                logger.info(f"Processing uploaded file: {f.filename} ({file_size / (1024*1024):.1f}MB)")
            
            path = os.path.join(upload_dir, f"{video_id}_{f.filename}")
            
            # Save file in chunks to avoid memory issues
            with open(path, 'wb') as out_file:
                chunk_size = 8192  # 8KB chunks
                total_written = 0
                
                # Read and write file in chunks
                try:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        total_written += len(chunk)
                        
                        # Log progress every 1MB
                        if total_written % (1024 * 1024) == 0:
                            logger.info(f"Written {total_written / (1024*1024):.1f}MB")
                            
                except Exception as e:
                    logger.error(f"Error reading file: {e}")
                    raise ValueError(f"Failed to read uploaded file: {str(e)}")
            
            # Now check the actual file size after saving
            actual_file_size = os.path.getsize(path)
            if actual_file_size > 100 * 1024 * 1024:  # 100MB limit
                os.remove(path)  # Clean up the oversized file
                raise ValueError(f"File too large: {actual_file_size / (1024*1024):.1f}MB (max 100MB)")
            
            logger.info(f"File saved successfully to: {path} ({actual_file_size / (1024*1024):.1f}MB)")
            
            # Validate the saved video file
            try:
                import cv2
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise ValueError("Saved file is not a valid video file")
                
                # Check if we can read at least one frame
                ret, frame = cap.read()
                if not ret:
                    raise ValueError("Cannot read frames from saved video file")
                
                cap.release()
                logger.info("Video file validation successful")
                
            except ImportError:
                logger.warning("OpenCV not available, skipping video validation")
            except Exception as e:
                logger.error(f"Video validation failed: {e}")
                # Remove the invalid file
                if os.path.exists(path):
                    os.remove(path)
                raise ValueError(f"Invalid video file: {e}")
            
            # Ensure file handle is properly closed
            try:
                f.close()
            except:
                pass  # File might already be closed
            
            return path, video_id
            
        # Check for YouTube URL
        elif 'yt_url' in request.form and request.form['yt_url'].strip():
            yt_url = request.form['yt_url'].strip()
            logger.info(f"Processing YouTube URL: {yt_url}")
            
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

            # Compress with ffmpeg for OCR-friendly video
            compressed_path = os.path.join(upload_dir, f"{video_id}.mp4")
            logger.info("Compressing video with FFmpeg...")
            
            result = subprocess.run([
                "ffmpeg", "-i", temp_file,
                "-vcodec", "libx264", "-crf", "28",
                "-preset", "fast",
                "-acodec", "aac", "-b:a", "96k",
                compressed_path
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg compression failed: {result.stderr}")

            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            logger.info(f"YouTube video processed successfully: {compressed_path}")
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
        # Ensure upload directory exists and is writable
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)
            logger.info(f"Created upload directory: {upload_dir}")
        
        if not os.access(upload_dir, os.W_OK):
            raise ValueError(f"Upload directory is not writable: {upload_dir}")
        
        video_id = str(uuid.uuid4())
        logger.info(f"Processing video with ID: {video_id}")
        
        # Check for uploaded video file data
        if video_file_data and video_file_data.get('content'):
            filename = video_file_data['filename']
            content = video_file_data['content']
            content_type = video_file_data.get('content_type', 'video/mp4')
            
            logger.info(f"Processing uploaded file data: {filename} ({len(content)} bytes)")
            logger.info(f"Data types in function - filename: {type(filename)}, content: {type(content)}, content_type: {type(content_type)}")
            
            # Validate data types
            if not isinstance(filename, str):
                raise ValueError(f"Invalid filename type: {type(filename)}")
            if not isinstance(content, bytes):
                raise ValueError(f"Invalid content type: {type(content)}")
            if not isinstance(content_type, str):
                raise ValueError(f"Invalid content_type: {type(content_type)}")
            
            # Check file size
            if len(content) > 100 * 1024 * 1024:  # 100MB limit
                raise ValueError(f"File too large: {len(content) / (1024*1024):.1f}MB (max 100MB)")
            
            path = os.path.join(upload_dir, f"{video_id}_{filename}")
            logger.info(f"Will save file to: {path}")
            
            # Save file content directly using a completely different approach
            try:
                # Use binary write mode and write the entire content at once
                with open(path, 'wb') as out_file:
                    # Write content in a single operation to avoid any seek issues
                    out_file.write(content)
                    out_file.flush()  # Ensure all data is written
                    os.fsync(out_file.fileno())  # Force sync to disk
                    logger.info(f"File content written to disk successfully")
                
                # Verify the file was written correctly
                if not os.path.exists(path):
                    raise ValueError("File was not created")
                
                actual_size = os.path.getsize(path)
                if actual_size != len(content):
                    raise ValueError(f"File size mismatch: expected {len(content)}, got {actual_size}")
                
            except Exception as e:
                logger.error(f"Failed to write file to disk: {e}")
                # Clean up any partial file
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
                raise ValueError(f"Failed to save file: {str(e)}")
            
            logger.info(f"File saved successfully to: {path} ({len(content) / (1024*1024):.1f}MB)")
            
            # Validate the saved video file
            try:
                import cv2
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise ValueError("Saved file is not a valid video file")
                
                # Check if we can read at least one frame
                ret, frame = cap.read()
                if not ret:
                    raise ValueError("Cannot read frames from saved video file")
                
                cap.release()
                logger.info("Video file validation successful")
                
            except ImportError:
                logger.warning("OpenCV not available, skipping video validation")
            except Exception as e:
                logger.error(f"Video validation failed: {e}")
                # Remove the invalid file
                if os.path.exists(path):
                    os.remove(path)
                raise ValueError(f"Invalid video file: {e}")
            
            return path, video_id
            
        # Check for YouTube URL
        elif yt_url_data and yt_url_data.strip():
            yt_url = yt_url_data.strip()
            logger.info(f"Processing YouTube URL: {yt_url}")
            
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

            # Compress with ffmpeg for OCR-friendly video
            compressed_path = os.path.join(upload_dir, f"{video_id}.mp4")
            logger.info("Compressing video with FFmpeg...")
            
            result = subprocess.run([
                "ffmpeg", "-i", temp_file,
                "-vcodec", "libx264", "-crf", "28",
                "-preset", "fast",
                "-acodec", "aac", "-b:a", "96k",
                compressed_path
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg compression failed: {result.stderr}")

            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            logger.info(f"YouTube video processed successfully: {compressed_path}")
            return compressed_path, video_id
            
        else:
            raise ValueError("No video file data or YouTube URL provided")
            
    except Exception as e:
        logger.error(f"Error in handle_video_upload_or_download_from_data: {str(e)}")
        raise

def handle_youtube_download(yt_url, upload_dir):
    """Handle YouTube download only."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure upload directory exists and is writable
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)
            logger.info(f"Created upload directory: {upload_dir}")
        
        if not os.access(upload_dir, os.W_OK):
            raise ValueError(f"Upload directory is not writable: {upload_dir}")
        
        video_id = str(uuid.uuid4())
        logger.info(f"Processing YouTube URL with ID: {video_id}")
        
        if not yt_url or not yt_url.strip():
            raise ValueError("No YouTube URL provided")
        
        yt_url = yt_url.strip()
        logger.info(f"Processing YouTube URL: {yt_url}")
        
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

        # Compress with ffmpeg for OCR-friendly video
        compressed_path = os.path.join(upload_dir, f"{video_id}.mp4")
        logger.info("Compressing video with FFmpeg...")
        
        result = subprocess.run([
            "ffmpeg", "-i", temp_file,
            "-vcodec", "libx264", "-crf", "28",
            "-preset", "fast",
            "-acodec", "aac", "-b:a", "96k",
            compressed_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg compression failed: {result.stderr}")

        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        logger.info(f"YouTube video processed successfully: {compressed_path}")
        return compressed_path, video_id
        
    except Exception as e:
        logger.error(f"Error in handle_youtube_download: {str(e)}")
        raise
