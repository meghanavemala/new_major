import os
import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import re
from flask import (
    Flask, render_template, request, jsonify, 
    send_from_directory, session, Response, stream_with_context
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Import utility modules
from utils.downloader import is_youtube_url, handle_youtube_download
from utils.transcriber import transcribe_video, SUPPORTED_LANGUAGES as TRANSCRIBE_LANGS
from utils.keyframes import extract_keyframes, get_keyframe_text_summary, extract_keyframes_for_time_range
from utils.clustering import cluster_segments, extract_keywords
from utils.summarizer import summarize_cluster
from utils.tts import text_to_speech, SUPPORTED_VOICES, get_available_voices
from utils.video_maker import make_summary_video, SUPPORTED_RESOLUTIONS, select_keyframes_for_topic
from utils.translator import (
    translate_text, translate_segments, get_available_languages,
    detect_language, LANGUAGE_MAPPINGS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(filename='app.log',encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'UPLOAD_DIR': 'uploads',
    'PROCESSED_DIR': 'processed',
    'MAX_CONTENT_LENGTH': 40 * 1024 * 1024,  # 40MB
    'MAX_VIDEO_DURATION': 40 * 60,  # 40 minutes in seconds
    'SUPPORTED_EXTENSIONS': {'mp4', 'avi', 'mov', 'mkv', 'webm'},
    'DEFAULT_LANGUAGE': 'en',
    'DEFAULT_VOICE': 'en-US-Standard-C',
    'DEFAULT_RESOLUTION': '480p',
    'ENABLE_DARK_MODE': True,
}

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = CONFIG['MAX_CONTENT_LENGTH']
# Ensure directories exist
for dir_path in [CONFIG['UPLOAD_DIR'], CONFIG['PROCESSED_DIR']]:
    os.makedirs(dir_path, exist_ok=True)

# Global state for tracking processing status
processing_status = {}

def get_processing_status(video_id: str) -> Dict:
    """Get the current processing status for a video."""
    return processing_status.get(video_id, {
        'status': 'not_started',
        'progress': 0,
        'message': 'Processing not started',
        'error': None
    })

def update_processing_status(video_id: str, status: str, progress: int, message: str, error: str = None) -> None:
    """Update the processing status for a video."""
    if video_id not in processing_status:
        processing_status[video_id] = {}
    
    processing_status[video_id].update({
        'status': status,
        'progress': progress,
        'message': message,
        'error': error,
        'last_updated': time.time()
    })

@app.route('/test')
def test():
    """Simple test route to verify the app is working."""
    return jsonify({'status': 'ok', 'message': 'Flask app is running'})

@app.route('/')
def index():
    """Render the main page with language and voice options."""
    # Get available voices grouped by language
    voices_by_lang = {}
    available_voices = get_available_voices()
    
    # voices_by_lang will have the structure: {lang_code: {'name': str, 'voices': list}}
    for lang_code, voices in available_voices.items():
        voices_by_lang[lang_code] = {
            'name': TRANSCRIBE_LANGS.get(lang_code, 'Unknown'),
            'voices': voices
        }
    
    # Sort languages by name
    sorted_languages = sorted(
        [(code, data['name']) for code, data in voices_by_lang.items()],
        key=lambda x: x[1]
    )
    
    # Get default language and voice
    default_lang = request.cookies.get('preferred_language', CONFIG['DEFAULT_LANGUAGE'])
    default_voice = request.cookies.get('preferred_voice', CONFIG['DEFAULT_VOICE'])
    dark_mode = request.cookies.get('dark_mode', 'true') == 'true'
    
    return render_template(
        'index.html',
        languages=sorted_languages,
        voices_by_lang=voices_by_lang,
        default_language=default_lang,
        default_voice=default_voice,
        dark_mode=dark_mode,
        max_file_size=CONFIG['MAX_CONTENT_LENGTH'],
        max_duration=CONFIG['MAX_VIDEO_DURATION'],
        supported_extensions=', '.join(CONFIG['SUPPORTED_EXTENSIONS']),
        resolutions=SUPPORTED_RESOLUTIONS,
        default_resolution=CONFIG['DEFAULT_RESOLUTION']
    )

@app.route('/api/process', methods=['POST'])
def process():
    """Handle video processing request with enhanced language support."""
    # Get request data
    source_language = request.form.get('source_language', 'auto')
    target_language = request.form.get('target_language', 'english')
    voice = request.form.get('voice', CONFIG['DEFAULT_VOICE'])
    resolution = request.form.get('resolution', CONFIG['DEFAULT_RESOLUTION'])
    summary_length = request.form.get('summary_length', 'medium')
    enable_ocr = request.form.get('enable_ocr', 'true').lower() == 'true'
    
    # Validate input (either video file or YouTube URL must be provided)
    video_file = request.files.get('video_file')
    yt_url = request.form.get('yt_url', '').strip()
    
    # Debug logging
    logger.info(f"Received video_file: {video_file}")
    logger.info(f"Received yt_url: {yt_url}")
    logger.info(f"All form fields: {list(request.form.keys())}")
    logger.info(f"All files: {list(request.files.keys())}")
    logger.info(f"Request content type: {request.content_type}")
    logger.info(f"Request content length: {request.content_length}")
    
    # Check if video_file exists and has content
    if video_file:
        logger.info(f"Video file details - filename: {video_file.filename}, content_type: {video_file.content_type}")
        if video_file.filename:
            logger.info(f"Video file has filename: {video_file.filename}")
        else:
            logger.info("Video file exists but has no filename")
    else:
        logger.info("No video_file in request.files")
    
    if not video_file and not yt_url:
        logger.error("Validation failed: No video file or YouTube URL provided")
        return jsonify({'error': 'Please provide either a video file or a YouTube URL'}), 400
    
    # Validate languages
    if source_language != 'auto' and source_language not in TRANSCRIBE_LANGS:
        return jsonify({'error': f'Unsupported source language: {source_language}'}), 400
    
    if target_language not in LANGUAGE_MAPPINGS:
        return jsonify({'error': f'Unsupported target language: {target_language}'}), 400
    
    # Generate a unique ID for this processing job
    video_id = str(int(time.time()))
    
    # Initialize processing status
    update_processing_status(video_id, 'initializing', 0, 'Initializing processing...')
    
    # COMPLETELY NEW APPROACH: Save file to disk immediately and pass the path
    video_file_path = None
    yt_url_data = None
    
    if 'video_file' in request.files and request.files['video_file']:
        f = request.files['video_file']
        if f and f.filename:
            try:
                # Generate a temporary file path
                import tempfile
                import uuid
                temp_id = str(uuid.uuid4())
                temp_filename = f"{temp_id}_{f.filename}"
                temp_path = os.path.join(CONFIG['UPLOAD_DIR'], temp_filename)
                
                logger.info(f"Saving uploaded file to temporary path: {temp_path}")
                
                # Save file to disk immediately - no memory operations
                with open(temp_path, 'wb') as temp_file:
                    # Read and write in one operation
                    temp_file.write(f.read())
                
                # Verify file was saved
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    video_file_path = temp_path
                    logger.info(f"File saved successfully: {temp_path} ({os.path.getsize(temp_path)} bytes)")
                else:
                    raise ValueError("Failed to save uploaded file")
                
                # Clear file reference immediately
                f = None
                
            except Exception as e:
                logger.error(f"Failed to save uploaded file: {e}")
                # Clean up any partial file
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                raise ValueError(f"Failed to save uploaded file: {str(e)}")
    
    if 'yt_url' in request.form and request.form['yt_url'].strip():
        yt_url_data = str(request.form['yt_url'].strip())
        logger.info(f"YouTube URL provided: {yt_url_data}")
    
    # Verify we have valid data before starting processing
    if not video_file_path and not yt_url_data:
        raise ValueError("No video file or YouTube URL provided")
    
    logger.info("File handling completed successfully, starting background processing...")
    
    # Force cleanup
    import gc
    gc.collect()
    logger.info("Garbage collection completed")
    
    # Start processing in background
    def process_video():
        nonlocal video_id
        try:
            logger.info(f"Starting background processing for video {video_id}")
            logger.info(f"Video file path available: {video_file_path is not None}")
            logger.info(f"YouTube URL data available: {yt_url_data is not None}")
            
            update_processing_status(video_id, 'downloading', 5, 'Downloading video...')
            
            # 1. Process the uploaded video file or YouTube URL
            try:
                if video_file_path:
                    # We already have the file saved, just use it
                    logger.info(f"Using already saved file: {video_file_path}")
                    video_path = video_file_path
                    # Keep the same video_id for consistency
                elif yt_url_data:
                    # Process YouTube URL
                    logger.info("Processing YouTube URL...")
                    
                    # Create a status callback function that updates the processing status
                    def update_download_status(status, progress, message):
                        update_processing_status(video_id, status, progress, message)
                    
                    video_path, downloaded_video_id = handle_youtube_download(
                        yt_url_data, 
                        CONFIG['UPLOAD_DIR'],
                        video_id=video_id,  # Pass the original video_id
                        status_callback=update_download_status  # Pass the status callback
                    )
                    # Use the downloaded video ID for the file path, but keep the original video_id for status updates
                    logger.info(f"YouTube video downloaded to: {video_path}")
                    logger.info(f"Downloaded video ID: {downloaded_video_id}, Processing video ID: {video_id}")
                    
                    # Status is now updated by the callback, no need to manually update here
                else:
                    raise ValueError("No video file or YouTube URL available")
                
                logger.info(f"Video successfully processed: {video_path}")
                logger.info(f"Moving to transcription stage with video_path: {video_path}")
                
            except Exception as e:
                logger.error(f"Failed to process video: {str(e)}")
                logger.error(f"Exception type: {type(e)}")
                logger.error(f"Exception details: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Update status to error so frontend sees the real error
                update_processing_status(
                    video_id,
                    'error',
                    0,
                    'Failed to download/process video',
                    str(e)
                )
                return  # Stop further processing
            
            # 2. Transcribe the video
            logger.info(f"Starting transcription for video: {video_path}")
            logger.info(f"Video file exists: {os.path.exists(video_path)}")
            logger.info(f"Video file size: {os.path.getsize(video_path) if os.path.exists(video_path) else 'N/A'} bytes")
            
            # Test if ffmpeg is available
            try:
                import subprocess
                result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
                logger.info(f"FFmpeg available: {result.returncode == 0}")
                if result.returncode != 0:
                    logger.error(f"FFmpeg error: {result.stderr}")
            except Exception as e:
                logger.error(f"FFmpeg test failed: {e}")
            
            update_processing_status(video_id, 'transcribing', 20, 'Transcribing audio...')
            
            # Use auto-detection if source language is auto
            transcription_language = source_language if source_language != 'auto' else 'english'
            logger.info(f"Transcription language: {transcription_language}")
            
            logger.info(f"Calling transcribe_video function...")
            logger.info(f"Function signature: transcribe_video({video_path}, {CONFIG['PROCESSED_DIR']}, {video_id}, {transcription_language})")
            
            # Test if the function is callable
            logger.info(f"transcribe_video function type: {type(transcribe_video)}")
            logger.info(f"transcribe_video function callable: {callable(transcribe_video)}")
            
            try:
                transcript_path, segments = transcribe_video(
                    video_path, 
                    CONFIG['PROCESSED_DIR'], 
                    video_id,
                    language=transcription_language
                )
                logger.info(f"Transcription completed. Transcript path: {transcript_path}, Segments: {len(segments) if segments else 0}")
            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise ValueError(f"Transcription failed: {str(e)}")
            
            if not segments:
                raise ValueError("No speech detected in the video")
                
            # Auto-detect language if needed
            detected_language = transcription_language
            if source_language == 'auto' and segments:
                sample_text = ' '.join([seg.get('text', '') for seg in segments[:5]])
                detected_lang = detect_language(sample_text)
                if detected_lang:
                    detected_language = detected_lang
                    logger.info(f"Auto-detected language: {detected_language}")
            
            # Translate segments if source and target languages are different
            if detected_language != target_language:
                update_processing_status(video_id, 'translating', 35, 'Translating content...')
                segments = translate_segments(segments, detected_language, target_language)
                logger.info(f"Translated from {detected_language} to {target_language}")
            
            # 3. Extract key frames with OCR
            update_processing_status(video_id, 'extracting', 40, 'Extracting key frames...')
            
            # Prepare OCR languages
            ocr_languages = [detected_language, target_language]
            if 'english' not in ocr_languages:
                ocr_languages.append('english')
            
            keyframes_dir = extract_keyframes(
                video_path, 
                CONFIG['PROCESSED_DIR'], 
                video_id,
                target_resolution=resolution,
                ocr_languages=ocr_languages,
                enable_ocr=enable_ocr
            )
            
            # Get OCR text summary for additional context
            ocr_summary = {}
            if enable_ocr and keyframes_dir:
                ocr_summary = get_keyframe_text_summary(keyframes_dir)
                if ocr_summary.get('high_confidence_text'):
                    logger.info(f"Extracted OCR text: {len(ocr_summary['high_confidence_text'])} characters")
            
            # 4. Cluster transcript into topics
            update_processing_status(video_id, 'clustering', 55, 'Analyzing content...')
            
            # Enhance segments with OCR context if available
            if enable_ocr and ocr_summary.get('high_confidence_text'):
                # Add OCR context to the first few segments for better clustering
                ocr_context = ocr_summary['high_confidence_text'][:500]  # Limit context
                if segments and ocr_context.strip():
                    segments[0]['text'] += f" [Visual context: {ocr_context}]"
            
            # Improved clustering with better parameters
            n_clusters = min(6, max(3, len(segments) // 8))  # Better cluster count
            
            clustered = cluster_segments(
                segments,
                language=target_language,
                method='kmeans',  # More efficient than LDA for this use case
                n_clusters=n_clusters,
                min_cluster_size=3,  # Ensure meaningful clusters
                similarity_threshold=0.7  # Higher threshold for better separation
            )
            
            # 5. Process each cluster
            summaries = []
            tts_paths = []
            cluster_keywords = []
            
            for cluster_id, cluster in enumerate(clustered):
                # Get keywords for this cluster
                cluster_texts = [seg['text'] for seg in cluster]
                keywords = extract_keywords(
                    cluster_texts, 
                    n_keywords=5,
                    language=target_language
                )
                cluster_keywords.append(keywords)
                
                # Generate summary in target language
                update_processing_status(
                    video_id, 
                    'summarizing', 
                    60 + (20 * cluster_id // len(clustered)),
                    f'Generating summary for topic {cluster_id + 1}...'
                )
                summary = summarize_cluster(
                    cluster,
                    language=target_language,
                    use_extractive=True
                )
                summaries.append(summary)
                
                # Generate TTS audio in target language
                tts_path = text_to_speech(
                    text=summary,
                    output_dir=CONFIG['PROCESSED_DIR'],
                    video_id=video_id,
                    cluster_id=cluster_id,
                    voice=voice,
                    language=target_language
                )
                tts_paths.append(tts_path)
            
            # 6. Create summary videos with better organization
            summary_videos = []
            
            # Create structured folders for this video
            video_base_dir = os.path.join(CONFIG['PROCESSED_DIR'], video_id)
            os.makedirs(video_base_dir, exist_ok=True)
            
            # Create subdirectories
            keyframes_dir_organized = os.path.join(video_base_dir, 'keyframes')
            audio_dir = os.path.join(video_base_dir, 'audio')
            videos_dir = os.path.join(video_base_dir, 'videos')
            os.makedirs(keyframes_dir_organized, exist_ok=True)
            os.makedirs(audio_dir, exist_ok=True)
            os.makedirs(videos_dir, exist_ok=True)
            
            # Load and organize keyframe metadata
            with open(os.path.join(keyframes_dir, 'keyframes_metadata.json'), 'r', encoding='utf-8') as f:
                keyframe_metadata = json.load(f)['keyframes']
            
            # Copy keyframes to organized directory and update paths
            for kf in keyframe_metadata:
                if os.path.exists(kf['filepath']):
                    new_path = os.path.join(keyframes_dir_organized, os.path.basename(kf['filepath']))
                    import shutil
                    shutil.copy2(kf['filepath'], new_path)
                    kf['filepath'] = new_path
            
            # Save organized metadata
            organized_metadata_path = os.path.join(video_base_dir, 'metadata.json')
            with open(organized_metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'video_id': video_id,
                    'keyframes': keyframe_metadata,
                    'segments': segments,
                    'clusters': [{'segments': cluster, 'keywords': keywords} for cluster, keywords in zip(clustered, cluster_keywords)]
                }, f, indent=2, ensure_ascii=False)
            
            # Process each cluster with more keyframes
            for cluster_id, (tts_path, keywords, cluster) in enumerate(zip(tts_paths, cluster_keywords, clustered)):
                update_processing_status(
                    video_id,
                    'rendering',
                    80 + (15 * cluster_id // len(tts_paths)),
                    f'Creating video for topic {cluster_id + 1}...'
                )
                
                # Get topic start/end times from cluster segments
                topic_start = min(seg['start'] for seg in cluster)
                topic_end = max(seg['end'] for seg in cluster)
                
                # Extract more keyframes for this topic (at least 30-40 for smooth video feel)
                topic_keyframes = select_keyframes_for_topic(keyframe_metadata, topic_start, topic_end)
                
                # If we don't have enough keyframes, extract more from the video for this time range
                if len(topic_keyframes) < 30:
                    additional_keyframes = extract_keyframes_for_time_range(
                        video_path, topic_start, topic_end, 30 - len(topic_keyframes)
                    )
                    topic_keyframes.extend(additional_keyframes)
                
                # Move TTS audio to organized directory
                audio_filename = f"topic_{cluster_id}_summary.wav"
                organized_audio_path = os.path.join(audio_dir, audio_filename)
                import shutil
                shutil.copy2(tts_path, organized_audio_path)
                
                # Create video
                output_path = os.path.join(videos_dir, f"topic_{cluster_id}_summary")
                video_out = make_summary_video(
                    keyframes=topic_keyframes,
                    tts_audio_path=organized_audio_path,
                    output_path=output_path,
                    target_width=854,
                    fps=30
                )
                
                if video_out:
                    # Store the relative path from processed directory
                    video_filename = os.path.relpath(video_out, CONFIG['PROCESSED_DIR'])
                    summary_videos.append(video_filename)
                    
                    # Update metadata with topic-specific info
                    topic_metadata = {
                        'topic_id': cluster_id,
                        'start_time': topic_start,
                        'end_time': topic_end,
                        'summary': summaries[cluster_id],
                        'keywords': keywords,
                        'keyframes_count': len(topic_keyframes),
                        'audio_file': audio_filename,
                        'video_file': video_filename
                    }
                    
                    # Save topic metadata
                    topic_metadata_path = os.path.join(video_base_dir, f'topic_{cluster_id}_metadata.json')
                    with open(topic_metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(topic_metadata, f, indent=2, ensure_ascii=False)
                else:
                    logger.error(f"Failed to create video for topic {cluster_id}")
                    summary_videos.append(None)
            
            # 7. Finalize with enhanced metadata
            result = {
                'video_id': video_id,
                'summaries': summaries,
                'keywords': cluster_keywords,
                'summary_videos': summary_videos,
                'source_language': detected_language,
                'target_language': target_language,
                'ocr_enabled': enable_ocr,
                'ocr_summary': ocr_summary if enable_ocr else {},
                'total_topics': len(clustered),
                'processing_stats': {
                    'segments_count': len(segments),
                    'keyframes_count': len(os.listdir(keyframes_dir)) if keyframes_dir and os.path.exists(keyframes_dir) else 0,
                    'translation_used': detected_language != target_language
                },
                'status': 'completed',
                'progress': 100,
                'message': 'Processing complete!'
            }
            
            # Update status with result
            processing_status[video_id].update(result)
            
            # Mark processing as complete
            update_processing_status(
                video_id,
                'completed',
                100,
                'Processing complete!',
                None
            )
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}\n{traceback.format_exc()}")
            update_processing_status(
                video_id,
                'error',
                0,
                'An error occurred during processing',
                str(e)
            )
        finally:
            # Cancel all timers if they exist
            if 'timeout_timer' in locals():
                timeout_timer.cancel()
            if 'watchdog_timer' in locals():
                watchdog_timer.cancel()
    
    # Start processing in background with timeout
    from threading import Thread, Timer
    
    def timeout_handler():
        logger.error(f"Processing timeout for video {video_id}")
        update_processing_status(
            video_id,
            'error',
            0,
            'Processing timeout - taking too long',
            'Processing exceeded maximum time limit'
        )
    
    # Set a timeout of 30 minutes for the entire process
    timeout_timer = Timer(30 * 60, timeout_handler)  # 30 minutes
    timeout_timer.start()
    
    # Add a watchdog timer to check if processing is stuck
    def watchdog_check():
        current_status = get_processing_status(video_id)
        if current_status.get('status') == 'downloading' and current_status.get('progress') == 5:
            # If stuck at 5% for more than 5 minutes, force error
            logger.error(f"Processing stuck at 5% for video {video_id}, forcing error")
            update_processing_status(
                video_id,
                'error',
                0,
                'Processing stuck - forcing error',
                'Processing got stuck at download stage'
            )
    
    watchdog_timer = Timer(5 * 60, watchdog_check)  # 5 minutes
    watchdog_timer.start()
    
    thread = Thread(target=process_video)
    thread.daemon = True
    thread.start()
    
    # Return initial response with video ID
    return jsonify({
        'video_id': video_id,
        'status': 'processing',
        'progress': 0,
        'message': 'Processing started'
    })

@app.route('/api/test-upload', methods=['POST'])
def test_upload():
    """Test endpoint to debug file upload issues."""
    logger.info(f"Test upload - files: {list(request.files.keys())}")
    logger.info(f"Test upload - form: {list(request.form.keys())}")
    logger.info(f"Test upload - content type: {request.content_type}")
    logger.info(f"Test upload - content length: {request.content_length}")
    
    if 'video_file' in request.files:
        f = request.files['video_file']
        if f and f.filename:
            # Get file size without seeking to avoid closing the file
            try:
                # Use content_length if available, otherwise read a small chunk to test
                if hasattr(f, 'content_length') and f.content_length:
                    file_size = f.content_length
                else:
                    # Read a small chunk to test if file is readable
                    f.seek(0)
                    test_chunk = f.read(1024)  # Read 1KB
                    file_size = len(test_chunk)
                    if test_chunk:
                        # File is readable, we can estimate size
                        file_size = "Readable (size unknown)"
                    else:
                        file_size = "Empty file"
                
                logger.info(f"Test upload - received file: {f.filename}, size: {file_size}")
                return jsonify({
                    'success': True, 
                    'filename': f.filename,
                    'size': file_size,
                    'content_type': getattr(f, 'content_type', 'unknown')
                })
            except Exception as e:
                logger.error(f"Test upload - error reading file: {e}")
                return jsonify({'success': False, 'error': f'File read error: {str(e)}'})
        else:
            logger.warning("Test upload - video_file exists but is empty or has no filename")
            return jsonify({'success': False, 'error': 'File is empty or has no filename'})
    else:
        logger.warning("Test upload - no video_file in request.files")
        return jsonify({'success': False, 'error': 'No video_file received in request.files'})

@app.route('/api/status/<video_id>')
def get_status(video_id: str):
    """Get the current status of a processing job."""
    status = get_processing_status(video_id)
    logger.info(f"Status request for {video_id}: {status}")
    return jsonify(status)

@app.route('/api/summary/<video_id>')
def get_summary(video_id: str):
    """Get the summary data for a processed video."""
    summary_path = os.path.join(CONFIG['PROCESSED_DIR'], f"{video_id}_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Summary not found'}), 404

@app.route('/api/stream/<video_id>/<int:cluster_id>')
def stream_video(video_id: str, cluster_id: int):
    """Stream a summary video."""
    video_path = os.path.join(CONFIG['PROCESSED_DIR'], f"{video_id}_summary_{cluster_id}.mp4")
    
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404
    
    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_from_directory(
            CONFIG['PROCESSED_DIR'],
            f"{video_id}_summary_{cluster_id}.mp4",
            as_attachment=False,
            mimetype='video/mp4'
        )
    start,end=0,0
    # Handle byte range requests for streaming
    def generate():
        nonlocal start,end
        with open(video_path, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            start = 0
            end = file_size - 1
            
            # Parse range header
            range_header = request.headers.get('Range')
            if range_header:
                range_match = re.search(r'bytes=(\d*)-(\d*)', range_header)
                if range_match.group(1):
                    start = int(range_match.group(1))
                if range_match.group(2):
                    end = int(range_match.group(2))
            
            chunk_size = 1024 * 1024  # 1MB chunks
            f.seek(start)
            
            while start <= end:
                chunk = f.read(min(chunk_size, end - start + 1))
                if not chunk:
                    break
                yield chunk
                start += len(chunk)
    
    # Send response with appropriate headers
    file_size = os.path.getsize(video_path)
    response = Response(
        stream_with_context(generate()),
        206,  # Partial Content
        mimetype='video/mp4',
        direct_passthrough=True
    )
    
    response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
    response.headers.add('Accept-Ranges', 'bytes')
    response.headers.add('Content-Length', str(end - start + 1))
    
    return response

@app.route('/processed/<path:path>')
def send_processed_file(path):
    """Serve processed files with proper caching headers."""
    response = send_from_directory(CONFIG['PROCESSED_DIR'], path)
    # Cache for 1 day
    response.headers['Cache-Control'] = 'public, max-age=86400'
    return response

@app.route('/srt/<video_id>')
def get_srt(video_id: str):
    """Get SRT subtitles for a video."""
    srt_path = os.path.join(CONFIG['PROCESSED_DIR'], f"{video_id}.srt")
    if os.path.exists(srt_path):
        response = send_from_directory(CONFIG['PROCESSED_DIR'], f"{video_id}.srt")
        response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        return response
    return jsonify({'error': 'Subtitles not found'}), 404

@app.route('/keyframes/<video_id>')
def get_keyframes(video_id: str):
    """Get list of keyframes for a video."""
    keyframes_dir = os.path.join(CONFIG['PROCESSED_DIR'], f"{video_id}_keyframes")
    if os.path.exists(keyframes_dir):
        try:
            files = sorted([f for f in os.listdir(keyframes_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            return jsonify({
                'keyframes': [f"/processed/{video_id}_keyframes/{f}" for f in files]
            })
        except Exception as e:
            logger.error(f"Error listing keyframes: {e}")
            return jsonify({'error': 'Error listing keyframes'}), 500
    return jsonify({'keyframes': []})

def cleanup_old_files():
    """Clean up old temporary files."""
    try:
        now = time.time()
        max_age = 24 * 3600  # 1 day in seconds
        
        for dir_path in [CONFIG['UPLOAD_DIR'], CONFIG['PROCESSED_DIR']]:
            if not os.path.exists(dir_path):
                continue
                
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                try:
                    # Delete files older than max_age
                    if os.path.isfile(file_path):
                        file_age = now - os.path.getmtime(file_path)
                        if file_age > max_age:
                            os.remove(file_path)
                            logger.info(f"Deleted old file: {file_path}")
                    # Clean up old temporary directories
                    elif os.path.isdir(file_path) and file_path.endswith('_temp'):
                        import shutil
                        shutil.rmtree(file_path, ignore_errors=True)
                        logger.info(f"Deleted temp directory: {file_path}")
                except Exception as e:
                    logger.error(f"Error cleaning up {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error in cleanup_old_files: {e}")

if __name__ == '__main__':
    # Ensure directories exist
    for dir_path in [CONFIG['UPLOAD_DIR'], CONFIG['PROCESSED_DIR']]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Set up cleanup job
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=cleanup_old_files, trigger='interval', hours=1)
    scheduler.start()
    
    # Start the Flask app
    try:
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), debug=False, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
