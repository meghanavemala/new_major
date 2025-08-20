import os
import json
import time
import logging
import traceback
import subprocess
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import re

import cv2
import numpy as np
from threading import Thread, Timer, Event
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.effects import normalize
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.config import change_settings

# Configure moviepy to use ffmpeg
change_settings({"FFMPEG_BINARY": "ffmpeg"})

# Local imports
from utils.video_maker import make_summary_video, add_subtitle_to_frame
from utils.keyframes import (
    extract_keyframes,
    extract_keyframes_for_time_range,
    get_keyframe_text_summary,
)
from utils.transcriber import transcribe_video
from utils.translator import translate_segments
from utils.topic_analyzer import analyze_topic_segments
from utils.summarizer import summarize_cluster
from utils.tts import text_to_speech

from utils.downloader import is_youtube_url, handle_youtube_download
from utils.transcriber import SUPPORTED_LANGUAGES as TRANSCRIBE_LANGS
from utils.clustering import cluster_segments, extract_keywords
from utils.topic_analyzer import get_keyframes_for_topic, calculate_keyframe_distribution
from utils.tts import SUPPORTED_VOICES, get_available_voices
from utils.video_maker import SUPPORTED_RESOLUTIONS, select_keyframes_for_topic, make_summary_video
from utils.translator import (
    translate_text,
    get_available_languages,
    detect_language,
    LANGUAGE_MAPPINGS,
)

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    session,
    Response,
    stream_with_context,
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
)
logger = logging.getLogger(__name__)


# Helper functions for video processing
def get_video_duration(video_path: str) -> float:
    """Get the duration of a video file in seconds."""
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception as e:
        logger.error(f"Error getting video duration for {video_path}: {str(e)}")
        return 0.0


def extract_audio(video_path: str, output_path: str) -> bool:
    """Extract audio from a video file."""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_path, logger=None)
        video.close()
        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"Error extracting audio from {video_path}: {str(e)}")
        return False


def create_transition_video(
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    output_path: str,
    duration: float = 1.5,
    transition_type: str = "fade",
    resolution: Tuple[int, int] = (1280, 720),
) -> bool:
    """Create a transition video between two frames."""
    try:
        # Ensure frames match target resolution
        start_frame = cv2.resize(start_frame, (resolution[0], resolution[1]))
        end_frame = cv2.resize(end_frame, (resolution[0], resolution[1]))

        temp_files: List[str] = []
        fps = 30
        frame_count = max(1, int(duration * fps))

        for i in range(frame_count):
            alpha = i / (frame_count - 1) if frame_count > 1 else 1.0
            if transition_type == "fade":
                blended = cv2.addWeighted(start_frame, 1 - alpha, end_frame, alpha, 0)
            elif transition_type == "slide":
                width = resolution[0]
                offset = int(alpha * width)
                blended = np.zeros_like(start_frame)
                if offset < width:
                    blended[:, : width - offset] = start_frame[:, offset:]
                    blended[:, width - offset :] = end_frame[:, :offset]
                else:
                    blended[:] = end_frame
            else:
                # Default to fade if unknown type
                blended = cv2.addWeighted(start_frame, 1 - alpha, end_frame, alpha, 0)

            temp_frame_path = os.path.join(
                os.path.dirname(output_path), f"temp_frame_{i:04d}.jpg"
            )
            cv2.imwrite(temp_frame_path, blended)
            temp_files.append(temp_frame_path)

        if temp_files:
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(os.path.dirname(temp_files[0]), "temp_frame_%04d.jpg"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "23",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg error creating transition: {result.stderr}")
                return False

        # Cleanup temp frames
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception:
                pass

        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"Error creating transition video: {str(e)}")
        return False


def capture_thumbnail(video_path: str, output_path: str, time_sec: float = 5.0) -> bool:
    """Capture a thumbnail from a video at the specified time."""
    try:
        video = VideoFileClip(video_path)
        time_sec = float(min(max(0, time_sec), max(0, video.duration - 0.1)))
        frame = video.get_frame(time_sec)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, frame_bgr)
        video.close()
        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"Error capturing thumbnail: {str(e)}")
        return False


def get_video_frame(video_path: str, frame_index: int = 0) -> Optional[np.ndarray]:
    """Get a specific frame from a video."""
    try:
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index < 0:
            frame_index = max(0, total_frames + frame_index)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video.read()
        video.release()
        return frame if ret else None
    except Exception as e:
        logger.error(f"Error getting video frame: {str(e)}")
        return None


CONFIG: Dict[str, Any] = {
    "UPLOAD_DIR": "uploads",
    "PROCESSED_DIR": "processed",
    "MAX_CONTENT_LENGTH": 100 * 1024 * 1024,
    "MAX_VIDEO_DURATION": 40 * 60,
    "SUPPORTED_EXTENSIONS": {"mp4", "avi", "mov", "mkv", "webm"},
    "DEFAULT_LANGUAGE": "en",
    "DEFAULT_VOICE": "en-US-Standard-C",
    "DEFAULT_RESOLUTION": "480p",
    "ENABLE_DARK_MODE": True,
}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = CONFIG["MAX_CONTENT_LENGTH"]

for dir_path in [CONFIG["UPLOAD_DIR"], CONFIG["PROCESSED_DIR"]]:
    os.makedirs(dir_path, exist_ok=True)

processing_status: Dict[str, Dict[str, Any]] = {}

# Generic stall detection threshold (seconds) for status updates
STALE_STALL_SECONDS = int(os.environ.get("STALE_STALL_SECONDS", "240"))


def get_processing_status(video_id: str) -> Dict[str, Any]:
    return processing_status.get(
        video_id,
        {
            "status": "not_started",
            "progress": 0,
            "message": "Processing not started",
            "error": None,
        },
    )


def update_processing_status(
    video_id: str, status: str, progress: int, message: str, error: Optional[str] = None
) -> None:
    if video_id not in processing_status:
        processing_status[video_id] = {}
    processing_status[video_id].update(
        {
            "status": status,
            "progress": progress,
            "message": message,
            "error": error,
            "last_updated": time.time(),
        }
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Return a clear JSON error when upload exceeds MAX_CONTENT_LENGTH."""
    limit_mb = int(app.config.get("MAX_CONTENT_LENGTH", 0)) // (1024 * 1024)
    return (
        jsonify(
            {
                "error": "File too large",
                "message": f"Max upload size is {limit_mb}MB",
            }
        ),
        413,
    )


@app.route("/test")
def test():
    return jsonify({"status": "ok", "message": "Flask app is running"})


@app.route("/")
def index():
    voices_by_lang: Dict[str, Dict[str, Any]] = {}
    available_voices = get_available_voices()
    for lang_code, voices in available_voices.items():
        voices_by_lang[lang_code] = {
            "name": TRANSCRIBE_LANGS.get(lang_code, "Unknown"),
            "voices": voices,
        }

    sorted_languages = sorted(
        [(code, data["name"]) for code, data in voices_by_lang.items()],
        key=lambda x: x[1],
    )

    default_lang = request.cookies.get("preferred_language", CONFIG["DEFAULT_LANGUAGE"])
    default_voice = request.cookies.get("preferred_voice", CONFIG["DEFAULT_VOICE"])
    dark_mode = request.cookies.get("dark_mode", "true") == "true"

    return render_template(
        "index.html",
        languages=sorted_languages,
        voices_by_lang=voices_by_lang,
        default_language=default_lang,
        default_voice=default_voice,
        dark_mode=dark_mode,
        max_file_size=CONFIG["MAX_CONTENT_LENGTH"],
        max_duration=CONFIG["MAX_VIDEO_DURATION"],
        supported_extensions=", ".join(CONFIG["SUPPORTED_EXTENSIONS"]),
        resolutions=SUPPORTED_RESOLUTIONS,
        default_resolution=CONFIG["DEFAULT_RESOLUTION"],
    )


@app.route("/api/process", methods=["POST"])
def process():
    source_language = request.form.get("source_language", "auto")
    target_language = request.form.get("target_language", "english")
    voice = request.form.get("voice", CONFIG["DEFAULT_VOICE"])
    resolution = request.form.get("resolution", CONFIG["DEFAULT_RESOLUTION"])
    summary_length = request.form.get("summary_length", "medium")
    enable_ocr = request.form.get("enable_ocr", "true").lower() == "true"

    video_file = request.files.get("video_file")
    yt_url = request.form.get("yt_url", "").strip()

    logger.info(f"Received video_file: {video_file}")
    logger.info(f"Received yt_url: {yt_url}")
    logger.info(f"All form fields: {list(request.form.keys())}")
    logger.info(f"All files: {list(request.files.keys())}")
    logger.info(f"Request content type: {request.content_type}")
    logger.info(f"Request content length: {request.content_length}")

    if video_file:
        logger.info(
            f"Video file details - filename: {video_file.filename}, content_type: {video_file.content_type}"
        )
        if video_file.filename:
            logger.info(f"Video file has filename: {video_file.filename}")
        else:
            logger.info("Video file exists but has no filename")
    else:
        logger.info("No video_file in request.files")

    if not video_file and not yt_url:
        logger.error("Validation failed: No video file or YouTube URL provided")
        return (
            jsonify({"error": "Please provide either a video file or a YouTube URL"}),
            400,
        )

    if source_language != "auto" and source_language not in TRANSCRIBE_LANGS:
        return jsonify({"error": f"Unsupported source language: {source_language}"}), 400
    if target_language not in LANGUAGE_MAPPINGS:
        return jsonify({"error": f"Unsupported target language: {target_language}"}), 400

    video_id = str(int(time.time()))
    update_processing_status(video_id, "initializing", 0, "Initializing processing...")

    video_file_path: Optional[str] = None
    yt_url_data: Optional[str] = None

    if "video_file" in request.files and request.files["video_file"]:
        f = request.files["video_file"]
        if f and f.filename:
            try:
                import uuid

                temp_id = str(uuid.uuid4())
                temp_filename = f"{temp_id}_{secure_filename(f.filename)}"
                temp_path = os.path.join(CONFIG["UPLOAD_DIR"], temp_filename)
                logger.info(f"Saving uploaded file to temporary path: {temp_path}")
                with open(temp_path, "wb") as temp_file:
                    temp_file.write(f.read())
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    video_file_path = temp_path
                    logger.info(
                        f"File saved successfully: {temp_path} ({os.path.getsize(temp_path)} bytes)"
                    )
                else:
                    raise ValueError("Failed to save uploaded file")
                f = None
            except Exception as e:
                logger.error(f"Failed to save uploaded file: {e}")
                if "temp_path" in locals() and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                raise ValueError(f"Failed to save uploaded file: {str(e)}")

    if "yt_url" in request.form and request.form["yt_url"].strip():
        yt_url_data = str(request.form["yt_url"].strip())
        logger.info(f"YouTube URL provided: {yt_url_data}")

    if not video_file_path and not yt_url_data:
        raise ValueError("No video file or YouTube URL provided")

    logger.info("File handling completed successfully, starting background processing...")
    import gc

    gc.collect()
    logger.info("Garbage collection completed")

    def process_video():
        nonlocal video_id
        try:
            logger.info(f"Starting background processing for video {video_id}")
            logger.info(f"Video file path available: {video_file_path is not None}")
            logger.info(f"YouTube URL data available: {yt_url_data is not None}")
            # Will hold all extracted keyframe metadata loaded from JSON
            keyframe_metadata_all: List[Dict[str, Any]] = []
            update_processing_status(video_id, "downloading", 5, "Downloading video...")
            try:
                if video_file_path:
                    logger.info(f"Using already saved file: {video_file_path}")
                    video_path = video_file_path
                elif yt_url_data:
                    def update_download_status(status, progress, message):
                        update_processing_status(video_id, status, progress, message)

                    video_path, downloaded_video_id = handle_youtube_download(
                        yt_url_data,
                        CONFIG["UPLOAD_DIR"],
                        video_id=video_id,
                        status_callback=update_download_status,
                    )
                    logger.info(f"YouTube video downloaded to: {video_path}")
                    logger.info(
                        f"Downloaded video ID: {downloaded_video_id}, Processing video ID: {video_id}"
                    )
                else:
                    raise ValueError("No video file or YouTube URL available")
                logger.info(f"Video successfully processed: {video_path}")
            except Exception as e:
                logger.error(f"Failed to process video: {str(e)}")
                logger.error(f"Exception type: {type(e)}")
                logger.error(f"Exception details: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                update_processing_status(
                    video_id, "error", 0, "Failed to download/process video", str(e)
                )
                return

            logger.info(f"Moving to transcription stage with video_path: {video_path}")

            try:
                result = subprocess.run(
                    ["ffmpeg", "-version"], capture_output=True, text=True
                )
                logger.info(f"FFmpeg available: {result.returncode == 0}")
                if result.returncode != 0:
                    logger.error(f"FFmpeg error: {result.stderr}")
            except Exception as e:
                logger.error(f"FFmpeg test failed: {e}")

            update_processing_status(video_id, "transcribing", 20, "Transcribing audio...")
            transcription_language = source_language if source_language != "auto" else "english"
            logger.info(f"Transcription language: {transcription_language}")
            logger.info(f"Calling transcribe_video function...")
            logger.info(
                f"Function signature: transcribe_video({video_path}, {CONFIG['PROCESSED_DIR']}, {video_id}, {transcription_language})"
            )
            logger.info(f"transcribe_video function type: {type(transcribe_video)}")
            logger.info(f"transcribe_video function callable: {callable(transcribe_video)}")

            # Start periodic heartbeat to prevent watchdog stalls during long transcription
            transcribe_hb_stop = Event()
            def _transcribe_heartbeat():
                try:
                    while not transcribe_hb_stop.is_set():
                        # Keep the same progress and message; this refreshes last_updated
                        update_processing_status(video_id, "transcribing", 20, "Transcribing audio...")
                        # Wait with wake-up capability
                        transcribe_hb_stop.wait(30)
                except Exception as hb_err:
                    logger.warning(f"Transcription heartbeat encountered an error: {hb_err}")

            transcribe_hb_thread = Thread(target=_transcribe_heartbeat, daemon=True)
            transcribe_hb_thread.start()

            try:
                transcript_path, segments = transcribe_video(
                    video_path,
                    CONFIG["PROCESSED_DIR"],
                    video_id,
                    language=transcription_language,
                )
                logger.info(
                    f"Transcription completed. Transcript path: {transcript_path}, Segments: {len(segments) if segments else 0}"
                )
            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise ValueError(f"Transcription failed: {str(e)}")
            finally:
                # Stop heartbeat thread
                try:
                    transcribe_hb_stop.set()
                    transcribe_hb_thread.join(timeout=1)
                except Exception:
                    pass

            if not segments:
                raise ValueError("No speech detected in the video")

            detected_language = transcription_language
            if source_language == "auto" and segments:
                sample_text = " ".join([seg.get("text", "") for seg in segments[:5]])
                detected_lang = detect_language(sample_text)
                if detected_lang:
                    detected_language = detected_lang
                    logger.info(f"Auto-detected language: {detected_language}")

            if detected_language != target_language:
                update_processing_status(video_id, "translating", 35, "Translating content...")
                segments = translate_segments(segments, detected_language, target_language)
                logger.info(f"Translated from {detected_language} to {target_language}")

            try:
                update_processing_status(
                    video_id, "extracting", 40, "Extracting key frames and analyzing visual content..."
                )

                ocr_languages = list(set([detected_language, target_language, "en"]))
                logger.info(f"Starting keyframe extraction with OCR languages: {ocr_languages}")

                keyframes_dir = extract_keyframes(
                    video_path,
                    CONFIG["PROCESSED_DIR"],
                    video_id,
                    target_resolution=resolution,
                    ocr_languages=ocr_languages,
                    enable_ocr=enable_ocr,
                    max_keyframes=100,
                    frame_interval=5  # frames to skip between keyframes (lower = more frequent)
                )

                ocr_summary: Dict[str, Any] = {}
                # Load keyframe metadata JSON for downstream usage
                metadata_file = os.path.join(keyframes_dir, "keyframes_metadata.json") if keyframes_dir else None
                if metadata_file and os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            md = json.load(f)
                            keyframe_metadata_all = md.get("keyframes", []) or []
                    except Exception as e:
                        logger.warning(f"Failed to load keyframe metadata JSON: {e}")

                if enable_ocr and keyframe_metadata_all:
                    logger.info("Processing OCR text from keyframes...")
                    ocr_summary = get_keyframe_text_summary(keyframe_metadata_all)
                    if ocr_summary.get("high_confidence_text"):
                        logger.info(
                            f"Extracted {len(ocr_summary['high_confidence_text'])} characters of high-confidence OCR text"
                        )
                    if ocr_summary.get("all_text"):
                        logger.info(
                            f"Total OCR text extracted: {len(ocr_summary['all_text'])} characters"
                        )

                if not keyframes_dir or not os.path.exists(keyframes_dir):
                    raise ValueError("Failed to extract keyframes from video")

            except Exception as e:
                logger.error(f"Keyframe extraction failed: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                update_processing_status(
                    video_id,
                    "error",
                    0,
                    "Failed to extract keyframes from video",
                    str(e),
                )
                return

            try:
                update_processing_status(
                    video_id, "analyzing", 55, "Analyzing content for coherent topics..."
                )
                if enable_ocr and ocr_summary.get("high_confidence_text"):
                    ocr_context = ocr_summary["high_confidence_text"]
                    if ocr_context.strip() and segments:
                        ocr_chunks = [ocr_context[i : i + 300] for i in range(0, len(ocr_context), 300)]
                        for i, chunk in enumerate(ocr_chunks):
                            seg_idx = min(i, len(segments) - 1)
                            segments[seg_idx]["text"] += f" [Visual context: {chunk}]"
                logger.info(f"Starting topic analysis on {len(segments)} segments...")
                # Removed unsupported parameters: min_segment_duration, max_topics, min_topic_duration, use_visual_context, visual_context
                topics = analyze_topic_segments(
                    segments=segments,
                    language=target_language
                )
                if not topics:
                    raise ValueError("No coherent topics could be identified in the content")
                logger.info(f"Identified {len(topics)} distinct topics: {[t['name'] for t in topics]}")
            except Exception as e:
                logger.error(f"Topic analysis failed: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                update_processing_status(
                    video_id,
                    "error",
                    0,
                    "Failed to analyze video content for topics",
                    str(e),
                )
                return

            logger.info(f"Identified {len(topics)} distinct topics: {[t['name'] for t in topics]}")
            try:
                summaries: List[str] = []
                tts_paths: List[str] = []
                topic_names: List[str] = []
                topic_keywords: List[List[str]] = []
                topic_metadata: List[Dict[str, Any]] = []

                for topic_idx, topic in enumerate(topics):
                    # utils/topic_analyzer.analyze_topic_segments returns topics with key 'id'
                    # and annotates original 'segments' in-place with segment['topic_id'].
                    topic_id = topic.get("id", topic.get("topic_id"))
                    topic_name = topic.get("name", f"Topic {topic_idx + 1}")
                    keywords = topic.get("keywords", [])

                    # Reconstruct the list of segments for this topic from the annotated segments
                    if topic_id is not None:
                        try:
                            topic_id_int = int(topic_id)
                        except Exception:
                            topic_id_int = topic_id
                        topic_segments = [
                            s for s in segments if s.get("topic_id") == topic_id_int
                        ]
                    else:
                        # Fallback: use time range if available
                        start = topic.get("start_time")
                        end = topic.get("end_time")
                        topic_segments = [
                            s
                            for s in segments
                            if start is not None
                            and end is not None
                            and s.get("start") is not None
                            and s.get("end") is not None
                            and s.get("start") >= start
                            and s.get("end") <= end
                        ]

                    progress = 60 + (25 * topic_idx // max(1, len(topics)))
                    update_processing_status(
                        video_id,
                        "summarizing",
                        progress,
                        f"Processing topic {topic_idx + 1}/{len(topics)}: {topic_name}...",
                    )

                    logger.info(f"Generating summary for topic {topic_id}: {topic_name}")
                    summary = summarize_cluster(
                        cluster=topic_segments,
                        language=target_language,
                        use_extractive=False,
                    )
                    if not summary or len(summary.strip()) < 10:
                        logger.warning(
                            f"Generated summary for topic {topic_id} is too short, using fallback"
                        )
                        summary = (
                            f"This segment discusses {topic_name}. "
                            f"Key points include: {', '.join(keywords[:3])}."
                        )
                    summaries.append(summary)
                    topic_names.append(topic_name)
                    topic_keywords.append(keywords)

                    try:
                        logger.info(f"Generating TTS for topic {topic_id}")
                        audio_path = text_to_speech(
                            text=summary,
                            output_dir=CONFIG["PROCESSED_DIR"],
                            video_id=video_id,
                            cluster_id=topic_id,
                            voice=voice,
                            language=target_language,
                            slow=False,
                        )
                        if not audio_path or not os.path.exists(audio_path):
                            raise FileNotFoundError("TTS file not generated")
                        tts_paths.append(audio_path)
                        topic_metadata.append(
                            {
                                "id": topic_id,
                                "name": topic_name,
                                "start_time": topic["start_time"],
                                "end_time": topic["end_time"],
                                "duration": topic["end_time"] - topic["start_time"],
                                "segment_count": len(topic_segments),
                                "summary": summary,
                                "audio_path": audio_path,
                                "keywords": keywords,
                            }
                        )
                    except Exception as tts_error:
                        logger.error(f"TTS generation failed for topic {topic_id}: {str(tts_error)}")
                        silence = AudioSegment.silent(duration=5000)
                        fallback_path = os.path.join(
                            CONFIG["PROCESSED_DIR"], f"{video_id}_topic_{topic_id}_fallback.wav"
                        )
                        silence.export(fallback_path, format="wav")
                        tts_paths.append(fallback_path)
                        logger.info(f"Created silent fallback audio at {fallback_path}")

                if not tts_paths:
                    raise ValueError("Failed to generate audio for any topics")

            except Exception as e:
                logger.error(f"Topic processing failed: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                update_processing_status(
                    video_id, "error", 0, "Failed to process video topics", str(e)
                )
                return

            # Generate per-topic summary videos using keyframes and TTS audio
            try:
                update_processing_status(
                    video_id,
                    "rendering",
                    75,
                    "Generating per-topic summary videos...",
                )

                # Resolve output resolution tuple
                out_resolution = SUPPORTED_RESOLUTIONS.get(
                    resolution, SUPPORTED_RESOLUTIONS.get(CONFIG["DEFAULT_RESOLUTION"], (854, 480))
                )

                topic_videos: List[Dict[str, Any]] = []
                total_topics = max(1, len(topic_metadata))

                for idx, tmeta in enumerate(topic_metadata):
                    # Determine how many keyframes to use for this topic
                    target_kf = calculate_keyframe_distribution(
                        tmeta.get("duration", 0.0)
                    )

                    # Filter existing keyframes by topic time window
                    topic_kfs = []
                    if keyframe_metadata_all:
                        topic_kfs = select_keyframes_for_topic(
                            keyframe_metadata_all,
                            tmeta["start_time"],
                            tmeta["end_time"],
                        )

                    # If we need more, extract additional frames from the time range
                    if len(topic_kfs) < target_kf:
                        try:
                            extra_needed = max(0, target_kf - len(topic_kfs))
                            if extra_needed > 0:
                                additional = extract_keyframes_for_time_range(
                                    video_path=video_path,
                                    start_time=tmeta["start_time"],
                                    end_time=tmeta["end_time"],
                                    max_keyframes=extra_needed,
                                )
                                topic_kfs = sorted(
                                    (topic_kfs or []) + (additional or []),
                                    key=lambda x: x.get("timestamp", 0),
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to extract additional keyframes for topic {tmeta.get('id')}: {e}"
                            )

                    # Cap to target count
                    if target_kf > 0 and len(topic_kfs) > target_kf:
                        topic_kfs = topic_kfs[:target_kf]

                    if not topic_kfs:
                        logger.warning(
                            f"No keyframes available for topic {tmeta.get('id')} - skipping video generation."
                        )
                        continue

                    # Build a safe output base path (without extension; maker adds .mp4)
                    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(tmeta.get("name", "topic"))).strip("_")
                    output_base = os.path.join(
                        CONFIG["PROCESSED_DIR"], f"{video_id}_topic_{tmeta.get('id')}_{safe_name}"
                    )

                    # Render topic summary video
                    topic_video_path = make_summary_video(
                        output_path=output_base,
                        keyframes=topic_kfs,
                        audio_path=tmeta.get("audio_path"),
                        resolution=out_resolution,
                        fps=30,
                        transition_type="fade",
                        transition_duration=0.5,
                        log_level="INFO",
                    )

                    if topic_video_path and os.path.exists(topic_video_path):
                        topic_videos.append(
                            {
                                "topic_id": tmeta.get("id"),
                                "name": tmeta.get("name"),
                                "keywords": tmeta.get("keywords", []),
                                "summary": tmeta.get("summary"),
                                "audio_path": tmeta.get("audio_path"),
                                "video_path": topic_video_path,
                                "video_filename": os.path.basename(topic_video_path),
                                "start_time": tmeta.get("start_time"),
                                "end_time": tmeta.get("end_time"),
                                "duration": tmeta.get("duration"),
                                "keyframe_count": len(topic_kfs),
                            }
                        )

                    # Update progress within rendering phase
                    render_progress = 75 + int(20 * (idx + 1) / total_topics)
                    update_processing_status(
                        video_id,
                        "rendering",
                        min(95, render_progress),
                        f"Generated {idx + 1}/{total_topics} topic videos",
                    )

                # Optionally concatenate topic videos into a final summary
                final_summary_video = None
                try:
                    if topic_videos:
                        clips = []
                        for tv in topic_videos:
                            try:
                                clips.append(VideoFileClip(tv["video_path"]))
                            except Exception as e:
                                logger.warning(f"Failed to load clip for concatenation: {e}")
                        if clips:
                            concatenated = concatenate_videoclips(clips, method="compose")
                            final_summary_video = os.path.join(
                                CONFIG["PROCESSED_DIR"], f"{video_id}_final_summary.mp4"
                            )
                            concatenated.write_videofile(
                                final_summary_video,
                                codec="libx264",
                                audio_codec="aac",
                                verbose=False,
                                logger=None,
                            )
                            # Close clips
                            try:
                                concatenated.close()
                                for c in clips:
                                    c.close()
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Failed to concatenate topic videos: {e}")

                result_message = "Processing complete!"
                result = {
                    "video_id": video_id,
                    "summaries": summaries,
                    "keywords": topic_keywords,
                    "topic_videos": topic_videos,
                    "final_summary_video": final_summary_video,
                    "status": "completed",
                    "progress": 100,
                    "message": result_message,
                }
                processing_status[video_id].update(result)
                update_processing_status(video_id, "completed", 100, result_message, None)

            except Exception as e:
                logger.error(f"Video rendering failed: {str(e)}\n{traceback.format_exc()}")
                update_processing_status(
                    video_id, "error", 0, "Failed during summary video generation", str(e)
                )
                return


        except Exception as e:
            logger.error(f"Error processing video: {str(e)}\n{traceback.format_exc()}")
            update_processing_status(
                video_id, "error", 0, "An error occurred during processing", str(e)
            )

    # Thread and Timer already imported at module level

    def timeout_handler():
        logger.error(f"Processing timeout for video {video_id}")
        update_processing_status(
            video_id,
            "error",
            0,
            "Processing timeout - taking too long",
            "Processing exceeded maximum time limit",
        )

    timeout_timer = Timer(30 * 60, timeout_handler)
    timeout_timer.start()

    def watchdog_check():
        current_status = get_processing_status(video_id)
        if current_status.get("status") == "downloading" and current_status.get("progress") == 5:
            logger.error(f"Processing stuck at 5% for video {video_id}, forcing error")
            update_processing_status(
                video_id,
                "error",
                0,
                "Processing stuck - forcing error",
                "Processing got stuck at download stage",
            )

    watchdog_timer = Timer(5 * 60, watchdog_check)
    watchdog_timer.start()

    # Heartbeat watchdog: if last_updated hasn't changed for STALE_STALL_SECONDS, fail fast
    def heartbeat_watchdog(prev_ts: Optional[float] = None):
        try:
            st = get_processing_status(video_id)
            status_val = st.get("status")
            last_ts = st.get("last_updated") or 0
            now = time.time()

            # If no progress heartbeat within threshold, mark as error
            if status_val not in ("completed", "error") and last_ts and now - last_ts > STALE_STALL_SECONDS:
                logger.error(
                    f"Processing stalled for video {video_id}: no heartbeat for {int(now - last_ts)}s"
                )
                update_processing_status(
                    video_id,
                    "error",
                    0,
                    "Processing stalled - no progress for too long",
                    "Watchdog detected stalled processing",
                )
        finally:
            # Reschedule periodic check (once per minute) only if still running
            st2 = get_processing_status(video_id)
            if st2.get("status") not in ("completed", "error"):
                Timer(60, heartbeat_watchdog).start()

    Timer(60, heartbeat_watchdog).start()

    thread = Thread(target=process_video)
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "video_id": video_id,
            "status": "processing",
            "progress": 0,
            "message": "Processing started",
        }
    )

@app.route("/api/status/<video_id>", methods=["GET"])
def api_status(video_id: str):
    """Return current processing status for a given video_id."""
    status = get_processing_status(video_id)
    # Include any extra result fields if present (e.g., summaries, keywords)
    extra = processing_status.get(video_id, {})
    merged = {**status}
    for key in ("summaries", "keywords", "topic_videos", "final_summary_video", "message", "progress", "status", "error", "video_id"):
        if key in extra:
            merged[key] = extra[key]
    merged.setdefault("video_id", video_id)
    return jsonify(merged)


@app.route("/processed/<path:filename>")
def serve_processed_file(filename: str):
    """Serve files from the processed directory for playback/download."""
    return send_from_directory(CONFIG["PROCESSED_DIR"], filename)


@app.route("/api/test-upload", methods=["POST"])
def api_test_upload():
    """Simple endpoint to verify file uploads from frontend."""
    # Case 1: File upload present
    if "video_file" in request.files and request.files["video_file"].filename != "":
        f = request.files["video_file"]
        # Do not persist; just read to measure size
        data = f.read()
        return jsonify({
            "success": True,
            "type": "file",
            "filename": f.filename,
            "size": len(data),
            "content_type": getattr(f, "content_type", None),
        })

    # Case 2: YouTube URL provided
    yt_url = (request.form.get("yt_url") or "").strip()
    if yt_url:
        if not is_youtube_url(yt_url):
            return jsonify({"success": False, "error": "Invalid YouTube URL"}), 400
        # Optionally try to extract minimal info without downloading
        title = None
        duration = None
        try:
            import yt_dlp  # local import to avoid overhead if not used
            ydl_opts = {"quiet": True, "no_warnings": True, "socket_timeout": 10}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(yt_url, download=False)
                title = info.get("title")
                duration = info.get("duration")
                yt_url = info.get("webpage_url") or yt_url
        except Exception as e:
            logger.warning(f"yt-dlp metadata fetch failed for test-upload: {e}")

        return jsonify({
            "success": True,
            "type": "url",
            "yt_url": yt_url,
            "title": title,
            "duration": duration,
            "message": "YouTube URL accepted for processing"
        })

    return jsonify({
        "success": False,
        "error": "Please provide a YouTube URL or upload a video file."
    }), 400


if __name__ == "__main__":
    # Start the Flask development server
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    logger.info(f"Starting Flask server on http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug, threaded=True)
