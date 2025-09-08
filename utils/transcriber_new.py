import os
import json
import logging
import torch
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple

from utils.gpu_config import get_device
from utils.audio_utils import extract_audio, enhance_audio_quality
from utils.config import SUPPORTED_LANGUAGES, get_smart_model_size

logger = logging.getLogger(__name__)

def transcribe_video(
    video_path: str,
    processed_dir: str,
    video_id: str,
    language: str = 'english',
    model_size: Optional[str] = None,
    quality_preference: str = 'balanced'
) -> Tuple[Optional[str], List[Dict]]:
    """Transcribe video to text using Faster-Whisper.
    
    This function takes a video file and transcribes its audio content using
    Faster-Whisper. It supports both GPU and CPU processing, with automatic
    fallback to CPU if GPU memory is insufficient.
    
    Args:
        video_path (str): Path to the input video file
        processed_dir (str): Directory to save processed files
        video_id (str): Unique identifier for the video
        language (str, optional): Language of the video. Defaults to 'english'
        model_size (Optional[str], optional): Size of Whisper model. Defaults to None
        quality_preference (str, optional): Quality preference. Defaults to 'balanced'
        
    Returns:
        Tuple[Optional[str], List[Dict]]: A tuple containing:
            - str or None: Path to the segments JSON file, or None if transcription failed
            - List[Dict]: List of transcribed segments, each containing start time,
                       end time, text, and word-level timing information
    """
    segments = []
    audio_path = None
    seg_path = None
    
    try:
        # Setup paths and extract audio
        os.makedirs(processed_dir, exist_ok=True)
        audio_path = os.path.join(processed_dir, f"{video_id}_audio.wav")
        seg_path = os.path.join(processed_dir, f"{video_id}_segments.json")
        
        if not extract_audio(video_path, audio_path):
            logger.error("Failed to extract audio from video")
            return None, []
        
        # Configure model and device
        lang_code = SUPPORTED_LANGUAGES.get(language.lower(), language.lower())
        device = get_device()
        if not model_size:
            model_size = get_smart_model_size(video_path, lang_code, quality_preference)
        
        # Check GPU memory and adjust model size if needed
        if device == 'cuda':
            total_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            if total_mem <= 4096:
                logger.warning("GPU has 4GB or less. Using 'small' model.")
                model_size = 'small'

        # Apply audio enhancement before transcription
        try:
            enhanced_audio_path = os.path.join(processed_dir, f"{video_id}_enhanced_audio.wav")
            if not enhance_audio_quality(audio_path, enhanced_audio_path):
                logger.warning("Audio enhancement failed, using original audio.")
            else:
                logger.info("Using enhanced audio for transcription.")
                audio_path = enhanced_audio_path
        except Exception as e:
            logger.warning(f"Audio enhancement failed, using original audio: {e}")
            
        logger.info(f"Transcribing with {model_size} model on {device}")
        
        try:
            from faster_whisper import WhisperModel
            
            if device == 'cuda':
                # Try GPU first
                try:
                    torch.cuda.empty_cache()
                    model = WhisperModel(model_size, device='cuda', compute_type="float16")
                    segments_fw, info = model.transcribe(
                        audio_path,
                        language=lang_code,
                        beam_size=5,
                        vad_filter=True
                    )
                    
                    if segments_fw:
                        for s in segments_fw:
                            segments.append({
                                'start': s.start,
                                'end': s.end,
                                'text': s.text.strip(),
                                'words': getattr(s, 'words', [])
                            })
                        logger.info(f"GPU transcription successful: {len(segments)} segments")
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        logger.warning("CUDA OOM error, falling back to CPU")
                        device = 'cpu'
                        model_size = 'small'
                        torch.cuda.empty_cache()
                    else:
                        raise
            
            # Use CPU if GPU failed or wasn't available
            if device == 'cpu' or not segments:
                model = WhisperModel(model_size, device='cpu', compute_type="int8")
                segments_fw, info = model.transcribe(
                    audio_path,
                    language=lang_code,
                    beam_size=5,
                    vad_filter=True
                )
                
                if segments_fw:
                    segments.clear()  # Clear any failed GPU attempt results
                    for s in segments_fw:
                        segments.append({
                            'start': s.start,
                            'end': s.end,
                            'text': s.text.strip(),
                            'words': getattr(s, 'words', [])
                        })
                    logger.info(f"CPU transcription successful: {len(segments)} segments")
            
            # Save results if we got any segments
            if segments:
                with open(seg_path, 'w', encoding='utf-8') as f:
                    json.dump(segments, f, indent=2, ensure_ascii=False)
                return seg_path, segments
            
            logger.error("No speech segments detected")
            return None, []
            
        except ImportError as e:
            logger.error(f"Failed to import Faster-Whisper: {e}")
            return None, []
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return None, []
    
    finally:
        # Clean up the enhanced audio file if it exists
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary audio file: {e}")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
