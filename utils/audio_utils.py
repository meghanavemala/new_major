import subprocess
import logging
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

def extract_audio(
    video_path: str, 
    output_path: str, 
    sample_rate: int = 16000
) -> bool:
    """Extract audio from video using ffmpeg.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save extracted audio
        sample_rate: Sample rate for output audio (Hz)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', video_path,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono audio
            '-f', 'wav',
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600  # 10 minutes safety timeout
        )
        logger.debug(f"FFmpeg audio extraction output: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e.stderr}")
        return False
    except subprocess.TimeoutExpired as e:
        logger.error(f"FFmpeg audio extraction timed out: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in extract_audio: {str(e)}", exc_info=True)
        return False

def enhance_audio_quality(audio_path: str, enhanced_path: str) -> bool:
    """
    Enhance audio quality for better transcription accuracy.
    
    Args:
        audio_path: Path to the input audio file
        enhanced_path: Path where the enhanced audio will be saved
        
    Returns:
        bool: True if enhancement was successful, False otherwise
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Apply noise reduction
        y_reduced = librosa.effects.preemphasis(y, coef=0.97)
        
        # Normalize audio
        y_normalized = librosa.util.normalize(y_reduced)
        
        # Save enhanced audio
        sf.write(enhanced_path, y_normalized, sr)
        
        return True
        
    except Exception as e:
        logger.error(f"Error enhancing audio: {str(e)}", exc_info=True)
        return False
