"""Audio validation utility for detecting presence of speech."""
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

def has_speech(audio_path: str, threshold: float = 0.01, min_duration: float = 0.5) -> bool:
    """
    Check if the audio file contains speech.
    
    Args:
        audio_path: Path to audio file
        threshold: Energy threshold for speech detection
        min_duration: Minimum duration of audio above threshold (in seconds)
        
    Returns:
        bool: True if speech is detected, False otherwise
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate frame-level energy
        frame_length = 2048
        hop_length = 512
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Count frames above threshold
        speech_frames = np.sum(energy > threshold)
        duration_above_threshold = speech_frames * hop_length / sr
        
        has_speech = duration_above_threshold >= min_duration
        
        if not has_speech:
            logger.warning(f"No significant speech detected in audio file: {audio_path}")
            logger.info(f"Speech duration: {duration_above_threshold:.2f}s, Threshold: {min_duration}s")
        
        return has_speech
        
    except Exception as e:
        logger.error(f"Error analyzing audio for speech: {str(e)}")
        return False
