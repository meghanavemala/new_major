import os
import subprocess
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from moviepy.editor import VideoFileClip


def get_video_duration(video_path: str) -> float:
    """Get the duration of a video file in seconds."""
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception:
        return 0.0


def extract_audio(video_path: str, output_path: str) -> bool:
    """Extract audio from a video file."""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_path, logger=None)
        video.close()
        return os.path.exists(output_path)
    except Exception:
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
                return False

        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception:
                pass

        return os.path.exists(output_path)
    except Exception:
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
    except Exception:
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
    except Exception:
        return None



