import time
from typing import Dict, Any, Optional


def init_status() -> Dict[str, Dict[str, Any]]:
    return {}


def get_status(store: Dict[str, Dict[str, Any]], video_id: str) -> Dict[str, Any]:
    return store.get(
        video_id,
        {
            "status": "not_started",
            "progress": 0,
            "message": "Processing not started",
            "error": None,
        },
    )


def update_status(
    store: Dict[str, Dict[str, Any]],
    video_id: str,
    status: str,
    progress: int,
    message: str,
    error: Optional[str] = None,
) -> None:
    if video_id not in store:
        store[video_id] = {}
    store[video_id].update(
        {
            "status": status,
            "progress": progress,
            "message": message,
            "error": error,
            "last_updated": time.time(),
        }
    )



