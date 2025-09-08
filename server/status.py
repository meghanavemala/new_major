import time
from typing import Dict, Any, Optional


def init_status() -> Dict[str, Dict[str, Any]]:
    return {}


def get_status(store: Dict[str, Dict[str, Any]], video_id: str) -> Dict[str, Any]:
    status = store.get(video_id)
    
    if not status:
        return {
            "status": "not_started",
            "progress": 0,
            "message": "Processing not started",
            "error": None,
        }
        
    # Check for stale status (no updates for 5 minutes)
    if 'last_updated' in status:
        if time.time() - status['last_updated'] > 300:  # 5 minutes
            return {
                "status": "error",
                "progress": status.get('progress', 0),
                "message": "Processing stalled",
                "error": "Process timed out - no updates received for 5 minutes"
            }
            
    return status


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



