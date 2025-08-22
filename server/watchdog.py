import time
from threading import Timer, Event
from typing import Callable, Dict, Any


def schedule_timeout(seconds: int, on_timeout: Callable[[], None]) -> Timer:
    timer = Timer(seconds, on_timeout)
    timer.start()
    return timer


def start_periodic_heartbeat(
    interval_seconds: int,
    update_fn: Callable[[], None],
) -> Event:
    stop_event = Event()

    def _beat():
        try:
            while not stop_event.is_set():
                update_fn()
                stop_event.wait(interval_seconds)
        except Exception:
            # Best-effort; avoid crashing thread
            pass

    t = Timer(0, _beat)
    t.daemon = True
    t.start()
    return stop_event


def start_stall_watchdog(
    store: Dict[str, Dict[str, Any]],
    video_id: str,
    max_stale_seconds: int,
    update_status: Callable[[Dict[str, Dict[str, Any]], str, str, int, str, str], None],
) -> None:
    def _check():
        st = store.get(video_id, {})
        last_ts = st.get("last_updated") or 0
        now = time.time()
        if st.get("status") not in ("completed", "error") and last_ts and now - last_ts > max_stale_seconds:
            update_status(store, video_id, "error", 0, "Processing stalled - no progress for too long", "Watchdog detected stalled processing")
        else:
            # reschedule
            Timer(60, _check).start()

    Timer(60, _check).start()






