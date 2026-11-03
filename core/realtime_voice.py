"""Real-time voice helpers for chunked / partial STT.

Works with the existing SeamlessM4T file-based transcriber by accepting
rolling audio buffers from the browser (MediaRecorder timeslices) and
coordinating partial vs final transcripts without a separate streaming ASR.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

# Per-session last partial request time (rate limit)
_last_partial: Dict[str, float] = defaultdict(float)
_lock = threading.Lock()

# Default: at most one partial STT every N seconds per session
DEFAULT_PARTIAL_MIN_INTERVAL = 0.65


def allow_partial(session_id: str, min_interval: float = DEFAULT_PARTIAL_MIN_INTERVAL) -> Tuple[bool, float]:
    """Return (allowed, retry_after_seconds)."""
    sid = (session_id or "default").strip() or "default"
    now = time.monotonic()
    with _lock:
        last = _last_partial[sid]
        wait = min_interval - (now - last)
        if wait > 0:
            return False, round(wait, 2)
        _last_partial[sid] = now
        return True, 0.0


def mark_final(session_id: str) -> None:
    """Reset rate limiter after a final utterance (optional)."""
    sid = (session_id or "default").strip() or "default"
    with _lock:
        _last_partial.pop(sid, None)


def is_short_audio_bytes(size: int, min_bytes: int = 2000) -> bool:
    """Heuristic: skip STT on tiny blobs (noise / empty)."""
    return size < min_bytes
