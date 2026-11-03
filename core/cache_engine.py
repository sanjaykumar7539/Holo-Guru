"""
core/cache_engine.py
Lightweight in-memory TTL cache — cuts latency for repeated questions
(common in a classroom: multiple students ask the same thing), repeated
RAG lookups, and generated visuals (charts/images are deterministic
enough per-prompt to be worth reusing).

Deliberately simple (dict + lock + TTL), not Redis — this is a single-
process Flask app on one device, a distributed cache would be pure
overhead here.
"""
import hashlib
import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_store: dict = {}
_lock = threading.Lock()
MAX_ENTRIES = 500

DEFAULT_TTL = 60 * 30   # 30 minutes — long enough to help a class period,
                         # short enough that stale personalization doesn't linger


def _key(namespace: str, *parts: str) -> str:
    raw = namespace + "::" + "::".join(p.strip().lower() for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()


def get(namespace: str, *parts: str) -> Optional[Any]:
    k = _key(namespace, *parts)
    with _lock:
        entry = _store.get(k)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del _store[k]
            return None
        return value


def set(namespace: str, *parts: str, value: Any, ttl: int = DEFAULT_TTL):
    k = _key(namespace, *parts)
    with _lock:
        _store[k] = (value, time.time() + ttl)
        if len(_store) > MAX_ENTRIES:
            oldest = min(_store, key=lambda item: _store[item][1])
            _store.pop(oldest, None)


def get_or_compute(namespace: str, *parts: str, compute: Callable[[], Any], ttl: int = DEFAULT_TTL) -> Any:
    cached = get(namespace, *parts)
    if cached is not None:
        logger.info(f"Cache hit [{namespace}]")
        return cached
    result = compute()
    if result is not None:
        set(namespace, *parts, value=result, ttl=ttl)
    return result


def clear_expired():
    now = time.time()
    with _lock:
        expired = [k for k, (_, exp) in _store.items() if now > exp]
        for k in expired:
            del _store[k]
    if expired:
        logger.info(f"Cache: cleared {len(expired)} expired entries")


def clear_all():
    with _lock:
        _store.clear()


def stats() -> dict:
    with _lock:
        return {"entries": len(_store)}
