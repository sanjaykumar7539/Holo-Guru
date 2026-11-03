"""Cache-Augmented Generation support.

CAG here means validated knowledge context is kept hot and reused before a
fresh retrieval round. This is separate from final-answer caching: the LLM can
still generate a fresh response using the cached evidence and conversation.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Dict, Optional, Sequence

import config

_LOCK = threading.RLock()
_STORE: "OrderedDict[str, tuple[float, Dict]]" = OrderedDict()


def _key(question: str, book_ids: Optional[Sequence[str]]) -> str:
    norm = " ".join((question or "").lower().split())
    scope = json.dumps(sorted(book_ids or []), separators=(",", ":"))
    return hashlib.sha256(f"{norm}|{scope}".encode()).hexdigest()


def get(question: str, book_ids: Optional[Sequence[str]] = None) -> Optional[Dict]:
    if not getattr(config, "CAG_ENABLED", True):
        return None
    key = _key(question, book_ids)
    now = time.time()
    with _LOCK:
        item = _STORE.get(key)
        if not item:
            return None
        expires, value = item
        if now >= expires:
            _STORE.pop(key, None)
            return None
        _STORE.move_to_end(key)
        return dict(value)


def put(question: str, book_ids: Optional[Sequence[str]], value: Dict) -> None:
    if not getattr(config, "CAG_ENABLED", True) or not value.get("context"):
        return
    key = _key(question, book_ids)
    ttl = int(getattr(config, "CAG_TTL_SECONDS", 7200))
    max_entries = int(getattr(config, "CAG_MAX_ENTRIES", 256))
    with _LOCK:
        _STORE[key] = (time.time() + ttl, dict(value))
        _STORE.move_to_end(key)
        while len(_STORE) > max_entries:
            _STORE.popitem(last=False)


def clear() -> None:
    with _LOCK:
        _STORE.clear()


def stats() -> Dict:
    with _LOCK:
        return {"entries": len(_STORE)}
