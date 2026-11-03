from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Dict, Any

_LOCK = threading.Lock()
_SAMPLES = defaultdict(lambda: deque(maxlen=512))
_COUNTS = defaultdict(int)
_ERRORS = defaultdict(int)
_SLOW = deque(maxlen=64)


def _pct(values, p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, math.ceil((p / 100.0) * len(vals)) - 1))
    return round(vals[idx], 2)


def record(name: str, duration_ms: float, ok: bool = True, slow_ms: float | None = None, meta: Dict[str, Any] | None = None):
    name = str(name or "unknown")[:120]
    ms = max(0.0, float(duration_ms))
    with _LOCK:
        _SAMPLES[name].append(ms)
        _COUNTS[name] += 1
        if not ok:
            _ERRORS[name] += 1
        if slow_ms is not None and ms >= slow_ms:
            _SLOW.appendleft({
                "name": name,
                "duration_ms": round(ms, 2),
                "ok": bool(ok),
                "ts": round(time.time(), 3),
                "meta": dict(meta or {}),
            })


def increment(name: str, amount: int = 1):
    with _LOCK:
        _COUNTS[name] += int(amount)


@contextmanager
def span(name: str, slow_ms: float | None = None, meta: Dict[str, Any] | None = None):
    started = time.perf_counter()
    ok = False
    try:
        yield
        ok = True
    finally:
        record(name, (time.perf_counter() - started) * 1000.0, ok=ok, slow_ms=slow_ms, meta=meta)


def snapshot() -> Dict[str, Any]:
    with _LOCK:
        metrics = {}
        keys = sorted(set(_COUNTS) | set(_SAMPLES))
        for name in keys:
            vals = list(_SAMPLES.get(name, ()))
            metrics[name] = {
                "count": int(_COUNTS.get(name, 0)),
                "errors": int(_ERRORS.get(name, 0)),
                "avg_ms": round(sum(vals) / len(vals), 2) if vals else 0.0,
                "p50_ms": _pct(vals, 50),
                "p95_ms": _pct(vals, 95),
                "p99_ms": _pct(vals, 99),
                "max_ms": round(max(vals), 2) if vals else 0.0,
                "samples": len(vals),
            }
        return {
            "metrics": metrics,
            "recent_slow_operations": list(_SLOW),
            "sample_window": 512,
        }


def reset():
    with _LOCK:
        _SAMPLES.clear()
        _COUNTS.clear()
        _ERRORS.clear()
        _SLOW.clear()
