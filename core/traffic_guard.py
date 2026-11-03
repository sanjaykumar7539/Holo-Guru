"""Crash-resistant admission control and lightweight runtime metrics.

Designed for a single Jetson-class deployment: bound expensive generations so
traffic spikes are rejected/queued by the web tier instead of exhausting RAM.
"""
import threading
import time
from collections import deque

class TrafficGuard:
    def __init__(self, max_concurrent=1, max_waiting=12, wait_timeout=8.0):
        self.max_concurrent = max(1, int(max_concurrent))
        self.max_waiting = max(0, int(max_waiting))
        self.wait_timeout = float(wait_timeout)
        self._slots = threading.BoundedSemaphore(self.max_concurrent)
        self._lock = threading.Lock()
        self.active = 0
        self.waiting = 0
        self.accepted = 0
        self.rejected = 0
        self.completed = 0
        self.failed = 0
        self.latencies = deque(maxlen=200)

    def acquire(self):
        start = time.monotonic()
        # Fast path: take a free inference slot without entering the waiting room.
        if self._slots.acquire(blocking=False):
            with self._lock:
                self.active += 1
                self.accepted += 1
            return start

        with self._lock:
            if self.waiting >= self.max_waiting:
                self.rejected += 1
                return None
            self.waiting += 1
        ok = self._slots.acquire(timeout=self.wait_timeout)
        with self._lock:
            self.waiting -= 1
            if not ok:
                self.rejected += 1
                return None
            self.active += 1
            self.accepted += 1
        return start

    def release(self, started_at, failed=False):
        elapsed = max(0.0, time.monotonic() - started_at)
        with self._lock:
            self.active = max(0, self.active - 1)
            self.completed += 1
            self.failed += int(bool(failed))
            self.latencies.append(elapsed)
        self._slots.release()

    def snapshot(self):
        with self._lock:
            vals = list(self.latencies)
            avg = sum(vals) / len(vals) if vals else 0.0
            return {
                "max_concurrent": self.max_concurrent, "max_waiting": self.max_waiting,
                "active": self.active, "waiting": self.waiting, "accepted": self.accepted,
                "rejected": self.rejected, "completed": self.completed, "failed": self.failed,
                "avg_generation_seconds": round(avg, 3),
            }
