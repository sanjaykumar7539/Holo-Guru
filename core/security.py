"""Input validation, prompt-injection detection and local request limiting."""
import hmac
import re
import time
from collections import defaultdict, deque


_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"reveal\s+(the\s+)?(system|developer)\s+prompt",
    r"show\s+(me\s+)?(api\s*keys?|passwords?|secrets?)",
    r"bypass\s+(the\s+)?(guardrails?|safety|restrictions?)",
    r"developer\s+mode|jailbreak|do\s+anything\s+now",
]
_COMPILED = [re.compile(pattern, re.I) for pattern in _PATTERNS]


def validate_question(value, max_chars=2000):
    if not isinstance(value, str):
        return False, "Question must be text."
    text = value.strip()
    if not text:
        return False, "Question is empty."
    if len(text) > max_chars:
        return False, f"Question exceeds {max_chars} characters."
    if any(ord(ch) < 9 or 13 < ord(ch) < 32 for ch in text):
        return False, "Question contains unsupported control characters."
    return True, text


def detect_prompt_injection(text):
    return next((p.pattern for p in _COMPILED if p.search(text)), None)


def check_teacher_token(provided, expected):
    """Return True if teacher auth is not required, or the token matches.

    When *expected* is empty, admin routes stay open (offline single-teacher).
    Comparison is constant-time when both values are non-empty.
    """
    expected = (expected or "").strip()
    if not expected:
        return True
    provided = (provided or "").strip()
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def check_room_pin(provided, expected):
    """Same semantics as check_teacher_token for optional room PIN."""
    return check_teacher_token(provided, expected)


def extract_room_pin(headers=None, args=None, json_body=None):
    """Read room PIN from JSON body, query args, or X-Room-Pin header."""
    if json_body and isinstance(json_body, dict):
        pin = json_body.get("room_pin") or json_body.get("pin")
        if pin is not None and str(pin).strip():
            return str(pin).strip()
    if args:
        pin = args.get("room_pin") or args.get("pin")
        if pin is not None and str(pin).strip():
            return str(pin).strip()
    headers = headers or {}
    pin = headers.get("X-Room-Pin") or headers.get("x-room-pin")
    if pin:
        return pin.strip()
    return ""


def extract_teacher_token(headers, form=None, json_body=None):

    """Read teacher token from header, form, or JSON body (in that order)."""
    headers = headers or {}
    # Preferred: X-Teacher-Token
    token = headers.get("X-Teacher-Token") or headers.get("x-teacher-token")
    if token:
        return token.strip()
    # Authorization: Bearer <token>
    auth = headers.get("Authorization") or headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if form:
        token = form.get("teacher_token") or form.get("token")
        if token:
            return str(token).strip()
    if json_body and isinstance(json_body, dict):
        token = json_body.get("teacher_token") or json_body.get("token")
        if token:
            return str(token).strip()
    return ""


class SlidingWindowLimiter:
    def __init__(self, limit=30, window_seconds=60):
        self.limit = limit
        self.window = window_seconds
        self.events = defaultdict(deque)

    def check(self, key):
        now = time.monotonic()
        events = self.events[key]
        while events and now - events[0] >= self.window:
            events.popleft()
        if len(events) >= self.limit:
            return False, max(1, int(self.window - (now - events[0])))
        events.append(now)
        return True, 0
