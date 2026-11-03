from __future__ import annotations

import time
from copy import deepcopy
from threading import Lock
from typing import Any, Dict

_ROOMS: Dict[str, Dict[str, Any]] = {}
_LOCK = Lock()
_PARTICIPANT_TTL = 12.0
_MAX_STROKES = 80
_MAX_POINTS = 120


def _now() -> float:
    return time.time()


def _clean_room(room_id: str | None) -> str:
    return (room_id or "demo").strip() or "demo"


def _clean_user(user_id: str | None) -> str:
    return (user_id or "user").strip() or "user"


def _default_room(room_id: str) -> Dict[str, Any]:
    ts = _now()
    return {
        "room_id": room_id,
        "participants": {},
        "scene": {"state": None, "ts": ts, "by": None},
        "annotation": {"pointer": None, "strokes": [], "ts": ts, "by": None},
    }


def _get_room(room_id: str) -> Dict[str, Any]:
    rid = _clean_room(room_id)
    if rid not in _ROOMS:
        _ROOMS[rid] = _default_room(rid)
    return _ROOMS[rid]


def _prune(room: Dict[str, Any]) -> None:
    now = _now()
    dead = [uid for uid, p in room["participants"].items() if now - p.get("last_seen", 0) > _PARTICIPANT_TTL]
    for uid in dead:
        room["participants"].pop(uid, None)


def join_room(room_id: str, user_id: str, name: str) -> Dict[str, Any]:
    rid = _clean_room(room_id)
    uid = _clean_user(user_id)
    with _LOCK:
        room = _get_room(rid)
        _prune(room)
        room["participants"][uid] = {"user_id": uid, "name": (name or uid)[:40], "last_seen": _now()}
        return snapshot(rid)


def leave_room(room_id: str, user_id: str) -> Dict[str, Any]:
    rid = _clean_room(room_id)
    uid = _clean_user(user_id)
    with _LOCK:
        room = _get_room(rid)
        room["participants"].pop(uid, None)
        _prune(room)
        return snapshot(rid)


def heartbeat(room_id: str, user_id: str) -> Dict[str, Any]:
    rid = _clean_room(room_id)
    uid = _clean_user(user_id)
    with _LOCK:
        room = _get_room(rid)
        if uid in room["participants"]:
            room["participants"][uid]["last_seen"] = _now()
        _prune(room)
        return snapshot(rid)


def update_scene(room_id: str, user_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    rid = _clean_room(room_id)
    uid = _clean_user(user_id)
    with _LOCK:
        room = _get_room(rid)
        room["scene"] = {"state": deepcopy(state), "ts": _now(), "by": uid}
        if uid in room["participants"]:
            room["participants"][uid]["last_seen"] = _now()
        _prune(room)
        return snapshot(rid)


def update_annotation(room_id: str, user_id: str, pointer=None, stroke=None, clear: bool = False) -> Dict[str, Any]:
    rid = _clean_room(room_id)
    uid = _clean_user(user_id)
    with _LOCK:
        room = _get_room(rid)
        ann = room["annotation"]
        if clear:
            ann["strokes"] = []
            ann["pointer"] = None
        if pointer is not None:
            ann["pointer"] = pointer
        if stroke:
            pts = stroke.get("points") or []
            pts = pts[:_MAX_POINTS]
            ann["strokes"].append({"points": pts, "color": stroke.get("color") or "cyan"})
            ann["strokes"] = ann["strokes"][-_MAX_STROKES:]
        ann["ts"] = _now()
        ann["by"] = uid
        if uid in room["participants"]:
            room["participants"][uid]["last_seen"] = _now()
        _prune(room)
        return snapshot(rid)


def snapshot(room_id: str) -> Dict[str, Any]:
    rid = _clean_room(room_id)
    room = _get_room(rid)
    _prune(room)
    return {
        "room_id": rid,
        "participants": [{"user_id": uid, "name": data["name"]} for uid, data in sorted(room["participants"].items())],
        "scene": deepcopy(room["scene"]),
        "annotation": deepcopy(room["annotation"]),
    }
