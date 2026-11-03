"""Safe local assistant features inspired by the second JARVIS archive."""
from __future__ import annotations

import datetime as _datetime
import os
from pathlib import Path
from typing import Any, Dict


_NOTES_DIR = Path(__file__).resolve().parent.parent / "data" / "assistant_notes"


def current_datetime() -> Dict[str, str]:
    now = _datetime.datetime.now().astimezone()
    return {
        "date": now.strftime("%b %d %Y"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": now.tzname() or "local",
    }


def system_status() -> Dict[str, str]:
    """Return lightweight local process/host information without shell access."""
    status = {
        "platform": os.name,
        "cpu_count": str(os.cpu_count() or 1),
        "process_id": str(os.getpid()),
    }
    try:
        import psutil
    except ImportError:
        status["memory"] = "unavailable (install psutil for memory metrics)"
        return status
    status["cpu_percent"] = f"{psutil.cpu_percent(interval=None):.1f}"
    memory = psutil.virtual_memory()
    status["memory_percent"] = f"{memory.percent:.1f}"
    battery = psutil.sensors_battery()
    if battery is not None:
        status["battery_percent"] = f"{battery.percent:.1f}"
    return status


def save_note(text: str) -> Dict[str, str]:
    note = str(text or "").strip()
    if not note:
        raise ValueError("Note text is required")
    if len(note) > 4000:
        raise ValueError("Note exceeds 4000 characters")
    _NOTES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = _NOTES_DIR / f"{stamp}-note.txt"
    path.write_text(note + "\n", encoding="utf-8")
    return {"path": str(path), "saved_at": _datetime.datetime.now().astimezone().isoformat(timespec="seconds")}


def daily_briefing(memory: Dict[str, Any], classroom: Dict[str, Any], learning: Dict[str, Any]) -> Dict[str, Any]:
    """Build a local-only briefing without contacting external services."""
    notes = list(_NOTES_DIR.glob("*.txt")) if _NOTES_DIR.exists() else []
    state = classroom.get("state") or {}
    mastery = classroom.get("mastery_gate") or {}
    return {
        "datetime": current_datetime(),
        "classroom": {
            "status": state.get("status", "standby"),
            "subject": state.get("subject"),
            "topic": state.get("topic"),
            "mastered_students": mastery.get("mastered_count", 0),
            "present_students": mastery.get("present_count", 0),
            "pending_students": len(mastery.get("pending", [])),
        },
        "memory": {
            "preferences": len(memory.get("preferences") or {}),
            "facts": sum(len(values) for values in (memory.get("user_facts") or {}).values()),
            "projects": len(memory.get("current_projects") or {}),
        },
        "learning": learning,
        "local_notes": len(notes),
    }
