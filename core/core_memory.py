"""Small persistent memory for user preferences and assistant context."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

import config

_PATH = Path(getattr(config, "CORE_MEMORY_PATH", Path(config.DATA_DIR) / "core_memory.json"))
_LOCK = threading.RLock()


def _defaults() -> Dict[str, Any]:
    return {
        "persona": {"role": "concise, helpful local AI co-pilot"},
        "user_facts": {},
        "preferences": {"wake_word": "Hey san", "response_style": "concise"},
        "current_projects": {},
    }


def _load() -> Dict[str, Any]:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _PATH.exists():
        data = _defaults()
        _PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data
    try:
        data = json.loads(_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = _defaults()
    return data if isinstance(data, dict) else _defaults()


def get_memory() -> Dict[str, Any]:
    with _LOCK:
        return _load()


def _save(data: Dict[str, Any]) -> None:
    _PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def remember(fact: str, category: str = "general") -> Dict[str, Any]:
    fact = str(fact or "").strip()
    category = str(category or "general").strip()[:40] or "general"
    if not fact:
        raise ValueError("Fact is required")
    if len(fact) > 500:
        raise ValueError("Fact exceeds 500 characters")
    with _LOCK:
        data = _load()
        bucket = data.setdefault("user_facts", {}).setdefault(category, [])
        if fact not in bucket:
            bucket.append(fact)
        _save(data)
        return data


def forget(fact: str, category: str = "general") -> Dict[str, Any]:
    with _LOCK:
        data = _load()
        bucket = data.get("user_facts", {}).get(category, [])
        if fact in bucket:
            bucket.remove(fact)
        _save(data)
        return data


def render_prompt_block() -> str:
    data = get_memory()
    lines = ["PERSISTENT ASSISTANT MEMORY"]
    for section in ("preferences", "user_facts", "current_projects"):
        values = data.get(section) or {}
        if not values:
            continue
        lines.append(f"{section}:")
        for key, value in values.items():
            if isinstance(value, list):
                for item in value[:20]:
                    lines.append(f"- {key}: {item}")
            else:
                lines.append(f"- {key}: {value}")
    return "\n".join(lines)
