"""Opt-in local feedback logs for improving assistant responses."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict

import config

_LOCK = threading.RLock()
_LAST_TURNS: Dict[str, Dict[str, Any]] = {}
_POSITIVE = re.compile(r"\b(perfect|thanks|thank you|exactly|good job|correct)\b", re.I)
_NEGATIVE = re.compile(r"\b(wrong|not like that|try again|incorrect|stop)\b", re.I)


def _enabled() -> bool:
    return bool(getattr(config, "ASSISTANT_LEARNING_ENABLED", True))


def _path(name: str) -> Path:
    return Path(config.DATA_DIR) / name


def _append(name: str, payload: Dict[str, Any]) -> None:
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def record_turn(session_id: str, question: str, answer: str, tools: list[str] | None = None) -> None:
    if not _enabled() or not question.strip() or not answer.strip():
        return
    with _LOCK:
        _LAST_TURNS[session_id] = {"question": question, "answer": answer, "tools": tools or [], "ts": time.time()}
        _append("live_sft.jsonl", {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "tools": tools or [],
            "ts": time.time(),
        })


def record_feedback(session_id: str, signal: str, comment: str = "") -> Dict[str, Any]:
    if not _enabled():
        return {"saved": False, "reason": "learning_disabled"}
    signal = signal.strip().lower()
    if signal not in {"positive", "negative"}:
        raise ValueError("signal must be positive or negative")
    with _LOCK:
        turn = _LAST_TURNS.get(session_id)
        if not turn:
            return {"saved": False, "reason": "no_recent_turn"}
        payload = {
            "prompt": turn["question"],
            "chosen": turn["answer"] if signal == "positive" else "",
            "rejected": turn["answer"] if signal == "negative" else "",
            "correction_hint": comment.strip()[:500],
            "signal": signal,
            "ts": time.time(),
        }
        _append("live_dpo.jsonl", payload)
        return {"saved": True, "signal": signal}


def infer_feedback(utterance: str) -> str | None:
    if _POSITIVE.search(utterance):
        return "positive"
    if _NEGATIVE.search(utterance):
        return "negative"
    return None


def stats() -> Dict[str, Any]:
    def count(name: str) -> int:
        path = _path(name)
        if not path.exists():
            return 0
        with path.open(encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    dpo = count("live_dpo.jsonl")
    target = int(getattr(config, "ASSISTANT_LEARNING_MIN_FEEDBACK", 200))
    return {"enabled": _enabled(), "sft_rows": count("live_sft.jsonl"), "feedback_rows": dpo, "retrain_ready": dpo >= target, "target": target}
