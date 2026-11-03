"""
core/personalizer.py
Per-student adaptive personalization layer ("LoRA-lite" MVP).

Full LoRA fine-tuning per student is heavy for an 8GB Jetson to do live.
This MVP achieves the same *behavioural* goal — the system teaches each
student differently based on their history — via:
  1. Tracking weak/strong topics per student in SQLite.
  2. Biasing RAG retrieval toward topics the student struggles with.
  3. Injecting a short personalization note into the system prompt.

This is upgrade-compatible: swap `build_personal_prompt()` internals for a
real per-student LoRA adapter later without touching callers.
"""
import logging
import sqlite3
import os
from collections import Counter
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "personalization.db")


def _get_conn():
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS topic_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            outcome TEXT NOT NULL,       -- 'weak' | 'strong'
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn


def record_outcome(student_id: str, topic: str, outcome: str):
    """outcome: 'weak' (confused/wrong answer) or 'strong' (understood/correct)."""
    if outcome not in ("weak", "strong"):
        return
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO topic_events (student_id, topic, outcome) VALUES (?, ?, ?)",
            (student_id, topic.strip().lower(), outcome),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"personalizer record_outcome error: {e}")


def get_weak_topics(student_id: str, limit: int = 5) -> List[str]:
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT topic FROM topic_events WHERE student_id=? AND outcome='weak' "
            "ORDER BY ts DESC LIMIT 50",
            (student_id,),
        ).fetchall()
        conn.close()
        counts = Counter(r[0] for r in rows)
        return [t for t, _ in counts.most_common(limit)]
    except Exception as e:
        logger.error(f"personalizer get_weak_topics error: {e}")
        return []


def build_personal_prompt(base_system_prompt: str, student_id: Optional[str]) -> str:
    """Append a short, targeted personalization note to the system prompt."""
    if not student_id:
        return base_system_prompt

    weak = get_weak_topics(student_id)
    if not weak:
        return base_system_prompt

    note = (
        f"\n\nPersonalization note: this student has previously struggled with: "
        f"{', '.join(weak)}. When these topics come up, explain more slowly, "
        f"use simpler words and a concrete example, and check understanding."
    )
    return base_system_prompt + note


def retrieval_query_boost(question: str, student_id: Optional[str]) -> str:
    """
    Lightly biases the RAG query by appending the student's weak topics,
    so retrieval favours chunks connecting the new question to known gaps.
    """
    if not student_id:
        return question
    weak = get_weak_topics(student_id, limit=2)
    if not weak:
        return question
    return f"{question} ({' '.join(weak)})"
