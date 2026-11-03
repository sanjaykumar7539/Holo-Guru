"""Classroom orchestration for HOLO-GURU.

This module deliberately contains no camera, microphone, LLM, or 3D code.
It is the deterministic teaching/state foundation that those adapters call.

Responsibilities
----------------
* timetable -> fast subject/staff-period switching without model reloads
* curriculum -> subject / unit / lesson / topic progression
* teaching-style profiles -> teacher-specific delivery preferences
* whole-class mastery gate -> do not advance until all PRESENT students master
* adaptive reteaching -> rotate explanation method instead of repeating verbatim
* visitor interruption -> pause, checkpoint, recall, resume
* individual student break -> never pause the class; only answer "excuse me"
* whole-class break -> pause/checkpoint/resume
* distraction -> temporal signal counter + gentle re-engagement action
* wellbeing -> caring reaction only; no diagnosis
* durable SQLite state -> survives application restarts

The API returns ACTIONS (e.g. ``pause_lesson``, ``short_recap``) rather than
performing UI/audio actions directly.  This keeps the state machine testable.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import config

_DB_PATH = getattr(config, "CLASSROOM_DB_PATH", os.path.join(config.BASE_DIR, "classroom_state.db"))
_TZ = ZoneInfo(getattr(config, "CLASSROOM_TIMEZONE", "Asia/Kolkata"))
_LOCK = threading.RLock()

RETEACH_STRATEGIES = [
    "real_world_example",
    "analogy",
    "visual_or_diagram",
    "step_by_step",
    "storytelling",
    "live_demonstration",
    "previous_topic_connection",
    "guided_discussion",
]

DEFAULT_TEACHING_STYLE = {
    "name": "balanced_real_teacher",
    "explanation_length": "short",
    "tone": "natural, warm, classroom-teacher style",
    "preferred_methods": ["simple_explanation", "real_world_example", "visual_or_diagram"],
    "avoid": ["long_lecture", "unnecessary_jargon", "test_like_questioning"],
}


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(_DB_PATH, timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db() -> None:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    with _LOCK, _connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS classroom_sessions (
                session_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS classroom_configs (
                session_id TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )


def _now_iso() -> str:
    return datetime.now(_TZ).isoformat(timespec="seconds")


def _default_state(session_id: str) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "status": "idle",  # idle|teaching|paused_visitor|paused_class_break|awaiting_recall
        "period": None,
        "subject": None,
        "staff_id": None,
        "staff_style": deepcopy(DEFAULT_TEACHING_STYLE),
        "unit": None,
        "lesson": None,
        "topic": None,
        "topic_index": -1,
        "lesson_index": -1,
        "unit_index": -1,
        "checkpoint": None,
        "recall_required": False,
        "short_recap_required": False,
        "reteach_round": 0,
        "reteach_strategy": None,
        "students": {},  # student_id -> present|absent
        "mastery": {},   # student_id -> mastered|not_mastered|pending|absent
        "distraction_streak": {},
        "active_break_students": [],
        "last_action": None,
        "updated_at": _now_iso(),
    }


def get_config(session_id: str) -> Dict[str, Any]:
    init_db()
    with _LOCK, _connect() as con:
        row = con.execute("SELECT config_json FROM classroom_configs WHERE session_id=?", (session_id,)).fetchone()
    return json.loads(row["config_json"]) if row else {
        "timetable": [],
        "curriculum": {},
        "staff_styles": {},
        "class_students": [],
    }


def save_config(session_id: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    init_db()
    normalized = {
        "timetable": cfg.get("timetable") or [],
        "curriculum": cfg.get("curriculum") or {},
        "staff_styles": cfg.get("staff_styles") or {},
        "class_students": cfg.get("class_students") or [],
    }
    with _LOCK, _connect() as con:
        con.execute(
            """INSERT INTO classroom_configs(session_id, config_json, updated_at)
               VALUES(?,?,?) ON CONFLICT(session_id) DO UPDATE SET
               config_json=excluded.config_json, updated_at=excluded.updated_at""",
            (session_id, json.dumps(normalized, ensure_ascii=False), _now_iso()),
        )
    state = get_state(session_id)
    if normalized["class_students"]:
        state["students"] = {str(s): "present" for s in normalized["class_students"]}
        _reset_mastery(state)
        save_state(state)
    return normalized


def get_state(session_id: str) -> Dict[str, Any]:
    init_db()
    with _LOCK, _connect() as con:
        row = con.execute("SELECT state_json FROM classroom_sessions WHERE session_id=?", (session_id,)).fetchone()
    if row:
        return json.loads(row["state_json"])
    state = _default_state(session_id)
    save_state(state)
    return state


def save_state(state: Dict[str, Any]) -> Dict[str, Any]:
    init_db()
    state["updated_at"] = _now_iso()
    with _LOCK, _connect() as con:
        con.execute(
            """INSERT INTO classroom_sessions(session_id, state_json, updated_at)
               VALUES(?,?,?) ON CONFLICT(session_id) DO UPDATE SET
               state_json=excluded.state_json, updated_at=excluded.updated_at""",
            (state["session_id"], json.dumps(state, ensure_ascii=False), state["updated_at"]),
        )
    return state


def _parse_hhmm(v: str) -> int:
    h, m = [int(x) for x in v.split(":", 1)]
    return h * 60 + m


def current_period(cfg: Dict[str, Any], now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    now = now or datetime.now(_TZ)
    weekday = now.strftime("%A").lower()
    minute = now.hour * 60 + now.minute
    for p in cfg.get("timetable", []):
        days = [str(d).lower() for d in p.get("days", [])]
        if days and weekday not in days:
            continue
        try:
            if _parse_hhmm(p["start"]) <= minute < _parse_hhmm(p["end"]):
                return deepcopy(p)
        except (KeyError, ValueError):
            continue
    return None


def next_period(cfg: Dict[str, Any], now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    now = now or datetime.now(_TZ)
    weekday = now.strftime("%A").lower()
    minute = now.hour * 60 + now.minute
    candidates = []
    for p in cfg.get("timetable", []):
        days = [str(d).lower() for d in p.get("days", [])]
        if days and weekday not in days:
            continue
        try:
            start = _parse_hhmm(p["start"])
        except (KeyError, ValueError):
            continue
        if start > minute:
            candidates.append((start, p))
    return deepcopy(min(candidates, key=lambda x: x[0])[1]) if candidates else None


def _style_for(cfg: Dict[str, Any], staff_id: Optional[str]) -> Dict[str, Any]:
    style = deepcopy(DEFAULT_TEACHING_STYLE)
    custom = (cfg.get("staff_styles") or {}).get(str(staff_id), {}) if staff_id else {}
    style.update(custom)
    return style


def sync_timetable(session_id: str, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Switch subject/staff context using lightweight state only; no model reload."""
    cfg, state = get_config(session_id), get_state(session_id)
    period = current_period(cfg, now)
    upcoming = next_period(cfg, now)
    action = "no_period"

    if period:
        period_key = f"{period.get('start','')}|{period.get('end','')}|{period.get('subject','')}|{period.get('staff_id','')}"
        old_key = (state.get("period") or {}).get("key") if state.get("period") else None
        if period_key != old_key:
            if state.get("topic"):
                state["checkpoint"] = _checkpoint(state, reason="period_switch")
            state["period"] = {**period, "key": period_key}
            state["subject"] = period.get("subject")
            state["staff_id"] = period.get("staff_id")
            state["staff_style"] = _style_for(cfg, state["staff_id"])
            state["status"] = "teaching"
            state["recall_required"] = False
            state["short_recap_required"] = False
            _load_first_or_saved_topic(state, cfg)
            action = "period_switched"
        else:
            action = "period_active"
    save_state(state)
    return {"action": action, "state": state, "next_period": upcoming}


def _curriculum_subject(cfg: Dict[str, Any], subject: Optional[str]) -> Dict[str, Any]:
    if not subject:
        return {}
    curriculum = cfg.get("curriculum") or {}
    return curriculum.get(subject) or curriculum.get(subject.lower()) or {}


def _load_first_or_saved_topic(state: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    data = _curriculum_subject(cfg, state.get("subject"))
    units = data.get("units") or []
    if not units:
        state.update({"unit": None, "lesson": None, "topic": None})
        return
    ui = min(max(state.get("unit_index", 0), 0), len(units) - 1)
    lessons = units[ui].get("lessons") or []
    if not lessons:
        state.update({"unit": units[ui].get("name"), "lesson": None, "topic": None})
        return
    li = min(max(state.get("lesson_index", 0), 0), len(lessons) - 1)
    topics = lessons[li].get("topics") or []
    ti = min(max(state.get("topic_index", 0), 0), max(len(topics) - 1, 0))
    state["unit_index"], state["lesson_index"], state["topic_index"] = ui, li, ti
    state["unit"] = units[ui].get("name")
    state["lesson"] = lessons[li].get("name")
    state["topic"] = (topics[ti].get("name") if isinstance(topics[ti], dict) else topics[ti]) if topics else None
    _reset_mastery(state)


def start_topic(session_id: str, subject: Optional[str] = None, unit_index: int = 0,
                lesson_index: int = 0, topic_index: int = 0) -> Dict[str, Any]:
    cfg, state = get_config(session_id), get_state(session_id)
    if subject:
        state["subject"] = subject
    state["unit_index"] = int(unit_index)
    state["lesson_index"] = int(lesson_index)
    state["topic_index"] = int(topic_index)
    _load_first_or_saved_topic(state, cfg)
    state["status"] = "teaching"
    state["reteach_round"] = 0
    state["reteach_strategy"] = None
    state["last_action"] = "topic_started"
    save_state(state)
    return {"action": "topic_started", "state": state}


def _reset_mastery(state: Dict[str, Any]) -> None:
    students = state.get("students") or {}
    state["mastery"] = {
        sid: ("pending" if status == "present" else "absent")
        for sid, status in students.items()
    }


def set_attendance(session_id: str, present: Iterable[str], absent: Iterable[str] = ()) -> Dict[str, Any]:
    state = get_state(session_id)
    state["students"] = {str(s): "present" for s in present}
    for s in absent:
        state["students"][str(s)] = "absent"
    _reset_mastery(state)
    save_state(state)
    return {"action": "attendance_updated", "state": state}


def record_mastery(session_id: str, student_id: str, understood: Optional[bool]) -> Dict[str, Any]:
    state = get_state(session_id)
    sid = str(student_id)
    if state.get("students", {}).get(sid) == "absent":
        status = "absent"
    elif understood is True:
        status = "mastered"
    elif understood is False:
        status = "not_mastered"
    else:
        status = "pending"
    state.setdefault("mastery", {})[sid] = status
    gate = mastery_gate(state)
    state["last_action"] = "mastery_recorded"
    save_state(state)
    return {"action": "mastery_recorded", "student_id": sid, "status": status, "gate": gate, "state": state}


def mastery_gate(state: Dict[str, Any]) -> Dict[str, Any]:
    present = [s for s, v in (state.get("students") or {}).items() if v == "present"]
    mastery = state.get("mastery") or {}
    pending = [s for s in present if mastery.get(s, "pending") == "pending"]
    weak = [s for s in present if mastery.get(s) == "not_mastered"]
    mastered = [s for s in present if mastery.get(s) == "mastered"]
    can_advance = bool(present) and not pending and not weak
    return {
        "can_advance": can_advance,
        "present_count": len(present),
        "mastered_count": len(mastered),
        "pending": pending,
        "needs_reteach": weak,
    }


def request_topic_advance(session_id: str, teacher_override: bool = False) -> Dict[str, Any]:
    cfg, state = get_config(session_id), get_state(session_id)
    gate = mastery_gate(state)
    if not teacher_override and not gate["can_advance"]:
        if gate["needs_reteach"]:
            strategy = _next_reteach_strategy(state)
            action = "reteach_required"
        else:
            strategy = None
            action = "mastery_check_required"
        state["last_action"] = action
        save_state(state)
        return {"action": action, "strategy": strategy, "gate": gate, "state": state}

    advanced = _advance_topic_indices(state, cfg)
    state["reteach_round"] = 0
    state["reteach_strategy"] = None
    _reset_mastery(state)
    state["last_action"] = advanced
    save_state(state)
    return {"action": advanced, "gate": gate, "state": state}


def _advance_topic_indices(state: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    data = _curriculum_subject(cfg, state.get("subject"))
    units = data.get("units") or []
    if not units:
        return "curriculum_missing"
    ui, li, ti = state["unit_index"], state["lesson_index"], state["topic_index"]
    if not 0 <= ui < len(units):
        return "curriculum_missing"
    lessons = units[ui].get("lessons") or []
    if not 0 <= li < len(lessons):
        return "curriculum_missing"
    topics = lessons[li].get("topics") or []
    if not 0 <= ti < len(topics):
        return "curriculum_missing"
    if ti + 1 < len(topics):
        state["topic_index"] += 1
        action = "next_topic"
    elif li + 1 < len(lessons):
        state["lesson_index"] += 1
        state["topic_index"] = 0
        action = "next_lesson"
    elif ui + 1 < len(units):
        state["unit_index"] += 1
        state["lesson_index"] = 0
        state["topic_index"] = 0
        action = "next_unit"
    else:
        state["status"] = "idle"
        return "subject_completed"
    _load_first_or_saved_topic(state, cfg)
    return action


def _next_reteach_strategy(state: Dict[str, Any]) -> str:
    state["reteach_round"] = int(state.get("reteach_round", 0)) + 1
    strategy = RETEACH_STRATEGIES[(state["reteach_round"] - 1) % len(RETEACH_STRATEGIES)]
    state["reteach_strategy"] = strategy
    return strategy


def _checkpoint(state: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "reason": reason,
        "subject": state.get("subject"),
        "unit": state.get("unit"),
        "lesson": state.get("lesson"),
        "topic": state.get("topic"),
        "topic_index": state.get("topic_index"),
        "saved_at": _now_iso(),
    }


def handle_event(session_id: str, event: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deterministic event state machine for real classroom behaviour."""
    payload = payload or {}
    state = get_state(session_id)
    event = (event or "").strip().lower()
    action: Dict[str, Any] = {"action": "ignored", "speak": None}

    if event == "visitor_enter":
        state["checkpoint"] = _checkpoint(state, "visitor")
        state["status"] = "paused_visitor"
        state["recall_required"] = False
        action = {"action": "pause_lesson", "speak": None}

    elif event == "visitor_leave" and state.get("status") == "paused_visitor":
        state["status"] = "awaiting_recall"
        state["recall_required"] = True
        action = {"action": "ask_recall", "speak": "Seri students, namma enga vittom?"}

    elif event == "student_recall_response" and state.get("recall_required"):
        correct = payload.get("correct")
        state["recall_required"] = False
        state["status"] = "teaching"
        if correct is True:
            state["short_recap_required"] = False
            action = {"action": "resume_from_checkpoint", "speak": "Correct, anga irundhu continue pannalaam."}
        else:
            state["short_recap_required"] = True
            action = {"action": "short_recap_then_resume", "speak": None}

    elif event == "student_break_leave":
        sid = str(payload.get("student_id") or "unknown")
        active = set(state.get("active_break_students") or [])
        active.add(sid)
        state["active_break_students"] = sorted(active)
        action = {"action": "continue_lesson", "speak": None}

    elif event == "student_break_return":
        sid = str(payload.get("student_id") or "unknown")
        active = set(state.get("active_break_students") or [])
        active.discard(sid)
        state["active_break_students"] = sorted(active)
        action = {"action": "continue_lesson", "speak": None}

    elif event == "student_excuse_me":
        action = {"action": "brief_acknowledgement", "speak": "Yes, come in."}

    elif event == "class_break_start":
        state["checkpoint"] = _checkpoint(state, "class_break")
        state["status"] = "paused_class_break"
        action = {"action": "pause_lesson", "speak": None}

    elif event == "class_break_end" and state.get("status") == "paused_class_break":
        state["status"] = "teaching"
        state["short_recap_required"] = True
        action = {"action": "short_recap_then_resume", "speak": None}

    elif event == "distraction_signal":
        sid = str(payload.get("student_id") or "class")
        detected = bool(payload.get("detected", True))
        streaks = state.setdefault("distraction_streak", {})
        streaks[sid] = (int(streaks.get(sid, 0)) + 1) if detected else 0
        threshold = int(getattr(config, "DISTRACTION_STREAK_THRESHOLD", 3))
        if streaks[sid] >= threshold:
            streaks[sid] = 0
            action = {"action": "gentle_reengage", "speak": payload.get("prompt") or "Students, inga konjam focus pannunga; indha point important."}
        else:
            action = {"action": "observe_only", "speak": None}

    elif event == "wellbeing_observation":
        # Deliberately no health/emotion diagnosis. React only to observable concern.
        action = {
            "action": "gentle_wellbeing_reaction",
            "speak": payload.get("prompt") or "Enna pa, tired-ah irukkiya? Thevai na konjam water kudichittu vaa.",
        }

    elif event == "student_says_fine":
        action = {"action": "continue_lesson", "speak": "Seri, gavanichukko."}

    state["last_action"] = action["action"]
    save_state(state)
    return {**action, "state": state}


def teaching_prompt(session_id: str) -> str:
    """Build compact, deterministic teacher behaviour instructions for the LLM."""
    state = get_state(session_id)
    style = state.get("staff_style") or DEFAULT_TEACHING_STYLE
    gate = mastery_gate(state)
    method = state.get("reteach_strategy") or "simple_explanation"
    return f"""
CLASSROOM TEACHING MODE
Current subject: {state.get('subject') or 'not set'}
Unit: {state.get('unit') or 'not set'}
Lesson: {state.get('lesson') or 'not set'}
Topic: {state.get('topic') or 'not set'}
Teacher style: {style.get('name', 'balanced_real_teacher')}
Preferred methods: {', '.join(style.get('preferred_methods', [])) or 'simple explanation, real-world example'}
Current method: {method}

Behaviour rules:
- Speak like a real classroom teacher: simple, short, natural, and clear.
- Teach only the current topic; do not jump ahead without the curriculum manager.
- Start with a concise explanation, then use a relevant real-world example or visual when it genuinely helps.
- Connect to the previous topic when useful. Use board-style key points, demonstration, storytelling, discussion, or analogy only when relevant.
- If reteaching is requested, DO NOT repeat the same wording; use the specified different method ({method}).
- After a topic explanation, invite natural student interaction/understanding feedback; do not turn every interaction into a test or quiz.
- Topic advancement is controlled externally. Never claim the class has mastered a topic unless the mastery gate says so.
- If some students have not mastered the topic, help them using a different method while keeping the explanation respectful and concise.
- If interruption state says pause, do not continue teaching until resumed.
- Individual student breaks must not interrupt the lesson. Do not recap for a returning student unless explicitly asked.
- For wellbeing observations, respond kindly without diagnosing health or emotion.
- Keep answers grounded in retrieved textbook evidence and cite sources when available.
Class mastery snapshot: {gate['mastered_count']}/{gate['present_count']} present students mastered; pending={len(gate['pending'])}; reteach={len(gate['needs_reteach'])}.
""".strip()


def state_summary(session_id: str) -> Dict[str, Any]:
    state = get_state(session_id)
    cfg = get_config(session_id)
    return {
        "state": state,
        "mastery_gate": mastery_gate(state),
        "current_period": current_period(cfg),
        "next_period": next_period(cfg),
    }
