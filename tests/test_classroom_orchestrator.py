import os
import tempfile

import core.classroom_orchestrator as c


def setup_module():
    fd, path = tempfile.mkstemp(prefix="hologuru_classroom_", suffix=".db")
    os.close(fd)
    os.unlink(path)
    c._DB_PATH = path
    c.init_db()


def test_mastery_gate_and_reteach():
    sid = "class-a"
    c.save_config(sid, {
        "class_students": ["s1", "s2"],
        "curriculum": {
            "Physics": {"units": [{"name": "Electricity", "lessons": [{"name": "Basics", "topics": ["Current", "Ohm's Law"]}]}]}
        },
    })
    c.start_topic(sid, subject="Physics")
    c.record_mastery(sid, "s1", True)
    c.record_mastery(sid, "s2", False)
    result = c.request_topic_advance(sid)
    assert result["action"] == "reteach_required"
    assert result["gate"]["can_advance"] is False
    c.record_mastery(sid, "s2", True)
    result = c.request_topic_advance(sid)
    assert result["action"] == "next_topic"
    assert result["state"]["topic"] == "Ohm's Law"


def test_visitor_pause_recall_resume():
    sid = "class-b"
    c.handle_event(sid, "visitor_enter")
    assert c.get_state(sid)["status"] == "paused_visitor"
    leave = c.handle_event(sid, "visitor_leave")
    assert leave["action"] == "ask_recall"
    resume = c.handle_event(sid, "student_recall_response", {"correct": False})
    assert resume["action"] == "short_recap_then_resume"
    assert c.get_state(sid)["status"] == "teaching"


def test_individual_break_never_pauses_class():
    sid = "class-c"
    state = c.get_state(sid)
    state["status"] = "teaching"
    c.save_state(state)
    leave = c.handle_event(sid, "student_break_leave", {"student_id": "s7"})
    ret = c.handle_event(sid, "student_break_return", {"student_id": "s7"})
    assert leave["action"] == "continue_lesson"
    assert ret["action"] == "continue_lesson"
    assert ret["state"]["status"] == "teaching"


def test_excuse_me_only_brief_acknowledgement():
    result = c.handle_event("class-d", "student_excuse_me")
    assert result["action"] == "brief_acknowledgement"
    assert result["speak"] == "Yes, come in."


def test_distraction_requires_temporal_streak():
    sid = "class-e"
    first = c.handle_event(sid, "distraction_signal", {"student_id": "s1", "detected": True})
    second = c.handle_event(sid, "distraction_signal", {"student_id": "s1", "detected": True})
    third = c.handle_event(sid, "distraction_signal", {"student_id": "s1", "detected": True})
    assert first["action"] == "observe_only"
    assert second["action"] == "observe_only"
    assert third["action"] == "gentle_reengage"
