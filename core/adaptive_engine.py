"""
core/adaptive_engine.py
Combines emotion_detector + gesture_detector + personalizer signals into a
single teaching decision, sitting between STT output and the ReAct loop.

Also applies NLP intent routing and Tamil/Tanglish morphology expansion so
retrieval and answer style match the student's question type.
"""
import logging
from typing import Dict, Optional

from core.personalizer import build_personal_prompt, retrieval_query_boost, record_outcome
from core.nlp_intent import classify_intent, intent_system_addon
from core.tamil_morphology import expand_query_for_retrieval, simplify_tamil_for_display

logger = logging.getLogger(__name__)

_confusion_streak: Dict[str, int] = {}
CONFUSION_THRESHOLD = 2


def process_signals(
    student_id: Optional[str],
    base_system_prompt: str,
    question: str,
    emotion_bucket: Optional[str] = None,
    gesture: Optional[str] = None,
    preferred_language: Optional[str] = None,
) -> Dict:
    sid = student_id or "anonymous"

    intent_info = classify_intent(question or "")
    addon = intent_system_addon(intent_info)

    decision = {
        "simplify": bool(intent_info.get("prefer_simple")),
        "switch_language": None,
        "pause": False,
        "wants_to_speak": False,
        "understood": None,
        "system_prompt": base_system_prompt,
        "boosted_question": question,
        "intent": intent_info.get("intent") or "unknown",
        "intent_confidence": float(intent_info.get("confidence") or 0.0),
        "intent_style": intent_info.get("style") or {},
        "skip_rag": bool(intent_info.get("skip_rag")),
        "question_display": simplify_tamil_for_display(question or ""),
    }

    if gesture == "palm_stop":
        decision["pause"] = True
    elif gesture == "raised_hand":
        decision["wants_to_speak"] = True
    elif gesture == "thumbs_up":
        decision["understood"] = True
        record_outcome(sid, _last_topic_hint(question), "strong")
        _confusion_streak[sid] = 0
    elif gesture == "thumbs_down":
        decision["understood"] = False
        record_outcome(sid, _last_topic_hint(question), "weak")
        _confusion_streak[sid] = _confusion_streak.get(sid, 0) + 1

    if emotion_bucket in ("confused", "frustrated"):
        _confusion_streak[sid] = _confusion_streak.get(sid, 0) + 1
    elif emotion_bucket == "engaged":
        _confusion_streak[sid] = 0

    if _confusion_streak.get(sid, 0) >= CONFUSION_THRESHOLD:
        decision["simplify"] = True
        if preferred_language and preferred_language != "en":
            decision["switch_language"] = preferred_language
        else:
            decision["switch_language"] = "ta"
        logger.info(
            "Adaptive: student %s confused %sx -> simplifying + language switch",
            sid, _confusion_streak[sid],
        )

    prompt = build_personal_prompt(base_system_prompt, sid)
    if addon:
        prompt += "\n\n" + addon
    if decision["simplify"]:
        prompt += (
            "\n\nThe student currently looks confused. Simplify your explanation: "
            "shorter sentences, one idea at a time, use a real-life example."
        )
    if decision["switch_language"] == "ta":
        prompt += (
            "\n\nPrefer clear Tamil or simple Tanglish phrasing when appropriate "
            "for this student."
        )
    decision["system_prompt"] = prompt

    boosted = retrieval_query_boost(question, sid)
    # Intent retrieval query already morphology-aware; expand again after personalizer.
    base_q = intent_info.get("retrieval_query") or boosted or question
    decision["boosted_question"] = expand_query_for_retrieval(base_q)

    return decision


def _last_topic_hint(question: str) -> str:
    words = [w for w in (question or "").strip().split() if len(w) > 3]
    return " ".join(words[:3]) if words else (question or "")[:30]
