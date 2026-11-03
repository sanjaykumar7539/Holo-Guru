"""NLP intent routing for HOLO-GURU classroom questions.

Lightweight, offline-friendly intent classifier (rules + cues).
Routes questions toward the right retrieval / response style without a
separate heavy NLU service.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.tamil_morphology import expand_query_for_retrieval, has_tamil, normalize_text

# Intent labels used across adaptive + RAG layers.
INTENTS = (
    "greeting",
    "definition",
    "factual",
    "how_to",
    "compare",
    "cause_effect",
    "explain_simple",
    "hologram_control",
    "quiz",
    "off_topic",
    "unknown",
)

_HOLO_CUES = re.compile(
    r"\b(holo|hologram|/holo|orbit|explode|x-?ray|slice|freeze|measure|clip|"
    r"show\s+(heart|engine|brain|cell|solar|anatomy|molecule|imported))\b",
    re.I,
)
_GREET = re.compile(
    r"^\s*(hi|hello|hey|vanakkam|vannakkam|வணக்கம்|thanks|thank\s*you|nandri|நன்றி|"
    r"ok|okay|sari|சரி|bye|good\s*(morning|night|evening))\s*[!.]*\s*$",
    re.I,
)
_DEFINE = re.compile(
    r"\b(define|definition|meaning|what\s+is|what's|what\s+are|enna\s+athu|enna|"
    r"என்ன|பொருள்|வரையறை|define\s+the)\b",
    re.I,
)
_HOW = re.compile(
    r"\b(how\s+to|how\s+do|how\s+does|how\s+can|steps?|procedure|method|"
    r"epadi|eppadi|எப்படி|முறை|படிகள்)\b",
    re.I,
)
_COMPARE = re.compile(
    r"\b(compare|difference|different|versus|vs\.?|between|oppidu|verupadu|"
    r"வேறுபாடு|ஒப்பிடு|இடையே)\b",
    re.I,
)
_CAUSE = re.compile(
    r"\b(why|cause|effect|because|reason|yean|yen|yaen|ஏன்|காரணம்|விளைவு)\b",
    re.I,
)
_SIMPLE = re.compile(
    r"\b(simple|simply|easy|eliya|eliyaga|simplify|slow|again|மீண்டும்|"
    r"எளிதாக|சுலபமாக|simple\s+words)\b",
    re.I,
)
_QUIZ = re.compile(
    r"\b(quiz|test\s+me|ask\s+me|question\s+me|mcq|practice)\b",
    re.I,
)
_OFF = re.compile(
    r"\b(weather|cricket|movie|song|joke|bitcoin|stock|celebrity|dating)\b",
    re.I,
)


def classify_intent(question: str) -> Dict[str, Any]:
    """Classify classroom question intent and return routing metadata."""
    q = normalize_text(question or "")
    low = q.lower()
    scores: Dict[str, float] = {k: 0.0 for k in INTENTS}

    if not q:
        return _result("unknown", 0.0, q)

    if _GREET.match(q) or len(q.split()) <= 2 and _GREET.search(q):
        scores["greeting"] += 3.0
    if _HOLO_CUES.search(q):
        scores["hologram_control"] += 3.0
    if _DEFINE.search(q):
        scores["definition"] += 2.2
    if _HOW.search(q):
        scores["how_to"] += 2.2
    if _COMPARE.search(q):
        scores["compare"] += 2.4
    if _CAUSE.search(q):
        scores["cause_effect"] += 2.0
    if _SIMPLE.search(q):
        scores["explain_simple"] += 2.0
    if _QUIZ.search(q):
        scores["quiz"] += 2.5
    if _OFF.search(q):
        scores["off_topic"] += 2.0

    # Default factual bias for content questions with some substance.
    if len(q.split()) >= 3 and scores["greeting"] < 1 and scores["hologram_control"] < 1:
        scores["factual"] += 1.0
    if has_tamil(q):
        # Slight boost to definition/how when Tamil interrogatives present.
        if re.search(r"என்ன|யார|எங்கே|எப்போது", q):
            scores["definition"] += 0.6
        if re.search(r"எப்படி|எவ்வாறு", q):
            scores["how_to"] += 0.6
        if re.search(r"ஏன்", q):
            scores["cause_effect"] += 0.6

    intent = max(scores, key=scores.get)
    conf = scores[intent]
    # Normalize confidence to 0..1 soft scale
    confidence = round(min(1.0, conf / 3.0), 3)
    if conf < 0.5:
        intent = "unknown"
        confidence = 0.2

    return _result(intent, confidence, q, scores)


def _result(intent: str, confidence: float, question: str, scores: Optional[Dict] = None) -> Dict[str, Any]:
    retrieval_query = expand_query_for_retrieval(question) if intent not in {"greeting", "hologram_control"} else question
    style = _style_for(intent)
    return {
        "intent": intent,
        "confidence": confidence,
        "retrieval_query": retrieval_query,
        "style": style,
        "skip_rag": intent in {"greeting", "hologram_control"},
        "prefer_simple": intent == "explain_simple" or style.get("simplify", False),
        "scores": scores or {},
    }


def _style_for(intent: str) -> Dict[str, Any]:
    if intent == "definition":
        return {"structure": "definition", "simplify": False, "list_steps": False}
    if intent == "how_to":
        return {"structure": "steps", "simplify": False, "list_steps": True}
    if intent == "compare":
        return {"structure": "compare", "simplify": False, "list_steps": False}
    if intent == "cause_effect":
        return {"structure": "cause_effect", "simplify": False, "list_steps": False}
    if intent == "explain_simple":
        return {"structure": "simple", "simplify": True, "list_steps": False}
    if intent == "quiz":
        return {"structure": "quiz", "simplify": False, "list_steps": False}
    if intent == "greeting":
        return {"structure": "chat", "simplify": False, "list_steps": False}
    if intent == "hologram_control":
        return {"structure": "control", "simplify": False, "list_steps": False}
    if intent == "off_topic":
        return {"structure": "redirect", "simplify": False, "list_steps": False}
    return {"structure": "factual", "simplify": False, "list_steps": False}


def intent_system_addon(intent_info: Dict[str, Any]) -> str:
    """Extra system-prompt guidance based on routed intent."""
    intent = intent_info.get("intent") or "unknown"
    style = intent_info.get("style") or {}
    lines: List[str] = [f"Student intent route: {intent}."]
    structure = style.get("structure")
    if structure == "definition":
        lines.append("Answer with a clear definition first, then a short example from the evidence.")
    elif structure == "steps":
        lines.append("Answer as numbered steps. Keep each step short and grounded in the excerpts.")
    elif structure == "compare":
        lines.append("Compare point-by-point (similarities then differences). Use only evidence.")
    elif structure == "cause_effect":
        lines.append("Explain cause → effect clearly. Do not invent causal links absent from evidence.")
    elif structure == "simple":
        lines.append("Use simpler words and shorter sentences. Avoid jargon unless defined.")
    elif structure == "quiz":
        lines.append("If asking a quiz question, keep it syllabus-grounded and wait for the student.")
    elif structure == "redirect":
        lines.append("Politely redirect to the uploaded lesson topics; do not answer off-topic chat.")
    elif structure == "control":
        lines.append("This is a hologram control request; acknowledge briefly if answered as text.")
    if style.get("list_steps"):
        lines.append("Prefer an ordered list format.")
    return " ".join(lines)
