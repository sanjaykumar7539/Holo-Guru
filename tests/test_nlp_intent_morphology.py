from core.nlp_intent import classify_intent, intent_system_addon
from core.tamil_morphology import (
    expand_query_for_retrieval,
    morphological_variants,
    stem_tamil_token,
    normalize_tanglish_token,
    has_tamil,
)
from core.adaptive_engine import process_signals


def test_intent_definition_and_how():
    d = classify_intent("What is Ohm's law?")
    assert d["intent"] in {"definition", "factual"}
    h = classify_intent("How to solve this problem step by step?")
    assert h["intent"] == "how_to"
    c = classify_intent("Difference between series and parallel circuits")
    assert c["intent"] == "compare"


def test_intent_greeting_and_holo():
    assert classify_intent("vanakkam")["intent"] == "greeting"
    assert classify_intent("/holo orbit")["intent"] == "hologram_control"
    assert classify_intent("வணக்கம்")["intent"] == "greeting"


def test_intent_tamil_cues():
    q = classify_intent("ஒளியின் வேகம் என்ன?")
    assert q["intent"] in {"definition", "factual", "unknown"}
    assert q["retrieval_query"]


def test_tanglish_map_and_stem():
    assert normalize_tanglish_token("epadi") == "how"
    assert normalize_tanglish_token("iruthayam") == "heart"
    # suffix peel on a longish Tamil-like token if present
    assert stem_tamil_token("புத்தகங்கள்") != ""


def test_query_expansion():
    variants = morphological_variants("epadi force work aagum")
    assert variants
    expanded = expand_query_for_retrieval("epadi force work")
    assert "how" in expanded.lower() or "epadi" in expanded.lower()


def test_adaptive_includes_intent():
    d = process_signals("s1", "BASE PROMPT", "Define photosynthesis", None, None, "en")
    assert d["intent"] in {"definition", "factual"}
    assert "intent" in d
    assert d["boosted_question"]
    assert "BASE PROMPT" in d["system_prompt"] or "definition" in d["system_prompt"].lower() or "Student intent" in d["system_prompt"]


def test_intent_system_addon_nonempty():
    info = classify_intent("Why does current flow?")
    addon = intent_system_addon(info)
    assert "intent" in addon.lower() or "cause" in addon.lower() or info["intent"] in addon
