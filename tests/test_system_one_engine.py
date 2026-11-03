import core.system_one_engine as s


def test_rule_route_relation_question():
    out = s._fallback_route("How are photosynthesis and respiration related?")
    assert out["route"] == "both"
    assert out["provider"] == "rules"


def test_rule_route_factual_question():
    out = s._fallback_route("Define photosynthesis")
    assert out["route"] == "rag"


def test_choice_normalization():
    assert s._normalize_choice({"choice": "GraphRAG"}) == "kag"
    assert s._normalize_choice({"choice": "RAG+KAG"}) == "both"
