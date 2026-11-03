from core import cag_engine
from core import knowledge_graph


def test_cag_roundtrip():
    cag_engine.clear()
    value = {"context": "grounded evidence", "stage": "book", "sources": ["Book"]}
    cag_engine.put("What is AI?", ["b1"], value)
    hit = cag_engine.get("What is AI?", ["b1"])
    assert hit is not None
    assert hit["context"] == "grounded evidence"


def test_knowledge_graph_index_and_search(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge_graph, "_DB", str(tmp_path / "kg.db"))
    knowledge_graph.index_chunks(
        "b1",
        "AI Book",
        [{"text": "Artificial Intelligence uses Machine Learning for predictive systems.", "page": 1}],
    )
    hits = knowledge_graph.search("relationship between Artificial Intelligence and Machine Learning", ["b1"])
    assert hits
    assert hits[0]["book_id"] == "b1"
