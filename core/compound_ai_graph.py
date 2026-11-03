"""LangGraph orchestration for HOLO-GURU's Compound AI retrieval system.

Routes each non-trivial question across CAG, Hybrid RAG and KAG/GraphRAG,
then fuses grounded evidence. The graph is intentionally retrieval-only; final
LLM streaming and LettuceDetect verification stay in react_engine.py.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, TypedDict

import config
from core import cag_engine
from core.knowledge_graph import search as graph_search, build_context as graph_context
from core.rag_engine import search as rag_search, build_context as rag_context
from core.system_one_engine import decide_retrieval_route

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except Exception:  # graceful fallback if dependency was not installed yet
    StateGraph = START = END = None
    LANGGRAPH_AVAILABLE = False

_RELATION_RE = re.compile(
    r"\b(relationship|related|relation|compare|difference|between|connect|connected|depends|"
    r"affect|cause|effect|why|how|architecture|flow|link|epadi|eppadi)\b", re.I
)


class RetrievalState(TypedDict, total=False):
    question: str
    collection: object
    top_k: int
    book_ids: Optional[List[str]]
    cag: Optional[Dict]
    rag_hits: List[Dict]
    graph_hits: List[Dict]
    system1: Dict
    result: Dict


def _check_cag(state: RetrievalState) -> RetrievalState:
    state["cag"] = cag_engine.get(state["question"], state.get("book_ids"))
    return state


def _cag_route(state: RetrievalState) -> str:
    return "cag_hit" if state.get("cag") else "retrieve"


def _use_cag(state: RetrievalState) -> RetrievalState:
    cached = dict(state["cag"] or {})
    cached["stage"] = "cag"
    cached["retrieval_path"] = ["CAG"]
    state["result"] = cached
    return state


def _retrieve(state: RetrievalState) -> RetrievalState:
    question = state["question"]
    top_k = state["top_k"]
    book_ids = state.get("book_ids")

    if getattr(config, "SYSTEM1_ENABLED", True):
        decision = decide_retrieval_route(question)
    else:
        decision = {"route": "both" if _RELATION_RE.search(question) else "rag", "provider": "legacy_rules", "confidence": 1.0}
    state["system1"] = decision
    route = decision.get("route", "rag")
    logger.info("System-One retrieval decision: route=%s provider=%s confidence=%s", route, decision.get("provider"), decision.get("confidence"))

    state["rag_hits"] = []
    state["graph_hits"] = []
    if route in {"rag", "both"}:
        state["rag_hits"] = rag_search(question, state["collection"], top_k, book_ids)
    if route in {"kag", "both"} and getattr(config, "KAG_ENABLED", True):
        state["graph_hits"] = graph_search(question, book_ids, limit=max(3, min(top_k, 6)))

    # Safety net: a graph-only miss must still try grounded document retrieval.
    if route == "kag" and not state["graph_hits"]:
        state["rag_hits"] = rag_search(question, state["collection"], top_k, book_ids)
        state["system1"]["fallback"] = "rag_on_graph_miss"
    return state


def _fuse(state: RetrievalState) -> RetrievalState:
    r_hits = state.get("rag_hits") or []
    g_hits = state.get("graph_hits") or []
    parts = []
    paths = []
    if r_hits:
        ctx = rag_context(r_hits, max_chars=int(getattr(config, "RAG_CONTEXT_MAX_CHARS", 4200)))
        if ctx:
            parts.append(ctx)
            paths.append("RAG")
    if g_hits:
        ctx = graph_context(g_hits, max_chars=int(getattr(config, "KAG_CONTEXT_MAX_CHARS", 1800)))
        if ctx:
            parts.append(
                "Relationship evidence from the local knowledge graph. Use it only when it agrees with the document evidence:\n" + ctx
            )
            paths.append("KAG")

    context = "\n\n===== GRAPH-AUGMENTED EVIDENCE =====\n\n".join(parts)
    sources = []
    for h in [*r_hits, *g_hits]:
        title = h.get("book_title")
        if title and title not in sources:
            sources.append(title)
    result = {
        "context": context,
        "stage": "compound" if len(paths) > 1 else ("book" if paths else "none"),
        "sources": sources,
        "retrieval_path": paths,
        "system1": state.get("system1", {}),
    }
    state["result"] = result
    if context:
        cag_engine.put(state["question"], state.get("book_ids"), result)
    return state


def _fallback_run(question: str, collection, top_k: int, book_ids=None) -> Dict:
    state: RetrievalState = {
        "question": question, "collection": collection, "top_k": top_k, "book_ids": book_ids,
    }
    _check_cag(state)
    if state.get("cag"):
        _use_cag(state)
    else:
        _retrieve(state)
        _fuse(state)
    return state["result"]


def _build_graph():
    if not LANGGRAPH_AVAILABLE:
        return None
    graph = StateGraph(RetrievalState)
    graph.add_node("check_cag", _check_cag)
    graph.add_node("use_cag", _use_cag)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("fuse", _fuse)
    graph.add_edge(START, "check_cag")
    graph.add_conditional_edges("check_cag", _cag_route, {"cag_hit": "use_cag", "retrieve": "retrieve"})
    graph.add_edge("use_cag", END)
    graph.add_edge("retrieve", "fuse")
    graph.add_edge("fuse", END)
    return graph.compile()


_GRAPH = _build_graph()


def gather(question: str, collection, top_k: int, book_ids=None) -> Dict:
    """Run CAG -> Hybrid RAG + optional KAG -> evidence fusion."""
    if _GRAPH is None:
        logger.warning("langgraph unavailable; using equivalent deterministic fallback")
        return _fallback_run(question, collection, top_k, book_ids)
    out = _GRAPH.invoke({
        "question": question,
        "collection": collection,
        "top_k": top_k,
        "book_ids": book_ids,
    })
    return out.get("result") or {"context": "", "stage": "none", "sources": [], "retrieval_path": []}
