"""High-quality RAG retrieval for HOLO-GURU.

Pipeline:
  query -> Weaviate hybrid search (BM25 + dense vector) -> BGE reranker
  -> best evidence -> grounded LLM context
"""
import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional

import weaviate.classes as wvc
from sentence_transformers import CrossEncoder

import config
from core.vector_store import embed_texts, make_book_filter
from core.tamil_morphology import expand_query_for_retrieval

logger = logging.getLogger(__name__)

_RELATION_WORDS = {
    "relationship", "related", "relation", "compare", "difference", "between",
    "connect", "connected", "depends", "affect", "cause", "effect", "why",
    "how", "epadi", "eppadi", "difference between",
}


def _retrieval_profile(query: str) -> Dict:
    """Tune hybrid retrieval automatically from the question shape."""
    q = query.lower()
    relation_like = any(term in q for term in _RELATION_WORDS)
    exact_like = bool(re.search(r"\b(define|meaning|formula|year|name|what is|enna)\b", q))

    if relation_like:
        return {"alpha": 0.75, "candidate_k": max(config.RAG_CANDIDATE_K, 18), "mode": "semantic"}
    if exact_like:
        return {"alpha": 0.45, "candidate_k": config.RAG_CANDIDATE_K, "mode": "balanced"}
    return {"alpha": config.HYBRID_ALPHA, "candidate_k": config.RAG_CANDIDATE_K, "mode": "hybrid"}


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    logger.info("Loading reranker: %s", config.RERANK_MODEL)
    return CrossEncoder(config.RERANK_MODEL)


def _rerank(query: str, hits: List[Dict], top_k: int) -> List[Dict]:
    if not hits:
        return []
    try:
        model = _get_reranker()
        pairs = [[query, h["text"]] for h in hits]
        scores = model.predict(pairs, show_progress_bar=False)
        for h, score in zip(hits, scores):
            h["rerank_score"] = float(score)
        hits.sort(key=lambda h: h["rerank_score"], reverse=True)
        return hits[:top_k]
    except Exception as exc:
        logger.warning("Reranker unavailable; using Weaviate order: %s", exc)
        return hits[:top_k]


def search(
    query: str,
    collection,
    top_k: int = 5,
    book_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """Hybrid search + cross-encoder reranking.

    Returns dictionaries containing text, source metadata, hybrid score and
    reranker score. Book filters are applied inside Weaviate.
    """
    if not query.strip():
        return []

    try:
        query_expanded = expand_query_for_retrieval(query)
        profile = _retrieval_profile(query_expanded)
        qvec = embed_texts([query_expanded], config.EMBED_MODEL)[0]
        response = collection.query.hybrid(
            query=query_expanded,
            vector=qvec,
            alpha=profile["alpha"],
            query_properties=["text^3", "section^2", "book_title"],
            filters=make_book_filter(book_ids),
            return_properties=["text", "book_title", "book_id", "page", "section"],
            return_metadata=wvc.query.MetadataQuery(score=True),
            limit=profile["candidate_k"],
        )

        candidates: List[Dict] = []
        for obj in response.objects:
            p = obj.properties
            text = (p.get("text") or "").strip()
            if not text:
                continue
            candidates.append({
                "text": text,
                "book_title": p.get("book_title") or "Unknown",
                "page": p.get("page") or "?",
                "book_id": p.get("book_id") or "",
                "section": p.get("section") or "",
                "score": float(obj.metadata.score or 0.0),
                "retrieval_mode": profile["mode"],
            })

        hits = _rerank(query, candidates, top_k)
        logger.debug("RAG: %d reranked hits for query='%s'", len(hits), query[:60])
        return hits

    except Exception as exc:
        logger.error("RAG search error: %s", exc)
        return []


def build_context(hits: List[Dict], max_chars: int = 4500) -> str:
    """Build a source-labelled evidence block for grounded generation."""
    if not hits:
        return ""

    parts = []
    total = 0
    for idx, h in enumerate(hits, 1):
        section = f" | Section: {h['section']}" if h.get("section") else ""
        snippet = (
            f"[Source {idx} | Book: {h['book_title']} | Page: {h['page']}{section}]\n"
            f"{h['text'].strip()}"
        )
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)

    if not parts:
        return ""

    return (
        "Use ONLY the evidence below. Prefer facts supported by multiple excerpts. "
        "If the evidence is incomplete or conflicting, say so instead of guessing. "
        "Answer clearly and cite the source number/book/page.\n\n"
        + "\n\n---\n\n".join(parts)
    )
