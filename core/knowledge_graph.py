"""Lightweight local Knowledge Graph for HOLO-GURU.

The graph is intentionally SQLite-backed so the Compound AI pipeline can run
on one local machine without Neo4j. During ingestion we extract conservative
entity co-occurrence edges from document sentences. At query time, matching
entities expand retrieval with relationship evidence (KAG / GraphRAG path).
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import config

_DB = getattr(config, "KNOWLEDGE_GRAPH_DB", os.path.join(config.DATA_DIR, "knowledge_graph.db"))
_LOCK = threading.RLock()

# Capitalised phrases + technical tokens. Kept conservative to avoid exploding
# the graph with every common word from a textbook.
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9+.#_-]{2,}(?:\s+[A-Z][A-Za-z0-9+.#_-]{2,}){0,3}|"
    r"[A-Za-z]+(?:AI|ML|LLM|RAG|API|CPU|GPU|IoT|NLP|CNN|LSTM|SQL|HTTP))\b"
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOP = {
    "The", "This", "That", "These", "Those", "There", "When", "Where", "What",
    "Which", "With", "From", "Into", "About", "Chapter", "Figure", "Table",
}


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB), exist_ok=True)
    con = sqlite3.connect(_DB, timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db() -> None:
    with _LOCK, _connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS kg_edges (
                book_id TEXT NOT NULL,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                sentence TEXT NOT NULL,
                page INTEGER,
                book_title TEXT,
                weight INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY(book_id, source, target, sentence)
            );
            CREATE INDEX IF NOT EXISTS idx_kg_source ON kg_edges(source);
            CREATE INDEX IF NOT EXISTS idx_kg_target ON kg_edges(target);
            CREATE INDEX IF NOT EXISTS idx_kg_book ON kg_edges(book_id);
            """
        )


def _entities(text: str) -> List[str]:
    found = []
    seen = set()
    for raw in _ENTITY_RE.findall(text or ""):
        ent = re.sub(r"\s+", " ", raw).strip(" .,:;()[]{}")
        if len(ent) < 3 or ent in _STOP:
            continue
        key = ent.casefold()
        if key not in seen:
            seen.add(key)
            found.append(ent)
    return found[:12]


def index_chunks(book_id: str, book_title: str, chunks: Sequence[Dict]) -> int:
    """Create entity co-occurrence edges from newly ingested chunks."""
    init_db()
    rows: List[Tuple] = []
    for chunk in chunks:
        page = int(chunk.get("page") or 0)
        for sentence in _SENTENCE_RE.split(chunk.get("text") or ""):
            sentence = sentence.strip()
            if len(sentence) < 30 or len(sentence) > 700:
                continue
            ents = _entities(sentence)
            if len(ents) < 2:
                continue
            # Limit each sentence to a bounded clique for predictable storage.
            for i, source in enumerate(ents[:8]):
                for target in ents[i + 1:8]:
                    a, b = sorted((source, target), key=str.casefold)
                    rows.append((book_id, a, b, sentence, page, book_title, 1))
    if not rows:
        return 0
    with _LOCK, _connect() as con:
        con.executemany(
            """INSERT INTO kg_edges(book_id, source, target, sentence, page, book_title, weight)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(book_id, source, target, sentence)
               DO UPDATE SET weight=weight+1""",
            rows,
        )
    return len(rows)


def delete_book(book_id: str) -> int:
    init_db()
    with _LOCK, _connect() as con:
        cur = con.execute("DELETE FROM kg_edges WHERE book_id=?", (book_id,))
        return cur.rowcount


def search(query: str, book_ids: Optional[Sequence[str]] = None, limit: int = 5) -> List[Dict]:
    """Return relationship evidence connected to entities mentioned in query."""
    init_db()
    terms = _entities(query)
    if not terms:
        # Also use meaningful lowercase query tokens as aliases.
        terms = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9+.#_-]{3,}", query) if w.lower() not in {
            "what", "when", "where", "which", "with", "from", "this", "that", "explain", "define", "difference", "between"
        }][:6]
    if not terms:
        return []

    clauses = []
    params: List[object] = []
    for term in terms[:6]:
        clauses.append("(lower(source) LIKE ? OR lower(target) LIKE ?)")
        pattern = f"%{term.lower()}%"
        params.extend([pattern, pattern])
    sql = "SELECT * FROM kg_edges WHERE (" + " OR ".join(clauses) + ")"
    if book_ids:
        placeholders = ",".join("?" for _ in book_ids)
        sql += f" AND book_id IN ({placeholders})"
        params.extend(book_ids)
    sql += " ORDER BY weight DESC LIMIT ?"
    params.append(max(limit * 4, limit))

    with _LOCK, _connect() as con:
        rows = con.execute(sql, params).fetchall()

    # De-duplicate by sentence and score by number of query terms touching edge.
    out, seen = [], set()
    for row in rows:
        sentence = row["sentence"]
        if sentence in seen:
            continue
        seen.add(sentence)
        edge_text = f"{row['source']} {row['target']}".lower()
        score = sum(1 for t in terms if t.lower() in edge_text)
        out.append({
            "text": sentence,
            "source": row["source"],
            "target": row["target"],
            "book_id": row["book_id"],
            "book_title": row["book_title"] or "Unknown",
            "page": row["page"] or "?",
            "score": score,
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:limit]


def build_context(hits: Sequence[Dict], max_chars: int = 2200) -> str:
    if not hits:
        return ""
    parts, total = [], 0
    for i, h in enumerate(hits, 1):
        block = (
            f"[Graph Source {i} | {h['book_title']} | Page: {h['page']} | "
            f"Relation: {h['source']} <-> {h['target']}]\n{h['text']}"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)
