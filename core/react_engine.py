"""
core/react_engine.py
ReAct (Reasoning + Acting) loop for HOLO-GURU.

Flow (strictly ordered, per project convention):
    THINK  -> does the question need external lookup?
    ACT 1  -> search uploaded books (Weaviate Hybrid RAG)
    ACT 2  -> if book context is empty/weak, search Wikipedia
    ACT 3  -> if Wikipedia has nothing, search DuckDuckGo
    OBSERVE-> pick the best evidence found, tag its source
    ANSWER -> stream final answer from the LLM, grounded in that evidence
    (if nothing found anywhere -> polite "not covered" message, no hallucination)

Latency features:
  - RAG/Wikipedia/DuckDuckGo lookups are cached (core/cache_engine.py) —
    repeated questions (common across a class) skip straight to the answer.
  - Simple/greeting questions skip the ReAct loop entirely ("ReAct only
    when needed") — no point spending a RAG search + fallback chain on
    "hi" or "thank you".
  - The token stream also emits sentence-boundary events so the caller
    can start TTS on the first sentence while the rest is still
    generating (sentence-level TTS, parallel with generation).

This module is intentionally a thin orchestrator — it does not replace
rag_engine.py or llm_engine.py, it sequences them.
"""
import logging
import config
import re
import time
from typing import Dict, Generator, List, Optional

from core.rag_engine import search as book_search, build_context
from core.compound_ai_graph import gather as compound_gather
from core.llm_engine import stream_answer
from core import cache_engine, observability
from core.hallucination_detector import verify as verify_hallucination

logger = logging.getLogger(__name__)

NOT_FOUND_MSG = (
    "This topic is not covered in the uploaded books, and I could not "
    "find a reliable answer online either. Please upload a relevant book "
    "or rephrase the question."
)

SENTENCE_END_RE = re.compile(r'[.!?।]\s')  # covers en/ta sentence enders + Tamil full stop variants

_GREETING_WORDS = {
    "hi", "hello", "hey", "vanakkam", "vannakam", "thanks", "thank", "thankyou",
    "bye", "ok", "okay", "sari", "nandri", "good", "morning", "night",
}


def is_simple_question(question: str) -> bool:
    """
    Heuristic: short greetings/pleasantries don't need book search or the
    ReAct fallback chain at all — answering directly is both faster and
    more natural than forcing "not covered in books" onto small talk.
    """
    words = question.strip().lower().split()
    if not words:
        return True
    if len(words) <= 3 and any(w.strip("!?.,") in _GREETING_WORDS for w in words):
        return True
    return False


# ── ACT 2: Wikipedia ────────────────────────────────────────────────────────

def _wikipedia_lookup(query: str, sentences: int = 3) -> Optional[Dict]:
    try:
        import wikipedia
        wikipedia.set_lang("en")
        results = wikipedia.search(query, results=1)
        if not results:
            return None
        summary = wikipedia.summary(results[0], sentences=sentences, auto_suggest=False)
        if not summary.strip():
            return None
        return {"text": summary, "source": f"Wikipedia: {results[0]}"}
    except Exception as e:
        logger.debug(f"Wikipedia lookup skipped/failed: {e}")
        return None


# ── ACT 3: DuckDuckGo ───────────────────────────────────────────────────────

def _duckduckgo_lookup(query: str) -> Optional[Dict]:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=1):
                snippet = (r.get("body") or "").strip()
                if snippet:
                    return {"text": snippet, "source": f"Web: {r.get('title', 'DuckDuckGo')}"}
    except Exception as e:
        logger.debug(f"DuckDuckGo lookup skipped/failed: {e}")
    return None


# ── Orchestrator ─────────────────────────────────────────────────────────────

def _rewrite_query(question: str) -> str:
    """Cheap deterministic rewrite: remove conversational filler and add textbook intent.

    No extra LLM call is used, which matters on an 8GB Jetson.
    """
    import re
    q = re.sub(r"\b(please|pls|can you|could you|tell me|explain me|sollu|konjam)\b", " ", question, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip()
    return q or question

def _evidence_is_weak(hits) -> bool:
    if not hits:
        return True
    # CrossEncoder scores are model-dependent, so avoid brittle absolute thresholds.
    # Treat a single tiny excerpt as weak; otherwise let grounding prompt handle uncertainty.
    useful = [h for h in hits if len((h.get("text") or "").strip()) >= 80]
    return len(useful) < 2

def _gather_context_uncached(
    question: str,
    collection,
    top_k: int,
    book_ids: Optional[List[str]] = None,
) -> Dict:
    # ACT 1 — book RAG (always tried first, per project rule)
    hits = book_search(question, collection, top_k, book_ids)
    retried = False
    if getattr(config, "AGENTIC_RAG_RETRY_ENABLED", True) and _evidence_is_weak(hits):
        rewritten = _rewrite_query(question)
        retry_k = max(top_k, getattr(config, "AGENTIC_RAG_RETRY_TOP_K", top_k))
        logger.info("Agentic RAG: weak evidence; retrying retrieval with rewritten/wider query")
        retry_hits = book_search(rewritten, collection, retry_k, book_ids)
        if len(retry_hits) > len(hits):
            hits = retry_hits
            retried = True
    if hits:
        context = build_context(hits)
        if context.strip():
            sources = list({h["book_title"] for h in hits})
            logger.info(f"ReAct: answered from books ({len(hits)} hits, retry={retried})")
            return {"context": context, "stage": "book_retry" if retried else "book", "sources": sources}

    # In classroom textbook mode, do not silently mix web knowledge with
    # uploaded-book evidence. This preserves source grounding and accuracy.
    if not getattr(config, "RAG_ALLOW_WEB_FALLBACK", False):
        logger.info("ReAct: no book evidence; web fallback disabled by classroom policy")
        return {"context": "", "stage": "none", "sources": []}

    # ACT 2 — Wikipedia fallback (explicitly opt-in only)
    wiki = _wikipedia_lookup(question)
    if wiki:
        logger.info("ReAct: falling back to Wikipedia")
        context = (
            "Give a short, direct answer in 9-12 lines. "
            "This information is from Wikipedia, not the uploaded books — "
            "mention that briefly.\n\n"
            f"[{wiki['source']}]\n{wiki['text'][:800]}"
        )
        return {"context": context, "stage": "wikipedia", "sources": [wiki["source"]]}

    # ACT 3 — DuckDuckGo fallback
    ddg = _duckduckgo_lookup(question)
    if ddg:
        logger.info("ReAct: falling back to DuckDuckGo")
        context = (
            "Give a short, direct answer in 9-12 lines. "
            "This information is from a web search, not the uploaded books — "
            "mention that briefly.\n\n"
            f"[{ddg['source']}]\n{ddg['text'][:800]}"
        )
        return {"context": context, "stage": "duckduckgo", "sources": [ddg["source"]]}

    # Nothing found anywhere
    logger.info("ReAct: no evidence found in books, Wikipedia, or web")
    return {"context": "", "stage": "none", "sources": []}


def gather_context(
    question: str,
    collection,
    top_k: int,
    book_ids: Optional[List[str]] = None,
) -> Dict:
    """Run the Compound AI retrieval layer.

    Primary path: LangGraph -> CAG -> Hybrid RAG + optional KAG/GraphRAG -> fusion.
    If no local evidence is found, the existing classroom web-fallback policy is
    preserved through the legacy ReAct lookup chain.
    """
    started = time.perf_counter()
    try:
        result = compound_gather(question, collection, top_k, book_ids)
    finally:
        observability.record(
            "rag.compound_retrieval",
            (time.perf_counter() - started) * 1000.0,
            slow_ms=1200,
        )
    if result.get("context"):
        logger.info("Compound retrieval path: %s", result.get("retrieval_path", []))
        return result

    # Preserve the existing opt-in web fallback behavior when local compound
    # retrieval cannot ground an answer.
    started = time.perf_counter()
    try:
        return _gather_context_uncached(question, collection, top_k, book_ids)
    finally:
        observability.record(
            "rag.fallback_retrieval",
            (time.perf_counter() - started) * 1000.0,
            slow_ms=1200,
        )


def stream_react_answer(
    question: str,
    collection,
    top_k: int,
    book_ids: Optional[List[str]],
    history: List[Dict],
    system_prompt: str,
    model_id: str,
    options: dict,
    load_in_4bit: bool = True,
) -> Generator[Dict, None, None]:
    """
    Full ReAct pipeline as a generator of events:
        {"type": "stage", "stage": ..., "sources": [...]}
        {"type": "token", "token": "..."}
        {"type": "sentence", "text": "..."}   # fires as each sentence completes,
                                                # so the caller can start TTS early
        {"type": "done"}
    """
    # "ReAct only when needed" — skip RAG + fallback chain for small talk
    if is_simple_question(question):
        logger.info("ReAct: simple/greeting question — skipping RAG + fallback chain")
        yield {"type": "stage", "stage": "direct", "sources": []}
        fast_options = {**options, "num_predict": min(options.get("num_predict", 800), 60)}
        sentence_buf = ""
        for token in stream_answer(
            question, "The student is making small talk — respond briefly and warmly, "
                       "no book lookup needed.",
            history, system_prompt, model_id, fast_options, load_in_4bit,
        ):
            yield {"type": "token", "token": token}
            sentence_buf += token
            if SENTENCE_END_RE.search(sentence_buf):
                yield {"type": "sentence", "text": sentence_buf.strip()}
                sentence_buf = ""
        if sentence_buf.strip():
            yield {"type": "sentence", "text": sentence_buf.strip()}
        yield {"type": "done"}
        return

    result = gather_context(question, collection, top_k, book_ids)

    yield {"type": "stage", "stage": result["stage"], "sources": result["sources"]}

    if result["stage"] == "none":
        yield {"type": "token", "token": NOT_FOUND_MSG}
        yield {"type": "sentence", "text": NOT_FOUND_MSG}
        yield {"type": "done"}
        return

    # Buffer grounded answers until LettuceDetect verifies them. This prevents an
    # unsupported sentence from being displayed/spoken before verification finishes.
    generated = []
    generation_started = time.perf_counter()
    try:
        for token in stream_answer(
            question,
            result["context"],
            history,
            system_prompt,
            model_id,
            options,
            load_in_4bit,
        ):
            generated.append(token)
    finally:
        observability.record(
            "llm.generate",
            (time.perf_counter() - generation_started) * 1000.0,
            slow_ms=2000,
            meta={"stage": result.get("stage", "none")},
        )

    answer = "".join(generated).strip()
    verify_started = time.perf_counter()
    try:
        lettuce = verify_hallucination(result["context"], question, answer)
    finally:
        observability.record(
            "grounding.verify",
            (time.perf_counter() - verify_started) * 1000.0,
            slow_ms=1000,
        )
    if lettuce.get("checked") and not lettuce.get("supported"):
        logger.warning("LettuceDetect blocked unsupported spans: %s", lettuce.get("spans"))
        yield {"type": "verification", "supported": False, "spans": lettuce.get("spans", [])}
        yield {"type": "token", "token": NOT_FOUND_MSG}
        yield {"type": "sentence", "text": NOT_FOUND_MSG}
    else:
        yield {"type": "verification", "supported": True, "checked": lettuce.get("checked", False)}
        # Preserve the existing event contract after verification.
        sentence_buf = ""
        for token in generated:
            yield {"type": "token", "token": token}
            sentence_buf += token
            if SENTENCE_END_RE.search(sentence_buf) and "[" not in sentence_buf:
                yield {"type": "sentence", "text": sentence_buf.strip()}
                sentence_buf = ""
        if sentence_buf.strip() and "[" not in sentence_buf:
            yield {"type": "sentence", "text": sentence_buf.strip()}

    yield {"type": "done"}
