"""System-One decision layer for HOLO-GURU.

Primary: LAYA (local/open, offline after model is present)
Fallback: JEV (hosted TypeSafe API, optional)
Final fallback: deterministic rules so HOLO-GURU never depends on either service.

This layer makes bounded routing/retry decisions. It never generates the final
teaching answer; Qwen remains the generator and LettuceDetect remains the
post-generation grounding verifier.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Dict, Optional

import requests
import config

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_LAYA_AGENT = None
_LAYA_FAILED = False

_RELATION_RE = re.compile(
    r"\b(relationship|related|relation|compare|difference|between|connect|connected|depends|"
    r"affect|cause|effect|why|how|architecture|flow|link|epadi|eppadi|versus|vs)\b", re.I
)


def _fallback_route(question: str) -> Dict:
    route = "both" if _RELATION_RE.search(question or "") else "rag"
    return {"route": route, "confidence": 0.50, "provider": "rules"}


def _normalize_choice(value) -> Optional[str]:
    if isinstance(value, dict):
        value = value.get("choice") or value.get("value") or value.get("label")
    if value is None:
        return None
    v = str(value).strip().lower()
    aliases = {
        "vector": "rag", "hybrid": "rag", "hybrid_rag": "rag",
        "graph": "kag", "graphrag": "kag", "knowledge_graph": "kag",
        "rag+kag": "both", "kag+rag": "both", "hybrid+graph": "both",
    }
    v = aliases.get(v, v)
    return v if v in {"rag", "kag", "both"} else None


def _extract_confidence(answer, default: float = 0.0) -> float:
    if not isinstance(answer, dict):
        return default
    for key in ("confidence", "probability", "prob", "score"):
        try:
            if key in answer:
                x = float(answer[key])
                return max(0.0, min(1.0, x if x <= 1 else x / 100.0))
        except (TypeError, ValueError, KeyError) as exc:
            logger.debug("Could not parse confidence from %r: %s", answer, exc)
    return default


def _get_laya_agent():
    global _LAYA_AGENT, _LAYA_FAILED
    if _LAYA_AGENT is not None:
        return _LAYA_AGENT
    if _LAYA_FAILED or not getattr(config, "LAYA_ENABLED", True):
        return None
    with _LOCK:
        if _LAYA_AGENT is not None:
            return _LAYA_AGENT
        try:
            import laya  # optional dependency
            model = getattr(config, "LAYA_MODEL", "convaiinnovations/laya")
            # Lazy-load to avoid consuming RAM at application startup.
            _LAYA_AGENT = laya.load(model)
            logger.info("LAYA System-One model loaded: %s", model)
        except Exception as exc:
            _LAYA_FAILED = True
            logger.warning("LAYA unavailable; falling back to JEV/rules: %s", exc)
            return None
    return _LAYA_AGENT


def _route_with_laya(question: str) -> Optional[Dict]:
    agent = _get_laya_agent()
    if agent is None:
        return None
    questions = {
        "retrieval_route": {
            "type": "choice",
            "instructions": "Choose the evidence path best suited to answer the student's question.",
            "criteria": {
                "rag": "factual or explanatory content best answered from document chunks",
                "kag": "entity relationships or multi-hop graph relationships are central",
                "both": "both document evidence and relationship evidence are useful",
            },
        }
    }
    try:
        result = agent.predict({"question": question}, questions)
        answer = (result.get("answers") or {}).get("retrieval_route") or {}
        route = _normalize_choice(answer)
        if route:
            return {"route": route, "confidence": _extract_confidence(answer, 0.75), "provider": "laya"}
    except Exception as exc:
        logger.warning("LAYA routing failed: %s", exc)
    return None


def _route_with_jev(question: str) -> Optional[Dict]:
    if not getattr(config, "JEV_ENABLED", False):
        return None
    api_key = getattr(config, "JEV_API_KEY", "")
    if not api_key:
        return None
    payload = {
        "model": getattr(config, "JEV_MODEL", "jev-latest"),
        "state": {"question": question},
        "questions": {
            "retrieval_route": {
                "type": "choice",
                "instructions": "Choose the evidence path best suited to answer the student's question.",
                "criteria": {
                    "rag": "factual or explanatory content best answered from document chunks",
                    "kag": "entity relationships or multi-hop graph relationships are central",
                    "both": "both document evidence and relationship evidence are useful",
                },
            }
        },
    }
    try:
        response = requests.post(
            getattr(config, "JEV_API_URL", "https://api.typesafe.ai/v1/systemone"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=float(getattr(config, "JEV_TIMEOUT_SECONDS", 1.5)),
        )
        response.raise_for_status()
        result = response.json()
        answer = (result.get("answers") or {}).get("retrieval_route") or {}
        route = _normalize_choice(answer)
        if route:
            return {"route": route, "confidence": _extract_confidence(answer, 0.75), "provider": "jev"}
    except Exception as exc:
        logger.warning("JEV routing failed: %s", exc)
    return None


def decide_retrieval_route(question: str) -> Dict:
    """Return {route: rag|kag|both, confidence, provider}.

    Provider order is configurable. Offline deployments should keep LAYA first.
    """
    order = str(getattr(config, "SYSTEM1_PROVIDER_ORDER", "laya,jev,rules")).lower().split(",")
    for provider in [p.strip() for p in order if p.strip()]:
        result = None
        if provider == "laya":
            result = _route_with_laya(question)
        elif provider == "jev":
            result = _route_with_jev(question)
        elif provider == "rules":
            result = _fallback_route(question)
        if result:
            return result
    return _fallback_route(question)


def should_retry(question: str, rag_hits: int, graph_hits: int, context_chars: int) -> Dict:
    """Cheap bounded retry decision.

    This intentionally uses deterministic evidence metrics. It avoids another
    model call on an 8GB Jetson while exposing the same typed-decision contract.
    """
    retry = context_chars < 160 or (rag_hits + graph_hits) == 0
    return {
        "retry": bool(retry),
        "confidence": 0.95 if (rag_hits + graph_hits) == 0 else 0.75,
        "provider": "evidence_metrics",
        "question": question,
    }
