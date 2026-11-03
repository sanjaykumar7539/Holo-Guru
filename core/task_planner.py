"""Local task planning for Holo-Guru's assistant layer.

The planner follows the useful part of HuggingGPT/JARVIS: decompose a
multi-intent request into small dependency-aware steps, then map each step to
an existing local capability. It does not execute arbitrary system commands.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


_VISUAL_RE = re.compile(r"\b(show|draw|diagram|visual|image|chart|picture)\b", re.I)
_COMPARE_RE = re.compile(r"\b(compare|difference|versus|vs\.?|between)\b", re.I)
_SUMMARY_RE = re.compile(r"\b(summary|summari[sz]e|key points|explain|teach)\b", re.I)
_QUIZ_RE = re.compile(r"\b(quiz|test me|ask me)\b", re.I)
_MULTI_RE = re.compile(r"\b(and then|then|after that|also|step by step|first .+ then)\b", re.I)
_TIME_RE = re.compile(r"\b(what time|current time|today'?s date|what date)\b", re.I)
_STATUS_RE = re.compile(r"\b(system status|cpu|ram|memory|battery|system health)\b", re.I)
_NOTE_RE = re.compile(r"\b(note this|take a note|save a note|remember this)\b", re.I)


def _step(step_id: str, task: str, capability: str, depends_on: List[str] | None = None) -> Dict[str, Any]:
    return {
        "id": step_id,
        "task": task,
        "capability": capability,
        "depends_on": depends_on or [],
        "status": "planned",
    }


def create_plan(question: str, *, has_books: bool = True) -> Dict[str, Any]:
    """Return a deterministic, UI-safe plan for a user request."""
    text = str(question or "").strip()
    steps: List[Dict[str, Any]] = []

    if _TIME_RE.search(text):
        steps.append(_step("datetime", "Read the local date and time", "datetime"))
        return {"planner": "holo-local-task-planner", "intent": "utility", "question": text,
                "steps": steps, "execution_policy": "local_safe_features_only"}
    if _STATUS_RE.search(text):
        steps.append(_step("status", "Read local system status", "system_status"))
        return {"planner": "holo-local-task-planner", "intent": "utility", "question": text,
                "steps": steps, "execution_policy": "local_safe_features_only"}
    if _NOTE_RE.search(text):
        steps.append(_step("note", "Save a note in the workspace data directory", "note"))
        return {"planner": "holo-local-task-planner", "intent": "utility", "question": text,
                "steps": steps, "execution_policy": "local_safe_features_only"}

    if has_books:
        steps.append(_step("retrieve", "Retrieve supporting evidence from uploaded books", "grounded_retrieval"))
    else:
        steps.append(_step("clarify_books", "Ask the user to upload a book", "knowledge_gate"))

    if _COMPARE_RE.search(text):
        steps.append(_step("compare", "Compare the requested concepts using retrieved evidence", "reasoning", ["retrieve"]))
    elif _SUMMARY_RE.search(text):
        steps.append(_step("explain", "Prepare a concise grounded explanation", "reasoning", ["retrieve"]))
    else:
        steps.append(_step("answer", "Answer the user's grounded question", "reasoning", ["retrieve"]))

    last = steps[-1]["id"]
    if _VISUAL_RE.search(text):
        steps.append(_step("visual", "Prepare a relevant visual aid when supported", "visual_aid", [last]))
        last = "visual"
    if _QUIZ_RE.search(text):
        steps.append(_step("quiz", "Offer a short understanding check", "classroom", [last]))
        last = "quiz"
    if _MULTI_RE.search(text):
        steps.append(_step("synthesize", "Combine the completed steps into one concise response", "response_synthesis", [last]))

    return {
        "planner": "holo-local-task-planner",
        "intent": "multi_step" if len(steps) > 2 else "single_step",
        "question": text,
        "steps": steps,
        "execution_policy": "grounded_local_tools_only",
    }
