"""Lightweight Tamil / Tanglish morphology helpers for retrieval.

No external NLP stack — designed for offline Jetson classroom use.
Provides:
  - Unicode normalization
  - Common Tamil inflectional suffix stripping (approximate stem)
  - Tanglish → normalized tokens for mixed queries
  - Query expansion variants for better hybrid RAG recall
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Set

# Longest-first Tamil suffixes (case, plural, common verbal/nominal endings).
# Order matters: try longer suffixes before shorter ones.
_TAMIL_SUFFIXES = (
    "ங்களிடமிருந்து",
    "ங்களிலிருந்து",
    "த்திலிருந்து",
    "த்திடமிருந்து",
    "ங்களுக்காக",
    "ங்களோடு",
    "ங்களுடன்",
    "ங்களுக்கு",
    "ங்களில்",
    "ங்களால்",
    "ங்களை",
    "ங்களின்",
    "ங்களும்",
    "த்திற்காக",
    "த்தினால்",
    "த்திற்கு",
    "த்திடம்",
    "த்துடன்",
    "த்தோடு",
    "த்தில்",
    "த்தை",
    "த்தின்",
    "த்து",
    "ங்கள",
    "ங்கள்",
    "ார்கள",
    "ார்கள்",
    "ுக்காக",
    "ிலிருந்து",
    "ிடமிருந்து",
    "ுடன",
    "ுடன்",
    "ோடு",
    "ுக்க",
    "ுக்கு",
    "ில்",
    "ால்",
    "ை",
    "ின்",
    "ும்",
    "ேன்",
    "ாய்",
    "ாக",
    "ான",
    "ுகள",
    "ுகள்",
)

# Common Tanglish educational stems → preferred retrieval forms (lowercase).
_TANGLISH_MAP = {
    "enna": "what",
    "epadi": "how",
    "eppadi": "how",
    "yean": "why",
    "yen": "why",
    "yaen": "why",
    "sollu": "explain",
    "sollunga": "explain",
    "pari": "lesson",
    "padam": "lesson",
    "paadam": "lesson",
    "kelvi": "question",
    "vidai": "answer",
    "porul": "meaning",
    "vilakkam": "explanation",
    "sutham": "formula",
    "soothiram": "formula",
    "vithai": "method",
    "muraimurai": "steps",
    "murai": "method",
    "verupadu": "difference",
    "oppidu": "compare",
    "kaaranam": "cause",
    "villai": "effect",
    "iyakkam": "motion",
    "veli": "force",
    "energy": "energy",
    "sakthi": "energy",
    "minnalsakthi": "electricity",
    "minnal": "electric",
    "thanni": "water",
    "kaatru": "air",
    "nilam": "earth",
    "sooriyan": "sun",
    "chandiran": "moon",
    "manithan": "human",
    "udambu": "body",
    "iruthayam": "heart",
    "mooLai": "brain",
    "moolai": "brain",
}

_WORD_RE = re.compile(r"[\w\u0B80-\u0BFF]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """NFC normalize and collapse whitespace."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_tamil_char(ch: str) -> bool:
    o = ord(ch)
    return 0x0B80 <= o <= 0x0BFF


def has_tamil(text: str) -> bool:
    return any(is_tamil_char(c) for c in (text or ""))


def stem_tamil_token(token: str) -> str:
    """Approximate stem by stripping common Tamil suffixes (longest first)."""
    if not token or len(token) < 3:
        return token
    # Keep pure ASCII tokens as-is (may be Tanglish handled elsewhere).
    if not has_tamil(token):
        return token
    stem = token
    changed = True
    # At most two suffix peels to avoid over-stemming short roots.
    rounds = 0
    while changed and rounds < 2 and len(stem) > 3:
        changed = False
        rounds += 1
        for suf in _TAMIL_SUFFIXES:
            if stem.endswith(suf) and len(stem) - len(suf) >= 2:
                stem = stem[: -len(suf)]
                changed = True
                break
    return stem or token


def normalize_tanglish_token(token: str) -> str:
    low = token.lower()
    return _TANGLISH_MAP.get(low, low)


def tokens(text: str) -> List[str]:
    return _WORD_RE.findall(normalize_text(text))


def morphological_variants(text: str, max_variants: int = 8) -> List[str]:
    """Return original + stemmed / mapped token variants for retrieval expansion."""
    raw = normalize_text(text)
    if not raw:
        return []
    variants: List[str] = [raw]
    seen: Set[str] = {raw.casefold()}
    toks = tokens(raw)
    stemmed_parts: List[str] = []
    mapped_parts: List[str] = []
    for tok in toks:
        st = stem_tamil_token(tok)
        mp = normalize_tanglish_token(tok)
        stemmed_parts.append(st)
        mapped_parts.append(mp if mp != tok.lower() else st)

    for candidate in (
        " ".join(stemmed_parts),
        " ".join(mapped_parts),
    ):
        key = candidate.casefold().strip()
        if key and key not in seen:
            seen.add(key)
            variants.append(candidate.strip())

    # Single-token stems as extra recall hints (short queries).
    if len(toks) <= 6:
        for tok in toks:
            st = stem_tamil_token(tok)
            if st and st.casefold() not in seen and has_tamil(st):
                seen.add(st.casefold())
                variants.append(st)
            mp = normalize_tanglish_token(tok)
            if mp and mp.casefold() not in seen:
                seen.add(mp.casefold())
                variants.append(mp)

    return variants[:max_variants]


def expand_query_for_retrieval(query: str) -> str:
    """Join morphological variants into one retrieval query string."""
    vars_ = morphological_variants(query)
    if not vars_:
        return query
    # Prefer original first; append unique tokens only (avoid duplicated stems).
    seen = set()
    out = []
    for variant in vars_:
        for tok in variant.split():
            key = tok.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(tok)
    return " ".join(out)[:500]


def simplify_tamil_for_display(text: str) -> str:
    """Light cleanup for TTS / simplified answers (does not rewrite meaning)."""
    t = normalize_text(text)
    # Normalize elongated punctuation common in chat Tamil.
    t = re.sub(r"[!]{2,}", "!", t)
    t = re.sub(r"[?]{2,}", "?", t)
    return t
