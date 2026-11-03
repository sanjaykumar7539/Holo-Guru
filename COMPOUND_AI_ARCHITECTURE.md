# HOLO-GURU Compound AI Architecture

## Name
**Compound AI System using Modular GraphRAG and Cache-Augmented Generation**

## Integrated technologies
- RAG: Weaviate hybrid BM25 + dense BGE-M3 retrieval, followed by BGE reranking.
- KAG / GraphRAG: local SQLite entity-relationship graph created from uploaded document sentences.
- CAG: reusable hot evidence-context cache, separate from final response caching.
- LangGraph: conditional routing between cache and fresh retrieval paths.
- Local LLM: existing Qwen3-8B generation.
- Grounding: existing LettuceDetect post-generation verification.

## Flow
Question
-> LangGraph router
-> CAG context lookup
-> if miss: Hybrid RAG
-> if relationship-style query: KAG/GraphRAG in parallel
-> evidence fusion
-> local LLM
-> grounding verification
-> cited answer

## Why local SQLite graph?
The project targets local/offline hardware. A separate graph database service would add memory, deployment and maintenance overhead. SQLite preserves a real persistent graph relationship layer while keeping the stack lightweight.

## System-One decision layer: LAYA + JEV

HOLO-GURU now has an optional fast decision layer in `core/system_one_engine.py`.
It does not replace Qwen or generate student-facing answers.

Provider order (default):

1. **LAYA** — local/open typed-decision model, lazy-loaded only when a routing decision is needed.
2. **JEV** — optional hosted TypeSafe System One fallback when `JEV_ENABLED=true` and `TYPESAFE_API_KEY` is configured.
3. **Rules** — deterministic zero-dependency fallback, so the application keeps working even when neither model is available.

The decision layer selects one of three evidence routes:

- `rag` — Weaviate hybrid document retrieval
- `kag` — local knowledge-graph retrieval
- `both` — retrieve both and fuse evidence

Flow:

`Question -> CAG check -> System-One (LAYA/JEV) route -> RAG/KAG/both -> evidence fusion -> Qwen -> LettuceDetect -> answer`

For offline deployments, keep `SYSTEM1_PROVIDER_ORDER=laya,jev,rules` and leave `JEV_ENABLED=false`.
To enable hosted JEV, set `JEV_ENABLED=true` and `TYPESAFE_API_KEY` server-side.
LAYA is intentionally lazy-loaded to reduce startup RAM pressure.

## Touchless hologram interaction (added)

The local camera pipeline now provides a second interaction channel alongside classroom emotion/gesture sensing. MediaPipe hand landmarks are converted into deterministic hologram commands without sending camera frames to any cloud service.

- Index finger: move the hologram using hand position.
- Index + middle finger: rotate the hologram.
- Thumb + index spread: zoom between a bounded 0.75x and 1.75x range.
- Pinch: select/highlight the hologram.
- Closed fist: reset position, zoom, and rotation.
- Existing raised-hand, thumbs-up/down, and palm-stop classroom gestures remain higher-priority commands.

The browser polls gesture sensing at roughly 500 ms for responsiveness while throttling emotion inference to roughly 2.5 seconds to reduce edge-device load. The current UI applies the commands to the hologram figure wrapper; the same normalized interaction payload can later drive a Three.js/Unity 3D teaching model without changing the camera API.
