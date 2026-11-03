## Responsive voice assistant

Latency reductions:
- Partial STT interval **0.65s**, MediaRecorder timeslice **600ms**
- VAD end-of-speech **~850ms** silence (was ~1.4s)
- **Barge-in**: mic stops assistant TTS immediately
- **Optimistic ask**: if a fresh partial transcript has ≥3 words, answer without waiting for final STT
- **Sentence TTS queue** with next-sentence prefetch while current audio plays
- Fixed missing `enqueueSentenceTTS` so streamed sentences speak as they arrive

## Real-time voice processing

- Browser **MediaRecorder timeslices** (~1s) stream audio while you speak.
- **Partial STT** via `POST /api/stt/partial` (rate-limited per session).
- Live interim transcript in the question box + hint bar.
- **VAD (energy-based)** auto-stops after ~1.4s of silence (or max ~15s).
- Final `POST /api/stt` on stop, then normal ask pipeline.
- Config: `REALTIME_VOICE_ENABLED`, `STT_PARTIAL_MIN_INTERVAL`.

## NLP intent routing & Tamil morphology

- **Intent routing** (`core/nlp_intent.py`): greets, definitions, how-to, compare, cause/effect, simplify, hologram control, quiz, off-topic.
- Routes answer **style** (steps / definition / compare) via system-prompt add-on.
- **Tamil morphology** (`core/tamil_morphology.py`): suffix stripping, Tanglish map, query expansion for hybrid RAG.
- Wired into `adaptive_engine` + `rag_engine`; SSE emits `intent` + `intent_confidence`.

# HOLO-GURU Secure Classroom Foundation

Offline-first classroom AI tutor with Weaviate hybrid retrieval, BGE-M3
embeddings, BGE reranking, adaptive classroom orchestration, STT/TTS, vision,
and optional chart/image/video teaching aids.

## Secure Teaching Flow

```text
Student voice/text
 -> validation + prompt-injection guard + rate limit
 -> classroom context and adaptive decision
 -> Weaviate BM25+dense hybrid retrieval
 -> BAAI/bge-reranker-v2-m3
 -> strict grounded local Qwen answer
 -> sentence SSE streaming and TTS
 -> optional chart/image/video aid
 -> audit event and lesson memory
```

Web fallback is disabled by default. Unsupported questions are rejected rather
than answered from model memory.

## Supported Knowledge Files

PDF, EPUB, DOCX, PPTX, XLSX, CSV, JSON, XML, HTML and TXT.

Files can be uploaded in the UI or placed inside `knowledge/`. Content hashes
prevent duplicate ingestion. A failed update does not delete previously indexed
knowledge.

## Windows Start

1. Install Python 3.11, Docker Desktop and FFmpeg.
2. Extract the ZIP.
3. Put lesson files in `knowledge/`.
4. Run `START_HOLO_GURU.bat`.
5. Open the HTTPS address printed in the terminal.

For later lesson updates, run `UPDATE_KNOWLEDGE.bat`.

## Manual Start

```bash
docker compose up -d weaviate
python -m venv .venv
python -m pip install -r requirements.txt
python sync_knowledge.py
python main.py
```

## Main Components

- `main.py` — single backend/frontend entry point
- `core/classroom_orchestrator.py` — classroom teaching state machine
- `core/vector_store.py` — BGE-M3 and Weaviate connection
- `core/rag_engine.py` — hybrid retrieval and BGE reranking
- `core/vision_engine.py` — visual-question support
- `core/visual_aid_generator.py` — charts and diagrams
- `core/image_gen.py`, `core/video_gen.py` — optional GPU teaching media
- `core/security.py` — validation, injection detection and rate limiting
- `core/audit.py` — append-only SQLite audit trail
- `core/cache_engine.py` — bounded TTL cache
- `sync_knowledge.py` — safe incremental ingestion

## Tests

```bash
python -m pytest -q
```

Full model, Weaviate and GPU-media validation requires the target machine and
downloaded model weights.

## Important Configuration

- `WEAVIATE_HOST`, `WEAVIATE_HTTP_PORT`, `WEAVIATE_GRPC_PORT`
- `RAG_ALLOW_WEB_FALLBACK=false`
- `MAX_QUESTION_CHARS=2000`
- `RATE_LIMIT_REQUESTS=30`
- `RATE_LIMIT_WINDOW_SECONDS=60`
- `CACHE_TTL_SECONDS=3600`
- `CACHE_MAX_ENTRIES=500`

Static SSL private keys are not distributed. A local certificate is generated
on first start when OpenSSL is available.

## Jetson production protection upgrade

This build adds bounded LLM admission control for the single-device deployment.
Defaults: one Qwen generation at a time, up to 12 waiting requests, and an
8-second queue wait. When capacity is exhausted `/api/ask` returns HTTP 503
with `Retry-After` instead of allowing unbounded work to exhaust RAM.
Operational counters are available at `/api/system/traffic`.

RAG also performs one controlled retrieval retry when evidence is weak. The
retry uses a cleaned query and wider Top-K, then continues through the existing
hybrid retrieval + BGE reranking + grounded generation pipeline. It deliberately
does not add MCP, GraphRAG, Redis, or a multi-agent stack to the single-Jetson
build.

Environment tuning: `LLM_MAX_CONCURRENT`, `LLM_MAX_WAITING`,
`LLM_QUEUE_WAIT_SECONDS`, `OVERLOAD_RETRY_AFTER_SECONDS`,
`AGENTIC_RAG_RETRY_ENABLED`, `AGENTIC_RAG_RETRY_TOP_K`.

### LettuceDetect grounding verification
Grounded teaching answers are optionally buffered and checked with LettuceDetect before display/TTS. Unsupported spans above the configured threshold are blocked and replaced by the safe not-found response. This preserves existing RAG grounding while adding a post-generation verification layer.

## Compound AI Retrieval (RAG + KAG/GraphRAG + CAG + LangGraph)

HOLO-GURU now uses a modular Compound AI retrieval layer:

1. **CAG** checks a hot, validated evidence-context cache first (separate from final-answer caching).
2. **Hybrid RAG** retrieves uploaded-book evidence with Weaviate BM25 + dense BGE-M3 search and BGE reranking.
3. **KAG / GraphRAG** uses a lightweight SQLite knowledge graph built during ingestion for relationship / multi-hop style questions.
4. **LangGraph** routes and fuses the retrieval paths before the existing local LLM generation.
5. **LettuceDetect** still verifies generated output against the fused evidence before it is shown/spoken.

The local graph intentionally avoids an extra Neo4j service so the system remains suitable for single-device/offline deployments.

Key modules:
- `core/compound_ai_graph.py` — LangGraph retrieval orchestration
- `core/cag_engine.py` — hot evidence-context CAG layer
- `core/knowledge_graph.py` — local KAG/GraphRAG relationship store
- `core/rag_engine.py` — Weaviate hybrid RAG + reranking

Runtime flow:
`Question -> LangGraph -> CAG hit? -> (yes: reuse evidence) / (no: Hybrid RAG + optional KAG) -> Fusion -> Local LLM -> Grounding verification -> Answer`

### Touchless hologram controls

Turn on **SENSE** in the UI to enable local camera-based interaction. Supported controls: point with the index finger to move, show index+middle to rotate, spread thumb+index to zoom, pinch to select/highlight, and make a closed fist to reset. Existing classroom gestures remain enabled. Camera frames are processed locally by MediaPipe/OpenCV.

### Advanced hologram scene controls

This build adds a more cinematic, Tony-Stark-inspired hologram HUD (implemented
from scratch, without franchise assets):

- Orbit / scan / X-ray toggles
- Explode-view percentage
- Layer cycling (surface → structure → internal → labels)
- Focus mode for a selected part/topic
- Compare mode and HUD panels
- Typed control commands in chat using `/holo ...`

Examples:
- `/holo orbit`
- `/holo explode 80`
- `/holo focus on engine`
- `/holo xray`
- `/holo reset`

### Advanced 3D hologram + voice scene control

The hologram panel now includes an offline custom 3D scene renderer and floating
HUD cards. It reacts to orbit, scan, X-ray, focus, layer and explode state without
requiring an external CDN. Voice STT can directly trigger scene commands when a
spoken phrase looks like a hologram command (for example: “Holo orbit”,
“explode view”, “X-ray”, “focus on core”, “next layer”, “reset hologram”).

The 3D renderer is intentionally lightweight so it can run locally alongside the
AI stack. A future Unity/Three.js asset pipeline can replace the demo geometry
without changing the backend scene-control API.

### Educational hologram model presets + part-aware AI

The hologram renderer now includes four offline procedural educational presets:

- **Heart** — right/left atrium, right/left ventricle, aorta, septum
- **Engine** — cylinders, piston, crankshaft, valve train
- **Molecule** — central atom and bonded atoms
- **Anatomy** — head, chest, abdomen, arms and pelvis

Click a rendered part, or pinch near it while SENSE is enabled, to select it.
The selected part is sent as UI context with the next question so phrases such as
“what does this do?” can be resolved to the selected component. The selected
part is not treated as a factual source; factual claims still require retrieved
book evidence.

Voice / typed examples:
- `show heart` / `/holo show heart`
- `show engine`
- `show molecule`
- `show anatomy`
- `focus on left ventricle`
- `explode view to 80`


### More advanced hologram features added

This build extends the hologram system further with:

- **More educational assets:** brain, cell, and solar-system presets in addition to heart, engine, molecule, and anatomy.
- **Interactive quiz mode:** identify highlighted parts by clicking or pinch-selecting them.
- **Guided tour mode:** automatically cycles focus through parts of the current model.
- **Scene bookmarks:** save and restore recent hologram states locally.
- **Event log panel:** track scene changes, selections, and quiz/tour events.

Extra commands:
- `/holo quiz mode`
- `/holo tour mode`
- `/holo save bookmark`
- `/holo load bookmark`
- `/holo show brain`
- `/holo show cell`
- `/holo show solar`




### Hologram 3D advanced features

- **MEASURE** — click two parts for distance (`/holo measure`).
- **ANIM+** — animated explode timeline (`/holo explode anim 100`).
- **CLIP** — multi-layer visibility (`/holo clip`, `/holo clip layer internal`).

### Hologram 3D module — new classroom features

- **SLICE** — cross-section plane (`section_enabled`, axis x/y/z, position 0–100%).
  Voice/commands: `/holo slice`, `/holo slice y 40`, `/holo cross section off`.
- **FREEZE** — pause orbit/scan animation for teaching discussion.
- **LABELS** — part name callouts on the 3D stage.
- **WIRE+** — denser structural edges for complex models.
- **Orbit speed** — `/holo orbit speed 1.5` (0.25×–3.0×).

UI buttons: SLICE · FREEZE · LABELS · WIRE+ in the hologram control dock.
HUD shows SLICE / FREEZE / LABELS / SPEED status.

### Security & code-quality hardening (this build)

- **Teacher token (optional):** set `TEACHER_TOKEN` to require header
  `X-Teacher-Token` (or `Authorization: Bearer …`) on knowledge **upload** and
  **delete**. Empty keeps offline single-teacher mode open.
- **Room PIN (optional):** set `HOLO_ROOM_PIN` to require `room_pin` (JSON body,
  query, or `X-Room-Pin`) on holo-room join / state / scene / annotation.
- **Upload size limit:** `MAX_UPLOAD_MB` (default 50). Oversized bodies return 413.
- **Safe client errors:** ingest/STT failures no longer return raw exception text.
- **Cleanup / parse paths:** temp-file and value-parse failures log at debug
  instead of silent `pass`; explode/confidence parse uses typed exceptions.

### Added on this build: features 3, 4, and 5

This update adds the three next-stage features requested:

1. **Live annotation / laser pointer**
   - Pointer mode for a laser-dot style highlight.
   - Draw mode for freehand annotation over the hologram stage.
   - Clear-annotation action.
   - Annotation sync is throttled for smoother classroom performance.

2. **Student multi-user interaction**
   - Lightweight room system with join/leave.
   - Participant list.
   - Shared hologram scene sync.
   - Shared annotation / pointer sync.
   - Implemented with simple low-latency polling and in-memory room state.

3. **Imported 3D model support (.obj / .glb)**
   - Import local `.obj` models.
   - Import local `.glb` models (basic uncompressed glTF/GLB geometry flow).
   - Imported mesh is normalized and rendered as a hologram wireframe in the existing offline renderer.

Performance-minded implementation details:
- Local rendering stays client-side for smoother interaction.
- Pointer sync is throttled.
- Room sync uses lightweight polling to avoid UI blocking.
- Annotation/history sizes are bounded to avoid runaway latency.

## Local performance observability + edge optimization

This build adds offline/local performance visibility without sending prompts or classroom data to a cloud tracing service.

### Metrics endpoint

`GET /api/system/metrics`

Tracks bounded in-memory timing samples for:
- HTTP endpoints
- `ask.ttft` (time to first generated token)
- `ask.stream.total`
- `rag.compound_retrieval`
- fallback retrieval
- `llm.generate`
- grounding verification
- STT transcription
- TTS synthesis

Each metric reports count, errors, average, p50, p95, p99 and max latency. Recent slow operations contain timings and small operational metadata only, not prompt text.

### Edge utilities

- `scripts/edge_profile.py` — prints conservative CPU/thread guidance.
- `scripts/benchmark_runtime.py` — lightweight local overhead benchmark; no model download.
- `scripts/optimize_openvino.py` — optional OpenVINO exporter for CPU deployments.
- `requirements-edge.txt` — optional edge export dependencies; normal startup does not require them.

`START_HOLO_GURU.bat` now derives a conservative CPU thread count and sets OMP/MKL/OpenBLAS/NumExpr thread limits if the user has not already configured them. This reduces CPU oversubscription and helps interaction smoothness on constrained systems.

Important: software tuning can reduce latency, but true end-to-end latency still depends on target CPU/GPU, model availability, thermals, input length, and concurrent load. Use `/api/system/metrics` plus the benchmark script on the actual target hardware to validate p50/p95 performance.
