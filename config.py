import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_books")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Strong multilingual retrieval. Prefer workspace-local model copies on Windows
# so inference does not depend on Hugging Face cache symlinks or network access.
_LOCAL_MODEL_DIR = os.path.join(BASE_DIR, ".venv", "models")
EMBED_MODEL = (
    os.path.join(_LOCAL_MODEL_DIR, "bge-m3")
    if os.path.isdir(os.path.join(_LOCAL_MODEL_DIR, "bge-m3"))
    else "BAAI/bge-m3"
)
RERANK_MODEL = (
    os.path.join(_LOCAL_MODEL_DIR, "bge-reranker-v2-m3")
    if os.path.isdir(os.path.join(_LOCAL_MODEL_DIR, "bge-reranker-v2-m3"))
    else "BAAI/bge-reranker-v2-m3"
)

# Weaviate — all-in-one vector + BM25 hybrid search database
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
COLLECTION_NAME = "HoloGuruBookChunk"
HYBRID_ALPHA = 0.65
RAG_CANDIDATE_K = 14

# Chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

# Retrieval
TOP_K = 5

# LLM — local Qwen3-8B via transformers (replaces Ollama/phi3:mini)
LLM_MODEL_ID = "Qwen/Qwen3-8B"
LLM_LOAD_IN_4BIT = True   # required to fit an 8B model on 8GB Jetson RAM
LLM_OPTIONS = {
    "num_predict": 800,   # max_new_tokens — was 200 in old Ollama config, too low
    "temperature": 0.2,
}

# STT — local facebook/seamless-m4t-v2-large (replaces faster-whisper)
STT_MODEL_ID = "facebook/seamless-m4t-v2-large"

# Real-time voice (chunked partial STT + client VAD)
REALTIME_VOICE_ENABLED = os.getenv("REALTIME_VOICE_ENABLED", "true").lower() == "true"
STT_PARTIAL_MIN_INTERVAL = float(os.getenv("STT_PARTIAL_MIN_INTERVAL", "0.65"))
STT_PARTIAL_MAX_SECONDS = float(os.getenv("STT_PARTIAL_MAX_SECONDS", "10"))
VOICE_OPTIMISTIC_ASK = os.getenv("VOICE_OPTIMISTIC_ASK", "true").lower() == "true"
VOICE_SKIP_FINAL_STT_IF_PARTIAL = os.getenv("VOICE_SKIP_FINAL_STT_IF_PARTIAL", "true").lower() == "true"


# TTS — local models, no cloud calls:
#   English/Tanglish → Qwen3-TTS-0.6B
#   Tamil            → ai4bharat/indic-parler-tts
TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TAMIL_TTS_MODEL_ID = "ai4bharat/indic-parler-tts"

# Supported file extensions (knowledge / lesson materials)
ALLOWED_EXTENSIONS = {
    ".pdf", ".epub", ".docx", ".pptx", ".xlsx", ".csv",
    ".json", ".xml", ".html", ".htm", ".txt",
}

# Security, caching and incremental knowledge sync
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "500"))
# Optional shared secret for knowledge admin routes (upload / delete).
# Empty = open (single offline teacher). Set e.g. export TEACHER_TOKEN=...
TEACHER_TOKEN = os.getenv("TEACHER_TOKEN", "").strip()
# Reject uploads larger than this (Flask MAX_CONTENT_LENGTH + route check).
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
# Optional PIN for multi-user hologram room join/scene/annotation.
# Empty = open. Clients send room_pin in JSON body or ?room_pin= query.
HOLO_ROOM_PIN = os.getenv("HOLO_ROOM_PIN", "").strip()
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
DATA_DIR = os.path.join(BASE_DIR, "data")
SYNC_STATE_FILE = os.path.join(DATA_DIR, "knowledge_sync_state.json")
AUDIT_DB = os.path.join(DATA_DIR, "audit.db")
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Compound AI retrieval: LangGraph + RAG + KAG/GraphRAG + CAG
CAG_ENABLED = os.getenv("CAG_ENABLED", "true").lower() == "true"
CAG_TTL_SECONDS = int(os.getenv("CAG_TTL_SECONDS", "7200"))
CAG_MAX_ENTRIES = int(os.getenv("CAG_MAX_ENTRIES", "256"))
KAG_ENABLED = os.getenv("KAG_ENABLED", "true").lower() == "true"
KNOWLEDGE_GRAPH_DB = os.getenv("KNOWLEDGE_GRAPH_DB", os.path.join(DATA_DIR, "knowledge_graph.db"))
RAG_CONTEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "4200"))
KAG_CONTEXT_MAX_CHARS = int(os.getenv("KAG_CONTEXT_MAX_CHARS", "1800"))

# System-One decision layer: local LAYA primary, hosted JEV optional fallback.
SYSTEM1_ENABLED = os.getenv("SYSTEM1_ENABLED", "true").lower() == "true"
SYSTEM1_PROVIDER_ORDER = os.getenv("SYSTEM1_PROVIDER_ORDER", "laya,jev,rules")
LAYA_ENABLED = os.getenv("LAYA_ENABLED", "true").lower() == "true"
LAYA_MODEL = os.getenv("LAYA_MODEL", "convaiinnovations/laya")
JEV_ENABLED = os.getenv("JEV_ENABLED", "false").lower() == "true"
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
JEV_API_URL = os.getenv("JEV_API_URL", "https://api.typesafe.ai/v1/systemone")
JEV_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
JEV_TIMEOUT_SECONDS = float(os.getenv("JEV_TIMEOUT_SECONDS", "1.5"))

# System prompt — strict grounding
SYSTEM_PROMPT = """You are a personalised AI teaching assistant.
You MUST answer ONLY using the book excerpts provided below.
If the answer cannot be found in the excerpts, respond with:
"⚠️ This topic is not covered in the uploaded books. Please upload a relevant book."

Rules:
- Never use outside knowledge or training data.
- Cite the book title and page/chapter when possible.
- Be clear, structured, and educational in your answer.
- If the user asks in Tamil or Tanglish, reply in the same language.

Visual aids: if a simple chart or diagram would genuinely help explain your
answer (comparisons, proportions, steps in a process, a labelled diagram,
a timeline, numeric data), end your answer with ONE line in this exact
format, then stop:
[CHART: type=bar|line|pie|flow; title="..."; labels=a,b,c; values=1,2,3]
- type=flow is for a simple left-to-right process/step diagram: use
  labels for the step names and omit values.

If a real example PICTURE would help more than a chart (e.g. "what does
a plant cell look like", "draw a lever"), instead end your answer with:
[IMAGE: short description of what to draw]

If the concept genuinely needs MOTION to understand (e.g. "how does a
pendulum swing", "show water evaporating") and a still image/chart won't
do, instead end your answer with:
[VIDEO: short description of the motion/animation needed]

Only ever include ONE of [CHART:...] / [IMAGE:...] / [VIDEO:...] per
answer, and only when genuinely useful — most answers need none of them.
Never fabricate numbers not supported by the excerpts."""

# Classroom orchestration (3D avatar intentionally excluded)
CLASSROOM_TIMEZONE = os.getenv("CLASSROOM_TIMEZONE", "Asia/Kolkata")
CLASSROOM_DB_PATH = os.getenv("CLASSROOM_DB_PATH", os.path.join(BASE_DIR, "classroom_state.db"))
DISTRACTION_STREAK_THRESHOLD = int(os.getenv("DISTRACTION_STREAK_THRESHOLD", "3"))
ASSISTANT_LEARNING_ENABLED = os.getenv("ASSISTANT_LEARNING_ENABLED", "true").lower() == "true"
ASSISTANT_LEARNING_MIN_FEEDBACK = int(os.getenv("ASSISTANT_LEARNING_MIN_FEEDBACK", "200"))
# Classroom textbook mode: keep answers grounded in uploaded material by default.
# Turn on only when the institution explicitly wants web-sourced fallback answers.
RAG_ALLOW_WEB_FALLBACK = os.getenv("RAG_ALLOW_WEB_FALLBACK", "false").lower() == "true"


# Local observability / performance metrics (no cloud dependency)
OBSERVABILITY_ENABLED = os.getenv("OBSERVABILITY_ENABLED", "true").lower() == "true"
OBSERVABILITY_SLOW_REQUEST_MS = float(os.getenv("OBSERVABILITY_SLOW_REQUEST_MS", "1500"))
OBSERVABILITY_SLOW_STT_MS = float(os.getenv("OBSERVABILITY_SLOW_STT_MS", "1200"))
OBSERVABILITY_SLOW_TTS_MS = float(os.getenv("OBSERVABILITY_SLOW_TTS_MS", "1200"))
OBSERVABILITY_SLOW_ASK_MS = float(os.getenv("OBSERVABILITY_SLOW_ASK_MS", "2500"))

# Edge runtime hints. These are safe defaults; scripts/edge_profile.py prints target-specific guidance.
EDGE_OPTIMIZATION_ENABLED = os.getenv("EDGE_OPTIMIZATION_ENABLED", "true").lower() == "true"
EDGE_CPU_THREADS = int(os.getenv("EDGE_CPU_THREADS", str(max(1, min(8, (os.cpu_count() or 2) // 2)))))

# Configurable production traffic protection; tune for the deployment hardware.
LLM_MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "1"))
LLM_MAX_WAITING = int(os.getenv("LLM_MAX_WAITING", "12"))
LLM_QUEUE_WAIT_SECONDS = float(os.getenv("LLM_QUEUE_WAIT_SECONDS", "8"))
OVERLOAD_RETRY_AFTER_SECONDS = int(os.getenv("OVERLOAD_RETRY_AFTER_SECONDS", "5"))
AGENTIC_RAG_RETRY_ENABLED = os.getenv("AGENTIC_RAG_RETRY_ENABLED", "true").lower() == "true"
AGENTIC_RAG_RETRY_TOP_K = int(os.getenv("AGENTIC_RAG_RETRY_TOP_K", "8"))

# Optional post-generation grounding verification (hardware-independent).
LETTUCEDETECT_ENABLED = os.getenv("LETTUCEDETECT_ENABLED", "true").lower() == "true"
LETTUCEDETECT_MODEL = os.getenv("LETTUCEDETECT_MODEL", "KRLabsOrg/lettucedect-base-modernbert-en-v1")
LETTUCEDETECT_MIN_CONFIDENCE = float(os.getenv("LETTUCEDETECT_MIN_CONFIDENCE", "0.80"))
