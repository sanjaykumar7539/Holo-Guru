"""
core/video_gen.py
Wan2.1-T2V-1.3B — generates a short example video when a concept genuinely
needs motion to explain (e.g. "show how a pendulum swings"), triggered via
a [VIDEO: ...] tag, same pattern as [CHART: ...] and [IMAGE: ...].

⚠️ HARDWARE: even the smallest 1.3B Wan2.1 variant needs a real GPU
(realistically 8GB+ VRAM, generation takes tens of seconds even on a
capable GPU). This is the heaviest/slowest of the three generation
features — use sparingly, and only on a GPU-equipped demo machine, never
on the Jetson. Fails closed exactly like image_gen.py: if unavailable,
the tag is stripped and the answer stays text-only.

Because this is slow, it is NOT run inline with the streamed answer like
[CHART]/[IMAGE] — the answer streams and finishes normally, and the
video is generated as a background job the frontend polls for, so a
30-60s video render never blocks the actual answer from being read/heard.
"""
import base64
import logging
import re
import tempfile
import threading
import uuid
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

VIDEO_TAG_RE = re.compile(r'\[VIDEO:\s*(?P<prompt>[^\]]+)\]', re.IGNORECASE)

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

_pipe = None
_lock = threading.Lock()

# job_id -> {"status": "running"|"done"|"error", "data_url": str|None}
_jobs: dict = {}
_jobs_lock = threading.Lock()


def _load_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    with _lock:
        if _pipe is not None:
            return _pipe
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(
                    "No CUDA GPU detected — skipping Wan2.1 load. "
                    "Video generation will be unavailable on this device."
                )
                return None

            from diffusers import WanPipeline
            logger.info(f"Loading video model '{MODEL_ID}'...")
            _pipe = WanPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
            _pipe.enable_model_cpu_offload()
            logger.info("✅ Wan2.1-T2V-1.3B loaded.")
        except ImportError:
            logger.warning("diffusers (with Wan support) not installed — run: pip install -U diffusers")
        except Exception as e:
            logger.error(f"Failed to load Wan2.1: {e}")
    return _pipe


def warmup():
    try:
        _load_pipe()
    except Exception as e:
        logger.warning(f"Video gen warmup skipped: {e}")


def extract_video_tag(answer_text: str) -> Tuple[str, Optional[str]]:
    """Splits answer_text into (clean_text, video_prompt_or_None)."""
    match = VIDEO_TAG_RE.search(answer_text)
    if not match:
        return answer_text, None
    clean_text = (answer_text[:match.start()] + answer_text[match.end():]).strip()
    return clean_text, match.group("prompt").strip()


def start_video_job(prompt: str) -> Optional[str]:
    """Kicks off generation in a background thread, returns a job_id immediately."""
    pipe = _load_pipe()
    if pipe is None:
        return None

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "data_url": None}

    def _run():
        try:
            from diffusers.utils import export_to_video
            result = pipe(
                prompt=f"Simple, clear educational animation: {prompt}",
                height=320, width=576,
                num_frames=33,     # ~2s at 16fps — kept short, this is slow to render
                guidance_scale=5.0,
            )
            frames = result.frames[0]

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                export_to_video(frames, f.name, fps=16)
                f.seek(0)
                video_bytes = open(f.name, "rb").read()

            b64 = base64.b64encode(video_bytes).decode()
            data_url = f"data:video/mp4;base64,{b64}"
            with _jobs_lock:
                _jobs[job_id] = {"status": "done", "data_url": data_url}
        except Exception as e:
            logger.error(f"Video generation error: {e}")
            with _jobs_lock:
                _jobs[job_id] = {"status": "error", "data_url": None}

    threading.Thread(target=_run, daemon=True).start()
    return job_id


def get_job_status(job_id: str) -> dict:
    with _jobs_lock:
        return _jobs.get(job_id, {"status": "not_found", "data_url": None})
