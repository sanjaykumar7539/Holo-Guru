"""
core/image_gen.py
FLUX.1-schnell — generates a real-time example image when the LLM decides
one would help explain a concept (e.g. "draw the water cycle", "show a
plant cell"). Triggered via an [IMAGE: ...] tag, same pattern as the
[CHART: ...] tag in visual_aid_generator.py.

⚠️ HARDWARE: FLUX.1-schnell (~12B params) needs a real GPU with several GB
of free VRAM (practically: 8GB+ even in fp16/bf16, more comfortably 12GB+).
This will NOT run on the Jetson Orin Nano (8GB unified memory, already
shared with the LLM/STT/TTS). It's intended for a GPU-equipped demo
machine — see docker-compose.yml GPU section. If no suitable GPU is
available, this module fails closed: the tag is stripped and the answer
is shown as text-only, it never crashes the request.
"""
import base64
import io
import logging
import re
import threading
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

IMAGE_TAG_RE = re.compile(r'\[IMAGE:\s*(?P<prompt>[^\]]+)\]', re.IGNORECASE)

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

_pipe = None
_lock = threading.Lock()


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
                    "No CUDA GPU detected — skipping FLUX.1-schnell load. "
                    "Image generation will be unavailable on this device."
                )
                return None

            from diffusers import FluxPipeline
            logger.info(f"Loading image model '{MODEL_ID}'...")
            _pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
            _pipe.enable_model_cpu_offload()   # keeps VRAM usage manageable
            logger.info("✅ FLUX.1-schnell loaded.")
        except ImportError:
            logger.warning("diffusers not installed — run: pip install diffusers")
        except Exception as e:
            logger.error(f"Failed to load FLUX.1-schnell: {e}")
    return _pipe


def warmup():
    try:
        _load_pipe()
    except Exception as e:
        logger.warning(f"Image gen warmup skipped: {e}")


def extract_image_tag(answer_text: str) -> Tuple[str, Optional[str]]:
    """Splits answer_text into (clean_text, image_prompt_or_None)."""
    match = IMAGE_TAG_RE.search(answer_text)
    if not match:
        return answer_text, None
    clean_text = (answer_text[:match.start()] + answer_text[match.end():]).strip()
    return clean_text, match.group("prompt").strip()


def generate_image(prompt: str) -> Optional[str]:
    """Returns a base64 PNG data URL, or None if generation isn't available."""
    pipe = _load_pipe()
    if pipe is None:
        return None
    try:
        # schnell is a distilled/fast variant — 4 steps is its intended range
        result = pipe(
            prompt=f"Simple educational diagram, clear labels, classroom teaching style: {prompt}",
            height=512, width=512,
            num_inference_steps=4,
            guidance_scale=0.0,
        )
        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return None
