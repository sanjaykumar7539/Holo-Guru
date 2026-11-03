"""
core/vision_engine.py
Handles "visual queries" — student points the camera at a textbook diagram,
worksheet, or object and asks about it. Uses Moondream2 (small, fast local
VLM —~1.9GB, practical to load alongside the other models on Jetson,
unlike a full Qwen-VL variant).

This is separate from emotion_detector.py / gesture_detector.py, which
read the student's face/hands, not what the student is showing the camera.
"""
import base64
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_lock = threading.Lock()

MODEL_ID = "vikhyatk/moondream2"


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading vision model '{MODEL_ID}'...")
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, trust_remote_code=True,
                torch_dtype=torch.float16, device_map="auto",
            )
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            logger.info("✅ Vision model loaded.")
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")


def warmup():
    try:
        _load_model()
    except Exception as e:
        logger.warning(f"Vision engine warmup skipped: {e}")


def _decode_frame(image_b64: str):
    from PIL import Image
    import io
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def answer_visual_query(image_b64: str, question: str) -> dict:
    """
    Input: base64 image (textbook page/diagram/object) + student's question.
    Output: {"answer": str, "ok": bool}
    """
    _load_model()
    if _model is None:
        return {"answer": "Visual question feature is not available right now.", "ok": False}

    try:
        image = _decode_frame(image_b64)
        enc_image = _model.encode_image(image)
        answer = _model.answer_question(enc_image, question, _tokenizer)
        return {"answer": answer.strip(), "ok": True}
    except Exception as e:
        logger.error(f"Visual query error: {e}")
        return {"answer": "I couldn't process that image, please try again.", "ok": False}
