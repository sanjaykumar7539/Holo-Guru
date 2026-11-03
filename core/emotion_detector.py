"""
core/emotion_detector.py
Lightweight, fully-local facial emotion detection for the adaptive teaching
loop. Runs on Jetson Orin Nano CPU/GPU.

Design:
  1. OpenCV Haar cascade finds the face (bundled with opencv-python, no
     download needed).
  2. If an ONNX emotion-classification model is present at EMOTION_MODEL_PATH
     (e.g. a FER+ / AffectNet model you place there), it is used for a real
     prediction: angry, disgust, fear, happy, neutral, sad, surprise.
  3. If no model file is found, falls back to a simple neutral reading rather
     than crashing — the adaptive engine treats "neutral" as "no signal,
     don't change teaching behaviour".

This module never sends video off the device — everything runs locally,
matching the project's fully-local / privacy-first requirement.
"""
import base64
import logging
import os
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Map raw emotions -> teaching-relevant buckets used by adaptive_engine.py
TEACHING_BUCKET = {
    "happy": "engaged",
    "surprise": "engaged",
    "neutral": "neutral",
    "sad": "confused",
    "fear": "confused",
    "angry": "frustrated",
    "disgust": "frustrated",
}

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
EMOTION_MODEL_PATH = os.path.join(_MODEL_DIR, "emotion_ferplus.onnx")

_face_cascade = None
_onnx_session = None
_model_checked = False


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        import cv2
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _get_onnx_session():
    """Lazily load the ONNX emotion model. Returns None if not installed."""
    global _onnx_session, _model_checked
    if _model_checked:
        return _onnx_session
    _model_checked = True

    if not os.path.exists(EMOTION_MODEL_PATH):
        logger.warning(
            f"Emotion model not found at {EMOTION_MODEL_PATH} — "
            "emotion detection will report 'neutral' only. "
            "Place a FER+ ONNX model there to enable real detection."
        )
        return None

    try:
        import onnxruntime as ort
        _onnx_session = ort.InferenceSession(
            EMOTION_MODEL_PATH, providers=["CPUExecutionProvider"]
        )
        logger.info("✅ Emotion ONNX model loaded.")
    except ImportError:
        logger.warning("onnxruntime not installed — run: pip install onnxruntime")
    except Exception as e:
        logger.error(f"Failed to load emotion model: {e}")

    return _onnx_session


def _decode_frame(image_b64: str) -> Optional[np.ndarray]:
    try:
        import cv2
        if "," in image_b64:  # strip data:image/jpeg;base64, prefix
            image_b64 = image_b64.split(",", 1)[1]
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Frame decode error: {e}")
        return None


def detect_emotion(image_b64: str) -> Dict:
    """
    Input: base64-encoded JPEG/PNG frame from the browser webcam.
    Output: {"emotion": "happy", "bucket": "engaged", "confidence": 0.0-1.0,
             "face_found": bool}
    """
    import cv2

    frame = _decode_frame(image_b64)
    if frame is None:
        return {"emotion": "neutral", "bucket": "neutral", "confidence": 0.0, "face_found": False}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _get_face_cascade().detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return {"emotion": "neutral", "bucket": "neutral", "confidence": 0.0, "face_found": False}

    # Largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y + h, x:x + w]

    session = _get_onnx_session()
    if session is None:
        return {"emotion": "neutral", "bucket": "neutral", "confidence": 0.3, "face_found": True}

    try:
        face_input = cv2.resize(face_roi, (64, 64)).astype(np.float32)
        face_input = face_input.reshape(1, 1, 64, 64)
        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: face_input})[0][0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        idx = int(np.argmax(probs))
        emotion = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else "neutral"
        return {
            "emotion": emotion,
            "bucket": TEACHING_BUCKET.get(emotion, "neutral"),
            "confidence": round(float(probs[idx]), 3),
            "face_found": True,
        }
    except Exception as e:
        logger.error(f"Emotion inference error: {e}")
        return {"emotion": "neutral", "bucket": "neutral", "confidence": 0.0, "face_found": True}
