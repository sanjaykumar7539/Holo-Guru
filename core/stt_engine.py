"""
core/stt_engine.py
Speech-to-text via facebook/seamless-m4t-v2-large (replaces faster-whisper).

seamless-m4t-v2-large supports ~100 languages including Tamil natively,
which faster-whisper's tiny/base models handled poorly — this should
meaningfully improve Tamil/Tanglish recognition accuracy.

Loaded once and kept resident (large model — repeated load/unload per
request would blow the sub-2s latency budget).
"""
import logging
import tempfile
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_model = None
_processor = None
_model_lock = threading.Lock()

# seamless uses 3-letter language codes
LANG_CODE_MAP = {"en": "eng", "ta": "tam", None: "eng"}


def _load_model(model_id: str):
    global _model, _processor
    if _model is not None:
        return

    with _model_lock:
        if _model is not None:
            return

        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        logger.info(f"Loading STT model '{model_id}'...")
        _processor = AutoProcessor.from_pretrained(model_id)
        _model = SeamlessM4Tv2Model.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        logger.info(f"✅ STT model '{model_id}' loaded.")


def warmup(model_id: str):
    try:
        _load_model(model_id)
        logger.info(f"STT model '{model_id}' warmed up.")
    except Exception as e:
        logger.warning(f"STT warmup skipped/failed: {e}")


def transcribe(audio_path: str, lang_hint: Optional[str], model_id: str) -> dict:
    """
    audio_path: path to a WAV/webm/ogg file on disk (already saved by the caller).
    lang_hint: "en" | "ta" | None (None = auto-detect not supported by this
               model in ASR mode, so we default to English unless told otherwise).
    Returns: {"text": str, "language": "en"|"ta"}
    """
    _load_model(model_id)

    import torch
    import torchaudio

    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:   # stereo -> mono
            waveform = waveform.mean(dim=0, keepdim=True)

        tgt_lang = LANG_CODE_MAP.get(lang_hint, "eng")

        inputs = _processor(
            audios=waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
        ).to(_model.device)

        with torch.no_grad():
            output_tokens = _model.generate(
                **inputs, tgt_lang=tgt_lang, generate_speech=False
            )

        text = _processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        detected_lang = "ta" if tgt_lang == "tam" else "en"

        return {"text": text.strip(), "language": detected_lang}

    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        return {"text": "", "language": lang_hint or "en"}
