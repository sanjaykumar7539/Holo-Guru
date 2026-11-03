"""
core/tts_engine.py
Primary: Qwen3-TTS-0.6B (local, offline) for English/Tanglish.
Tamil:   ai4bharat/indic-parler-tts (local, offline) — no Tamil support in
         Qwen3-TTS, and this replaces the old edge-tts cloud call, so the
         whole TTS stack is now fully local, matching the project's
         fully-local requirement.
Final fallback: espeak-ng (fully offline, any language) if either model
         fails to load or errors at inference time.
"""
import io
import logging
import re
import subprocess
import tempfile
import threading
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Qwen3-TTS (local, English/Tanglish only) ────────────────────────────────
QWEN_TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
QWEN_TTS_SPEAKERS = {"male": "Ryan", "female": "Vivian"}

_qwen_tts_model = None
_qwen_tts_lock = threading.Lock()


def _load_qwen_tts():
    global _qwen_tts_model
    if _qwen_tts_model is not None:
        return _qwen_tts_model
    with _qwen_tts_lock:
        if _qwen_tts_model is not None:
            return _qwen_tts_model
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            logger.info(f"Loading Qwen3-TTS '{QWEN_TTS_MODEL_ID}'...")
            _qwen_tts_model = Qwen3TTSModel.from_pretrained(
                QWEN_TTS_MODEL_ID, device_map="auto", dtype=torch.float16
            )
            logger.info("✅ Qwen3-TTS loaded.")
        except ImportError:
            logger.warning("qwen-tts not installed — run: pip install qwen-tts")
        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS: {e}")
    return _qwen_tts_model


def _qwen_tts_synth(text: str, gender: Optional[str] = None) -> Optional[bytes]:
    model = _load_qwen_tts()
    if model is None:
        return None
    try:
        import soundfile as sf
        speaker = QWEN_TTS_SPEAKERS.get((gender or DEFAULT_GENDER).lower(), QWEN_TTS_SPEAKERS["female"])
        wavs, sr = model.generate_custom_voice(text=text, language="English", speaker=speaker)
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Qwen3-TTS synth error: {e}")
        return None


# ── ai4bharat/indic-parler-tts (local, Tamil) ───────────────────────────────
INDIC_TTS_MODEL_ID = "ai4bharat/indic-parler-tts"

_indic_tts_model = None
_indic_tts_tokenizer = None
_indic_desc_tokenizer = None
_indic_tts_lock = threading.Lock()

INDIC_TTS_DESCRIPTIONS = {
    "male": "A male Indian speaker with a clear Tamil accent delivers speech "
            "at a moderate speed and pitch. The recording is very close up and high quality.",
    "female": "A female Indian speaker with a clear Tamil accent delivers speech "
               "at a moderate speed and pitch. The recording is very close up and high quality.",
}


def _load_indic_tts():
    global _indic_tts_model, _indic_tts_tokenizer, _indic_desc_tokenizer
    if _indic_tts_model is not None:
        return _indic_tts_model
    with _indic_tts_lock:
        if _indic_tts_model is not None:
            return _indic_tts_model
        try:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            logger.info(f"Loading Tamil TTS '{INDIC_TTS_MODEL_ID}'...")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            _indic_tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
                INDIC_TTS_MODEL_ID
            ).to(device)
            _indic_tts_tokenizer = AutoTokenizer.from_pretrained(INDIC_TTS_MODEL_ID)
            _indic_desc_tokenizer = AutoTokenizer.from_pretrained(
                _indic_tts_model.config.text_encoder._name_or_path
            )
            logger.info("✅ Tamil TTS (indic-parler-tts) loaded.")
        except ImportError:
            logger.warning(
                "parler-tts not installed — run: "
                "pip install git+https://github.com/huggingface/parler-tts.git"
            )
        except Exception as e:
            logger.error(f"Failed to load indic-parler-tts: {e}")
    return _indic_tts_model


def _indic_tts_synth(text: str, gender: Optional[str] = None) -> Optional[bytes]:
    model = _load_indic_tts()
    if model is None:
        return None
    try:
        import torch
        import soundfile as sf

        device = model.device
        g = (gender or DEFAULT_GENDER).lower()
        description = INDIC_TTS_DESCRIPTIONS.get(g, INDIC_TTS_DESCRIPTIONS["female"])

        desc_ids = _indic_desc_tokenizer(description, return_tensors="pt").to(device)
        prompt_ids = _indic_tts_tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            generation = model.generate(
                input_ids=desc_ids.input_ids,
                attention_mask=desc_ids.attention_mask,
                prompt_input_ids=prompt_ids.input_ids,
                prompt_attention_mask=prompt_ids.attention_mask,
            )

        audio = generation.cpu().numpy().squeeze()
        buf = io.BytesIO()
        sf.write(buf, audio, model.config.sampling_rate, format="WAV")
        return buf.getvalue()
    except Exception as e:
        logger.error(f"indic-parler-tts synth error: {e}")
        return None


def warmup_tts_models():
    """Call once at app startup so the first real answer isn't slow."""
    try:
        _load_qwen_tts()
    except Exception as e:
        logger.warning(f"Qwen3-TTS warmup skipped: {e}")
    try:
        _load_indic_tts()
    except Exception as e:
        logger.warning(f"indic-parler-tts warmup skipped: {e}")


DEFAULT_GENDER = "female"
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")

# Common Tanglish words (Tamil words written in English)
_TANGLISH = {
    "enna", "epdi", "sollu", "theriyum", "illai", "oru", "naan", "nee",
    "avan", "aval", "inge", "ange", "paaru", "keelu", "seri", "poi",
    "vaa", "da", "pa", "ma", "anna", "akka", "thambi", "vanakkam",
    "nandri", "romba", "konjam", "yenna", "solren", "puriyala",
}


def detect_language(text: str) -> str:
    """Return 'ta' | 'tanglish' | 'en'"""
    if _TAMIL_RE.search(text):
        return "ta"
    words = set(re.findall(r"\b\w+\b", text.lower()))
    if words & _TANGLISH:
        return "tanglish"
    return "en"


# ── espeak-ng fallback (fully offline) ────────────────────────────────────────

def _espeak_synth(text: str, lang: str, gender: Optional[str] = None) -> Optional[bytes]:
    try:
        base_voice = "ta" if lang == "ta" else "en-in"
        # espeak-ng variant suffixes: +m3 (male), +f3 (female)
        variant = "+m3" if (gender or DEFAULT_GENDER).lower() == "male" else "+f3"
        voice_flag = base_voice + variant
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        subprocess.run(
            ["espeak-ng", "-v", voice_flag, "-s", "148", "-w", tmp, text],
            check=True, capture_output=True,
        )
        with open(tmp, "rb") as f:
            data = f.read()
        os.unlink(tmp)
        return data
    except Exception as e:
        logger.error(f"espeak-ng error: {e}")
        return None


# ── Text cleaner ──────────────────────────────────────────────────────────────

def _clean(text: str, max_chars: int = 120) -> str:
    """
    Extract first sentence only — keeps TTS under 1-2 seconds.
    Stripping markdown and emoji first, then cut at first sentence end.
    """
    # Strip markdown and common emojis
    t = re.sub(r"[*_`#]", "", text)
    t = re.sub(r"[⚠️❌✅📚🎓📗📎🇮🇳]", "", t)
    t = re.sub(r"\s+", " ", t).strip()

    # Split on sentence boundaries and take the first non-empty sentence
    sentences = re.split(r"(?<=[.?!])\s+", t)
    first = next((s.strip() for s in sentences if len(s.strip()) > 8), t)

    # Hard cap — never send more than max_chars to TTS server
    if len(first) > max_chars:
        # Cut at last word boundary before max_chars
        cut = first[:max_chars]
        last_space = cut.rfind(" ")
        first = cut[:last_space] + "." if last_space > 20 else cut + "."

    return first


# ── Public API ────────────────────────────────────────────────────────────────

def synthesise(text: str, lang: Optional[str] = None, gender: Optional[str] = None) -> tuple:
    """
    Returns (audio_bytes, mime_type, detected_lang)
    `gender` ("male"|"female") picks a natural-sounding voice matching the
    student's chosen avatar profile — defaults to female if not provided.
    Fully local: Qwen3-TTS for English/Tanglish, indic-parler-tts for Tamil,
    espeak-ng as last-resort fallback for either.
    """
    if not text or not text.strip():
        return None, "audio/wav", "en"

    clean_text = _clean(text)
    if not clean_text:
        return None, "audio/wav", "en"

    detected = lang or detect_language(clean_text)

    # Primary path — Qwen3-TTS (local) for English/Tanglish
    if detected in ("en", "tanglish"):
        audio = _qwen_tts_synth(clean_text, gender)
        if audio:
            logger.info(f"TTS Qwen3-TTS [{gender or DEFAULT_GENDER}] {len(audio)}B")
            return audio, "audio/wav", detected

    # Primary path — indic-parler-tts (local) for Tamil
    if detected == "ta":
        audio = _indic_tts_synth(clean_text, gender)
        if audio:
            logger.info(f"TTS indic-parler-tts [{gender or DEFAULT_GENDER}] {len(audio)}B")
            return audio, "audio/wav", detected

    # Fallback: espeak-ng (either model unavailable/errored)
    audio = _espeak_synth(clean_text, detected, gender)
    if audio:
        logger.info(f"TTS espeak fallback {len(audio)}B")
        return audio, "audio/wav", detected

    return None, "audio/wav", detected
