"""
core/llm_engine.py
Local Qwen3-8B inference via transformers (replaces Ollama/phi3:mini).

Loaded once at startup (see warmup()) and kept resident in memory — an 8B
model is too slow to reload per-request. 4-bit quantization (bitsandbytes)
is used to fit an 8B model on an 8GB Jetson Orin Nano; even so, this is
tight — see models/README_LLM.md for memory notes.
"""
import logging
import threading
from typing import Generator, List, Dict, Optional

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_model_lock = threading.Lock()


def _load_model(model_id: str, load_in_4bit: bool = True):
    global _model, _tokenizer
    if _model is not None:
        return

    with _model_lock:
        if _model is not None:   # re-check after acquiring lock
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading LLM '{model_id}' (4-bit={load_in_4bit})...")
        _tokenizer = AutoTokenizer.from_pretrained(model_id)

        kwargs = dict(torch_dtype="auto", device_map="auto")
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                logger.warning("bitsandbytes not installed — loading full precision (needs much more RAM). "
                                "Run: pip install bitsandbytes")

        _model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        logger.info(f"✅ LLM '{model_id}' loaded.")


def warmup(model_id: str, load_in_4bit: bool = True):
    """Call once at app startup so the first real question isn't slow."""
    try:
        _load_model(model_id, load_in_4bit)
        # tiny generation to force CUDA kernels/cache to initialize
        list(stream_answer("hi", "hi", [], "You are a helpful assistant.",
                            model_id, {"max_new_tokens": 5}, load_in_4bit))
        logger.info(f"Model '{model_id}' warmed up.")
    except Exception as e:
        logger.warning(f"Warmup skipped/failed: {e}")


def stream_answer(
    question: str,
    context: str,
    history: List[Dict],
    system_prompt: str,
    model_id: str,
    options: dict,
    load_in_4bit: bool = True,
) -> Generator[str, None, None]:

    if not context.strip():
        yield "This topic is not covered in the uploaded books."
        return

    _load_model(model_id, load_in_4bit)

    import torch
    from transformers import TextIteratorStreamer

    context = context[:1200]
    history = history[-2:]

    current_user_msg = f"Book excerpts:\n{context}\n\nQuestion: {question}"

    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": current_user_msg})

    try:
        text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _tokenizer(text, return_tensors="pt").to(_model.device)

        streamer = TextIteratorStreamer(
            _tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=options.get("num_predict", 800),
            temperature=options.get("temperature", 0.2),
            do_sample=options.get("temperature", 0.2) > 0,
        )

        thread = threading.Thread(target=_model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            if token:
                yield token

        thread.join()

    except Exception as e:
        logger.error(f"LLM stream error: {e}")
        yield f"\n\nLLM error: {e}"
