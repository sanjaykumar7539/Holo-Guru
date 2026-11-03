"""
main.py — HOLO-GURU single entry point.

Combines what used to be app.py (Flask app + SSL setup) and api/routes.py
(all API endpoints) into one file, so the whole backend + the route that
serves the frontend (templates/index.html) can be started with just:

    python main.py

Model engines (core/llm_engine.py, stt_engine.py, tts_engine.py,
react_engine.py, adaptive_engine.py, personalizer.py, emotion_detector.py,
gesture_detector.py, rag_engine.py, ingestion.py) stay as separate modules —
merging heavy model code into this file would make it unreadable and
break the "surgical patch" workflow the team already uses. "Combined" here
means one file to run, not one file containing everything.
"""
import os
import io
import uuid
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from collections import defaultdict

from flask import Flask, request, jsonify, Response, stream_with_context, render_template, g

import config
from core.ingestion import ingest_book, list_books, delete_book
from core.llm_engine import warmup as warmup_llm
from core.tts_engine import synthesise, detect_language
from core.react_engine import stream_react_answer
from core.stt_engine import transcribe
from core import realtime_voice, observability
from core.emotion_detector import detect_emotion
from core.gesture_detector import detect_gesture
from core.adaptive_engine import process_signals
from core.visual_aid_generator import extract_chart_tag, render_chart
from core.vision_engine import answer_visual_query
from core.image_gen import extract_image_tag, generate_image
from core.hologram_engine import get_state as get_hologram_state, apply_action as apply_hologram_action, apply_command as apply_hologram_command
from core.holo_room import join_room, leave_room, heartbeat as room_heartbeat, snapshot as room_snapshot, update_scene as room_update_scene, update_annotation as room_update_annotation
from core.video_gen import extract_video_tag, start_video_job, get_job_status
from core import cache_engine, cag_engine
from core.security import validate_question, detect_prompt_injection, SlidingWindowLimiter, check_teacher_token, extract_teacher_token, check_room_pin, extract_room_pin
from core.audit import AuditLog
from core.traffic_guard import TrafficGuard
from core.vector_store import connect as connect_weaviate, ensure_collection
from core.task_planner import create_plan
from core.assistant_features import current_datetime, system_status, save_note, daily_briefing
from core.core_memory import get_memory, remember, forget, render_prompt_block
from core.learning import record_turn, record_feedback, stats as learning_stats
from core.classroom_orchestrator import (
    save_config as save_classroom_config,
    state_summary as classroom_state_summary,
    sync_timetable, start_topic, set_attendance, record_mastery,
    request_topic_advance, handle_event as handle_classroom_event, teaching_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── In-memory state ──────────────────────────────────────────────────────────
_conversations: dict = defaultdict(list)
MAX_HISTORY_TURNS = 10
_profiles: dict = {}   # session_id -> {"gender": ..., "age_group": ...}


# ── SSL cert using system openssl (no pyopenssl needed) ───────────────────────
def _ensure_ssl_cert(cert_path: str, key_path: str) -> bool:
    if os.path.exists(cert_path) and os.path.exists(key_path):
        logger.info("✅ SSL cert already exists.")
        return True

    logger.info("Generating self-signed SSL certificate using openssl…")
    try:
        subprocess.run([
            "openssl", "req", "-x509",
            "-newkey", "rsa:2048",
            "-keyout", key_path,
            "-out",    cert_path,
            "-days",   "365",
            "-nodes",
            "-subj",   "/C=IN/O=HoloGuru/CN=localhost",
        ], check=True, capture_output=True)
        logger.info(f"✅ SSL cert generated → {cert_path}")
        return True
    except FileNotFoundError:
        logger.error("openssl not found. Install: sudo apt install openssl")
    except subprocess.CalledProcessError as e:
        logger.error(f"openssl failed: {e.stderr.decode()}")
    return False


def create_app():
    app = Flask(__name__)

    for key in dir(config):
        if key.isupper():
            app.config[key] = getattr(config, key)

    try:
        vector_client = connect_weaviate(
            host=config.WEAVIATE_HOST,
            http_port=config.WEAVIATE_HTTP_PORT,
            grpc_port=config.WEAVIATE_GRPC_PORT,
        )
        collection = ensure_collection(vector_client, config.COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Weaviate connection failed. Start it with `docker compose up -d weaviate` "
            "before starting HOLO-GURU."
        ) from exc

    app.config["VECTOR_CLIENT"] = vector_client
    app.config["COLLECTION"] = collection
    logger.info(f"Weaviate ready — collection '{config.COLLECTION_NAME}'")

    app.config["SECRET_KEY"] = os.urandom(24)
    cache_engine.MAX_ENTRIES = app.config["CACHE_MAX_ENTRIES"]
    limiter = SlidingWindowLimiter(
        app.config["RATE_LIMIT_REQUESTS"], app.config["RATE_LIMIT_WINDOW_SECONDS"]
    )
    audit = AuditLog(app.config["AUDIT_DB"])
    traffic_guard = TrafficGuard(
        app.config["LLM_MAX_CONCURRENT"], app.config["LLM_MAX_WAITING"],
        app.config["LLM_QUEUE_WAIT_SECONDS"],
    )
    # Reject oversized bodies early (Flask aborts with 413).
    app.config["MAX_CONTENT_LENGTH"] = int(app.config.get("MAX_UPLOAD_MB", 50)) * 1024 * 1024

    @app.before_request
    def observability_request_start():
        if app.config.get("OBSERVABILITY_ENABLED", True):
            g._holo_request_started = time.perf_counter()

    @app.after_request
    def observability_request_end(response):
        started = getattr(g, "_holo_request_started", None)
        if started is not None and app.config.get("OBSERVABILITY_ENABLED", True):
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            observability.record(
                f"http.{request.endpoint or 'unknown'}",
                elapsed_ms,
                ok=response.status_code < 500,
                slow_ms=app.config.get("OBSERVABILITY_SLOW_REQUEST_MS", 1500),
                meta={"method": request.method, "status": response.status_code},
            )
            response.headers["X-HoloGuru-Latency-Ms"] = f"{elapsed_ms:.2f}"
        return response

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(self), geolocation=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; script-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        )
        if request.is_secure:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    # NOTE (hardware): only the LLM is warmed up eagerly at startup. STT
    # (seamless-m4t-v2-large), TTS (Qwen3-TTS / indic-parler-tts), and the
    # vision model (Moondream2, visual queries) load lazily on first use —
    # loading all five models resident at once will almost certainly
    # exceed 8GB RAM on the Jetson Orin Nano.
    warmup_llm(config.LLM_MODEL_ID, config.LLM_LOAD_IN_4BIT)

    def _cfg(key):
        return app.config[key]

    def _collection():
        return app.config["COLLECTION"]

    # ── Frontend ──────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    # ── Upload book ───────────────────────────────────────────────────────
    def _require_teacher(action="admin"):
        """Gate knowledge admin routes when TEACHER_TOKEN is configured."""
        expected = _cfg("TEACHER_TOKEN")
        provided = extract_teacher_token(request.headers, request.form, request.get_json(silent=True))
        if check_teacher_token(provided, expected):
            return None
        audit.write(
            "teacher_auth_denied",
            client_ip=request.remote_addr or "",
            details={"action": action},
        )
        return jsonify({"error": "Teacher authorization required."}), 401

    @app.route("/api/upload", methods=["POST"])
    def upload():
        denied = _require_teacher("upload")
        if denied:
            return denied

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"error": "No file provided"}), 400

        ext = Path(f.filename).suffix.lower()
        allowed = _cfg("ALLOWED_EXTENSIONS")

        if ext not in allowed:
            return jsonify({"error": f"Unsupported format. Use: {', '.join(sorted(allowed))}"}), 400

        # Content-Length check (defense in depth alongside MAX_CONTENT_LENGTH).
        max_bytes = int(_cfg("MAX_UPLOAD_MB")) * 1024 * 1024
        content_length = request.content_length
        if content_length is not None and content_length > max_bytes:
            return jsonify({"error": f"File too large. Maximum is {_cfg('MAX_UPLOAD_MB')} MB."}), 413

        save_path = os.path.join(_cfg("UPLOAD_DIR"), f"{uuid.uuid4().hex}{ext}")
        try:
            f.save(save_path)
            if os.path.getsize(save_path) > max_bytes:
                os.remove(save_path)
                return jsonify({"error": f"File too large. Maximum is {_cfg('MAX_UPLOAD_MB')} MB."}), 413

            info = ingest_book(save_path, _collection(), _cfg("CHUNK_SIZE"), _cfg("CHUNK_OVERLAP"))
            if not info.get("duplicate"):
                cache_engine.clear_all()
                cag_engine.clear()
            audit.write("knowledge_upload", client_ip=request.remote_addr or "", details=info)
            return jsonify({"success": True, **info})
        except Exception as e:
            logger.error(f"Ingest error: {e}")
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except OSError as cleanup_err:
                    logger.debug("Could not remove failed upload %s: %s", save_path, cleanup_err)
            return jsonify({"error": "Failed to process the uploaded file."}), 500

    # ── List / delete books ──────────────────────────────────────────────
    @app.route("/api/books", methods=["GET"])
    def books():
        return jsonify(list_books(_collection()))

    @app.route("/api/books/<book_id>", methods=["DELETE"])
    def delete(book_id):
        denied = _require_teacher("delete")
        if denied:
            return denied

        book_id = (book_id or "").strip()
        if not book_id or len(book_id) > 128:
            return jsonify({"error": "Invalid book id."}), 400

        n = delete_book(book_id, _collection())
        if n:
            cache_engine.clear_all()
            cag_engine.clear()
            audit.write(
                "knowledge_delete",
                client_ip=request.remote_addr or "",
                details={"book_id": book_id, "chunks": n},
            )
        return jsonify({"deleted_chunks": n})

    # ── Ask question (streaming, ReAct + adaptive engine) ────────────────
    @app.route("/api/ask", methods=["POST"])
    def ask():
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        book_ids = data.get("book_ids") or None
        sess_id = (data.get("session_id") or "default").strip()

        emotion_bucket = data.get("emotion_bucket")
        gesture_in = data.get("gesture")
        preferred_lang = data.get("preferred_lang")
        hologram_context = data.get("hologram_context") or {}
        holo_asset = str(hologram_context.get("asset") or "").strip().lower()[:40]
        holo_part = str(hologram_context.get("selected_part") or "").strip()[:80]
        holo_layer = str(hologram_context.get("layer") or "").strip().lower()[:40]
        if holo_part.lower() in {"auto", "none", "null"}:
            holo_part = ""

        client_ip = request.remote_addr or "unknown"
        allowed, retry_after = limiter.check(f"{client_ip}:{sess_id}")
        if not allowed:
            audit.write("rate_limit", sess_id, client_ip)
            response = jsonify({"error": "Too many requests. Please try again shortly."})
            response.status_code = 429
            response.headers["Retry-After"] = str(retry_after)
            return response

        valid, checked = validate_question(question, _cfg("MAX_QUESTION_CHARS"))
        if not valid:
            return jsonify({"error": checked}), 400
        question = checked
        injection = detect_prompt_injection(question)
        if injection:
            audit.write("prompt_injection_blocked", sess_id, client_ip, {"pattern": injection})
            return jsonify({"error": "Protected system instructions cannot be changed or exposed."}), 400
        task_plan = create_plan(question, has_books=bool(list_books(_collection())))

        history = _conversations[sess_id]
        books_key = json.dumps(sorted(book_ids or []))
        holo_key = json.dumps({"asset": holo_asset, "part": holo_part, "layer": holo_layer}, sort_keys=True)
        cached = None if history else cache_engine.get(
            "grounded_answer", question, books_key, preferred_lang or "auto", holo_key
        )
        if cached:
            def cached_stream():
                yield f"data: {json.dumps({'stage': cached['stage'], 'sources': cached['sources']})}\n\n"
                yield f"data: {json.dumps({'token': cached['answer']})}\n\n"
                yield "data: [DONE]\n\n"
            response = Response(stream_with_context(cached_stream()), mimetype="text/event-stream")
            response.headers.update({"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "X-HoloGuru-Cache": "HIT"})
            return response

        admission_started = traffic_guard.acquire()
        if admission_started is None:
            audit.write("overload_rejected", sess_id, client_ip)
            response = jsonify({"error": "Holo-Guru is busy. Please retry shortly."})
            response.status_code = 503
            response.headers["Retry-After"] = str(_cfg("OVERLOAD_RETRY_AFTER_SECONDS"))
            return response

        classroom_directive = teaching_prompt(sess_id)
        classroom_system_prompt = (
            _cfg("SYSTEM_PROMPT") + "\n\n" + classroom_directive
            + "\n\n" + render_prompt_block()
        )

        if holo_asset or holo_part:
            visual_context = (
                "\n\nHOLOGRAM INTERFACE CONTEXT (UI state, not a factual source): "
                f"displayed model={holo_asset or 'unknown'}; "
                f"selected part={holo_part or 'none'}; layer={holo_layer or 'surface'}. "
                "If the student's wording uses pronouns such as 'this', 'it', or 'that part', "
                "interpret them using this UI selection. All factual claims must still come from retrieved evidence."
            )
            classroom_system_prompt += visual_context

        question_for_ai = question
        if holo_part:
            question_for_ai = f"{question}\nCurrent hologram selection: {holo_asset or 'model'} / {holo_part}."
        elif holo_asset and any(token in question.lower().split() for token in {"this", "it", "model", "object"}):
            question_for_ai = f"{question}\nCurrent hologram model: {holo_asset}."

        decision = process_signals(
            student_id=sess_id,
            base_system_prompt=classroom_system_prompt,
            question=question_for_ai,
            emotion_bucket=emotion_bucket,
            gesture=gesture_in,
            preferred_language=preferred_lang,
        )

        if decision["pause"]:
            return jsonify({"paused": True})

        model_used = _cfg("LLM_MODEL_ID")
        full_reply = []
        response_meta = {"stage": "none", "sources": []}

        def generate():
            stream_started = time.perf_counter()
            first_token_recorded = False
            q_lang = decision["switch_language"] or detect_language(question)
            yield f"data: {json.dumps({'lang': q_lang})}\n\n"
            yield f"data: {json.dumps({'plan': task_plan})}\n\n"

            if decision.get("intent"):
                yield f"data: {json.dumps({'intent': decision.get('intent'), 'intent_confidence': decision.get('intent_confidence', 0)})}\n\n"

            if decision["simplify"]:
                yield f"data: {json.dumps({'adaptive': 'simplifying'})}\n\n"

            for event in stream_react_answer(
                decision["boosted_question"],
                _collection(),
                _cfg("TOP_K"),
                book_ids,
                history,
                decision["system_prompt"],
                model_used,
                _cfg("LLM_OPTIONS"),
                _cfg("LLM_LOAD_IN_4BIT"),
            ):
                if event["type"] == "stage":
                    response_meta.update({"stage": event["stage"], "sources": event["sources"]})
                    yield f"data: {json.dumps({'sources': event['sources'], 'stage': event['stage']})}\n\n"
                elif event["type"] == "token":
                    if not first_token_recorded:
                        observability.record(
                            "ask.ttft",
                            (time.perf_counter() - stream_started) * 1000.0,
                            slow_ms=_cfg("OBSERVABILITY_SLOW_ASK_MS"),
                            meta={"stage": response_meta.get("stage", "none")},
                        )
                        first_token_recorded = True
                    full_reply.append(event["token"])
                    yield f"data: {json.dumps({'token': event['token']})}\n\n"
                elif event["type"] == "sentence":
                    # Fired as each sentence completes — lets the frontend start
                    # TTS on sentence 1 while the LLM is still generating the rest
                    # (parallel scheduling), instead of waiting for the full answer.
                    yield f"data: {json.dumps({'sentence': event['text']})}\n\n"

            assistant_answer = "".join(full_reply)

            # Visual aid: parse trailing [CHART: ...] / [IMAGE: ...] / [VIDEO: ...]
            # tag, render/generate it, tell the frontend to strip the raw tag
            # text and show the result. Only one tag is expected per answer.
            clean_text, chart_spec = extract_chart_tag(assistant_answer)
            if chart_spec:
                chart_data_url = render_chart(chart_spec)
                if chart_data_url:
                    yield f"data: {json.dumps({'chart': chart_data_url})}\n\n"
                assistant_answer = clean_text

            clean_text, image_prompt = extract_image_tag(assistant_answer)
            if image_prompt:
                image_data_url = cache_engine.get_or_compute(
                    "gen_image", image_prompt,
                    compute=lambda: generate_image(image_prompt),
                    ttl=60 * 60,
                )
                if image_data_url:
                    yield f"data: {json.dumps({'image': image_data_url})}\n\n"
                else:
                    logger.info("Image generation unavailable/failed — showing text only")
                assistant_answer = clean_text

            clean_text, video_prompt = extract_video_tag(assistant_answer)
            if video_prompt:
                # Slow (tens of seconds) — kicked off as a background job so it
                # never blocks the answer; frontend polls /api/video-status/<id>.
                job_id = start_video_job(video_prompt)
                if job_id:
                    yield f"data: {json.dumps({'video_job': job_id})}\n\n"
                else:
                    logger.info("Video generation unavailable — showing text only")
                assistant_answer = clean_text

            if not history and response_meta["stage"] == "book" and response_meta["sources"] and assistant_answer.strip():
                cache_engine.set(
                    "grounded_answer", question, books_key, preferred_lang or "auto", holo_key,
                    value={"answer": assistant_answer, **response_meta},
                    ttl=_cfg("CACHE_TTL_SECONDS"),
                )
            audit.write("grounded_answer", sess_id, client_ip, {
                "stage": response_meta["stage"], "sources": response_meta["sources"], "cached": False,
            })

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": assistant_answer})
            record_turn(sess_id, question, assistant_answer)

            max_msgs = MAX_HISTORY_TURNS * 2
            if len(history) > max_msgs:
                _conversations[sess_id] = history[-max_msgs:]

            yield "data: [DONE]\n\n"

        def protected_generate():
            failed = True
            started = time.perf_counter()
            try:
                yield from generate()
                failed = False
            finally:
                observability.record(
                    "ask.stream.total",
                    (time.perf_counter() - started) * 1000.0,
                    ok=not failed,
                    slow_ms=_cfg("OBSERVABILITY_SLOW_ASK_MS"),
                    meta={"stage": response_meta.get("stage", "none")},
                )
                traffic_guard.release(admission_started, failed=failed)

        response = Response(
            stream_with_context(protected_generate()),
            mimetype="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "X-HoloGuru-Cache": "MISS"},
        )
        return response

    @app.route("/api/assistant/plan", methods=["POST"])
    def assistant_plan():
        data = request.get_json(silent=True) or {}
        question = str(data.get("question") or "").strip()
        valid, checked = validate_question(question, _cfg("MAX_QUESTION_CHARS"))
        if not valid:
            return jsonify({"error": checked}), 400
        injection = detect_prompt_injection(checked)
        if injection:
            return jsonify({"error": "Protected system instructions cannot be changed or exposed."}), 400
        return jsonify(create_plan(checked, has_books=bool(list_books(_collection()))))

    @app.route("/api/assistant/feature", methods=["POST"])
    def assistant_feature():
        data = request.get_json(silent=True) or {}
        feature = str(data.get("feature") or "").strip().lower()
        if feature == "datetime":
            return jsonify({"feature": feature, "result": current_datetime()})
        if feature == "system_status":
            return jsonify({"feature": feature, "result": system_status()})
        if feature == "note":
            try:
                result = save_note(data.get("text") or "")
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            return jsonify({"feature": feature, "result": result})
        return jsonify({"error": "Unsupported assistant feature"}), 400

    @app.route("/api/assistant/memory", methods=["GET", "POST", "DELETE"])
    def assistant_memory():
        data = request.get_json(silent=True) or {}
        if request.method == "GET":
            return jsonify(get_memory())
        fact = str(data.get("fact") or "").strip()
        category = str(data.get("category") or "general").strip()
        try:
            memory = forget(fact, category) if request.method == "DELETE" else remember(fact, category)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"saved": request.method != "DELETE", "memory": memory})

    @app.route("/api/assistant/feedback", methods=["POST"])
    def assistant_feedback():
        data = request.get_json(silent=True) or {}
        sess_id = (data.get("session_id") or "default").strip()
        try:
            result = record_feedback(sess_id, str(data.get("signal") or ""), str(data.get("comment") or ""))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        audit.write("assistant_feedback", sess_id, request.remote_addr or "", {
            "signal": str(data.get("signal") or "").strip().lower(),
            "saved": result.get("saved", False),
        })
        return jsonify(result)

    @app.route("/api/assistant/learning", methods=["GET"])
    def assistant_learning():
        return jsonify(learning_stats())

    @app.route("/api/assistant/briefing", methods=["GET"])
    def assistant_briefing():
        sess_id = (request.args.get("session_id") or "default").strip()
        return jsonify(daily_briefing(
            get_memory(),
            classroom_state_summary(sess_id),
            learning_stats(),
        ))

    @app.route("/api/system/traffic", methods=["GET"])
    def traffic_status():
        """Operational counters for health dashboards/load testing."""
        return jsonify(traffic_guard.snapshot())

    @app.route("/api/system/metrics", methods=["GET"])
    def system_metrics():
        """Local performance snapshot; timings/counters only, no prompt content."""
        return jsonify(observability.snapshot())

    # ── Classroom teaching orchestration (no 3D/avatar dependency) ─────────
    @app.route("/api/classroom/configure", methods=["POST"])
    def classroom_configure():
        data = request.get_json(force=True)
        sess_id = (data.get("session_id") or "default").strip()
        cfg = save_classroom_config(sess_id, data)
        return jsonify({"saved": True, "config": cfg, **classroom_state_summary(sess_id)})

    @app.route("/api/classroom/state", methods=["GET"])
    def classroom_state():
        sess_id = (request.args.get("session_id") or "default").strip()
        return jsonify(classroom_state_summary(sess_id))

    @app.route("/api/classroom/sync-period", methods=["POST"])
    def classroom_sync_period():
        data = request.get_json(silent=True) or {}
        sess_id = (data.get("session_id") or "default").strip()
        return jsonify(sync_timetable(sess_id))

    @app.route("/api/classroom/start-topic", methods=["POST"])
    def classroom_start_topic():
        data = request.get_json(force=True)
        sess_id = (data.get("session_id") or "default").strip()
        return jsonify(start_topic(
            sess_id,
            subject=data.get("subject"),
            unit_index=int(data.get("unit_index", 0)),
            lesson_index=int(data.get("lesson_index", 0)),
            topic_index=int(data.get("topic_index", 0)),
        ))

    @app.route("/api/classroom/attendance", methods=["POST"])
    def classroom_attendance():
        data = request.get_json(force=True)
        sess_id = (data.get("session_id") or "default").strip()
        return jsonify(set_attendance(sess_id, data.get("present") or [], data.get("absent") or []))

    @app.route("/api/classroom/mastery", methods=["POST"])
    def classroom_mastery():
        data = request.get_json(force=True)
        sess_id = (data.get("session_id") or "default").strip()
        sid = str(data.get("student_id") or "").strip()
        if not sid:
            return jsonify({"error": "student_id required"}), 400
        understood = data.get("understood")
        if understood not in (True, False, None):
            return jsonify({"error": "understood must be true, false, or null"}), 400
        return jsonify(record_mastery(sess_id, sid, understood))

    @app.route("/api/classroom/advance", methods=["POST"])
    def classroom_advance():
        data = request.get_json(force=True)
        sess_id = (data.get("session_id") or "default").strip()
        return jsonify(request_topic_advance(sess_id, bool(data.get("teacher_override", False))))

    @app.route("/api/classroom/event", methods=["POST"])
    def classroom_event():
        data = request.get_json(force=True)
        sess_id = (data.get("session_id") or "default").strip()
        event = (data.get("event") or "").strip()
        if not event:
            return jsonify({"error": "event required"}), 400
        return jsonify(handle_classroom_event(sess_id, event, data.get("payload") or {}))

    # ── Clear history ─────────────────────────────────────────────────────
    @app.route("/api/history/clear", methods=["POST"])
    def clear_history():
        data = request.get_json(force=True)
        sess_id = (data.get("session_id") or "default").strip()
        _conversations.pop(sess_id, None)
        return jsonify({"cleared": True})

    # ── Health ────────────────────────────────────────────────────────────
    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok", "books_loaded": len(list_books(_collection())),
            "vector_store": "weaviate", "retrieval": "hybrid+bge-reranker-v2-m3",
            "classroom_orchestrator": True, "vision": True,
            "grounding": "strict", "web_fallback": _cfg("RAG_ALLOW_WEB_FALLBACK"),
            "cache": cache_engine.stats(),
            "cag": cag_engine.stats(),
        })

    # ── TTS ───────────────────────────────────────────────────────────────
    @app.route("/api/tts", methods=["POST"])
    def tts():
        data = request.get_json(force=True)
        text = (data.get("text") or "").strip()
        lang = data.get("lang") or None
        sess_id = (data.get("session_id") or "default").strip()
        gender = data.get("gender") or _profiles.get(sess_id, {}).get("gender")

        if not text:
            return jsonify({"error": "No text"}), 400

        with observability.span(
            "tts.synthesise",
            slow_ms=_cfg("OBSERVABILITY_SLOW_TTS_MS"),
            meta={"lang": lang or "auto"},
        ):
            audio, mime, detected = synthesise(text, lang, gender)
        if not audio:
            return jsonify({"error": "TTS failed"}), 500

        return Response(
            audio, mimetype=mime,
            headers={"X-Detected-Language": detected, "Cache-Control": "no-cache"},
        )

    # ── Student profile ───────────────────────────────────────────────────
    @app.route("/api/profile", methods=["POST"])
    def set_profile():
        data = request.get_json(force=True)
        sess_id = (data.get("session_id") or "default").strip()
        _profiles[sess_id] = {
            "gender": (data.get("gender") or "female").lower(),
            "age_group": data.get("age_group") or "",
        }
        return jsonify({"saved": True, "profile": _profiles[sess_id]})

    # ── Emotion / gesture sensing ─────────────────────────────────────────
    @app.route("/api/emotion", methods=["POST"])
    def emotion():
        data = request.get_json(force=True)
        image_b64 = data.get("image") or ""
        if not image_b64:
            return jsonify({"error": "No image"}), 400
        return jsonify(detect_emotion(image_b64))

    @app.route("/api/gesture", methods=["POST"])
    def gesture():
        data = request.get_json(force=True)
        image_b64 = data.get("image") or ""
        if not image_b64:
            return jsonify({"error": "No image"}), 400
        return jsonify(detect_gesture(image_b64))

    @app.route("/api/hologram/state", methods=["GET"])
    def hologram_state():
        session_id = request.args.get("session_id") or "default"
        return jsonify(get_hologram_state(session_id))

    @app.route("/api/hologram/action", methods=["POST"])
    def hologram_action():
        data = request.get_json(force=True, silent=True) or {}
        session_id = data.get("session_id") or "default"
        action = data.get("action", "")
        value = data.get("value")
        state = apply_hologram_action(session_id, action, value)
        return jsonify({"ok": True, "state": state})

    @app.route("/api/hologram/command", methods=["POST"])
    def hologram_command():
        data = request.get_json(force=True, silent=True) or {}
        session_id = data.get("session_id") or "default"
        command = data.get("command", "")
        result = apply_hologram_command(session_id, command)
        return jsonify({"ok": True, **result})

    def _require_room_pin(data=None, args=None, action="room"):
        """Gate multi-user room mutations when HOLO_ROOM_PIN is configured."""
        expected = _cfg("HOLO_ROOM_PIN")
        provided = extract_room_pin(request.headers, args or request.args, data)
        if check_room_pin(provided, expected):
            return None
        audit.write(
            "room_pin_denied",
            client_ip=request.remote_addr or "",
            details={"action": action},
        )
        return jsonify({"error": "Room PIN required or invalid."}), 401

    @app.route("/api/holo-room/join", methods=["POST"])
    def holo_room_join():
        data = request.get_json(force=True, silent=True) or {}
        denied = _require_room_pin(data, action="join")
        if denied:
            return denied
        room_id = (data.get("room_id") or "demo")
        user_id = (data.get("user_id") or "")
        name = data.get("name") or data.get("user_id") or "student"
        if not str(user_id).strip():
            return jsonify({"error": "user_id is required."}), 400
        return jsonify(join_room(room_id, user_id, name))

    @app.route("/api/holo-room/leave", methods=["POST"])
    def holo_room_leave():
        data = request.get_json(force=True, silent=True) or {}
        # Leave does not require PIN so clients can exit after a pin change.
        return jsonify(leave_room(data.get("room_id"), data.get("user_id")))

    @app.route("/api/holo-room/state", methods=["GET"])
    def holo_room_state():
        # Heartbeat/snapshot requires PIN when configured (prevents silent observers).
        denied = _require_room_pin(args=request.args, action="state")
        if denied:
            return denied
        room_id = request.args.get("room_id") or "demo"
        user_id = request.args.get("user_id") or "user"
        return jsonify(room_heartbeat(room_id, user_id))

    @app.route("/api/holo-room/scene", methods=["POST"])
    def holo_room_scene():
        data = request.get_json(force=True, silent=True) or {}
        denied = _require_room_pin(data, action="scene")
        if denied:
            return denied
        state = data.get("state")
        if state is not None and not isinstance(state, dict):
            return jsonify({"error": "state must be an object."}), 400
        return jsonify(room_update_scene(data.get("room_id"), data.get("user_id"), state or {}))

    @app.route("/api/holo-room/annotation", methods=["POST"])
    def holo_room_annotation():
        data = request.get_json(force=True, silent=True) or {}
        denied = _require_room_pin(data, action="annotation")
        if denied:
            return denied
        return jsonify(room_update_annotation(
            data.get("room_id"),
            data.get("user_id"),
            pointer=data.get("pointer"),
            stroke=data.get("stroke"),
            clear=bool(data.get("clear")),
        ))

    # ── Visual query: student shows a diagram/page/object, asks about it ───
    # POST /api/visual-query  { "image": "data:image/jpeg;base64,...", "question": "..." }
    @app.route("/api/visual-query", methods=["POST"])
    def visual_query():
        data = request.get_json(force=True)
        image_b64 = data.get("image") or ""
        question = (data.get("question") or "What is shown in this image?").strip()
        if not image_b64:
            return jsonify({"error": "No image"}), 400
        result = answer_visual_query(image_b64, question)
        return jsonify(result)

    # ── Video generation job polling (Wan2.1 runs async, tens of seconds) ──
    @app.route("/api/video-status/<job_id>", methods=["GET"])
    def video_status(job_id):
        return jsonify(get_job_status(job_id))

    # ── Detect language ───────────────────────────────────────────────────
    @app.route("/api/detect-lang", methods=["POST"])
    def detect_lang_route():
        data = request.get_json(force=True)
        text = (data.get("text") or "").strip()
        return jsonify({"lang": detect_language(text) if text else "en"})

    # ── STT helpers (final + real-time partial) ───────────────────────────
    def _run_stt_on_upload(audio_file, lang_hint=None):
        """Save upload → ffmpeg wav → transcribe. Returns (result_dict|None, error_response|None)."""
        ext = Path(audio_file.filename).suffix.lower() if audio_file.filename else ".webm"
        if ext not in {".webm", ".mp3", ".wav", ".ogg", ".m4a", ".mp4"}:
            ext = ".webm"
        tmp_path = None
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                tmp_path = f.name
                audio_file.save(tmp_path)

            if realtime_voice.is_short_audio_bytes(os.path.getsize(tmp_path)):
                return {"text": "", "language": lang_hint or "en", "empty": True}, None

            wav_path = tmp_path + ".wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
                check=True, capture_output=True,
            )

            stt_lang = None
            if lang_hint == "ta":
                stt_lang = "ta"
            elif lang_hint in ("en", "tanglish"):
                stt_lang = "en"

            with observability.span(
                "stt.transcribe",
                slow_ms=_cfg("OBSERVABILITY_SLOW_STT_MS"),
                meta={"lang": stt_lang or "auto"},
            ):
                result = transcribe(wav_path, stt_lang, _cfg("STT_MODEL_ID"))
            return result, None
        except Exception as e:
            logger.error(f"STT error: {e}")
            return None, (jsonify({"error": "Speech recognition failed. Please try again."}), 500)
        finally:
            for p in (tmp_path, wav_path):
                if p:
                    try:
                        os.unlink(p)
                    except OSError as cleanup_err:
                        logger.debug("Could not remove temp audio %s: %s", p, cleanup_err)

    @app.route("/api/stt", methods=["POST"])
    def stt():
        if "audio" not in request.files:
            return jsonify({"error": "No audio file"}), 400

        audio_file = request.files["audio"]
        lang_hint = (request.form.get("lang") or "").strip() or None
        session_id = (request.form.get("session_id") or "default").strip()

        result, err = _run_stt_on_upload(audio_file, lang_hint)
        if err:
            return err
        realtime_voice.mark_final(session_id)
        text = (result or {}).get("text") or ""
        detected = (result or {}).get("language") or (lang_hint or "en")
        logger.info(f"STT final: '{text[:60]}' [{detected}]")
        return jsonify({"text": text, "lang": detected, "partial": False})

    @app.route("/api/stt/partial", methods=["POST"])
    def stt_partial():
        """Real-time interim transcript from a rolling MediaRecorder buffer."""
        if not _cfg("REALTIME_VOICE_ENABLED"):
            return jsonify({"error": "Real-time voice disabled", "partial": True}), 503

        if "audio" not in request.files:
            return jsonify({"error": "No audio file"}), 400

        session_id = (request.form.get("session_id") or "default").strip()
        lang_hint = (request.form.get("lang") or "").strip() or None
        min_interval = float(_cfg("STT_PARTIAL_MIN_INTERVAL") or 1.2)

        allowed, retry_after = realtime_voice.allow_partial(session_id, min_interval)
        if not allowed:
            return jsonify({
                "text": "",
                "lang": lang_hint or "en",
                "partial": True,
                "skipped": True,
                "retry_after": retry_after,
            }), 200

        audio_file = request.files["audio"]
        result, err = _run_stt_on_upload(audio_file, lang_hint)
        if err:
            # Soft-fail partials so UI keeps listening
            return jsonify({"text": "", "lang": lang_hint or "en", "partial": True, "error": True}), 200

        text = (result or {}).get("text") or ""
        detected = (result or {}).get("language") or (lang_hint or "en")
        if text:
            logger.info(f"STT partial: '{text[:60]}' [{detected}]")
        return jsonify({
            "text": text,
            "lang": detected,
            "partial": True,
            "empty": bool((result or {}).get("empty")),
        })

    return app


if __name__ == "__main__":
    app = create_app()

    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    CERT_FILE = os.path.join(BASE_DIR, "ssl_cert.pem")
    KEY_FILE  = os.path.join(BASE_DIR, "ssl_key.pem")

    ssl_ready = _ensure_ssl_cert(CERT_FILE, KEY_FILE)

    if ssl_ready:
        ssl_ctx = (CERT_FILE, KEY_FILE)
        logger.info("🔒 HOLO-GURU → https://0.0.0.0:7000")
        logger.info("👉 Open in browser: https://192.168.0.109:7000")
        logger.info("   First time: click 'Advanced' → 'Proceed to site' to accept cert")
    else:
        ssl_ctx = None
        logger.warning("⚠️  HTTP only — mic/camera will not work on network. Fix: sudo apt install openssl")
        logger.info("🚀 HOLO-GURU → http://0.0.0.0:7000")

    app.run(
        host="0.0.0.0",
        port=7000,
        debug=False,
        threaded=True,
        ssl_context=ssl_ctx,
    )
