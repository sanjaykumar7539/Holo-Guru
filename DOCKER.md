# HOLO-GURU — Docker deployment (generic, any OS)

This setup is NOT Jetson-specific — it runs on any machine with Docker:
Windows, Linux, Mac, a cloud GPU box, or a plain laptop with no GPU at
all (CPU mode). Point it at whatever hardware you're testing on right
now, and re-target to Jetson later when the board is available again —
that's just a Dockerfile base-image swap, not an app rewrite.

## GPU host (recommended — has an NVIDIA GPU + driver)

```bash
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
docker compose up -d --build
```

## CPU-only host (no GPU / just testing functionality)

Open `docker-compose.yml` and comment out (or delete) the `deploy:`,
`runtime: nvidia`, and `NVIDIA_*` lines, then:

```bash
docker compose up -d --build
```

Everything still works — the LLM/STT/TTS/vision code all falls back to
CPU automatically. It will be noticeably slower (expect several seconds
per answer instead of the sub-2s target), which is fine for checking
that a feature works correctly before testing performance on real
target hardware.

## Open

```
https://localhost:7000
```

(or `https://<machine-lan-ip>:7000` from another device on the same
network). Browser will warn about the self-signed cert — click
Advanced → Proceed. HTTPS is required for mic/camera to work from
another device on the LAN.

## Logs / stop

```bash
docker compose logs -f
docker compose down
```

## First run

Downloads Qwen3-8B, seamless-m4t-v2-large, Qwen3-TTS-0.6B,
indic-parler-tts, and Moondream2 into the `hf-cache` volume — several GB
total, expect this to take a while once. Subsequent restarts reuse the
cached weights.

## Memory note

`mem_limit: 8g` in docker-compose.yml is a safe default for an 8GB
device — raise it if testing on a machine with more RAM available.
Five models (LLM + STT + 2×TTS + vision) contending for RAM is real
pressure even on a normal PC if none have GPU offload; the lazy-loading
pattern in `core/llm_engine.py` / `stt_engine.py` / `tts_engine.py` /
`vision_engine.py` avoids keeping all five resident unless a request
actually needs them, but heavy concurrent use can still spike close to
whatever limit you set.

## Emotion model

`models/emotion_ferplus.onnx` is not bundled. Download it once and place
it at `./models/emotion_ferplus.onnx` on the host before starting the
container — it's picked up via the `./models` volume mount. Without it,
emotion detection runs in face-detection-only mode (always reports
"neutral", never crashes).

## Moving to Jetson later

When the Jetson Orin Nano is available again, swap the base image in
`Dockerfile` from `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` to a
Jetson L4T image (e.g. `dustynv/l4t-pytorch:r36.2.0`) and remove the
`pip install torch` line (L4T images ship Jetson-compatible torch
already) — the rest of the Dockerfile, docker-compose.yml, and all
application code stay the same.

## Image / video generation (FLUX.1-schnell, Wan2.1-T2V-1.3B)

These are the heaviest features in the stack and are GPU-only:
- Both check `torch.cuda.is_available()` at load time and simply skip
  themselves (no crash) if there's no CUDA GPU — the answer still shows
  as text, just without the generated picture/video.
- FLUX.1-schnell needs real free VRAM (comfortably 12GB+, tight even at
  8GB). Wan2.1-T2V-1.3B is similar and also slow (tens of seconds per
  clip), which is why video generation runs as a background job the
  frontend polls (`/api/video-status/<job_id>`) instead of blocking the
  answer.
- Neither is expected to run on the Jetson Orin Nano — demo these on
  whatever GPU machine is available (a gaming laptop with an RTX card,
  a cloud GPU instance, etc.), same as any other CUDA-only feature here.
