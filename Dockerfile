# HOLO-GURU — generic Dockerfile (any OS/host: Windows, Linux, Mac dev
# machines, cloud GPU boxes). Not Jetson/ARM64-specific — works with a
# standard NVIDIA GPU (via nvidia-container-toolkit) or CPU-only.
#
# Build:   docker build -t holo-guru:latest .
# Run:     see docker-compose.yml

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System packages: python3, ffmpeg (audio conversion for STT), openssl
# (HTTPS cert generation — mic/camera require HTTPS on non-localhost),
# espeak-ng (final TTS fallback), git (parler-tts installs from GitHub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-venv \
    ffmpeg \
    openssl \
    espeak-ng \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

COPY requirements.txt .

# Standard PyPI torch/torchaudio (CUDA build) — works on any x86_64/arm64
# machine with a normal NVIDIA GPU + driver, no Jetson-specific wheels.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Models are NOT baked into the image (multi-GB each) — they download to
# this volume-mounted cache on first run and persist across container
# restarts. See docker-compose.yml `hf-cache` volume.
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 7000

CMD ["python3", "main.py"]
