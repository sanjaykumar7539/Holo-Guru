# Presentation note: Tools & Resources deviations

The CMTI problem statement suggests AVA-Kinetics and LibriSpeech as
reference datasets, and Intel AI PC (GPU+NPU) as reference hardware.
HOLO-GURU deviates from both — here's the justification to have ready
for judges.

## Datasets: AVA-Kinetics / LibriSpeech — not used, and why

- **AVA-Kinetics** is a general human-action-recognition dataset (people
  doing everyday activities in video clips). It is not annotated for
  classroom engagement/confusion signals, and is not language- or
  curriculum-specific. HOLO-GURU's emotion detection instead uses a
  FER-trained model (7-class facial expression) purpose-built for the
  actual signal needed — confused/frustrated/engaged — rather than
  general action recognition.
- **LibriSpeech** is English-only, read audiobook speech — it doesn't
  cover Tamil or Tanglish code-switching, which is the core requirement
  for this project's target students. facebook/seamless-m4t-v2-large was
  chosen instead because it's pretrained across ~100 languages including
  Tamil, giving real code-switching support LibriSpeech-trained models
  don't have.
- Both are suggested as *reference* datasets in the problem statement,
  not mandatory ones — the project's actual grounding data is the
  uploaded Samacheer Kalvi / CBSE textbooks themselves (RAG over real
  curriculum content), which is a stronger fit for "answers grounded in
  the syllabus" than a general-purpose benchmark dataset would be.

## Hardware: Intel AI PC (GPU+NPU) — Jetson Orin Nano used instead

- Docker deployment abstracts this: the containerized build runs on any
  CUDA-capable device the compose file targets, so the project isn't
  locked to Jetson specifically — swapping the base image class (Intel
  OpenVINO / oneAPI equivalents) is a deployment-config change, not an
  architecture change.
- Jetson Orin Nano was chosen for this build because it's a purpose-built
  edge-AI board (Tegra GPU, unified memory, low power draw) well suited
  to a classroom device meant to run fully offline, unhooked from a
  desktop/PC form factor — closer to what a standalone classroom
  hologram unit would actually ship as.
