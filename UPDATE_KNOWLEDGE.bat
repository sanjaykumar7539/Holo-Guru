@echo off
setlocal
cd /d "%~dp0"
docker compose up -d weaviate
if not exist ".venv\Scripts\python.exe" (
  echo Run START_HOLO_GURU.bat once before updating knowledge.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python sync_knowledge.py
pause
