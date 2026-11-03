@echo off
setlocal
cd /d "%~dp0"
docker compose up -d weaviate
if errorlevel 1 goto :error
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 goto :error
)
call ".venv\Scripts\activate.bat"
set "HF_HOME=%CD%\.venv\hf-cache"
set "HF_HUB_CACHE=%HF_HOME%\hub"
for /f %%i in ('python -c "import os; print(max(1,min(8,(os.cpu_count() or 2)//2)))"') do set EDGE_CPU_THREADS=%%i
if not defined OMP_NUM_THREADS set OMP_NUM_THREADS=%EDGE_CPU_THREADS%
if not defined MKL_NUM_THREADS set MKL_NUM_THREADS=%EDGE_CPU_THREADS%
if not defined OPENBLAS_NUM_THREADS set OPENBLAS_NUM_THREADS=%EDGE_CPU_THREADS%
if not defined NUMEXPR_NUM_THREADS set NUMEXPR_NUM_THREADS=%EDGE_CPU_THREADS%
echo Edge CPU threads: %EDGE_CPU_THREADS%
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
if not exist "knowledge" mkdir knowledge
if not exist "data" mkdir data
python sync_knowledge.py
python main.py
goto :eof
:error
echo HOLO-GURU startup failed. Review the error above.
pause
exit /b 1
