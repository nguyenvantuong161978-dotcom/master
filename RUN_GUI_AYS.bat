@echo off
cd /d "%~dp0"
REM === Config cho may RTX 3060 12GB / 32GB RAM ===
set VE3_PARALLEL=2
set VE3_CLIP_WORKERS=8
set VE3_WORKER_REDUCE=3
set VE3_GPU_SLOTS=2
start "" pythonw GUI.pyw
