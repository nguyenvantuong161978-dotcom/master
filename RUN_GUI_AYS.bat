@echo off
cd /d "%~dp0"
REM === Config cho may i5-13400F 10c/16t / 32GB RAM / RTX 3060 12GB ===
set VE3_PARALLEL=3
set VE3_CLIP_WORKERS=8
set VE3_WORKER_REDUCE=4
set VE3_GPU_SLOTS=2
set VE3_BATCH_SIZE=20
set VE3_REENCODE_WORKERS=6
start "" pythonw GUI.pyw
