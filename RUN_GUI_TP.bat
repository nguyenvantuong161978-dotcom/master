@echo off
cd /d "%~dp0"
REM === Config cho may Dual Xeon 44 cores / 128GB RAM / GTX 1660 Ti ===
set VE3_PARALLEL=3
set VE3_CLIP_WORKERS=12
set VE3_WORKER_REDUCE=2
start "" pythonw GUI.pyw
