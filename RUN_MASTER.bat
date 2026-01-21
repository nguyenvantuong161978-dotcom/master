@echo off
echo ============================================================
echo   VE3 TOOL - MASTER SERVER
echo ============================================================
echo   Chay 2 script nen:
echo   1. run_srt.py - Voice to SRT (chi tao SRT)
echo   2. run_edit.py - Merge video MP4
echo ============================================================
echo.

REM Kiem tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python khong tim thay!
    pause
    exit /b 1
)

REM Chay run_srt.py nen
echo [MASTER] Starting run_srt.py in background...
start /min cmd /c "python run_srt.py & pause"

REM Doi 2 giay
timeout /t 2 /nobreak >nul

REM Chay run_edit.py nen
echo [MASTER] Starting run_edit.py in background...
start /min cmd /c "python run_edit.py & pause"

echo.
echo ============================================================
echo   MASTER SERVER STARTED!
echo   - run_srt.py: Dang chay nen (minimized)
echo   - run_edit.py: Dang chay nen (minimized)
echo ============================================================
echo.
echo Nhan phim bat ky de dong cua so nay (2 script van chay nen)
pause
