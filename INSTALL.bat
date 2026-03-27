@echo off
chcp 65001 >nul
echo ============================================
echo   CAI DAT VE3 TOOL - TAT CA TRONG 1
echo ============================================
echo.

cd /d "%~dp0"

echo [1/6] Kiem tra Python...
python --version
if errorlevel 1 (
    echo [ERROR] Chua cai Python! Tai tai: https://www.python.org/downloads/
    echo Nho tick "Add Python to PATH" khi cai.
    pause
    exit /b
)

echo.
echo [2/6] Tao thu muc can thiet...
if not exist "D:\AUTO\VISUAL" mkdir "D:\AUTO\VISUAL"
if not exist "D:\AUTO\done" mkdir "D:\AUTO\done"
if not exist "D:\AUTO\ve3-tool-simple\config" mkdir "D:\AUTO\ve3-tool-simple\config"
if not exist "D:\upload" mkdir "D:\upload"
echo Thu muc OK!

echo.
echo [3/6] Cai thu vien co ban...
pip install openpyxl Pillow gspread google-auth flask opencv-python numpy pyyaml

echo.
echo [4/6] Cai Whisper (cho tao SRT tu giong noi)...
pip install openai-whisper whisper-timestamped

echo.
echo [5/6] Cai PyTorch (cho remove background trong thumb)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo [6/6] Kiem tra FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARN] FFmpeg chua cai!
    echo   1. Tai tai: https://www.gyan.dev/ffmpeg/builds/
    echo   2. Giai nen, copy ffmpeg.exe va ffprobe.exe vao C:\Windows\
    echo   Hoac them folder bin vao System PATH.
) else (
    echo FFmpeg OK!
)

echo.
echo ============================================
echo   CAI DAT XONG!
echo ============================================
echo.
echo Con can lam:
echo   1. Copy 3 file config tu may chinh sang D:\AUTO\ve3-tool-simple\config\
echo      - config.json   (cau hinh chinh)
echo      - creds.json    (Google Sheets)
echo      - key.json      (Google TTS)
echo   2. Chay RUN_ve3.bat de khoi dong tool
echo.
pause
