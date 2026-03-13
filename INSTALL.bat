@echo off
chcp 65001 >nul
echo ============================================
echo   CAI DAT THU VIEN CHO VE3 TOOL
echo ============================================
echo.

cd /d "%~dp0"

echo [1] Kiem tra Python...
python --version
if errorlevel 1 (
    echo [ERROR] Chua cai Python! Tai tai: https://www.python.org/downloads/
    echo Nho tick "Add Python to PATH" khi cai.
    pause
    exit /b
)

echo.
echo [2] Cai thu vien co ban...
pip install openpyxl Pillow gspread google-auth flask opencv-python numpy

echo.
echo [3] Cai PyTorch (cho remove background trong thumb designer)...
echo Dang cai PyTorch CPU version (nho, nhanh)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo [4] Kiem tra FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARN] FFmpeg chua cai! Can cai FFmpeg va them vao PATH.
    echo Tai tai: https://www.gyan.dev/ffmpeg/builds/
    echo Giai nen va them folder bin vao System PATH.
) else (
    echo FFmpeg OK!
)

echo.
echo ============================================
echo   CAI DAT XONG!
echo ============================================
echo.
echo Luu y:
echo - Copy file config/creds.json tu may chinh sang may nay
echo   (de tool co the dien Google Sheet)
echo - Chay RUN_GUI_TP.bat de khoi dong tool
echo.
pause
