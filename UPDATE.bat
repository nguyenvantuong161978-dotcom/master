@echo off
chcp 65001 >nul
echo ============================================
echo   UPDATE FROM GITHUB (No Git Required)
echo ============================================
echo.

cd /d "%~dp0"
python UPDATE.py

pause
