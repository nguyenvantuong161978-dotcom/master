@echo off
chcp 65001 >nul
echo ============================================
echo   UPLOAD TO GITHUB (main branch)
echo ============================================
echo.

cd /d "%~dp0"

echo [1] Adding all files...
git add -A

echo.
set /p msg="Commit message (Enter = auto): "
if "%msg%"=="" set msg=Update %date% %time%

echo.
echo [2] Committing: %msg%
git commit -m "%msg%"

echo.
echo [3] Pushing to GitHub (main)...
git push -u origin main

echo.
echo ============================================
echo   DONE! Check: https://github.com/nguyenvantuong161978-dotcom/master
echo ============================================
pause
