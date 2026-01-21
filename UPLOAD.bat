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
echo [2] Committing...
git commit -m "Update %date% %time%"

echo.
echo [3] Pushing to GitHub (main)...
git push -u origin main

echo.
echo ============================================
echo   DONE!
echo   URL: https://github.com/nguyenvantuong161978-dotcom/master
echo ============================================
pause
