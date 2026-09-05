@echo off
setlocal

cd /d "%~dp0.."

if "%PORT%"=="" set "PORT=8765"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 .\battle_review_viewer\serve_viewer.py
  exit /b %errorlevel%
)

where python >nul 2>nul
if %errorlevel%==0 (
  python .\battle_review_viewer\serve_viewer.py
  exit /b %errorlevel%
)

echo Python not found. Install Python or use the py launcher.
pause
exit /b 1
