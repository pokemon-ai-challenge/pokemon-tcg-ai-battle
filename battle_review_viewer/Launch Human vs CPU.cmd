@echo off
setlocal

cd /d "%~dp0.."

set "PORT="

for /f %%p in ('py -3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()" 2^>nul') do set "PORT=%%p"
if not defined PORT (
  for /f %%p in ('python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()" 2^>nul') do set "PORT=%%p"
)
if not defined PORT (
  echo Python not found. Install Python or use the py launcher.
  pause
  exit /b 1
)

set "TARGET_URL=http://127.0.0.1:%PORT%/?mode=live&autostart=1&cpu=self"
start "Battle Review Viewer Server" cmd /k "set PORT=%PORT% && set VIEWER_NO_OPEN=1 && call ""%cd%\battle_review_viewer\run_viewer_server.cmd"""

timeout /t 2 /nobreak >nul
start "" "%TARGET_URL%"

endlocal
