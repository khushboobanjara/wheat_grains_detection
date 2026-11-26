@echo off
REM One-click launcher: creates a virtual environment on first run, installs packages, starts the dashboard.
cd /d "%~dp0"
if not exist venv (
    python -m venv venv
    call venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)
start "" http://localhost:8000
python server.py
pause
