#!/usr/bin/env bash
# One-click launcher: creates a virtual environment on first run, installs packages, starts the dashboard.
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source venv/bin/activate
fi
python server.py
