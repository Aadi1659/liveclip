#!/bin/bash
# Double-click to open liveclip in your browser. The first run sets everything up (a few minutes), after that it opens fast.
cd "$(dirname "$0")"
echo "liveclip is starting..."

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing FFmpeg (one time)..."; brew install ffmpeg
  else
    echo ""
    echo "liveclip needs FFmpeg. Install Homebrew from https://brew.sh, then run:  brew install ffmpeg"
    echo "Then double-click 'Start liveclip' again."
    read -p "Press Enter to close."; exit 1
  fi
fi

PY=""
for p in python3.12 python3.11 python3.10 python3; do
  if command -v $p >/dev/null 2>&1; then PY=$p; break; fi
done
if [ -z "$PY" ]; then
  echo "liveclip needs Python. Install it from https://www.python.org/downloads/ and double-click again."
  read -p "Press Enter to close."; exit 1
fi

if [ ! -d .venv ]; then
  echo "Setting up (one time, a few minutes)..."
  $PY -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt || {
    echo "Setup failed. Send a screenshot of this window to the person who shared liveclip."; read -p "Press Enter to close."; exit 1; }
fi

echo "Opening liveclip in your browser. Keep this window open while you use it."
.venv/bin/python app.py
