# liveclip: instructions for Claude Code

This folder is a live-stream clipper. When the user gives you a live YouTube or Twitch link and asks you to clip it:

1. **Setup (first time only).** Check `ffmpeg -version`. If it's missing, tell the user how to install it (Mac: `brew install ffmpeg`, Windows: `winget install ffmpeg`) and wait. Make a venv with Python 3.10-3.12 (`python3.11 -m venv .venv`; faster-whisper has no wheels for very new Pythons yet), then `.venv/bin/pip install -r requirements.txt` (Windows: `.venv\Scripts\pip`).
2. **Before running**, remind the user in one line: only clip streamers who allow clipping or run a clipping program.
3. **Run it in the background** so it keeps going: `.venv/bin/python liveclip.py "<url>"`, plus any options the user wants (see README.md). It runs until the stream ends or the user stops it.
4. While it runs, tell the user about new clips as they land in `clips/` (`clips/clips.csv` has the chat rate and top chat words for each one).
5. **To change behaviour** ("longer clips", "fewer clips", "only when chat says W"), prefer the flags (`--pre/--post`, `--spike`, `--cooldown`). Only edit `liveclip.py` when no flag covers it.

How it works: `chatread.py` reads the chat (no login). `liveclip.py` records with Streamlink in 6 s segments and watches the chat rate in 8 s windows. When the rate is `--spike` times the stream's normal rate, it cuts `--pre` s before and `--post` s after into a 9:16 MP4 and burns in Whisper captions (drawn with Pillow, so FFmpeg doesn't need libass).


`liveclip_colab.ipynb` is the phone version: a Colab notebook that clones this repo, installs the requirements and runs `liveclip.py` with `--device auto`, showing each clip inline with a download button. Regenerate it rather than hand-editing the JSON if the form changes.
