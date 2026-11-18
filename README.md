# liveclip: clip a live stream while it's still live

Most clippers download the whole stream after it ends, then spend hours looking for moments. By then the first clips are already up and have taken the views.

liveclip works **during** the stream:

1. **Streamlink** records the live stream from the second you start it.
2. **A script reads the live chat** (YouTube or Twitch). No login and no API key are needed.
3. **When the chat suddenly explodes**, that moment is a highlight. **FFmpeg** cuts the last ~30 seconds into a vertical 9:16 clip.
4. **Whisper** (it runs on your own computer) adds captions.

All four tools are free and open source. The catch: it only works while the streamer is live.

> **Only clip streamers who allow clipping** (or who run an official clipping program). Clips without permission get copyright strikes and no payout.

---

## Setup (one time, ~10 minutes)

You need **Python 3.10-3.12** and **FFmpeg**.

**Mac**
```bash
brew install python@3.11 ffmpeg
```
**Windows:** install Python 3.11 from python.org (tick "Add to PATH"), and FFmpeg with `winget install ffmpeg`.

Then download this repo and set it up:
```bash
git clone https://github.com/Aadi1659/liveclip.git
cd liveclip
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run it

```bash
python liveclip.py "https://www.youtube.com/watch?v=LIVE_VIDEO_ID"
```
- Leave it running while the stream is live. Clips land in `clips/` as `clip_<date>_<time>.mp4`, with a matching `.srt` caption file.
- **Press Enter** to clip the last ~30 seconds by hand.
- **Ctrl+C** to stop. Clips that are still being made get finished first.
- `clips/clips.csv` logs every clip: when it was made, how fast the chat was and the top chat words.

The first time you run it, Whisper downloads its model (~1.5 GB for `medium`) while the stream records, so start it a few minutes early.

## Useful options

| Option | What it does |
| --- | --- |
| `--pre 25 --post 6` | Seconds kept before/after the chat spike. Chat reacts a few seconds late, so most of the clip comes *before* the spike |
| `--spike 3` | How sudden the spike must be (3 = chat 3x faster than normal). Lower it for more clips, raise it for fewer |
| `--cooldown 60` | Minimum seconds between two auto clips |
| `--captions hinglish` | Hindi/English speech in Roman letters, the way people type it (default). `english` = translated to English, `off` = no captions |
| `--model medium` | Whisper size. `medium` (default) handles Hinglish well, about 10 s per clip. `small` is faster but garbles Hinglish |
| `--layout crop` | Center-crop to 9:16 instead of the blurred-background layout |
| `--max-clips 10 --duration 3600` | Stop after 10 clips or after 1 hour |

## Use it with Claude Code

Open the `liveclip` folder in Claude Code and just say:
> "Clip this stream: https://www.youtube.com/watch?v=..."

Claude reads `CLAUDE.md`, sets everything up and runs it for you. You can also ask it to change things ("make clips 45 seconds", "only clip when chat says W or LOL").

## Honest notes
- A chat spike means *something* happened. It doesn't guarantee a good clip, so watch each clip before you post it.
- Big streams (50K+ viewers) have fast chat all the time. If you get too many clips, raise `--spike` to 4-5.
- Captions are auto-generated and can have mistakes. Fix them in Edits/CapCut, or edit the `.srt` file.
- Clipping programs pay per view and usually have caps. Read each campaign's rules before you post.
