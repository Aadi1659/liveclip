"""liveclip app: paste a video link (or drop a file), mark the facecam and the gameplay, get vertical clips with captions.

Runs on your own laptop: python app.py  ->  http://127.0.0.1:7860 (the page is static/index.html)
Finished videos have no live chat, so moments are found from loudness peaks (shouting, hype, big reactions).
For live streams use liveclip.py (it clips from chat spikes while the stream is on).
"""
import os, re, shutil, subprocess, sys, tempfile, threading, time, wave, webbrowser
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "clips")
WORK = os.path.join(tempfile.gettempdir(), "liveclip-app")
os.makedirs(OUT, exist_ok=True); os.makedirs(WORK, exist_ok=True)
W, H = 1080, 1920
STYLES = ["Bold white", "Yellow pop", "White box"]
HINGLISH = ("Nahi yaar, yaha se rasta hi nahi hai. Bhai yeh dekho, agar aap naye ho toh follow kar lo. "
            "Main abhi helicopter dhoondh raha hoon, kya scene hai, chalo chalo. Haan bhai, mast hai.")


class Cancelled(Exception):
    pass


CANCEL = threading.Event()   # set by the Cancel button
PROC = {"p": None}           # the ffmpeg / yt-dlp process running right now, so Cancel can stop it


def check_cancel():
    if CANCEL.is_set(): raise Cancelled()


def run(*cmd, on_line=None):
    """Run a command. on_line gets each stdout line (for progress). Cancel kills it."""
    check_cancel()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE if on_line else subprocess.DEVNULL, stderr=subprocess.PIPE,
                         text=True, bufsize=1)
    PROC["p"] = p
    err = []
    t = threading.Thread(target=lambda: err.extend(p.stderr), daemon=True); t.start()
    try:
        if on_line:
            for line in p.stdout: on_line(line.strip())
        p.wait(); t.join(2)
    finally:
        PROC["p"] = None
    check_cancel()
    if p.returncode:
        raise RuntimeError(("".join(err)).strip().splitlines()[-1:] or "command failed")


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


# ---------- 3. find the loudest moments ----------
def find_moments(src, n, length):
    wav = os.path.join(WORK, "all.wav")
    run("ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "8000", "-c:a", "pcm_s16le", wav)
    with wave.open(wav) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    sec = len(x) // 8000
    if sec < length: return [0.0]
    rms = np.sqrt((x[:sec * 8000].reshape(sec, 8000) ** 2).mean(1)) + 1
    db = 20 * np.log10(rms)
    score = np.convolve(db - np.median(db), np.ones(6) / 6, "same")   # sustained loud bursts, not single clicks
    picks = []
    for i in np.argsort(score)[::-1]:
        start = max(0, min(sec - length, i - int(length * 0.6)))      # the peak lands ~60% into the clip
        if all(abs(start - p) >= length for p in picks): picks.append(start)
        if len(picks) >= n: break
    return sorted(float(p) for p in picks)


# ---------- 4. captions ----------
_model, _lock = None, threading.Lock()


def whisper():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


def captions(wav, lang):
    kw = dict(word_timestamps=True, vad_filter=True, condition_on_previous_text=False)
    if lang == "Hinglish": kw.update(language="en", initial_prompt=HINGLISH)
    elif lang == "English (translate)": kw.update(task="translate")
    segs, _ = whisper().transcribe(wav, **kw)
    words = [w for s in segs for w in (s.words or [])]
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= 3 or w.start - cur[-1].end > 0.6): out.append(cur); cur = []
        cur.append(w)
    if cur: out.append(cur)
    return [(c[0].start, c[-1].end, " ".join(x.word.strip() for x in c).strip(" ,.")) for c in out]


def font(size):
    for f in ["/System/Library/Fonts/Supplemental/Arial Black.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "C:/Windows/Fonts/ariblk.ttf", "C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(f): return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


def caption_png(text, style, path):
    up = style != "White box"
    text = text.upper() if up else text
    f = font(76 if style == "Yellow pop" else 64)
    d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d0.textlength(t, font=f) > W - 180 and cur: lines.append(cur); cur = w
        else: cur = t
    if cur: lines.append(cur)
    lh = int(f.size * 1.25)
    im = Image.new("RGBA", (W, lh * len(lines) + 40), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=f); x = (W - tw) / 2; y = 20 + i * lh
        if style == "White box":
            d.rounded_rectangle((x - 24, y - 8, x + tw + 24, y + lh - 4), radius=18, fill=(255, 255, 255, 255))
            d.text((x, y), ln, font=f, fill=(0, 0, 0, 255))
        else:
            col = (255, 225, 77, 255) if style == "Yellow pop" else (255, 255, 255, 255)
            d.text((x, y), ln, font=f, fill=col, stroke_width=8, stroke_fill=(0, 0, 0, 255))
    im.save(path)
    return im.height


# ---------- 5. cut one vertical clip ----------
def make_clip(st, t0, length, face_pct, style, lang, name, on_step=None):
    src, tmp = st["src"], os.path.join(WORK, name)
    os.makedirs(tmp, exist_ok=True)
    face, game = st["boxes"]["face"], st["boxes"]["game"]
    pw, ph = Image.open(os.path.join(WORK, "preview.jpg")).size

    def crop(b):  # preview pixels -> ffmpeg crop on the source (same size as the preview)
        x0, y0, x1, y1 = b
        return f"crop={max(2, x1 - x0) // 2 * 2}:{max(2, y1 - y0) // 2 * 2}:{x0}:{y0}"

    if face and game:
        fh = int(H * face_pct / 100) // 2 * 2
        vf = (f"[0:v]split[a][b];[a]{crop(face)},scale={W}:{fh}:force_original_aspect_ratio=increase,crop={W}:{fh}[f];"
              f"[b]{crop(game)},scale={W}:{H - fh}:force_original_aspect_ratio=increase,crop={W}:{H - fh}[g];[f][g]vstack[v0]")
        cap_y = fh
    else:  # no boxes: whole frame over a blurred copy
        sel = crop(game) + "," if game else ""
        vf = (f"[0:v]{sel}split[a][b];[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=8:2,"
              f"scale={W}:{H}[bg];[b]scale={W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]")
        fh, cap_y = 0, int(H * 0.68)
    inputs, last = ["-ss", f"{t0:.2f}", "-t", f"{length:.2f}", "-i", src], "v0"
    if lang != "Off":
        if on_step: on_step(0.05, "Writing captions")
        wav = os.path.join(tmp, "a.wav")
        run("ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.2f}", "-t", f"{length:.2f}", "-i", src, "-vn", "-ac", "1",
            "-ar", "16000", "-c:a", "pcm_s16le", wav)
        caps = captions(wav, lang)
        check_cancel()
        for i, (s, e, txt) in enumerate(caps):
            png = os.path.join(tmp, f"c{i}.png"); h = caption_png(txt, style, png); inputs += ["-i", png]
            vf += f";[{last}][{i + 1}:v]overlay=0:{max(0, cap_y - h // 2)}:enable='between(t,{s:.2f},{e + 0.1:.2f})'[v{i + 1}]"
            last = f"v{i + 1}"
    out = os.path.join(OUT, name + ".mp4")
    check_cancel()
    if on_step: on_step(0.35, "Rendering")

    def prog(line):  # ffmpeg -progress: out_time_us=12345678
        if line.startswith("out_time_us=") and on_step:
            try: on_step(0.35 + 0.65 * min(1, int(line.split("=")[1]) / 1e6 / length), "Rendering")
            except ValueError: pass
    try:
        run("ffmpeg", "-v", "error", "-y", "-progress", "pipe:1", "-nostats", *inputs, "-filter_complex", vf, "-map", f"[{last}]", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out, on_line=prog)
    except Cancelled:
        if os.path.exists(out): os.remove(out)   # no half-made clip left in the folder
        raise
    return out



# ---------- web API ----------
app = FastAPI()
ST = {"src": None, "dur": 0, "boxes": {"face": None, "game": None}}
JOB = {"busy": False, "kind": "", "progress": 0.0, "msg": "", "clips": [], "error": "", "cancelled": False, "result": None}
DL = re.compile(r"\[download\]\s+([\d.]+)%(?:\s+of\s+~?\s*([\d.]+\w+))?(?:.*?at\s+(\S+))?(?:.*?ETA\s+(\S+))?")


def preview_frame():
    prev = os.path.join(WORK, "preview.jpg")
    run("ffmpeg", "-v", "error", "-y", "-ss", f"{ST['dur'] * 0.3:.1f}", "-i", ST["src"], "-frames:v", "1", prev)
    w, h = Image.open(prev).size
    return {"preview": f"/api/preview.jpg?t={time.time():.0f}", "w": w, "h": h, "dur": round(ST["dur"])}


def fresh_job(kind, msg):
    CANCEL.clear()
    JOB.update(busy=True, kind=kind, progress=0.0, msg=msg, clips=[], error="", cancelled=False, result=None)


def finish(fn):
    try:
        fn()
    except Cancelled:
        JOB.update(cancelled=True, error="", msg="Cancelled")
    except Exception as e:
        JOB.update(error=str(e), msg="Something went wrong")
    finally:
        JOB["busy"] = False


def download(url, src):
    files = {"n": 0}   # yt-dlp fetches video, then sound, then joins them

    def line(t):
        if "Destination:" in t: files["n"] += 1
        elif t.startswith("[Merger]"): JOB.update(progress=0.97, msg="Joining video and sound")
        m = DL.search(t)
        if not m: return
        pct = float(m.group(1)) / 100
        prog = pct * 0.85 if files["n"] <= 1 else 0.85 + pct * 0.12
        what = "Downloading video" if files["n"] <= 1 else "Downloading sound"
        extra = " · ".join(x for x in [m.group(2) and f"{m.group(2)}", m.group(3) and m.group(3) != "Unknown" and m.group(3),
                                         m.group(4) and m.group(4) != "Unknown" and f"{m.group(4)} left"] if x)
        JOB.update(progress=max(JOB["progress"], min(prog, 0.97)), msg=f"{what} {pct:.0%}" + (f" · {extra}" if extra else ""))

    def go():
        try:
            run(sys.executable, "-m", "yt_dlp", "--newline", "-f", "bv*[height<=720]+ba/b[height<=720]/b",
                "--merge-output-format", "mp4", "-o", src, "--no-playlist", url, on_line=line)
        except RuntimeError as e:
            raise RuntimeError(f"Couldn't download that link. Check it opens in your browser. ({e})")
        loaded(src)
    finish(go)


def loaded(src):
    JOB.update(progress=0.99, msg="Making the preview")
    ST.update(src=src, dur=duration(src), boxes={"face": None, "game": None})
    JOB.update(progress=1.0, msg="Loaded", result=preview_frame())


@app.post("/api/load")
async def api_load(url: str = Form(""), file: UploadFile | None = File(None)):
    if JOB["busy"]: raise HTTPException(409, "Still working, wait for it to finish or hit Cancel.")
    shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK)
    src = os.path.join(WORK, "src.mp4")
    if file is not None and file.filename:   # the page shows the upload bar itself
        with open(src, "wb") as f: shutil.copyfileobj(file.file, f)
        fresh_job("load", "Reading your video")
        finish(lambda: loaded(src))
        if JOB["error"]: raise HTTPException(400, JOB["error"])
        return {"ok": True, "now": JOB["result"]}
    if not url.strip(): raise HTTPException(400, "Paste a link or choose a video file first.")
    ST.update(src=None)
    fresh_job("load", "Starting the download")
    threading.Thread(target=download, args=(url.strip(), src), daemon=True).start()
    return {"ok": True}


@app.post("/api/cancel")
def api_cancel():
    if JOB["busy"]:
        CANCEL.set(); JOB["msg"] = "Cancelling…"
        p = PROC["p"]
        if p and p.poll() is None: p.kill()
    return {"ok": True}


@app.get("/api/preview.jpg")
def api_preview():
    return FileResponse(os.path.join(WORK, "preview.jpg"))


def worker(opts):
    def go():
        JOB.update(progress=0.02, msg="Finding the best moments")
        starts = find_moments(ST["src"], int(opts["n"]), int(opts["length"]))
        base = "clip_" + time.strftime("%Y%m%d_%H%M%S")
        n = len(starts)
        for k, t0 in enumerate(starts):
            def step(f, what, k=k):
                JOB.update(progress=0.06 + 0.94 * (k + f) / n, msg=f"Clip {k + 1} of {n} · {what}")
            step(0, "Starting")
            out = make_clip(ST, t0, int(opts["length"]), int(opts["face_pct"]), opts["style"], opts["lang"],
                            f"{base}_{int(t0)}s", on_step=step)
            JOB["clips"].append("/clips/" + os.path.basename(out))
        JOB.update(progress=1.0, msg=f"Done: {n} clips")
    finish(go)


@app.post("/api/clip")
def api_clip(opts: dict):
    if not ST["src"]: raise HTTPException(400, "Load a video first.")
    if JOB["busy"]: raise HTTPException(409, "Already making clips.")
    def box(b):
        return tuple(int(v) for v in b) if b and abs(b[2] - b[0]) > 8 and abs(b[3] - b[1]) > 8 else None
    ST["boxes"] = {"face": box(opts.get("face")), "game": box(opts.get("game"))}
    fresh_job("clip", "Starting")
    threading.Thread(target=worker, args=(opts,), daemon=True).start()
    return {"ok": True}


@app.get("/api/status")
def api_status():
    return JOB


@app.post("/api/open-folder")
def api_open_folder():
    subprocess.Popen(["open" if sys.platform == "darwin" else "explorer" if os.name == "nt" else "xdg-open", OUT])
    return {"ok": True}


app.mount("/clips", StaticFiles(directory=OUT), name="clips")
app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    if "--no-browser" not in sys.argv:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"liveclip is running at http://127.0.0.1:{port}  (keep this window open)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
