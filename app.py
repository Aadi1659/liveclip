"""liveclip app: paste a video link (or drop a file), mark the facecam and the gameplay, get vertical clips with captions.

Runs on your own laptop: python app.py  ->  http://127.0.0.1:7860
Finished videos have no live chat, so moments are found from loudness peaks (shouting, hype, big reactions).
For live streams use liveclip.py (it clips from chat spikes while the stream is on).
"""
import os, re, shutil, subprocess, sys, tempfile, threading, wave
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "clips")
WORK = os.path.join(tempfile.gettempdir(), "liveclip-app")
os.makedirs(OUT, exist_ok=True); os.makedirs(WORK, exist_ok=True)
W, H = 1080, 1920
STYLES = ["Bold white", "Yellow pop", "White box"]
HINGLISH = ("Nahi yaar, yaha se rasta hi nahi hai. Bhai yeh dekho, agar aap naye ho toh follow kar lo. "
            "Main abhi helicopter dhoondh raha hoon, kya scene hai, chalo chalo. Haan bhai, mast hai.")


def run(*cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


# ---------- 1. get the video + a preview frame ----------
def load(url, file, progress=gr.Progress()):
    shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK)
    src = os.path.join(WORK, "src.mp4")
    if file:
        progress(0.2, desc="Copying your video")
        shutil.copy(file, src)
    elif url and url.strip():
        progress(0.1, desc="Downloading the video (720p)")
        run(sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=720]+ba/b[height<=720]/b", "--merge-output-format", "mp4",
            "-o", src, "--no-playlist", url.strip())
    else:
        raise gr.Error("Paste a link or drop a video file first.")
    progress(0.9, desc="Making the preview")
    d = duration(src)
    prev = os.path.join(WORK, "preview.jpg")
    run("ffmpeg", "-v", "error", "-y", "-ss", f"{d * 0.3:.1f}", "-i", src, "-frames:v", "1", prev)
    st = {"src": src, "dur": d, "boxes": {"face": None, "game": None}, "pts": []}
    return Image.open(prev).convert("RGB"), st, f"Loaded: {int(d // 60)} min {int(d % 60)} s. Now mark the boxes."


# ---------- 2. click two corners for each box ----------
def draw(st):
    im = Image.open(os.path.join(WORK, "preview.jpg")).convert("RGB")
    d = ImageDraw.Draw(im)
    for key, col in (("face", (255, 120, 80)), ("game", (80, 200, 255))):
        b = st["boxes"][key]
        if b:
            d.rectangle(b, outline=col, width=6)
            d.text((b[0] + 10, b[1] + 8), "FACE" if key == "face" else "GAMEPLAY", fill=col)
    for p in st["pts"]:
        d.ellipse((p[0] - 8, p[1] - 8, p[0] + 8, p[1] + 8), fill=(255, 255, 255))
    return im


def click(st, which, evt: gr.SelectData):
    if not st: raise gr.Error("Load a video first.")
    st["pts"].append(tuple(evt.index))
    key = "face" if which.startswith("Face") else "game"
    if len(st["pts"]) == 2:
        (x0, y0), (x1, y1) = st["pts"]
        st["boxes"][key] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        st["pts"] = []
        msg = f"{'Face' if key == 'face' else 'Gameplay'} box set. " + \
              ("Now pick 'Gameplay box' and click its two corners." if key == "face" and not st["boxes"]["game"] else "Ready to clip.")
    else:
        msg = "Now click the opposite corner."
    return draw(st), st, msg


def reset_boxes(st):
    if st: st["boxes"] = {"face": None, "game": None}; st["pts"] = []
    return (draw(st) if st else None), st, "Boxes cleared."


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
def make_clip(st, t0, length, face_pct, style, lang, name):
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
        wav = os.path.join(tmp, "a.wav")
        run("ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.2f}", "-t", f"{length:.2f}", "-i", src, "-vn", "-ac", "1",
            "-ar", "16000", "-c:a", "pcm_s16le", wav)
        for i, (s, e, txt) in enumerate(captions(wav, lang)):
            png = os.path.join(tmp, f"c{i}.png"); h = caption_png(txt, style, png); inputs += ["-i", png]
            vf += f";[{last}][{i + 1}:v]overlay=0:{max(0, cap_y - h // 2)}:enable='between(t,{s:.2f},{e + 0.1:.2f})'[v{i + 1}]"
            last = f"v{i + 1}"
    out = os.path.join(OUT, name + ".mp4")
    run("ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", vf, "-map", f"[{last}]", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out)
    return out


def clip_all(st, n, length, face_pct, style, lang, progress=gr.Progress()):
    if not st: raise gr.Error("Load a video first.")
    progress(0.05, desc="Finding the best moments")
    starts = find_moments(st["src"], int(n), int(length))
    base = "clip_" + __import__("time").strftime("%Y%m%d_%H%M%S")
    outs = []
    for k, t0 in enumerate(starts):
        progress((k + 0.5) / len(starts), desc=f"Making clip {k + 1} of {len(starts)}")
        outs.append(make_clip(st, t0, int(length), face_pct, style, lang, f"{base}_{int(t0)}s"))
    return outs, outs, f"Done: {len(outs)} clips saved in the 'clips' folder."


# ---------- UI ----------
CSS = """
.gradio-container {max-width: 1100px !important}
#hero h1 {font-size: 44px; margin: 0}
#hero p {opacity: .7; margin-top: 4px}
"""
with gr.Blocks(title="liveclip", css=CSS, theme=gr.themes.Soft(primary_hue="orange")) as ui:
    st = gr.State(None)
    gr.Markdown("# Long video. Big moments.\nPaste a link, mark the face and the gameplay, get vertical clips with captions.",
                elem_id="hero")
    with gr.Row():
        url = gr.Textbox(label="1. Video link (YouTube / Twitch)", placeholder="https://www.youtube.com/watch?v=...", scale=4)
        file = gr.File(label="or drop a video file", file_types=["video"], type="filepath", scale=2)
    get = gr.Button("Get preview", variant="primary")
    status = gr.Markdown()
    with gr.Row():
        with gr.Column(scale=3):
            prev = gr.Image(label="2. Click two corners for each box", interactive=False, type="pil")
        with gr.Column(scale=2):
            which = gr.Radio(["Face box", "Gameplay box"], value="Face box", label="Which box are you marking?")
            clear = gr.Button("Clear boxes")
            gr.Markdown("**3. Settings**")
            n = gr.Slider(1, 10, value=4, step=1, label="Number of clips")
            length = gr.Slider(15, 60, value=30, step=5, label="Clip length (seconds)")
            face_pct = gr.Slider(25, 50, value=35, step=1, label="Face size (% of the screen)")
            style = gr.Radio(STYLES, value="Yellow pop", label="Caption style")
            lang = gr.Radio(["Hinglish", "English (translate)", "Off"], value="Hinglish", label="Captions")
            go = gr.Button("Make clips", variant="primary", size="lg")
    gallery = gr.Gallery(label="Your clips", columns=4, height="auto")
    files = gr.Files(label="Download")
    gr.Markdown("Only clip creators who allow it or run a clipping program. Clips without permission get strikes and no payout.")

    get.click(load, [url, file], [prev, st, status])
    prev.select(click, [st, which], [prev, st, status])
    clear.click(reset_boxes, st, [prev, st, status])
    go.click(clip_all, [st, n, length, face_pct, style, lang], [gallery, files, status])

if __name__ == "__main__":
    ui.queue().launch(inbrowser=True, allowed_paths=[OUT])
