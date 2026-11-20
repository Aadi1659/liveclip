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
BG, CARD, LINE, TEXT, MUTED, ACC = "#12061f", "rgba(18, 8, 30, 0.82)", "rgba(255,255,255,0.10)", "#f7f2ff", "#b9a9cc", "#ff6b35"
_t = gr.themes.Base(primary_hue=gr.themes.colors.orange, neutral_hue=gr.themes.colors.zinc,
                    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
                    radius_size=gr.themes.sizes.radius_lg)
_v = dict(
    body_background_fill=BG, body_text_color=TEXT, body_text_color_subdued=MUTED,
    background_fill_primary=CARD, background_fill_secondary=BG,
    block_background_fill=CARD, block_border_color=LINE, block_border_width="1px", block_radius="18px",
    block_shadow="none", block_padding="18px",
    block_label_background_fill="transparent", block_label_text_color=MUTED, block_label_border_width="0px",
    block_label_text_size="13px", block_label_text_weight="600", block_label_padding="0 0 6px 0",
    block_title_text_color=MUTED, block_title_text_size="13px", block_title_text_weight="600",
    block_title_background_fill="transparent", block_title_padding="0 0 8px 0",
    input_background_fill="rgba(10, 4, 18, 0.65)", input_border_color=LINE, input_border_color_focus=ACC, input_radius="12px",
    input_text_size="15px", input_padding="14px",
    border_color_primary=LINE, border_color_accent=ACC,
    color_accent=ACC, color_accent_soft="#2a1a12",
    slider_color=ACC, checkbox_label_background_fill="rgba(10, 4, 18, 0.65)", checkbox_label_background_fill_selected="#2a1a12",
    checkbox_label_border_color=LINE, checkbox_label_border_color_selected=ACC,
    checkbox_label_text_color=TEXT, checkbox_label_text_color_selected=TEXT, checkbox_background_color_selected=ACC,
    button_large_radius="14px", button_medium_radius="12px", button_large_text_size="16px",
    button_primary_background_fill=f"linear-gradient(135deg, {ACC}, #ff3d6e)",
    button_primary_background_fill_hover="linear-gradient(135deg, #ff7d4d, #ff5580)",
    button_primary_text_color="#fff", button_primary_border_color="transparent",
    button_secondary_background_fill="rgba(255,255,255,0.06)", button_secondary_background_fill_hover="rgba(255,255,255,0.12)",
    button_secondary_text_color=TEXT, button_secondary_border_color=LINE,
)
_v = {k: v for k, v in _v.items() if hasattr(_t, k)}
THEME = _t.set(**_v, **{k + "_dark": v for k, v in _v.items() if hasattr(_t, k + "_dark")})

CSS = """
.gradio-container {max-width: 1080px !important; margin: 0 auto !important; padding: 28px 20px 60px !important}
footer {display: none !important}
#hero {text-align: center; padding: 26px 0 30px}
#hero .eyebrow {display: inline-block; font-size: 12px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
  color: #ff8a5c; background: #2a1a12; border: 1px solid #4a2a1a; padding: 6px 12px; border-radius: 999px}
#hero h1 {font-size: 56px; line-height: 1.05; font-weight: 800; letter-spacing: -0.03em; margin: 18px 0 12px;
  background: linear-gradient(180deg, #ffffff, #b9b6ae); -webkit-background-clip: text; background-clip: text; color: transparent}
#hero h1 span {background: linear-gradient(135deg, #ff6b35, #ff3d6e); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent}
#hero p {color: #8b8a93; font-size: 17px; margin: 0}
.step {display: flex; align-items: center; gap: 12px; margin: 26px 0 10px}
.step b {width: 28px; height: 28px; border-radius: 999px; display: grid; place-items: center; font-size: 13px;
  background: linear-gradient(135deg, #ff6b35, #ff3d6e); color: #fff}
.step span {font-size: 18px; font-weight: 700; color: #f2efe9}
.step em {font-style: normal; color: #8b8a93; font-size: 14px; margin-left: 4px}
#status {text-align: center; color: #8b8a93; min-height: 0}
.hint {color: #8b8a93; font-size: 13px; line-height: 1.55; padding: 4px 6px}
#status p {margin: 6px 0}
#prev img {border-radius: 12px; cursor: crosshair}
#note {text-align: center; color: #5f5e66; font-size: 13px; margin-top: 28px}
/* GTA 6 / Vice City sunset */
body, gradio-app, .gradio-container, .main, .wrap {background: transparent !important}
html {background: #12061f}
body::before {content: ""; position: fixed; inset: 0; z-index: -2;
  background:
    radial-gradient(1200px 600px at 85% 8%, rgba(255, 61, 110, 0.35), transparent 60%),
    radial-gradient(900px 500px at 10% 20%, rgba(124, 58, 237, 0.35), transparent 60%),
    linear-gradient(180deg, #12061f 0%, #2a0b45 30%, #6b1557 58%, #c2185b 78%, #ff6b35 100%);}
body::after {content: ""; position: fixed; left: 0; right: 0; bottom: 0; height: 38vh; z-index: -1; pointer-events: none;
  background: repeating-linear-gradient(0deg, rgba(255,255,255,0.05) 0 1px, transparent 1px 34px),
              repeating-linear-gradient(90deg, rgba(255,255,255,0.05) 0 1px, transparent 1px 34px);
  mask-image: linear-gradient(0deg, black, transparent); -webkit-mask-image: linear-gradient(0deg, black, transparent);
  transform: perspective(400px) rotateX(55deg); transform-origin: bottom}
.block, .form, .gr-group, .tabs {backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px)}
#hero h1 {text-shadow: 0 0 40px rgba(255, 61, 110, 0.25)}
#hero .eyebrow {background: rgba(255, 107, 53, 0.15); border-color: rgba(255, 107, 53, 0.4)}
#hero p, .step em, #status, .hint {color: #d3c4e6 !important}
#note {color: #fff3ea; opacity: .85}
.tab-wrapper, [role=tablist], .tab-container {background: rgba(10, 4, 18, 0.85) !important; border-color: rgba(255,255,255,0.08) !important}
[role=tab] {color: #d3c4e6 !important; font-weight: 600}
.tab-nav button.selected, button[role=tab][aria-selected=true] {color: #ff8a5c !important; border-color: #ff6b35 !important}
"""


def step(n, title, hint=""):
    gr.HTML(f'<div class="step"><b>{n}</b><span>{title}</span><em>{hint}</em></div>')


with gr.Blocks(title="liveclip") as ui:
    st = gr.State(None)
    gr.HTML('<div id="hero"><div class="eyebrow">liveclip · free &amp; open source</div>'
            '<h1>Long video.<br><span>Big moments.</span></h1>'
            '<p>Paste a stream, mark the face and the gameplay, get vertical clips with captions.</p></div>')

    step(1, "Add your video", "a YouTube / Twitch link, or a file from your laptop")
    with gr.Group():
        with gr.Tabs():
            with gr.Tab("Paste a link"):
                url = gr.Textbox(show_label=False, placeholder="https://www.youtube.com/watch?v=...", container=False)
            with gr.Tab("Upload a file"):
                file = gr.File(show_label=False, file_types=["video"], type="filepath", height=140)
    get = gr.Button("Get preview  →", variant="primary", size="lg")
    status = gr.Markdown(elem_id="status")

    step(2, "Mark the boxes", "pick a box, then click two opposite corners on the frame")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            prev = gr.Image(show_label=False, interactive=False, type="pil", elem_id="prev", height=420)
        with gr.Column(scale=2):
            which = gr.Radio(["Face box", "Gameplay box"], value="Face box", label="Now marking")
            clear = gr.Button("Clear boxes", variant="secondary")
            gr.HTML('<div class="hint"><b style="color:#ff8a5c">Face box</b> = the streamer\'s camera.<br>'
                    '<b style="color:#5cc8ff">Gameplay box</b> = the game.<br>Skip the face box for a gameplay-only clip.</div>')

    step(3, "Choose your style")
    with gr.Group():
        with gr.Row():
            n = gr.Slider(1, 10, value=4, step=1, label="Number of clips")
            length = gr.Slider(15, 60, value=30, step=5, label="Clip length (sec)")
            face_pct = gr.Slider(25, 50, value=35, step=1, label="Face size (%)")
        with gr.Row():
            style = gr.Radio(STYLES, value="Yellow pop", label="Caption style")
            lang = gr.Radio(["Hinglish", "English (translate)", "Off"], value="Hinglish", label="Captions")
    go = gr.Button("Make clips  ✦", variant="primary", size="lg")

    step(4, "Your clips", "saved in the clips folder too")
    gallery = gr.Gallery(show_label=False, columns=4, height="auto", object_fit="contain")
    files = gr.Files(label="Download")
    gr.Markdown("Only clip creators who allow it or run a clipping program. Clips without permission get strikes and no payout.",
                elem_id="note")

    get.click(load, [url, file], [prev, st, status])
    prev.select(click, [st, which], [prev, st, status])
    clear.click(reset_boxes, st, [prev, st, status])
    go.click(clip_all, [st, n, length, face_pct, style, lang], [gallery, files, status])

if __name__ == "__main__":
    ui.queue().launch(inbrowser=True, allowed_paths=[OUT], theme=THEME, css=CSS,
                      js="() => { document.body.classList.add('dark'); }")
