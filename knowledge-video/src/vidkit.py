"""Shared toolkit for the Knowledge Video series: drawing helpers, formulas, subtitles,
narration and assembly into an MP4. Each episode defines SCENES and renderers and calls render()."""

import io
import json
import math
import os
import subprocess
import sys
import wave

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from narrate import Narrator  # noqa: E402

W, H, FPS = 1280, 720, 24
BG = (7, 11, 20)
CYAN = (34, 211, 238)
VIOLET = (167, 139, 250)
TEXT = (226, 232, 240)
MUTED = (148, 163, 184)
GREEN = (52, 211, 153)
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def font(size, mono=False):
    path = MONO if mono and os.path.exists(MONO) else FONT
    return ImageFont.truetype(path, size)


F_TITLE, F_H, F_BODY, F_SMALL, F_SUB = font(64), font(40), font(30), font(24), font(30)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def background():
    img = Image.new("RGB", (W, H), BG)
    px = np.asarray(img).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    glow = np.exp(-(((xx - W * .5) / 520) ** 2 + ((yy + 80) / 360) ** 2))
    px += glow[..., None] * np.array([10, 40, 50])
    grid = ((xx % 48 == 0) | (yy % 48 == 0)) * 7.0
    px += grid[..., None]
    return Image.fromarray(np.clip(px, 0, 255).astype(np.uint8))


BG_IMG = background()


def text_c(d, xy, s, f, fill=TEXT, anchor="mm"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def header(d, idx, title):
    d.text((60, 52), f"// {idx:02d}", font=font(22, mono=True), fill=CYAN, anchor="lm")
    d.text((140, 52), title, font=F_H, fill=TEXT, anchor="lm")
    d.line([(60, 88), (W - 60, 88)], fill=(30, 41, 59), width=1)


def to_rgb(arr, size):
    a = np.clip((arr + 1) * 127.5, 0, 255).astype(np.uint8)
    img = Image.fromarray(a, "L").resize((size, size), Image.NEAREST)
    return img.convert("RGB")


def framed(canvas, img, xy, label=None, color=CYAN):
    x, y = xy
    canvas.paste(img, (x, y))
    d = ImageDraw.Draw(canvas)
    d.rectangle([x - 2, y - 2, x + img.width + 1, y + img.height + 1], outline=color, width=2)
    if label:
        d.text((x + img.width / 2, y + img.height + 22), label, font=F_SMALL, fill=MUTED, anchor="mm")


_formula_cache = {}


def formula(tex, size=34, color="#e2e8f0"):
    key = (tex, size, color)
    if key not in _formula_cache:
        fig = plt.figure(figsize=(0.01, 0.01), dpi=100)
        fig.text(0, 0, tex, fontsize=size, color=color)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=100, transparent=True, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        buf.seek(0)
        _formula_cache[key] = Image.open(buf).convert("RGBA")
    return _formula_cache[key]


def paste_center(canvas, img, cx, cy):
    canvas.paste(img, (int(cx - img.width / 2), int(cy - img.height / 2)), img)


def arrow(d, a, b, color, width=4):
    d.line([a, b], fill=color, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (-1, 1):
        p = (b[0] - 16 * math.cos(ang + s * .45), b[1] - 16 * math.sin(ang + s * .45))
        d.line([b, p], fill=color, width=width)


def points(d, pts, box, color, r=2.2, extent=3.2):
    x0, y0, x1, y1 = box
    cx, cy, sc = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / (2 * extent)
    for x, y in pts:
        px, py = cx + x * sc, cy - y * sc
        if x0 < px < x1 and y0 < py < y1:
            d.ellipse([px - r, py - r, px + r, py + r], fill=color)


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def wrap(s, f, width):
    lines, cur = [], ""
    for ch in s:
        if f.getlength(cur + ch) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def subtitle(canvas, s):
    d = ImageDraw.Draw(canvas, "RGBA")
    lines = wrap(s, F_SUB, W - 220)
    h = 46 * len(lines) + 20
    top = H - 30 - h
    d.rounded_rectangle([90, top, W - 90, H - 30], radius=14, fill=(2, 6, 16, 205))
    for i, ln in enumerate(lines):
        d.text((W / 2, top + 10 + 23 + 46 * i), ln, font=F_SUB, fill=(240, 244, 248), anchor="mm")



def render(scenes, renderers, out_dir, name):
    """Narrate scenes, draw frames and mux <out_dir>/<name>.mp4.

    renderers[scene](canvas, u, seg, su): u = scene progress, seg = sentence index, su = progress in sentence.

    Env: TTS_MODELS (model dir), VOICE (default "matcha"), ASR_MODEL (optional, verifies subtitle timing).
    """
    narrator = Narrator(os.environ["TTS_MODELS"], os.environ.get("VOICE", "matcha"), asr=os.environ.get("ASR_MODEL"))
    sr = narrator.sr
    os.makedirs(out_dir, exist_ok=True)
    lead, tail = 0.5, 0.9
    audio, timeline, starts = [], [], []  # timeline: (scene, seg, start, end, text, scene_start, scene_end)
    t = 0.0
    for scene, sents in scenes:
        scene_start = t
        starts.append({"name": scene, "start": round(scene_start, 2)})
        audio.append(np.zeros(int(lead * sr), np.float32)); t += lead
        x, spans = narrator.scene(sents)
        segs = [(i, t + s0, t + s1, s) for i, (s, (s0, s1)) in enumerate(zip(sents, spans))]
        audio.append(x); t += len(x) / sr
        audio.append(np.zeros(int(tail * sr), np.float32)); t += tail
        timeline += [(scene, *sg, scene_start, t) for sg in segs]
        print(f"scene {scene}: start {scene_start:.1f}s, {t - scene_start:.1f}s", flush=True)
    total = t
    with open(os.path.join(out_dir, "timeline.json"), "w") as f:
        json.dump({"scenes": starts,
                   "subtitles": [{"start": round(i[2], 2), "end": round(i[3], 2), "text": i[4]} for i in timeline]},
                  f, ensure_ascii=False, indent=1)
    pcm = np.concatenate(audio)
    pcm = pcm / max(1e-6, np.abs(pcm).max()) * 0.9
    wav_path = os.path.join(out_dir, "narration.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())
    with open(os.path.join(out_dir, "narration.txt"), "w") as f:
        for _, sents in scenes:
            f.write("".join(sents) + "\n")

    out = os.path.join(out_dir, name + ".mp4")
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", wav_path,
        "-c:v", "libx264", "-preset", "medium", "-crf", "26", "-pix_fmt", "yuv420p", "-tune", "animation",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-shortest", out,
    ], stdin=subprocess.PIPE)
    nframes = int(math.ceil(total * FPS))
    for f_i in range(nframes):
        ts = f_i / FPS
        cur = None
        for item in timeline:
            if item[5] <= ts < item[6] and (cur is None or item[2] <= ts):
                cur = item
        cur = cur or timeline[-1]
        scene, seg, s0, s1, text, sc0, sc1 = cur
        # progress within the current sentence (until the next sentence starts / the scene ends)
        nxt = next((i[2] for i in timeline if i[0] == scene and i[1] == seg + 1), sc1)
        canvas = BG_IMG.copy()
        renderers[scene](canvas, (ts - sc0) / max(1e-6, sc1 - sc0), seg, min(1.0, max(0.0, (ts - s0) / max(1e-6, nxt - s0))))
        if s0 - 0.2 <= ts <= s1 + 0.3:
            subtitle(canvas, text)
        ff.stdin.write(canvas.tobytes())
        if f_i % 960 == 0:
            print(f"frame {f_i}/{nframes}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("done", out, f"{total:.1f}s")
    return out
