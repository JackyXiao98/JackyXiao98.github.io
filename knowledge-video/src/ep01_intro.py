"""Knowledge Video 01 — 扩散模型介绍：三个例子看懂 Diffusion.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python ep01_intro.py <out_dir>

All diffusion examples are computed for real: forward noising uses the closed form
x_t = sqrt(abar) x0 + sqrt(1-abar) eps, and reverse sampling uses DDIM with the exact
optimal denoiser E[x0 | x_t] for the toy data distributions.
"""

import io
import math
import os
import subprocess
import sys
import wave


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 24
SR = 44100
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
# Narration (one list of sentences per scene)
# ---------------------------------------------------------------------------
SCENES = [
    ("title", [
        "大家好。",
        "这一期我们用三个小例子，直观地理解扩散模型，是怎样从一团随机噪声中，生成一张图片的。",
    ]),
    ("idea", [
        "扩散模型的核心思想只有两步。",
        "第一步叫前向过程：把一张真实图片一点一点加上噪声，直到它变成纯粹的随机噪声。",
        "第二步叫反向过程：训练一个神经网络，学会把噪声一步一步去掉。",
        "学会以后，我们从纯噪声出发，反复去噪，就能得到一张全新的图片。",
    ]),
    ("forward_img", [
        "先看第一个例子：给一张笑脸图片加噪声。",
        "时间从零走到一，图片里保留的原始信号越来越少，噪声越来越多。",
        "到最后，笑脸完全看不出来了，只剩下标准的高斯噪声。",
        "前向过程有一个很方便的公式：任意时刻的带噪图片，等于原图乘以一个系数，再加上噪声乘以另一个系数。",
        "所以我们可以一步直接算出任意时刻的加噪结果，不需要一步一步地模拟。",
    ]),
    ("training", [
        "那么神经网络到底学什么呢？",
        "训练时，我们随机取一张图片，随机选一个时刻，加上一份已知的噪声。",
        "然后让网络看着带噪的图片，去预测我们加进去的那份噪声。",
        "损失函数就是预测噪声和真实噪声之间的均方误差。",
        "任务很简单，但只要预测得足够准，网络就学会了数据的分布。",
    ]),
    ("forward_2d", [
        "第二个例子在二维平面上。",
        "每个点代表一个数据样本，它们排成了一个爱心的形状。",
        "我们对所有点同时做前向加噪，爱心逐渐散开，最后变成一团圆形的高斯分布。",
    ]),
    ("reverse_2d", [
        "现在反过来，从一团随机的点出发。",
        "每一步，去噪器都会估计每个点应该往哪里移动。",
        "可以看到，点先整体收缩，再慢慢找到爱心的轮廓，最后重新排成了爱心。这就是生成。",
        "这里用的是可以精确计算的最优去噪器，在真实的模型里，这个去噪器由神经网络来近似。",
    ]),
    ("reverse_img", [
        "第三个例子回到图片。",
        "我们的小数据集里有四张简单的图：笑脸、星星、爱心和圆环。",
        "从四份不同的随机噪声出发，反复去噪。",
        "注意观察：一开始先出现模糊的整体轮廓，然后细节才慢慢变清晰。",
        "这种先粗后细的生成顺序，正是扩散模型的一个典型特点。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，前向过程逐步加噪，可以用一个公式直接算出任意时刻的结果。",
        "第二，训练目标是预测噪声，用简单的均方误差。",
        "第三，生成时从纯噪声出发，一步一步去噪。",
        "今天的例子就到这里，感谢大家的观看。",
    ]),
]

# ---------------------------------------------------------------------------
# Diffusion helpers
# ---------------------------------------------------------------------------


def abar(t):
    """Cosine noise schedule, t in [0, 1]."""
    s = 0.008
    f = lambda u: np.cos((u + s) / (1 + s) * np.pi / 2) ** 2
    return np.clip(f(t) / f(0.0), 1e-5, 1.0)


def draw_shape(kind, n=32):
    img = Image.new("L", (n * 4, n * 4), 0)
    d = ImageDraw.Draw(img)
    S = n * 4
    if kind == "smile":
        d.ellipse([S * .1, S * .1, S * .9, S * .9], outline=255, width=S // 14)
        d.ellipse([S * .3, S * .32, S * .4, S * .45], fill=255)
        d.ellipse([S * .6, S * .32, S * .7, S * .45], fill=255)
        d.arc([S * .28, S * .35, S * .72, S * .75], 20, 160, fill=255, width=S // 14)
    elif kind == "star":
        pts = []
        for i in range(10):
            r = S * (.42 if i % 2 == 0 else .18)
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((S / 2 + r * math.cos(a), S / 2 + r * math.sin(a)))
        d.polygon(pts, fill=255)
    elif kind == "heart":
        ts = np.linspace(0, 2 * np.pi, 200)
        x = 16 * np.sin(ts) ** 3
        y = -(13 * np.cos(ts) - 5 * np.cos(2 * ts) - 2 * np.cos(3 * ts) - np.cos(4 * ts))
        pts = list(zip(S / 2 + x * S / 40, S / 2 + y * S / 40))
        d.polygon(pts, fill=255)
    elif kind == "ring":
        d.ellipse([S * .12, S * .12, S * .88, S * .88], outline=255, width=S // 9)
        d.ellipse([S * .4, S * .4, S * .6, S * .6], fill=255)
    img = img.resize((n, n), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 127.5 - 1.0


def heart_points(n):
    ts = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(ts) ** 3
    y = 13 * np.cos(ts) - 5 * np.cos(2 * ts) - 2 * np.cos(3 * ts) - np.cos(4 * ts)
    return np.stack([x, y], 1) / 9.0


def ddim_gmm(xT, mus, sigma0, steps):
    """DDIM sampling with the exact denoiser of an equal-weight GMM. Returns trajectory."""
    ts = np.linspace(1.0, 0.0, steps + 1)
    x = xT.copy()
    traj = [x.copy()]
    for i in range(steps):
        a, a2 = abar(ts[i]), abar(ts[i + 1])
        var = a * sigma0 ** 2 + (1 - a)
        diff = x[:, None, :] - np.sqrt(a) * mus[None]          # (N, K, D)
        logw = -(diff ** 2).sum(-1) / (2 * var)
        logw -= logw.max(1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(1, keepdims=True)
        gain = np.sqrt(a) * sigma0 ** 2 / var
        x0 = (w[:, :, None] * (mus[None] + gain * diff)).sum(1)
        eps = (x - np.sqrt(a) * x0) / np.sqrt(1 - a)
        x = np.sqrt(a2) * x0 + np.sqrt(1 - a2) * eps
        traj.append(x.copy())
    return np.array(traj)


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
    d.text((60, 52), f"// 0{idx}", font=font(22, mono=True), fill=CYAN, anchor="lm")
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


# ---------------------------------------------------------------------------
# Data for the examples
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
SMILE64 = draw_shape("smile", 64)
EPS64 = rng.standard_normal(SMILE64.shape).astype(np.float32)

HEART = heart_points(240)
N2D = 1400
X0_2D = HEART[rng.integers(0, len(HEART), N2D)] + 0.06 * rng.standard_normal((N2D, 2))
EPS_2D = rng.standard_normal((N2D, 2))
REV_STEPS = 120
TRAJ_2D = ddim_gmm(rng.standard_normal((N2D, 2)), HEART, 0.06, REV_STEPS)

DATASET = np.stack([draw_shape(k, 32) for k in ("smile", "star", "heart", "ring")]).reshape(4, -1)


def pick_seeds():
    """Find 4 noise seeds whose samples land on 4 different training images."""
    found, s = {}, 0
    while len(found) < 4 and s < 500:
        x = np.random.default_rng(1000 + s).standard_normal((1, DATASET.shape[1]))
        traj = ddim_gmm(x, DATASET, 0.0, 60)
        k = int(np.argmin(((DATASET - traj[-1]) ** 2).sum(1)))
        found.setdefault(k, traj)
        s += 1
    return [found[k] for k in sorted(found)]


TRAJ_IMG = pick_seeds()

# ---------------------------------------------------------------------------
# Scene renderers: f(canvas, u in [0,1], seg_index) -> None
# ---------------------------------------------------------------------------


def sc_title(c, u, seg):
    d = ImageDraw.Draw(c)
    t = 1 - ease(u * 1.3)
    noisy = np.sqrt(abar(t)) * SMILE64 + np.sqrt(1 - abar(t)) * EPS64
    framed(c, to_rgb(noisy, 220), (W // 2 - 110, 110))
    text_c(d, (W / 2, 420), "扩散模型介绍", F_TITLE, CYAN)
    text_c(d, (W / 2, 485), "从噪声到图像 · 三个直观例子", F_BODY, TEXT)
    text_c(d, (W / 2, 535), "Diffusion Models · 101", font(24, mono=True), MUTED)


def sc_idea(c, u, seg):
    d = ImageDraw.Draw(c)
    header(d, 1, "核心思想：加噪与去噪")
    ts = [0, .35, .65, 1.0]
    xs = [170, 470, 770, 1070]
    for x, t in zip(xs, ts):
        img = np.sqrt(abar(t)) * SMILE64 + np.sqrt(1 - abar(t)) * EPS64
        framed(c, to_rgb(img, 170), (x - 85, 150), f"t = {t:.2f}")
    show_fwd, show_rev = seg >= 1, seg >= 2
    if show_fwd:
        arrow(d, (120, 390), (1150, 390), CYAN)
        text_c(d, (W / 2, 420), "前向过程：逐步加噪（固定的，不需要学习）", F_BODY, CYAN)
    if show_rev:
        arrow(d, (1150, 470), (120, 470), VIOLET)
        text_c(d, (W / 2, 500), "反向过程：逐步去噪（由神经网络学习）", F_BODY, VIOLET)


def sc_forward_img(c, u, seg):
    d = ImageDraw.Draw(c)
    header(d, 2, "例子一：给笑脸加噪声")
    t = ease(min(u / 0.55, 1.0))
    a = abar(t)
    img = np.sqrt(a) * SMILE64 + np.sqrt(1 - a) * EPS64
    framed(c, to_rgb(img, 300), (110, 130), None)
    text_c(d, (260, 460), f"t = {t:.2f}", font(30, mono=True), CYAN)
    # signal / noise bars
    bx = 520
    d.text((bx, 170), "信号比例", font=F_SMALL, fill=MUTED)
    c.paste(formula(r"$\sqrt{\bar\alpha_t}$", 20, "#94a3b8"), (bx + 110, 160), formula(r"$\sqrt{\bar\alpha_t}$", 20, "#94a3b8"))
    d.rounded_rectangle([bx, 205, bx + 560, 233], 8, outline=(51, 65, 85))
    d.rounded_rectangle([bx, 205, bx + max(8, int(560 * np.sqrt(a))), 233], 8, fill=CYAN)
    d.text((bx, 260), "噪声比例", font=F_SMALL, fill=MUTED)
    c.paste(formula(r"$\sqrt{1-\bar\alpha_t}$", 20, "#94a3b8"), (bx + 110, 250), formula(r"$\sqrt{1-\bar\alpha_t}$", 20, "#94a3b8"))
    d.rounded_rectangle([bx, 295, bx + 560, 323], 8, outline=(51, 65, 85))
    d.rounded_rectangle([bx, 295, bx + max(8, int(560 * np.sqrt(1 - a))), 323], 8, fill=VIOLET)
    if seg >= 3:
        paste_center(c, formula(r"$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)$", 30), 800, 400)
        text_c(d, (800, 460), "一步算出任意时刻的加噪结果", F_SMALL, GREEN)


def sc_training(c, u, seg):
    d = ImageDraw.Draw(c)
    header(d, 3, "训练：让网络预测噪声")
    t = 0.45
    a = abar(t)
    noisy = np.sqrt(a) * SMILE64 + np.sqrt(1 - a) * EPS64
    framed(c, to_rgb(SMILE64, 130), (80, 130), "原图 x0")
    if seg >= 1:
        framed(c, to_rgb(EPS64 * .5, 130), (80, 330), "噪声 ε", VIOLET)
        arrow(d, (230, 330), (330, 330), MUTED, 3)
        framed(c, to_rgb(noisy, 150), (345, 255), "带噪图 x_t")
    if seg >= 2:
        arrow(d, (510, 330), (590, 330), MUTED, 3)
        d.rounded_rectangle([600, 270, 790, 390], 16, outline=CYAN, width=3)
        text_c(d, (695, 315), "神经网络", F_BODY, CYAN)
        paste_center(c, formula(r"$\epsilon_\theta(x_t,t)$", 22, "#94a3b8"), 695, 355)
        arrow(d, (800, 330), (880, 330), MUTED, 3)
        pred = EPS64 * .5 + 0.25 * np.random.default_rng(3).standard_normal(EPS64.shape)
        framed(c, to_rgb(pred, 150), (895, 255), "预测的噪声", VIOLET)
    if seg >= 3:
        paste_center(c, formula(r"$\mathcal{L}=\mathbb{E}_{x_0,t,\epsilon}\,\|\epsilon-\epsilon_\theta(x_t,t)\|^2$", 30), W / 2, 500)


def sc_forward_2d(c, u, seg):
    d = ImageDraw.Draw(c)
    header(d, 4, "例子二：二维数据加噪")
    t = ease(max(0.0, (u - 0.3) / 0.6))
    a = abar(t)
    pts = np.sqrt(a) * X0_2D + np.sqrt(1 - a) * EPS_2D
    box = (420, 110, 860, 550)
    d.rectangle(box, outline=(30, 41, 59))
    points(d, pts, box, CYAN)
    text_c(d, (1060, 300), f"t = {t:.2f}", font(34, mono=True), CYAN)
    text_c(d, (1060, 350), "数据 → 高斯", F_SMALL, MUTED)


def sc_reverse_2d(c, u, seg):
    d = ImageDraw.Draw(c)
    header(d, 5, "例子二：从噪声生成爱心")
    k = int(ease(min(max((u - 0.12) / 0.62, 0), 1)) * REV_STEPS)
    box = (420, 110, 860, 550)
    d.rectangle(box, outline=(30, 41, 59))
    # trails for a few particles
    x0, y0, x1, y1 = box
    cx, cy, sc = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 6.4
    for j in range(0, N2D, 70):
        tr = TRAJ_2D[: k + 1, j]
        if len(tr) > 1:
            d.line([(cx + p[0] * sc, cy - p[1] * sc) for p in tr], fill=(90, 70, 160), width=1)
    points(d, TRAJ_2D[k], box, VIOLET)
    t = 1 - k / REV_STEPS
    text_c(d, (1060, 300), f"t = {t:.2f}", font(34, mono=True), VIOLET)
    text_c(d, (1060, 350), f"去噪步数 {k}/{REV_STEPS}", F_SMALL, MUTED)


def sc_reverse_img(c, u, seg):
    d = ImageDraw.Draw(c)
    header(d, 6, "例子三：从噪声生成图片")
    if seg <= 1:
        for i, img in enumerate(DATASET):
            framed(c, to_rgb(img.reshape(32, 32), 200), (110 + i * 280, 200), ["笑脸", "星星", "爱心", "圆环"][i])
        text_c(d, (W / 2, 140), "训练集（4 张图）", F_BODY, MUTED)
        return
    steps = TRAJ_IMG[0].shape[0] - 1
    k = int(ease(min(max((u - 0.3) / 0.55, 0), 1)) * steps)
    for i, tr in enumerate(TRAJ_IMG):
        framed(c, to_rgb(tr[k][0].reshape(32, 32), 200), (110 + i * 280, 200), f"样本 {i + 1}", VIOLET)
    text_c(d, (W / 2, 140), f"去噪步数 {k}/{steps}", F_BODY, VIOLET)


def sc_summary(c, u, seg):
    d = ImageDraw.Draw(c)
    header(d, 7, "总结")
    items = [
        ("前向过程", "逐步加噪，闭式公式直接算出任意时刻"),
        ("训练目标", "预测加入的噪声，均方误差损失"),
        ("生成过程", "从纯噪声出发，一步步去噪"),
    ]
    for i, (k, v) in enumerate(items):
        if seg >= i + 1:
            y = 170 + i * 110
            d.rounded_rectangle([160, y, W - 160, y + 86], 14, outline=(30, 41, 59), fill=(12, 19, 34))
            d.text((200, y + 43), f"0{i + 1}", font=font(30, mono=True), fill=CYAN, anchor="lm")
            d.text((280, y + 43), k, font=F_BODY, fill=TEXT, anchor="lm")
            d.text((470, y + 43), v, font=F_BODY, fill=MUTED, anchor="lm")


RENDER = {
    "title": sc_title, "idea": sc_idea, "forward_img": sc_forward_img, "training": sc_training,
    "forward_2d": sc_forward_2d, "reverse_2d": sc_reverse_2d, "reverse_img": sc_reverse_img,
    "summary": sc_summary,
}

# ---------------------------------------------------------------------------
# TTS + assembly
# ---------------------------------------------------------------------------


def main():
    import vidkit
    # episode 01 renderers take (canvas, u, seg); vidkit also passes the sentence progress
    renderers = {k: (lambda f: lambda c, u, seg, su: f(c, u, seg))(f) for k, f in RENDER.items()}
    vidkit.render(SCENES, renderers, sys.argv[1], "diffusion-intro")


if __name__ == "__main__":
    main()
