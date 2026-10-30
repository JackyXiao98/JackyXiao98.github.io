"""Knowledge Video 09 — Diffusion Transformer 与视频生成.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python ep09_dit.py <out_dir>

Patchify is shown on a real image. The video part is a real toy experiment: 8-frame clips of
an ellipse whose axes change linearly; per-frame model (PCA-16 + GMM per frame, frames sampled
independently) vs joint model (PCA-24 over whole clips + GMM, sampled jointly), both with exact
DDIM denoisers. Flicker = mean RMS difference between consecutive frames. DiT facts are from
Peebles & Xie 2023 (DiT-XL/2, ImageNet 256, latent 32x32x4, patch 2 -> 256 tokens).
The token count for a video is an explicitly labelled example calculation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw

import diff2d as D
import latent as L
import video_toy as V
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL, F_TITLE

VTR = V.make_videos(4000, 0)
FRAMES = VTR.reshape(-1, 1024)
PF = L.PCA(FRAMES[::2], 16)
ZF = PF.enc(FRAMES[::4])
SDF = ZF.std()
MU1, VAR1, W1 = L.fit_gmm(ZF / SDF, 100, 0)
PV = L.PCA(VTR.reshape(4000, -1), 24)
ZV = PV.enc(VTR.reshape(4000, -1))
SDV = ZV.std()
MUJ, VARJ, WJ = L.fit_gmm(ZV / SDV, 100, 0)
NV = 200
INDEP = np.stack([PF.dec(L.ddim_latent(np.random.default_rng(100 + f).standard_normal((NV, 16)), MU1, VAR1, W1, 50, D.abar) * SDF)
                  for f in range(8)], 1)
JOINT = PV.dec(L.ddim_latent(np.random.default_rng(7).standard_normal((NV, 24)), MUJ, VARJ, WJ, 50, D.abar) * SDV).reshape(NV, 8, 1024)
REAL = V.make_videos(NV, 5)
FL_REAL, FL_JOINT, FL_INDEP = V.flicker(REAL), V.flicker(JOINT), V.flicker(INDEP)
print(f"flicker real {FL_REAL:.4f} joint {FL_JOINT:.4f} independent {FL_INDEP:.4f}")
assert 2.7 < FL_INDEP / FL_REAL < 3.7 and FL_JOINT < 1.05 * FL_REAL

# example token count (stated as an example in the narration)
TOK = (720 // 8 // 2) * (1280 // 8 // 2) * (5 * 24 // 4)
assert TOK == 108000

# a real image for the patchify demo (the smiley from episode 01, 64x64)
from ep01_intro import SMILE64  # noqa: E402

# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------
SCENES = [
    ("title", [
        "前几期的去噪网络，大多是卷积结构。近几年，越来越多的模型改用了变换器，也就是扩散变换器。",
        "这一期，我们讲扩散变换器的结构，以及扩散模型怎样从图片扩展到视频。",
    ]),
    ("patchify", [
        "变换器处理的是一串词元，怎样把图片变成词元呢？",
        "做法是把潜空间里的图片切成小方块，每个方块展平以后，映射成一个向量，再加上位置编码。",
        "比如一张两百五十六乘两百五十六的图片，潜变量是三十二乘三十二，按二乘二切块，就得到二百五十六个词元。",
    ]),
    ("block", [
        "所有词元一起经过多层变换器模块，模块里的自注意力，让每个方块都能和其他所有方块交换信息。",
        "时间步和类别这些条件，通过自适应层归一化注入，也就是根据条件，动态地产生缩放和平移的参数。",
        "扩散变换器还把每个模块最后的缩放参数初始化为零，让模块一开始近似恒等映射，训练更加稳定。",
    ]),
    ("scaling", [
        "变换器的一个好处，是规模扩展的规律很清楚。",
        "原论文发现，模型的计算量越大，生成质量越好，最大的模型在图像网络数据集上，取得了当时最好的结果。",
    ]),
    ("video", [
        "视频可以看成一串图片，但如果每一帧单独生成，前后帧就会不连贯。",
        "我们用一个小例子来演示，每段视频有八帧，椭圆的大小随时间平滑变化。",
        "如果对每一帧独立地生成，相邻帧的平均差异约为真实视频的三倍，画面剧烈跳变；把八帧放在一起联合建模，生成的视频就很连贯。",
    ]),
    ("spacetime", [
        "真实的视频模型，会把视频压缩到时空潜空间，再切成时空小块，作为词元。",
        "注意力在所有时空词元之间进行，模型就能同时照顾每一帧的画面，以及帧与帧之间的运动。",
        "代价是词元数量非常多。按一个示例设置估算，一段五秒的高清视频，可能有十万个以上的词元，而注意力的计算量，随词元数的平方增长。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，扩散变换器把潜空间图片切成小块作为词元，用变换器来去噪。",
        "第二，条件通过自适应层归一化注入，模型规模越大，效果越好。",
        "第三，视频需要联合建模所有帧，才能保持连贯。",
        "第四，时空小块让变换器统一处理画面和运动，代价是巨大的计算量。",
        "这个系列到这里就告一段落了，感谢大家的观看。",
    ]),
]

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def fpaste(c, tex, cx, cy, size=28, color="#e2e8f0"):
    vk.paste_center(c, vk.formula(tex, size, color), cx, cy)


def frame_img(v, size):
    a = (np.clip(v.reshape(32, 32), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(a, "L").resize((size, size), Image.NEAREST).convert("RGB")


def box(d, xy, label, col=CYAN, w=200, h=70):
    x, y = xy
    d.rounded_rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 12, outline=col, width=3, fill=(12, 19, 34))
    vk.text_c(d, (x, y), label, F_SMALL, col)


SMILE_IMG = vk.to_rgb(SMILE64, 320)


def sc_title(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.text_c(d, (640, 190), "Knowledge Video · 09", vk.font(24, mono=True), MUTED)
    vk.text_c(d, (640, 270), "扩散变换器与视频生成", F_TITLE, CYAN)
    vk.text_c(d, (640, 340), "Diffusion Transformer (DiT)", F_BODY, TEXT)


def sc_patchify(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "切块：图片 → 词元")
    c.paste(SMILE_IMG, (110, 150))
    if seg >= 1:
        n = 8
        cell = 320 // n
        for i in range(1, n):
            d.line([(110 + i * cell, 150), (110 + i * cell, 470)], fill=CYAN, width=1)
            d.line([(110, 150 + i * cell), (430, 150 + i * cell)], fill=CYAN, width=1)
        k = int(n * n * (vk.ease(su) if seg == 1 else 1))
        for j in range(k):
            r, q = divmod(j, n)
            tile = SMILE_IMG.crop((q * cell, r * cell, q * cell + cell, r * cell + cell)).resize((26, 26))
            x = 520 + (j % 16) * 42
            y = 130 + (j // 16) * 50
            c.paste(tile, (x, y))
            d.rectangle([x + 28, y + 2, x + 36, y + 24], fill=(76, 58, 140))
        vk.text_c(d, (870, 345), "每个方块 → 一个向量（+ 位置编码）", F_SMALL, MUTED)
    if seg >= 2:
        vk.text_c(d, (870, 420), "256×256 图片 → 32×32×4 潜变量", F_BODY, TEXT)
        vk.text_c(d, (870, 470), "2×2 切块 → 16 × 16 = 256 个词元", F_BODY, CYAN)


def sc_block(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "DiT 模块：自注意力 + 自适应层归一化")
    xs = 300
    for i, (lab, col) in enumerate([("LayerNorm", MUTED), ("缩放 γ / 平移 β", GREEN), ("自注意力", CYAN),
                                    ("LayerNorm", MUTED), ("缩放 γ / 平移 β", GREEN), ("前馈网络", CYAN)]):
        box(d, (xs, 150 + i * 70), lab, col, w=240, h=50)
        if i < 5:
            vk.arrow(d, (xs, 175 + i * 70), (xs, 195 + i * 70), MUTED, 2)
    if seg >= 1:
        box(d, (680, 290), "条件：时间步 t、类别 c", VIOLET, w=300, h=60)
        vk.arrow(d, (530, 290), (420, 220), VIOLET, 2)
        vk.arrow(d, (530, 300), (420, 430), VIOLET, 2)
        fpaste(c, r"$h=\gamma(t,c)\odot\mathrm{LayerNorm}(x)+\beta(t,c)$", 960, 400, 26, "#34d399")
    if seg >= 2:
        fpaste(c, r"$x\leftarrow x+\alpha(t,c)\odot\mathrm{Block}(h)$", 960, 490, 26)
        vk.text_c(d, (960, 540), "adaLN-Zero：α 初始化为 0，模块一开始 ≈ 恒等映射", F_SMALL, MUTED)
    if seg == 0:
        vk.text_c(d, (880, 300), "所有词元两两交换信息", F_BODY, CYAN)


def sc_scaling(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "规模扩展")
    vk.text_c(d, (640, 230), "计算量 ↑   →   生成质量 ↑", F_TITLE, CYAN)
    if seg >= 1:
        vk.text_c(d, (640, 340), "DiT-XL/2：ImageNet 256×256，FID 2.27（Peebles & Xie, 2023）", F_BODY, TEXT)
        vk.text_c(d, (640, 390), "（FID 越低越好；论文发表时的最好结果）", F_SMALL, MUTED)


def sc_video(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "视频：逐帧独立 vs 联合建模")
    rows = [("真实视频", REAL, TEXT), ("联合建模", JOINT, CYAN), ("逐帧独立", INDEP, (248, 113, 113))]
    show = 1 if seg <= 1 else 3
    for r, (name, vids, col) in enumerate(rows[:show]):
        y = 120 + r * 150
        vk.text_c(d, (140, y + 55), name, F_SMALL, col)
        for f in range(8):
            c.paste(frame_img(vids[r % 2][f], 100), (220 + f * 112, y))
        fl = {0: FL_REAL, 1: FL_JOINT, 2: FL_INDEP}[r]
        d.text((220 + 8 * 112 + 10, y + 50), f"{fl:.3f}", font=vk.font(24, mono=True), fill=col, anchor="lm")
    if seg >= 2:
        vk.text_c(d, (640, 590), "右侧数字：相邻帧的平均差异（越小越连贯）", F_SMALL, MUTED)


def sc_spacetime(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "时空小块（spacetime patches）")
    for f in range(5):
        ox, oy = 150 + f * 40, 160 + f * 22
        d.rectangle([ox, oy, ox + 260, oy + 180], outline=(51, 65, 85), fill=(10, 16, 28))
        for i in range(1, 6):
            d.line([(ox + i * 43, oy), (ox + i * 43, oy + 180)], fill=(30, 41, 59))
        for j in range(1, 4):
            d.line([(ox, oy + j * 45), (ox + 260, oy + j * 45)], fill=(30, 41, 59))
    d.rectangle([310, 248, 396, 338], outline=CYAN, width=3)
    vk.text_c(d, (290, 470), "时间 × 高 × 宽 的小块 → 词元", F_SMALL, CYAN)
    if seg >= 1:
        vk.text_c(d, (880, 220), "注意力覆盖所有时空词元", F_BODY, TEXT)
        vk.text_c(d, (880, 265), "同时建模画面与运动", F_SMALL, MUTED)
    if seg >= 2:
        vk.text_c(d, (880, 360), "示例估算（假设）：1280×720，5 秒，24 帧/秒", F_SMALL, MUTED)
        vk.text_c(d, (880, 395), "空间压缩 8×，时间压缩 4×，2×2 切块", F_SMALL, MUTED)
        vk.text_c(d, (880, 450), f"45 × 80 × 30 = {TOK:,} 个词元", vk.font(32), VIOLET)
        fpaste(c, r"$\mathrm{cost}_{\mathrm{attention}}\ \propto\ N^2$", 880, 520, 28, "#a78bfa")


def sc_summary(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 6, "总结")
    items = [
        ("切块成词元", "潜空间图片切成小块，变换器去噪"),
        ("adaLN", "条件注入；规模越大效果越好"),
        ("视频连贯", "联合建模所有帧"),
        ("时空小块", "统一处理画面和运动，计算量巨大"),
    ]
    for i, (k, v) in enumerate(items):
        if seg >= i + 1:
            y = 130 + i * 95
            d.rounded_rectangle([140, y, 1140, y + 76], 14, outline=(30, 41, 59), fill=(12, 19, 34))
            d.text((180, y + 38), f"0{i + 1}", font=vk.font(30, mono=True), fill=CYAN, anchor="lm")
            d.text((260, y + 38), k, font=F_BODY, fill=TEXT, anchor="lm")
            d.text((470, y + 38), v, font=F_BODY, fill=MUTED, anchor="lm")
    if seg >= 5:
        vk.text_c(d, (640, 530), "系列完结 · 感谢观看", F_SMALL, VIOLET)


RENDER = {"title": sc_title, "patchify": sc_patchify, "block": sc_block, "scaling": sc_scaling,
          "video": sc_video, "spacetime": sc_spacetime, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "dit-video")
