"""Video Generation 02 — 视频扩散 Transformer：时空注意力.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python vg02_dit.py <out_dir>

Real computations: attention pair counts of full 3D vs factorised (spatial + temporal) attention
for the example token grid of episode 01; 3D RoPE on random vectors (q . k depends only on the
offset in time, height and width); the size of one dense attention matrix in FP16.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Video Generation"

# 1. pair counts, example grid of episode 01: T = 31 latent frames, S = 45 x 80 tokens per frame
T, S = 31, 45 * 80
FULL = (T * S) ** 2
FACT = T * S ** 2 + S * T ** 2
SAVE = FULL / FACT
assert 30 < SAVE < 31

# 2. 3D RoPE: head dim split into time / height / width parts, each rotated by its own coordinate
rng = np.random.default_rng(0)
DH = 96
PARTS = [32, 32, 32]


def rope_1d(x, m):
    d = len(x)
    th = m / 10000 ** (2 * np.arange(d // 2) / d)
    c, s = np.cos(th), np.sin(th)
    out = np.empty_like(x)
    out[0::2], out[1::2] = x[0::2] * c - x[1::2] * s, x[0::2] * s + x[1::2] * c
    return out


def rope_3d(x, pos):
    out, i = [], 0
    for n, m in zip(PARTS, pos):
        out.append(rope_1d(x[i:i + n], m))
        i += n
    return np.concatenate(out)


QV, KV = rng.standard_normal(DH), rng.standard_normal(DH)
OFF = (1, 2, -3)
PAIRS3 = [((0, 0, 5), None), ((4, 10, 20), None), ((12, 30, 7), None)]
DOTS = [float(rope_3d(QV, p) @ rope_3d(KV, tuple(a + o for a, o in zip(p, OFF)))) for p, _ in PAIRS3]
DOT_OTHER = float(rope_3d(QV, (0, 0, 5)) @ rope_3d(KV, (2, 2, 2)))
assert np.ptp(DOTS) < 1e-9 and abs(DOT_OTHER - DOTS[0]) > 0.1

# 3. one dense attention matrix in FP16
N = T * S
GB = N * N * 2 / 1e9
assert 24 < GB < 26

SCENES = [
    ("title", [
        "上一期我们看到，一段视频压缩以后仍然有十万个左右的词元。这一期看视频扩散变换器，怎样在这么多的时空词元上做注意力。",
    ]),
    ("block", [
        "视频扩散变换器和图片的版本几乎一样。所有时空词元排成一个序列，每个模块里先做自注意力，让词元之间交换信息。",
        "然后通过交叉注意力读取文本提示的信息，再经过前馈网络。扩散的时间步通过自适应层归一化注入，这些在扩散模型系列的最后一期讲过。",
    ]),
    ("factor", [
        "最直接的做法是完整的三维注意力，每个词元和所有帧的所有词元计算关系。",
        f"另一种做法是分解注意力，先在每一帧内部做空间注意力，再让同一个位置在不同帧之间做时间注意力。按上一期的例子算，需要计算的词元对少了大约{int_zh(round(SAVE))}倍。",
        "代价是，一个物体从一帧的左边移到下一帧的右边，信息要先在帧内传递，再在帧间传递，需要两跳才能对上。早期的视频模型常在图片模型里插入时间注意力层，而近期的主流模型大多改用完整的三维注意力。",
    ]),
    ("rope", [
        "位置信息同样要扩展到三维。常见的做法是三维旋转位置编码，把每个注意力头的维度分成三份，分别按时间、高度和宽度旋转。",
        "我们用随机向量验证了一下，只要时间、高度、宽度三个方向的相对偏移相同，点积就完全相同，和绝对位置无关。",
    ]),
    ("memory", [
        f"最后看一下规模。十万个词元的注意力矩阵，用半精度存储，单个注意力头就要大约{int_zh(round(GB))}个吉字节。",
        "所以实际计算从来不把整个矩阵存下来，而是像闪电注意力那样分块计算。即便如此，计算量本身仍然和词元数的平方成正比，这是后面几期加速方法要解决的问题。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，视频扩散变换器把所有时空词元排成一个序列，用自注意力、交叉注意力和前馈网络处理。",
        "第二，分解注意力便宜很多，但跨帧跨位置的信息需要两跳；主流模型多用完整的三维注意力，配合三维旋转位置编码。",
        "第三，注意力的计算量随词元数平方增长。下一期，我们看怎样逐段地、实时地生成视频，也就是自回归生成和自我强迫训练。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "视频扩散 Transformer", "时空注意力 · 3D RoPE", SERIES)


def sc_block(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "视频 DiT 模块")
    items = [("时空词元 T × H × W", TEXT, 0), ("自注意力（时空）", CYAN, 0), ("交叉注意力 ← 文本", VIOLET, 1),
             ("前馈网络", GREEN, 1), ("adaLN ← 时间步 t", AMBER, 1)]
    for i, (t, col, s) in enumerate(items):
        on = seg >= s
        y = 150 + i * 85
        d.rounded_rectangle([420, y, 860, y + 60], 12, outline=col if on else (30, 41, 59), width=2, fill=(12, 19, 34))
        vk.text_c(d, (640, y + 30), t, F_BODY, col if on else MUTED)
        if i < 4:
            vk.arrow(d, (640, y + 62), (640, y + 83), MUTED, 2)


def grid(d, x0, y0, mode, hl=(2, 1, 1), n=4, cell=22, gap=16):
    """n frames of n x n tokens; highlight what token hl=(frame, row, col) attends to."""
    f0, r0, c0 = hl
    for f in range(n):
        for r in range(n):
            for cc in range(n):
                x = x0 + f * (n * cell + gap) + cc * cell
                y = y0 + r * cell
                on = mode == "full" or (mode == "spatial" and f == f0) or (mode == "temporal" and r == r0 and cc == c0)
                col = AMBER if (f, r, cc) == hl else (CYAN if on else (30, 41, 59))
                d.rectangle([x, y, x + cell - 3, y + cell - 3], fill=col)


def sc_factor(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "完整 3D 注意力 vs 分解注意力")
    grid(d, 120, 160, "full")
    d.text((120, 260), "完整 3D：看所有帧的所有词元", font=F_SMALL, fill=CYAN)
    if seg >= 1:
        grid(d, 120, 320, "spatial")
        d.text((120, 420), "空间注意力：同一帧", font=F_SMALL, fill=CYAN)
        grid(d, 120, 480, "temporal")
        d.text((120, 580), "时间注意力：同一位置", font=F_SMALL, fill=CYAN)
        tv.fpaste(c, r"$(TS)^2\ \ \mathrm{vs}\ \ TS^2+ST^2$", 960, 230, 30)
        vk.text_c(d, (960, 300), f"T = {T}, S = {S:,}: {FULL:.2e} vs {FACT:.2e}", vk.font(20, mono=True), TEXT)
        vk.text_c(d, (960, 350), f"少约 {SAVE:.0f} 倍", F_BODY, GREEN)
    if seg >= 2:
        vk.text_c(d, (960, 450), "跨帧又跨位置：需要两跳", F_BODY, RED)
        vk.text_c(d, (960, 510), "早期：图片模型 + 时间层（如 AnimateDiff）", F_SMALL, MUTED)
        vk.text_c(d, (960, 545), "近期：完整 3D 注意力", F_SMALL, MUTED)
        vk.text_c(d, (960, 575), "（如 CogVideoX、HunyuanVideo、Wan）", F_SMALL, MUTED)


def sc_rope(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "3D RoPE")
    x0 = 220
    for i, (lab, col) in enumerate([("时间 t", CYAN), ("高度 h", VIOLET), ("宽度 w", GREEN)]):
        d.rectangle([x0 + i * 280, 160, x0 + i * 280 + 270, 210], fill=col)
        vk.text_c(d, (x0 + i * 280 + 135, 185), f"{lab}：{PARTS[i]} 维", F_SMALL, (10, 14, 22))
    d.text((x0, 220), f"一个注意力头的 {DH} 维", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.text((230, 300), f"偏移 (Δt, Δh, Δw) = {OFF}", font=F_BODY, fill=TEXT)
        for i, ((p, _), v) in enumerate(zip(PAIRS3, DOTS)):
            q = tuple(a + o for a, o in zip(p, OFF))
            d.text((230, 360 + i * 46), f"q @ {p!s:<12} k @ {q!s:<12} → {v:+.6f}", font=vk.font(22, mono=True), fill=CYAN)
        d.text((230, 520), f"q @ (0, 0, 5)   k @ (2, 2, 2)    → {DOT_OTHER:+.6f}", font=vk.font(22, mono=True), fill=MUTED)
        d.text((230, 560), "最后一行偏移不同，点积也不同", font=F_SMALL, fill=MUTED)


def sc_memory(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "规模")
    tv.fpaste(c, r"$111600^2\times 2\ \mathrm{bytes}\approx 24.9\ \mathrm{GB}$", 640, 200, 36, "#fbbf24")
    vk.text_c(d, (640, 260), "单个注意力头的完整注意力矩阵（FP16）", F_SMALL, MUTED)
    if seg >= 1:
        for i in range(6):
            for j in range(6):
                on = (i, j) == (2, 3)
                d.rectangle([460 + j * 50, 320 + i * 40, 505 + j * 50, 355 + i * 40], fill=AMBER if on else (30, 41, 59))
        vk.text_c(d, (640, 580), "分块计算（FlashAttention）：不存整张矩阵；但计算量仍 ∝ N²", F_BODY, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("视频 DiT", "自注意力 + 交叉注意力 + FFN"), ("时空注意力", "分解便宜；主流用完整 3D + 3D RoPE"),
                      ("瓶颈", "计算量随词元数平方增长")], seg, "下一期：自回归视频生成与 Self-Forcing")


RENDER = {"title": sc_title, "block": sc_block, "factor": sc_factor, "rope": sc_rope, "memory": sc_memory,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vg-dit")
