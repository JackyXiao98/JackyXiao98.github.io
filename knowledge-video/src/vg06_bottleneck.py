"""Video Generation 06 — 注意力为什么是瓶颈.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python vg06_bottleneck.py <out_dir>

Real computations: share of a Transformer block's FLOPs spent in attention as a function of the
number of tokens (example width d = 3072, FFN 4d); how concentrated softmax attention is on toy video
tokens (vgx.toy_video_tokens: 8 x 16 x 16 grid) — mass in the top key blocks, and the share of
weights below the uniform level 1/N.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tviz as tv
import vgx
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Video Generation"

# 1. FLOPs per token per layer: projections + FFN = 24 d^2, attention (QK^T and PV) = 4 N d
DM = 3072
share = lambda n: n / (6 * DM + n)
NS = np.unique(np.logspace(2, 6, 60).astype(int))
SHARES = share(NS)
S_IMG, S_VID = share(4096), share(111600)
assert abs(S_IMG - 0.182) < 0.001 and abs(S_VID - 0.858) < 0.001

# 2. concentration on toy video tokens
Q, K, V = vgx.toy_video_tokens()
O, P = vgx.softmax_attn(Q, K, V)
M = vgx.block_mass(P)
NB = len(M)
TOP = {f: float((np.sort(M, 1)[:, ::-1][:, :int(round(f * NB))].sum(1) / M.sum(1)).mean()) for f in (0.05, 0.2)}
N = len(P)
SMALL = float((P < 1 / N).mean())
SMALL_MASS = float(P[P < 1 / N].sum() / N)
print(TOP, SMALL, SMALL_MASS)
assert 0.6 < TOP[0.05] < 0.7 and 0.85 < TOP[0.2] < 0.95 and SMALL > 0.9 and SMALL_MASS < 0.1
QB = 13  # a query block to display

SCENES = [
    ("title", [
        "前面我们用蒸馏减少了步数。这一期看每一步内部，注意力为什么成了最大的瓶颈，以及它的权重有什么特点。",
    ]),
    ("flops", [
        "一个变换器模块的计算分成两部分。线性层和前馈网络，对每个词元的计算量是固定的；而注意力要和所有词元打交道，对每个词元的计算量随词元总数增长。",
        f"以宽度三千零七十二的模型为例，一张图片的四千零九十六个词元，注意力只占整个模块计算量的{pct_zh(S_IMG)}。",
        f"换成第一期的视频，约十一万个词元，注意力的占比上升到{pct_zh(S_VID)}。视频模型的大部分时间，都花在注意力上。",
    ]),
    ("sparse", [
        "好消息是，注意力的权重非常集中。我们在一个玩具视频的词元上计算了注意力，把键按六十四个一组分块。",
        f"每个查询只看权重最大的百分之五的块，就拿到了{pct_zh(TOP[0.05])}的注意力；看百分之二十的块，就拿到了{pct_zh(TOP[0.2])}。",
        f"另一方面，{pct_zh(SMALL)}的权重都小于平均水平，但它们加起来还有大约{pct_zh(SMALL_MASS)}。这条长长的尾巴，直接丢掉会带来误差。",
    ]),
    ("directions", [
        "所以加速注意力有两个方向。第一个方向是让每一次计算更便宜，比如用低比特的整数来做矩阵乘法，这就是下一期的量化注意力。",
        "第二个方向是少算一些，重要的部分精确计算，不重要的部分近似或者跳过，这就是最后一期的稀疏线性注意力。两者可以叠加使用。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，注意力的计算量随词元数平方增长，视频模型里它占了绝大部分计算。",
        "第二，注意力权重高度集中，少数块拿走了大部分权重，但长尾也不能忽略。",
        "第三，加速的两个方向是算得更便宜，和算得更少。下一期介绍量化注意力。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "注意力为什么是瓶颈", "计算占比 · 权重有多集中", SERIES)


def sc_flops(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, f"注意力占模块计算量的比例（示例：d = {DM}，FFN = 4d）")
    tv.fpaste(c, r"$\frac{4Nd}{24d^2+4Nd}=\frac{N}{6d+N}$", 640, 150, 30)
    if seg >= 1:
        tv.line_chart(d, (180, 210, 1100, 560), NS, [("注意力占比", SHARES * 100, CYAN)], xlog=True, ylim=(0, 100),
                      xlabel="词元数 N", ylabel="%", xticks=[(100, "100"), (1000, "1K"), (10000, "10K"), (100000, "100K"), (1000000, "1M")])
        for n, v, col, t in [(4096, S_IMG, VIOLET, "图片 4,096"), (111600, S_VID, RED, "视频 111,600")]:
            if seg >= (1 if n == 4096 else 2):
                x = 180 + (np.log10(n) - 2) / 4 * 920
                y = 560 - v * 350
                d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col)
                d.text((x + 12, y + 6), f"{t}：{v:.0%}", font=F_SMALL, fill=col)


def sc_sparse(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "注意力权重有多集中（玩具视频词元，8 × 16 × 16）")
    row = M[QB] / M[QB].sum()
    tv.heat(d, (120, 170, 600, 230), row[None] ** 0.5, vmin=0, vmax=np.sqrt(row.max()))
    d.text((120, 240), f"一个查询块对 {NB} 个键块的注意力（开方显示）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        tv.bar_chart(d, (700, 180, 1150, 450), ["前 5% 的块", "前 20% 的块"], [TOP[0.05] * 100, TOP[0.2] * 100], [CYAN, VIOLET],
                     fmt="{:.0f}%", vmax=110)
        vk.text_c(d, (925, 515), "拿到的注意力（所有查询平均）", F_SMALL, MUTED)
    if seg >= 2:
        vk.text_c(d, (360, 380), f"{SMALL:.0%} 的权重 < 1/N", vk.font(34), AMBER)
        vk.text_c(d, (360, 440), f"但加起来仍有约 {SMALL_MASS:.0%} → 长尾不能直接丢", F_BODY, TEXT)


def sc_directions(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "两个加速方向")
    for i, (t, sub, col) in enumerate([("算得更便宜", "低比特量化：SageAttention（下一期）", CYAN),
                                       ("算得更少", "稀疏 + 线性：SLA（最后一期）", VIOLET)]):
        if seg >= i:
            x = 140 + i * 520
            d.rounded_rectangle([x, 200, x + 480, 360], 16, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 240, 250), t, vk.font(34), col)
            vk.text_c(d, (x + 240, 310), sub, F_SMALL, TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 460), "两者可以叠加", F_BODY, GREEN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("计算", "注意力 ∝ N²，视频中占绝大部分"), ("集中", "少数块拿走大部分权重"),
                      ("方向", "算得更便宜 · 算得更少")], seg, "下一期：SageAttention —— 量化注意力")


RENDER = {"title": sc_title, "flops": sc_flops, "sparse": sc_sparse, "directions": sc_directions, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vg-bottleneck")
