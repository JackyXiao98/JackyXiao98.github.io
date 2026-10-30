"""Video Generation 08 — SLA：稀疏-线性注意力.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python vg08_sla.py <out_dir>

Real computations (toy video tokens, vgx.toy_video_tokens; 2,048 tokens in 64-token blocks): exact
attention vs block-sparse attention on the top 5/10/20% key blocks, and sparse + a linear-attention
branch for the marginal blocks whose output projection is fitted by least squares on half of the
queries (standing in for fine-tuning) and evaluated on the other half; FLOP fractions for the
example video of episode 01.
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

Q, K, V = vgx.toy_video_tokens()
O, P = vgx.softmax_attn(Q, K, V)
M = vgx.block_mass(P)
N = len(Q)
perm = np.random.default_rng(0).permutation(N)
TR, TE = perm[:N // 2], perm[N // 2:]
KEEPS = [0.05, 0.1, 0.2]
ERR_S, ERR_SL = {}, {}
for k in KEEPS:
    Os, Osl = vgx.sla(Q, K, V, M, k, 0.5, fit_idx=TR)
    ERR_S[k], ERR_SL[k] = vgx.rel_err(Os[TE], O[TE]), vgx.rel_err(Osl[TE], O[TE])
print(ERR_S, ERR_SL)
assert all(ERR_SL[k] < 0.75 * ERR_S[k] for k in KEEPS) and 0.08 < ERR_S[0.05] < 0.095 and 0.045 < ERR_SL[0.05] < 0.055

# FLOPs per head, example video of episode 01: N = 111,600 tokens, head dim 128
NV, DHV = 111600, 128
FULL = 2 * NV * NV * DHV
LIN = 2 * NV * DHV * DHV
FRAC = 0.05 + LIN / FULL
assert abs(LIN / FULL - DHV / NV) < 1e-12 and 0.94 < 1 - FRAC < 0.95

SCENES = [
    ("title", [
        "这是视频生成系列的最后一期。上一期让每次计算更便宜，这一期让计算更少，介绍稀疏线性注意力。",
    ]),
    ("idea", [
        "第六期我们看到，注意力权重高度集中，但长尾也不能直接丢掉。稀疏线性注意力把注意力权重的块分成三类。",
        "权重最大的少数关键块，用精确的稀疏注意力计算；权重极小的块直接跳过；中间大量的边缘块，用线性注意力近似。",
        "线性注意力先把键和值合并成一个小矩阵，再和查询相乘，计算量和词元数成线性关系，而不是平方关系。它不够精确，但用来补充长尾正合适。",
    ]),
    ("toy", [
        f"我们在第六期的玩具视频词元上试一试。只保留百分之五的关键块，丢掉其余部分，输出的相对误差是{pct_zh(ERR_S[0.05], 1)}。",
        f"加上线性注意力的分支以后，误差降到{pct_zh(ERR_SL[0.05], 1)}。保留百分之十和百分之二十的关键块时，也都有类似的改善。",
        "这里线性分支的输出要经过一个投影矩阵，我们用一半的查询拟合它，在另一半上测试。真实的方法则是对整个模型做少量的微调，让模型适应这种新的注意力。",
    ]),
    ("cost", [
        "计算量能省多少？以第一期的视频为例，十一万多个词元，每个注意力头一百二十八维。",
        f"关键块占百分之五，线性分支的计算量只相当于完整注意力的千分之一左右，加起来，注意力的计算量减少了约{pct_zh(1 - FRAC)}。",
    ]),
    ("recap", [
        "回顾整个视频生成系列。我们从时空潜空间和视频扩散变换器出发，讲了自回归生成和自我强迫训练，又用分布匹配蒸馏把几十步压缩到几步。",
        "然后分析了注意力这个最大的瓶颈，用量化注意力让每次计算更便宜，用稀疏线性注意力让计算更少。这些方法叠加起来，正在让高质量的视频生成走向实时。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "SLA：稀疏-线性注意力", "关键的精确算，边缘的近似算，其余跳过", SERIES)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "三类注意力块")
    nb = len(M)
    row = M[13] / M[13].sum()
    order = np.argsort(-row)
    cls = np.zeros(nb, int)
    cls[order[:2]] = 2
    cls[order[2:nb // 2]] = 1
    for j in range(nb):
        col = [(30, 41, 59), VIOLET, AMBER][cls[j]] if seg >= 1 else CYAN
        h = 20 + 140 * np.sqrt(row[j] / row.max())
        d.rectangle([140 + j * 31, 330 - h, 165 + j * 31, 330], fill=col)
    d.text((140, 345), "一个查询块对各键块的注意力（开方显示）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        for i, (t, sub, col) in enumerate([("关键", "精确稀疏注意力", AMBER), ("边缘", "线性注意力", VIOLET),
                                           ("可忽略", "跳过", MUTED)]):
            x = 140 + i * 340
            d.rounded_rectangle([x, 400, x + 300, 480], 12, outline=col, width=2, fill=(12, 19, 34))
            d.text((x + 20, 440), t, font=F_BODY, fill=col, anchor="lm")
            d.text((x + 280, 440), sub, font=F_SMALL, fill=TEXT, anchor="rm")
    if seg >= 2:
        tv.fpaste(c, r"$\phi(Q)\left(\phi(K)^\top V\right):\ O(Nd^2)\ \ \mathrm{vs}\ \ O(N^2d)$", 640, 560, 28)


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "玩具实验：输出的相对误差（越小越好）")
    x0, x1, y0, y1 = 200, 1080, 170, 520
    gw = (x1 - x0) / 3
    vmax = ERR_S[0.05] * 115
    d.line([(x0, y1), (x1, y1)], fill=(71, 85, 105), width=2)
    for i, k in enumerate(KEEPS):
        if i > 0 and seg < 1:
            continue
        for j, (v, col) in enumerate([(ERR_S[k] * 100, RED), (ERR_SL[k] * 100, CYAN)]):
            if j == 1 and seg < 1:
                continue
            bx = x0 + i * gw + gw * 0.15 + j * gw * 0.36
            top = y1 - v / vmax * (y1 - y0)
            d.rectangle([bx, top, bx + gw * 0.3, y1], fill=col)
            d.text((bx + gw * 0.15, top - 14), f"{v:.1f}%", font=vk.font(20, mono=True), fill=col, anchor="mm")
        vk.text_c(d, (x0 + i * gw + gw / 2, y1 + 26), f"保留 {k:.0%} 的关键块", F_SMALL, MUTED)
    d.text((x1, y0 - 10), "■ 只用稀疏", font=F_SMALL, fill=RED, anchor="ra")
    if seg >= 1:
        d.text((x1, y0 + 20), "■ 稀疏 + 线性", font=F_SMALL, fill=CYAN, anchor="ra")
    if seg >= 2:
        vk.text_c(d, (640, 600), "线性分支的投影：在一半查询上拟合，另一半上测试（代替微调）", F_SMALL, AMBER)


def sc_cost(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, f"计算量（示例：N = {NV:,}，每头 {DHV} 维）")
    tv.fpaste(c, r"$\frac{\mathrm{linear}}{\mathrm{full}}=\frac{2Nd^2}{2N^2d}=\frac{d}{N}\approx 0.1\%$", 640, 180, 32)
    if seg >= 1:
        x0, w = 200, 880
        d.rectangle([x0, 300, x0 + w, 360], outline=(71, 85, 105), width=2)
        d.rectangle([x0, 300, x0 + w * FRAC, 360], fill=AMBER)
        d.text((x0 + w * FRAC + 10, 330), f"SLA ≈ {FRAC:.1%}（5% 关键块 + 线性分支）", font=F_SMALL, fill=AMBER, anchor="lm")
        d.text((x0, 380), "完整注意力 = 100%", font=F_SMALL, fill=MUTED)
        vk.text_c(d, (640, 480), f"注意力计算量减少约 {1 - FRAC:.0%}", vk.font(36), GREEN)


EPS_ = ["视频生成概览", "时空注意力", "Self-Forcing", "少步生成", "DMD 蒸馏", "注意力瓶颈", "SageAttention", "SLA"]


def sc_recap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "系列回顾")
    k = 5 if seg == 0 else 8
    for i, t in enumerate(EPS_):
        x, y = 170 + (i % 4) * 245, 200 + (i // 4) * 170
        on = i < k
        d.rounded_rectangle([x, y, x + 215, y + 120], 14, outline=CYAN if on else (30, 41, 59), width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 107, y + 36), f"{i + 1:02d}", vk.font(26, mono=True), CYAN if on else MUTED)
        vk.text_c(d, (x + 107, y + 82), t, F_SMALL, TEXT if on else MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 590), "感谢观看 · Video Generation 系列完", F_BODY, VIOLET)


RENDER = {"title": sc_title, "idea": sc_idea, "toy": sc_toy, "cost": sc_cost, "recap": sc_recap}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vg-sla")
