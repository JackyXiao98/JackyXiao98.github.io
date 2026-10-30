"""Video Generation 07 — SageAttention：量化注意力.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python vg07_sage.py <out_dir>

Real computations (numpy, 2,048 tokens, head dim 64): attention with Q and K quantised to INT8 (one
scale per block of 128 rows), on keys with a few large channel-wise offsets shared by all tokens
(the kind of outlier SageAttention reports in real models); error without and with smoothing K
(subtracting its mean over tokens), the exactness of that smoothing, and P.V in FP16.
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

rng = np.random.default_rng(0)
N, DH = 2048, 64
Q, K0, V = rng.standard_normal((N, DH)), rng.standard_normal((N, DH)), rng.standard_normal((N, DH))
BIAS = np.zeros(DH)
IDX = rng.choice(DH, 6, replace=False)
BIAS[IDX] = rng.choice([-1, 1], 6) * rng.uniform(8, 15, 6)
K = K0 + BIAS
q8 = vgx.int8_blocks
O, _ = vgx.softmax_attn(Q, K, V)
O0, _ = vgx.softmax_attn(Q, K0, V)
E_DIRECT = vgx.rel_err(vgx.softmax_attn(q8(Q), q8(K), V)[0], O)
KS = K - K.mean(0)
E_EXACT = vgx.rel_err(vgx.softmax_attn(Q, KS, V)[0], O)
O_SM, P_SM = vgx.softmax_attn(q8(Q), q8(KS), V)
E_SMOOTH = vgx.rel_err(O_SM, O)
E_NOBIAS = vgx.rel_err(vgx.softmax_attn(q8(Q), q8(K0), V)[0], O0)
O16 = P_SM.astype(np.float16).astype(np.float32) @ V.astype(np.float16).astype(np.float32)
E_FP16 = vgx.rel_err(O16, O)
CH_MAX = np.abs(K).max(0)
print(E_DIRECT, E_SMOOTH, E_EXACT, E_NOBIAS, E_FP16)
assert 0.035 < E_DIRECT < 0.045 and 0.012 < E_SMOOTH < 0.015 and E_EXACT < 1e-12
assert abs(E_SMOOTH - E_NOBIAS) < 0.001 and abs(E_FP16 - E_SMOOTH) < 0.001

SCENES = [
    ("title", [
        "上一期我们看到，视频模型的大部分时间花在注意力上。这一期介绍第一种加速方法，量化注意力，让每一次计算都更便宜。",
    ]),
    ("int8", [
        "注意力的核心是两次矩阵乘法，查询乘以键的转置，以及注意力权重乘以值。",
        "在支持整数八位运算的显卡上，整数八位矩阵乘法的吞吐量是半精度的两倍。所以如果能把查询和键量化成八位整数，计算就能快很多。",
        "量化就是用一个缩放系数，把浮点数映射到负一百二十七到一百二十七之间的整数。每一块数据用自己的缩放系数，块里绝对值最大的数决定了精度。",
    ]),
    ("outlier", [
        "问题在于键里的离群值。研究者发现，在真实模型里，键的某些通道对所有词元都有一个很大的共同偏移。",
        f"我们构造了这样的数据，六个通道带有八到十五的偏移，其他通道的最大值只有{dec_zh(np.median(CH_MAX), 1)}左右。直接量化，缩放系数被离群值撑大，普通数值的精度损失严重，注意力输出的相对误差是{pct_zh(E_DIRECT, 1)}。",
    ]),
    ("smooth", [
        "解决办法很巧妙，把键减去它在所有词元上的平均值。",
        "这不会改变注意力的结果。因为每个查询和这个平均向量的点积，对所有键都是同一个常数，而软最大函数对整行加一个常数不敏感。我们验证了，减去平均以后输出的差别只有十的负十五次方。",
        f"平滑以后再量化，误差从{pct_zh(E_DIRECT, 1)}降到了{pct_zh(E_SMOOTH, 1)}，和没有离群值的情况几乎一样。注意力权重乘以值这一步，用半精度计算，几乎不增加误差。",
    ]),
    ("practice", [
        "这就是量化注意力方法的核心思路，平滑键，把查询和键量化到八位整数，权重和值用更高的精度。它不需要重新训练，可以直接替换现有的注意力实现。",
        "后续版本进一步把权重乘以值这一步也换成了八位浮点数，并加入更细粒度的量化。具体的加速倍数和画质对比，见视频下方的笔记。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，整数八位矩阵乘法的吞吐量是半精度的两倍，量化查询和键可以加速注意力。",
        "第二，键在某些通道上的共同偏移会严重损害量化精度。",
        "第三，减去键的平均值不改变结果，却能让量化误差回到正常水平。最后一期，我们看另一种思路，少算一些，稀疏线性注意力。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "SageAttention", "量化注意力：每次计算更便宜", SERIES)


def sc_int8(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "INT8 量化")
    tv.fpaste(c, r"$O=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$", 640, 150, 32)
    if seg >= 1:
        tv.bar_chart(d, (200, 230, 620, 480), ["FP16", "INT8"], [1, 2], [MUTED, CYAN], fmt="{:.0f}×", vmax=2.4)
        vk.text_c(d, (410, 545), "矩阵乘法吞吐量（相对，A100 / H100）", F_SMALL, MUTED)
    if seg >= 2:
        tv.fpaste(c, r"$\hat x=s\cdot\mathrm{round}(x/s),\qquad s=\frac{\max|x_{\mathrm{block}}|}{127}$", 940, 300, 28)
        vk.text_c(d, (940, 380), "块里最大的数决定了精度", F_BODY, AMBER)


def sc_outlier(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "K 的通道离群值")
    tv.bar_chart(d, (120, 170, 1160, 450), [str(i) if i % 8 == 0 else "" for i in range(DH)], list(CH_MAX),
                 [RED if i in IDX else CYAN for i in range(DH)], fmt="", vmax=CH_MAX.max() * 1.1)
    d.text((120, 500), f"K 每个通道的最大绝对值（{N} 个词元）：6 个通道带有所有词元共同的偏移", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 580), f"直接量化：注意力输出相对误差 {E_DIRECT:.1%}", vk.font(32), RED)


def sc_smooth(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "平滑 K：减去所有词元的平均")
    tv.fpaste(c, r"$\tilde K=K-\mathbf{1}\bar k^\top,\qquad Q\tilde K^\top=QK^\top-(Q\bar k)\mathbf{1}^\top$", 640, 150, 28)
    if seg >= 1:
        vk.text_c(d, (640, 215), f"每行减去同一个常数 → softmax 不变（实测差异 {E_EXACT:.0e}）", F_BODY, GREEN)
    if seg >= 2:
        tv.bar_chart(d, (220, 280, 1060, 560), ["直接量化", "平滑后量化", "无离群值时量化", "+ PV 用 FP16"],
                     [E_DIRECT * 100, E_SMOOTH * 100, E_NOBIAS * 100, E_FP16 * 100], [RED, CYAN, MUTED, VIOLET],
                     fmt="{:.2f}%", vmax=E_DIRECT * 120)
        vk.text_c(d, (640, 640), "注意力输出的相对误差", F_SMALL, MUTED)


def sc_practice(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "SageAttention（Zhang et al., 2025）")
    steps = [("平滑 K", AMBER), ("Q、K → INT8", CYAN), ("softmax", TEXT), ("P·V 更高精度", VIOLET)]
    for i, (t, col) in enumerate(steps):
        x = 120 + i * 270
        d.rounded_rectangle([x, 200, x + 230, 280], 12, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 115, 240), t, F_BODY, col)
        if i < 3:
            vk.arrow(d, (x + 235, 240), (x + 265, 240), MUTED, 2)
    vk.text_c(d, (640, 340), "即插即用：不需要重新训练", F_BODY, GREEN)
    if seg >= 1:
        vk.text_c(d, (640, 420), "SageAttention2：P·V 用 FP8，更细粒度的量化", F_SMALL, TEXT)
        vk.text_c(d, (640, 460), "加速倍数与画质：见笔记（论文报告）", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("INT8", "吞吐量是 FP16 的两倍"), ("离群值", "K 的通道偏移损害量化精度"),
                      ("平滑", "减去均值：结果不变，误差回到正常")], seg, "最后一期：SLA —— 稀疏-线性注意力")


RENDER = {"title": sc_title, "int8": sc_int8, "outlier": sc_outlier, "smooth": sc_smooth, "practice": sc_practice,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vg-sage")
