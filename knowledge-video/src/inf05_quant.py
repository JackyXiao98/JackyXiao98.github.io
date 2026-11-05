"""LLM Inference 05 — 量化：用更少的位存权重.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] INF_CACHE=<dir> python inf05_quant.py <out_dir>

Real computations (infx.exp_quant, numpy): round-to-nearest symmetric quantisation of every weight matrix in the
two Transformer layers of the trained toy model (embeddings and output layer kept in float32) to 8 / 6 / 4 / 3 / 2
bits, with one scale per matrix ("per tensor") or one scale per output column ("per channel"); held-out loss on
256 sequences. Outlier test: one FFN hidden unit per layer is scaled up 30x on its input weights and down 30x on its
output weights - the function is unchanged because ReLU is positively homogeneous - and quantisation is repeated.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import infx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "LLM Inference"
CACHE = os.environ.get("INF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "infcache"))

Q = infx.cached(CACHE, "quant", infx.exp_quant, cache_dir=CACHE)
B0 = Q["base"]
BITS = [8, 6, 4, 3, 2]
PT = {b: Q[(b, False)] for b in BITS}
PC = {b: Q[(b, True)] for b in BITS}
W = np.array([-0.9, -0.31, 0.05, 0.42, 0.77])
s4 = np.abs(W).max() / 7
W4 = np.round(W / s4) * s4
print(round(B0, 4), {b: (round(PT[b], 3), round(PC[b], 3)) for b in BITS}, {k: round(v, 3) for k, v in Q.items() if isinstance(k, tuple) and k[0] == "out"}, Q["outlier_same"], W4)
assert round(B0, 3) == 1.660 and abs(Q["outlier_same"] - B0) < 1e-6
assert round(PT[8], 3) == 1.661 and round(PC[8], 3) == 1.660 and round(PT[4], 2) == 1.75 and round(PC[4], 2) == 1.70
assert round(PT[3], 2) == 2.53 and round(PC[3], 2) == 1.94 and PT[2] > 4 and PC[2] > 3.9
assert round(Q[("out", 8, False)], 2) == 1.71 and round(Q[("out", 4, False)], 2) == 3.93 and round(Q[("out", 4, True)], 2) == 1.71

SCENES = [
    ("title", [
        "第一期说过，解码时的瓶颈是读权重。那么最直接的加速办法就是让权重变小：用八位、四位甚至更少的位来存每个数。这一期讲量化。",
    ]),
    ("how", [
        "最简单的量化是这样的：找到一组权重里绝对值最大的数，用它定一个缩放因子，把所有权重除以这个因子，四舍五入到最近的整数。四位有符号整数只能表示负八到七，所以每个权重只能落在十六个格子之一上。用的时候再乘回缩放因子。",
        "权重从十六位变成四位，显存和每一步要读的数据都变成四分之一。代价是舍入带来的误差。",
    ]),
    ("bits", [
        f"我们把小模型两层里的所有权重矩阵量化，看损失怎样变化。原始损失是{dec_zh(B0, 3)}。八位几乎没有变化。四位时，整个矩阵共用一个缩放因子，损失升到{dec_zh(PT[4], 2)}；每一列各用一个缩放因子，只升到{dec_zh(PC[4], 2)}。",
        f"三位时差别更大：共用一个因子是{dec_zh(PT[3], 2)}，按列是{dec_zh(PC[3], 2)}。到了两位，两种方法都坏掉了。缩放因子分得越细，误差越小。",
    ]),
    ("outlier", [
        "真实的大模型里有一个麻烦：少数权重或激活特别大，叫离群值。我们做一个实验，把前馈网络里一个神经元的输入权重放大三十倍，输出权重缩小三十倍。因为线性整流函数的性质，模型的输出完全不变。",
        f"但量化就出问题了。整个矩阵共用一个缩放因子时，这个大数把缩放因子撑大，其他权重都挤在零附近的几个格子里。八位时损失就升到{dec_zh(Q[('out', 8, False)], 2)}，四位时直接升到{dec_zh(Q[('out', 4, False)], 2)}。按列缩放把这个大数隔离在自己那一列里，四位时损失只有{dec_zh(Q[('out', 4, True)], 2)}。",
    ]),
    ("real", [
        "真实系统用了更多的技巧。大模型八位矩阵乘法的工作发现，激活里的离群值集中在少数几个特征维度，就把这些维度单独用十六位计算。",
        "四位的训练后量化方法，比如逐列量化并补偿误差的方法，和根据激活大小保护重要权重的方法，都能让大模型在四位下只损失很少的质量。还有一类方法在训练时就考虑量化，效果更好，但需要重新训练。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，量化用更少的位存权重，显存和读取量成比例减少。",
        "第二，缩放因子分得越细，误差越小；在我们的实验里，四位按列量化只损失约零点零四。",
        "第三，离群值会撑大缩放因子，需要专门处理。下一期，我们看投机解码：先猜后验。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "量化", "用更少的位存权重", SERIES)


def sc_how(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "对称量化：缩放 → 取整 → 乘回（4 位：-8 … 7）")
    x0, x1, y = 140, 1140, 330
    d.line([(x0, y), (x1, y)], fill=MUTED, width=2)
    sx = lambda v: x0 + (v + 1) / 2 * (x1 - x0)
    for k in range(-8, 8):
        v = k * s4
        d.line([(sx(v), y - 12), (sx(v), y + 12)], fill=(71, 85, 105), width=2)
    for w, q in zip(W, W4):
        d.ellipse([sx(w) - 8, y - 60 - 8, sx(w) + 8, y - 60 + 8], fill=CYAN)
        vk.arrow(d, (sx(w), y - 50), (sx(q), y - 14), AMBER, 2)
        d.text((sx(w), y - 92), f"{w:.2f}", font=vk.font(20, mono=True), fill=CYAN, anchor="mm")
    tv.fpaste(c, r"$s=\max|w|/7,\quad \hat w = s\cdot\mathrm{round}(w/s)$", 640, 440, 36)
    if seg >= 1:
        vk.text_c(d, (640, 560), "16 位 → 4 位：显存与读取量 ÷ 4", F_BODY, GREEN)


def sc_bits(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "位数与损失（两层所有权重矩阵）")
    xs = list(range(len(BITS)))
    ser = [("整个矩阵一个缩放因子", np.minimum([PT[b] for b in BITS], 3.0), RED), ("每列一个缩放因子", np.minimum([PC[b] for b in BITS], 3.0), GREEN)]
    tv.line_chart(d, (120, 200, 720, 580), xs, ser, ylim=(1.6, 3.05), xticks=[(i, str(b)) for i, b in enumerate(BITS)], xlabel="位数", legend_xy=(780, 220))
    d.text((120, 165), "验证损失（2 位时超出画面）", font=F_SMALL, fill=MUTED)
    f = vk.font(22, mono=True)
    d.text((780, 330), "位数  一个因子  每列", font=F_SMALL, fill=MUTED)
    for i, b in enumerate(BITS):
        d.text((780, 370 + i * 38), f"{b:>2}    {PT[b]:.3f}    {PC[b]:.3f}", font=f, fill=TEXT)
    d.text((780, 570), f"原始 {B0:.3f}", font=F_SMALL, fill=MUTED)


def sc_outlier(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "离群值：一个大数撑大缩放因子")
    vk.text_c(d, (640, 180), "一个神经元：输入权重 × 30，输出权重 ÷ 30 → 模型输出完全不变", F_SMALL, TEXT)
    if seg >= 1:
        labs = ["8 位 · 一个因子", "4 位 · 一个因子", "4 位 · 每列"]
        vals = [Q[("out", 8, False)], Q[("out", 4, False)], Q[("out", 4, True)]]
        tv.bar_chart(d, (200, 250, 900, 600), labs, vals, [AMBER, RED, GREEN], fmt="{:.3f}", vmax=4.4)
        d.line([(200, 600 - B0 / 4.4 * 350), (900, 600 - B0 / 4.4 * 350)], fill=MUTED, width=1)
        d.text((910, 600 - B0 / 4.4 * 350), f"原始 {B0:.3f}", font=F_SMALL, fill=MUTED, anchor="lm")


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "真实系统的做法")
    rows = [("LLM.int8()（2022）", "离群特征维度单独用 16 位计算", CYAN)]
    if seg >= 1:
        rows += [("GPTQ（2022）", "逐列量化，并补偿已产生的误差", GREEN), ("AWQ（2023）", "按激活大小保护重要的权重通道", AMBER),
                 ("量化感知训练", "训练时就考虑量化，需要重新训练", VIOLET)]
    for i, (a, b, col) in enumerate(rows):
        y = 180 + i * 95
        d.rounded_rectangle([140, y, 1140, y + 70], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((170, y + 35), a, font=F_BODY, fill=col, anchor="lm")
        d.text((560, y + 35), b, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("量化", "更少的位：显存与读取量减少"), ("粒度", "按列缩放误差更小"),
                      ("离群值", "撑大缩放因子，需要专门处理")], seg, "下一期：投机解码")


RENDER = {"title": sc_title, "how": sc_how, "bits": sc_bits, "outlier": sc_outlier, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "inf-quant")
