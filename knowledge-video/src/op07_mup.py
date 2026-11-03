"""Optimization 07 — 超参数迁移：μP.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] OP_CACHE=<dir> python op07_mup.py <out_dir>

Real computations (opx.mup, JAX): MLP [24, w, w, 12] predicting the next token of a 12-token second-order
Markov language from the one-hot of the previous two tokens (2,000 examples, full batch), Adam for 100 steps,
learning rate 2^-16 ... 2^0, widths w = 64, 256, 1024; training loss after 100 steps. Standard
parametrisation (SP): every layer uses the same learning rate, init N(0, 1/fan_in). muP (Adam): hidden layer
learning rate divided by m = w / 64, logits multiplied by 1/m. "Threshold lr" = smallest grid learning rate
with loss <= 1.2.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import opx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Optimization"
CACHE = os.environ.get("OP_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "opcache"))

M = opx.cached(CACHE, "mup", opx.mup)
LRS = M["lrs"]
WS = sorted(M["sp"])
thr = lambda v: int(np.where(np.nan_to_num(v, nan=9) <= 1.2)[0][0])
TH = {p: {w: thr(M[p][w]) for w in WS} for p in ("sp", "mup")}
print(TH, {w: M["mup"][w][7] for w in WS}, {w: M["sp"][w][7] for w in WS})
assert WS == [64, 256, 1024] and list(LRS[[0, -1]]) == [2.0 ** -16, 1.0]
assert TH["sp"] == {64: 7, 256: 5, 1024: 3} and TH["mup"] == {64: 7, 256: 7, 1024: 7}
assert max(M["mup"][w][7] for w in WS) - min(M["mup"][w][7] for w in WS) < 0.01
assert M["sp"][1024][7] < 0.95 and M["sp"][64][7] > 1.1
COLS = {64: CYAN, 256: VIOLET, 1024: GREEN}

SCENES = [
    ("title", [
        "训练一个大模型很贵，没法在它身上反复试学习率。一个自然的想法是，在小模型上调好超参数，再直接用到大模型上。这一期看这件事为什么通常行不通，以及最大更新参数化怎样让它行得通。",
    ]),
    ("problem", [
        "在标准的参数化下，网络变宽以后，每一步更新对输出的影响会跟着变大。宽度翻倍，同样的学习率就相当于更大的步子。所以小模型上最好的学习率，放到大模型上往往太大。",
    ]),
    ("sp", [
        "我们训练三个宽度不同的网络，六十四、二百五十六和一千零二十四，每个都扫描十七个学习率，看训练一百步以后的损失。",
        "在标准参数化下，三条曲线左右错开。要让损失降到一点二以下，宽度六十四需要的学习率，是宽度一千零二十四的十六倍。宽度每大四倍，曲线就向左移动四倍。",
    ]),
    ("mup", [
        "最大更新参数化按照宽度，调整每一层的学习率和输出的缩放。对于亚当，中间层的学习率除以宽度的倍数，输出再乘以宽度倍数的倒数。",
        "这样设置以后，三条曲线几乎完全重合。在同一个学习率下，三个宽度的损失相差不到百分之一。在小模型上找到的最佳学习率，可以直接用在大模型上。",
    ]),
    ("practice", [
        "二零二二年的研究用这个方法，在一个小模型上调好超参数，迁移到六十七亿参数的模型，效果超过了直接在大模型上调参的版本，而调参成本只有一小部分。",
        "今天很多大模型的训练，都在用最大更新参数化或者类似的思路，结合缩放定律，先在小模型上找好学习率、初始化等超参数，再放大。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，标准参数化下，网络变宽，同样的学习率等于更大的步子，最佳学习率会漂移。",
        "第二，最大更新参数化按宽度调整各层的学习率和缩放，让不同宽度的学习率曲线重合。",
        "第三，这样可以在小模型上调参，再迁移到大模型。下一期，我们看新一代优化器，香波和穆昂。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "超参数迁移：μP", "在小模型上调参，直接用到大模型", SERIES)


def sc_problem(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "网络变宽，同样的学习率 = 更大的步子")
    for i, (w, h) in enumerate([(64, 60), (256, 120), (1024, 200)]):
        x = 220 + i * 330
        d.rounded_rectangle([x, 380 - h, x + 140, 380 + h], 12, outline=COLS[w], width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 70, 380 + h + 30), f"宽度 {w}", F_SMALL, COLS[w])
    vk.text_c(d, (640, 650), "小模型上的最佳学习率，放到大模型上往往太大", F_BODY, AMBER)


def _curves(d, par, n=3):
    series = [(f"宽度 {w}", np.nan_to_num(M[par][w], nan=2.5), COLS[w]) for w in WS[:n]]
    tv.line_chart(d, (140, 170, 820, 580), LRS, series, xlog=True, ylim=(0.9, 2.5), xlabel="学习率（对数）", ylabel="训练 100 步后的损失",
                  legend_xy=(880, 470), xticks=[(2.0 ** k, f"2^{k}") for k in (-16, -12, -8, -4, 0)])
    y = 580 - (1.2 - 0.9) / 1.6 * 410
    d.line([(140, y), (820, y)], fill=MUTED, width=1)
    d.text((825, y), "1.2", font=F_SMALL, fill=MUTED, anchor="lm")


def sc_sp(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "标准参数化（SP）：曲线随宽度左移")
    _curves(d, "sp")
    if seg >= 1:
        d.text((880, 200), "损失 ≤ 1.2 所需的最小学习率", font=F_SMALL, fill=MUTED)
        for i, w in enumerate(WS):
            d.text((880, 245 + i * 45), f"宽度 {w}", font=F_SMALL, fill=COLS[w])
            d.text((1210, 245 + i * 45), f"2^{int(np.log2(LRS[TH['sp'][w]]))}", font=vk.font(24, mono=True), fill=COLS[w], anchor="ra")
        d.text((880, 400), "宽度 × 4 → 学习率 ÷ 4", font=F_BODY, fill=RED)


def sc_mup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "μP（Yang & Hu, 2021）：曲线重合")
    if seg == 0:
        tv.fpaste(c, r"$\eta_{\mathrm{hidden}}=\eta/m,\qquad \mathrm{logits}=\frac{1}{m}\,W_{\mathrm{out}}h,\qquad m=\frac{\mathrm{width}}{64}$", 640, 300, 34)
        vk.text_c(d, (640, 400), "Adam 下的 μP：中间层学习率 ÷ m，输出乘 1/m", F_SMALL, AMBER)
        return
    _curves(d, "mup")
    d.text((880, 200), f"学习率 2^{int(np.log2(LRS[7]))} 时的损失", font=F_SMALL, fill=MUTED)
    for i, w in enumerate(WS):
        d.text((880, 245 + i * 45), f"宽度 {w}", font=F_SMALL, fill=COLS[w])
        d.text((1210, 245 + i * 45), f"{M['mup'][w][7]:.3f}", font=vk.font(24, mono=True), fill=COLS[w], anchor="ra")
    d.text((880, 400), "最佳学习率不随宽度漂移", font=F_BODY, fill=GREEN)


def sc_practice(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "μTransfer（Yang et al., 2022）")
    for i, (t, col) in enumerate([("小模型上调好超参数", CYAN), ("直接迁移", AMBER), ("67 亿参数模型", GREEN)]):
        x = 120 + i * 370
        d.rounded_rectangle([x, 200, x + 320, 290], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 160, 245), t, F_SMALL, col)
        if i < 2:
            vk.arrow(d, (x + 325, 245), (x + 365, 245), MUTED, 3)
    vk.text_c(d, (640, 360), "效果超过直接在大模型上调参，成本只有一小部分", F_SMALL, TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 470), "μP / 类似思路 + 缩放定律：先在小模型上找好超参数，再放大", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("SP", "变宽 → 最佳学习率漂移"), ("μP", "按宽度调整学习率与缩放，曲线重合"),
                      ("迁移", "小模型调参 → 大模型直接用")], seg, "下一期：Shampoo 与 Muon")


RENDER = {"title": sc_title, "problem": sc_problem, "sp": sc_sp, "mup": sc_mup, "practice": sc_practice, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "op-mup")
