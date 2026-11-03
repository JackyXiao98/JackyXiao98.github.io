"""Interpretability 05 — 叠加：概念比神经元多的时候.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MI_CACHE=<dir> python mi05_superposition.py <out_dir>

Real computations (mix.superposition, JAX): Toy Models of Superposition - 5 features x_i in [0, 1], each
zero with probability S (sparsity), importance 0.9^i; h = W x in 2 dimensions; x' = ReLU(W^T h + b);
weighted MSE, Adam, 10,000 steps of 1,024 samples, for S = 0, 0.7, 0.9, 0.97, 0.99. A feature counts as
represented if its column of W has norm > 0.5.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import mix
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Interpretability"
CACHE = os.environ.get("MI_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "micache"))

SUP = mix.cached(CACHE, "sup", mix.superposition)
SS = sorted(SUP)
NR = {s: SUP[s]["n_repr"] for s in SS}
ANG = {s: np.degrees(np.arctan2(SUP[s]["W"][1], SUP[s]["W"][0])) for s in SS}


def gaps(s):
    a = np.sort(ANG[s][SUP[s]["norms"] > 0.5] % 360)
    return np.diff(np.concatenate([a, [a[0] + 360]]))


print(NR, {s: np.round(gaps(s)) for s in SS})
assert NR == {0.0: 2, 0.7: 4, 0.9: 5, 0.97: 5, 0.99: 5}
assert np.allclose(gaps(0.7), 90, atol=3) and np.allclose(gaps(0.9), 72, atol=3) and SUP[0.0]["norms"][:2].min() > 0.99
assert abs((ANG[0.0][0] - ANG[0.0][1]) % 360 - 269) < 3       # the two dense features are orthogonal
assert SUP[0.9]["b"].max() < -0.19
COLS = [CYAN, VIOLET, GREEN, AMBER, RED]

SCENES = [
    ("title", [
        "我们希望每个神经元对应一个清楚的概念。但在真实的模型里，一个神经元常常同时对好几个毫不相关的东西有反应。这一期看背后的一个原因，叠加。",
    ]),
    ("poly", [
        "研究者在图像和语言模型里都发现了这样的神经元，比如同一个神经元，既对猫脸有反应，也对汽车的前脸有反应，这叫多义神经元。",
        "一个解释是，模型想表示的概念，远比它拥有的神经元多。它不得不把多个概念压进同一组维度，让它们共用空间，这就是叠加。",
    ]),
    ("toy", [
        "二零二二年的玩具模型把这件事看得很清楚。有五个特征，但隐藏层只有两维。模型要把五个特征压进两维，再还原出来。特征越靠前越重要。",
        "关键的参数是稀疏度，也就是每个特征有多大概率为零。我们把稀疏度从零一直调到零点九九，看两维空间里放了哪些特征。",
    ]),
    ("result", [
        "稀疏度为零，也就是所有特征总是同时出现时，模型只表示了最重要的两个特征，它们互相垂直，其余三个直接放弃。",
        "稀疏度为零点七时，模型放进了四个特征，排成一个正方形，每两个相对的特征共用一条直线。",
        "稀疏度达到零点九以后，五个特征全部放了进去，排成一个正五边形，相邻两个特征之间是七十二度。",
    ]),
    ("why", [
        "为什么稀疏时可以叠加？因为特征很少同时出现。当只有一个特征出现时，它在其他特征方向上的投影是干扰，模型用负的偏置，再加上激活函数，把这些小的干扰过滤掉。",
        "代价是，偶尔两个特征同时出现时会互相干扰。特征越稀疏，这种情况越少，叠加就越划算。",
    ]),
    ("implication", [
        "这对可解释性有一个重要的含义。在叠加的情况下，单个神经元的方向不对应任何一个特征，一个神经元必然是多义的。",
        "所以，要找到干净的特征，不能只看单个神经元，而要在激活空间里找到正确的方向。下一期的稀疏自编码器，就是专门用来找这些方向的工具。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，当概念比维度多时，模型会把多个特征叠加在同一组维度里。",
        "第二，在玩具模型里，特征越稀疏，叠加越多，五个特征可以排成五边形挤进两维。",
        "第三，叠加让单个神经元变得多义。下一期，我们用稀疏自编码器把叠加拆开。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "叠加", "概念比神经元多的时候", SERIES)


def sc_poly(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "多义神经元")
    d.ellipse([560, 190, 720, 350], outline=AMBER, width=3, fill=(12, 19, 34))
    vk.text_c(d, (640, 270), "神经元", F_BODY, AMBER)
    for (x, t) in ((280, "猫脸"), (1000, "汽车前脸")):
        d.rounded_rectangle([x - 110, 230, x + 110, 310], 12, outline=CYAN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x, 270), t, F_BODY, CYAN)
    vk.arrow(d, (395, 270), (555, 270), MUTED, 3)
    vk.arrow(d, (885, 270), (725, 270), MUTED, 3)
    if seg >= 1:
        vk.text_c(d, (640, 450), "概念远多于神经元 → 多个概念共用同一组维度 = 叠加", F_BODY, TEXT)


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "玩具模型（Elhage et al., 2022）")
    tv.fpaste(c, r"$h=Wx\in\mathbb{R}^2,\qquad x'=\mathrm{ReLU}(W^{\top}h+b),\qquad x\in[0,1]^5$", 640, 200, 32)
    for i in range(5):
        x = 330 + i * 130
        d.rounded_rectangle([x, 270, x + 100, 330], 10, outline=COLS[i], width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 50, 300), f"特征{i + 1}", F_SMALL, COLS[i])
        vk.text_c(d, (x + 50, 355), f"0.9^{i}", vk.font(18, mono=True), MUTED)
    d.text((330, 385), "重要性", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 480), "稀疏度 S：每个特征为 0 的概率 · S = 0 → 0.99", F_BODY, AMBER)


def _panel(d, s, cx, cy, r=110):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(30, 41, 59), width=1)
    d.line([(cx - r, cy), (cx + r, cy)], fill=(30, 41, 59))
    d.line([(cx, cy - r), (cx, cy + r)], fill=(30, 41, 59))
    W = SUP[s]["W"]
    for i in range(5):
        x, y = W[:, i] / 1.2 * r
        if np.hypot(x, y) > 5:
            vk.arrow(d, (cx, cy), (cx + x, cy - y), COLS[i], 3)
    vk.text_c(d, (cx, cy + r + 30), f"S = {s}", vk.font(22, mono=True), TEXT)
    vk.text_c(d, (cx, cy + r + 62), f"{NR[s]} 个特征", F_SMALL, AMBER)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "两维隐藏空间里的特征方向（W 的列）")
    show = [[0.0], [0.0, 0.7], SS][min(seg, 2)]
    for i, s in enumerate(SS):
        if s in show:
            _panel(d, s, 150 + i * 245, 330)
    for i in range(5):
        d.text((150 + i * 150, 560), f"■ 特征{i + 1}", font=F_SMALL, fill=COLS[i])
    msg = ["只表示最重要的 2 个，互相垂直", "4 个：正方形，相对的两个共用一条直线", "5 个：正五边形，相邻 72°"][min(seg, 2)]
    vk.text_c(d, (640, 640), msg, F_BODY, TEXT)


def sc_why(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "为什么稀疏时可以叠加？")
    _panel(d, 0.9, 260, 320, 130)
    d.text((520, 200), "只有一个特征出现 → 在其他方向上的投影 = 干扰", font=F_SMALL, fill=TEXT)
    d.text((520, 250), f"负偏置 b ≈ {SUP[0.9]['b'].mean():.2f} + ReLU → 过滤小干扰", font=F_SMALL, fill=GREEN)
    if seg >= 1:
        d.text((520, 340), "代价：两个特征同时出现 → 互相干扰", font=F_SMALL, fill=RED)
        d.text((520, 390), "越稀疏 → 同时出现越少 → 叠加越划算", font=F_BODY, fill=AMBER)


def sc_implication(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "含义：神经元方向 ≠ 特征方向")
    _panel(d, 0.9, 300, 330, 140)
    d.line([(160, 330), (440, 330)], fill=MUTED, width=3)
    d.text((445, 330), "神经元 1", font=F_SMALL, fill=MUTED, anchor="lm")
    vk.text_c(d, (850, 260), "每个神经元方向都和好几个特征有投影", F_SMALL, TEXT)
    vk.text_c(d, (850, 300), "→ 必然多义", F_BODY, RED)
    if seg >= 1:
        vk.text_c(d, (850, 420), "要在激活空间里找正确的方向", F_BODY, GREEN)
        vk.text_c(d, (850, 470), "→ 下一期：稀疏自编码器", F_SMALL, VIOLET)


def sc_summary(c, u, seg, su):
    tv.summary(c, 6, [("叠加", "概念多于维度时共用空间"), ("稀疏度", "越稀疏叠加越多：5 个挤进 2 维"),
                      ("多义", "单个神经元难以解释")], seg, "下一期：稀疏自编码器")


RENDER = {"title": sc_title, "poly": sc_poly, "toy": sc_toy, "result": sc_result, "why": sc_why, "implication": sc_implication, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "mi-superposition")
