"""Optimization 01 — 梯度下降：在损失地形上走下坡.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] OP_CACHE=<dir> python op01_gd.py <out_dir>

Real computations (opx.gd_quadratic, opx.condition): f(x, y) = 0.5 (x^2 + 10 y^2) from (1, 1), gradient
descent with learning rate 0.05 / 0.15 / 0.19 / 0.21 for 60 steps; steps for f to drop by a factor of 10^6
on f = 0.5 (x^2 / kappa + y^2) with the best constant learning rate 2 / (L + mu), kappa = 10 ... 10,000.
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

G = opx.cached(CACHE, "gdq", opx.gd_quadratic)
RUNS = G["runs"]
CO = opx.cached(CACHE, "cond", opx.condition)
KS = sorted(CO)
GD = [CO[k]["gd"] for k in KS]
print({lr: RUNS[lr]["f"][[10, 60]] for lr in RUNS}, GD)
assert list(G["lam"]) == [1.0, 10.0] and RUNS[0.05]["f"][60] > 1e-3 and RUNS[0.15]["f"][20] < 1e-3
assert RUNS[0.21]["f"][60] > 1e5 and RUNS[0.19]["f"][60] < 1e-4 and 2 / G["lam"].max() == 0.2
assert GD == [35, 346, 3454, 34539]

SCENES = [
    ("title", [
        "欢迎来到深度学习优化系列。训练一个神经网络，本质上就是在一个巨大的损失地形上找低点。这一期从最基本的方法讲起，梯度下降。",
    ]),
    ("idea", [
        "梯度指向函数上升最快的方向。梯度下降就是每一步沿着梯度的反方向，走一小步。走多远由学习率决定，它是训练中最重要的一个超参数。",
    ]),
    ("bowl", [
        "我们用一个最简单的地形来看，一个椭圆形的碗。它在一个方向上很平缓，在另一个方向上陡十倍。从同一个点出发，试四个不同的学习率。",
        "学习率零点零五时，每一步都很稳，但走得太慢，六十步后还没到底。学习率零点一五时，二十步左右就基本到了谷底。",
        "学习率零点一九时，在陡的方向上来回振荡，但振幅越来越小，最终还是收敛了。学习率零点二一时，振荡越来越大，损失一路爆炸。",
    ]),
    ("threshold", [
        "这个界限可以精确算出来。在陡的方向上，每一步会把坐标乘上一减去学习率乘以曲率。当学习率超过二除以最大曲率时，这个乘数的绝对值大于一，就必然发散。这里最大曲率是十，界限正好是零点二。",
        "所以学习率的上限，由地形最陡的方向决定。",
    ]),
    ("cond", [
        "但走得快慢，却受最平缓的方向限制。最陡和最平缓的曲率之比叫条件数。学习率被陡的方向限制住以后，在平缓的方向上就只能慢慢挪。",
        f"我们用最好的固定学习率，算出损失下降一百万倍需要的步数。条件数是十时，需要{int_zh(GD[0])}步；条件数一百时，{int_zh(GD[1])}步；一千时，{big_zh(GD[2])}步；一万时，{big_zh(GD[3])}步。步数和条件数几乎成正比。",
        "神经网络的损失地形，在不同方向上的曲率可以相差非常多倍，这就是为什么单纯的梯度下降训练大模型会很慢。后面几期的方法，都在想办法解决这个问题。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，梯度下降沿着梯度的反方向走，学习率决定步子大小。",
        "第二，学习率超过二除以最大曲率就必然发散，上限由最陡的方向决定。",
        "第三，收敛的步数和条件数成正比，病态的地形会让梯度下降很慢。下一期，我们看随机梯度和批量大小。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "梯度下降", "在损失地形上走下坡", SERIES)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "沿着梯度的反方向走一小步")
    tv.fpaste(c, r"$\theta_{t+1}=\theta_t-\eta\,\nabla f(\theta_t)$", 640, 250, 46)
    vk.text_c(d, (640, 360), "η：学习率 · 最重要的超参数", F_BODY, AMBER)


COL = {0.05: CYAN, 0.15: GREEN, 0.19: AMBER, 0.21: RED}


def _bowl(d, box, lrs, n):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx, sy = (x1 - x0) / 2 / 1.25, (y1 - y0) / 2 / 1.6
    for lev in (0.02, 0.1, 0.25, 0.5, 0.75):
        rx, ry = np.sqrt(2 * lev) * sx, np.sqrt(2 * lev / 10) * sy
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=(40, 52, 72), width=1)
    for lr in lrs:
        P = RUNS[lr]["path"][: n + 1]
        P = P[(np.abs(P[:, 0]) < 1.25) & (np.abs(P[:, 1]) < 1.6)]
        pts = [(cx + p[0] * sx, cy - p[1] * sy) for p in P]
        if len(pts) > 1:
            d.line(pts, fill=COL[lr], width=2)
        for q in pts:
            d.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=COL[lr])


def sc_bowl(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "f = ½(x² + 10y²)，从 (1, 1) 出发")
    lrs = [[], [0.05, 0.15], [0.05, 0.15, 0.19, 0.21]][min(seg, 2)]
    n = 60
    _bowl(d, (90, 150, 690, 620), lrs, n)
    shown = lrs if lrs else []
    if shown:
        series = [(f"η = {lr}", np.clip(RUNS[lr]["f"], 1e-8, 1e6), COL[lr]) for lr in shown]
        tv.line_chart(d, (770, 170, 1200, 560), np.arange(61), series, ylog=True, ylim=(-8, 6), xlabel="步数", ylabel="损失（对数）",
                      legend_xy=(790, 420), xticks=[(0, "0"), (30, "30"), (60, "60")])


def sc_threshold(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "发散的界限：η < 2 / L")
    tv.fpaste(c, r"$y_{t+1}=(1-\eta\,\lambda)\,y_t,\qquad |1-\eta\lambda|<1\ \Leftrightarrow\ \eta<\frac{2}{\lambda}$", 640, 230, 36)
    vk.text_c(d, (640, 340), "最大曲率 L = 10 → η < 0.2", F_BODY, AMBER)
    for i, lr in enumerate([0.05, 0.15, 0.19, 0.21]):
        m = 1 - lr * 10
        vk.text_c(d, (250 + i * 260, 440), f"η = {lr}", vk.font(26, mono=True), COL[lr])
        vk.text_c(d, (250 + i * 260, 485), f"乘数 {m:+.1f}", F_SMALL, TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 600), "学习率的上限由最陡的方向决定", F_BODY, RED)


def sc_cond(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "条件数 κ = 最大曲率 / 最小曲率")
    vk.text_c(d, (640, 160), "η 被陡的方向限制 → 在平缓的方向上慢慢挪", F_SMALL, MUTED)
    if seg >= 1:
        tv.bar_chart(d, (200, 240, 1080, 560), [f"κ = {k:,}" for k in KS], [np.log10(g) for g in GD], [CYAN, CYAN, AMBER, RED], fmt="", vmax=5.2)
        for i, g in enumerate(GD):
            x = 200 + i * 220 + 110
            top = 560 - np.log10(g) / 5.2 * 320
            d.text((x, top - 18), f"{g:,} 步", font=F_SMALL, fill=TEXT, anchor="mm")
        d.text((200, 200), "损失下降 100 万倍所需步数（最佳固定 η，对数刻度）", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        vk.text_c(d, (640, 640), "神经网络：不同方向曲率相差悬殊 → 梯度下降很慢", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("梯度下降", "沿负梯度走，η 决定步子"), ("上限", "η > 2/L 必然发散"),
                      ("条件数", "步数 ∝ κ，病态地形很慢")], seg, "下一期：随机梯度与批量大小")


RENDER = {"title": sc_title, "idea": sc_idea, "bowl": sc_bowl, "threshold": sc_threshold, "cond": sc_cond, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "op-gd")
