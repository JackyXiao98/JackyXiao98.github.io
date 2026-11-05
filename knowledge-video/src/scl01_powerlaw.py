"""Scaling Laws 01 — 什么是缩放定律：幂律与对数坐标.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python scl01_powerlaw.py <out_dir>

Computations: an illustrative power law y = 10 x^-0.5 on linear and log-log axes (labelled as an example); the
parametric fit reported by Hoffmann et al. 2022 (Chinchilla, approach 3): L(N, D) = E + A / N^a + B / D^b with
E = 1.69, A = 406.4, B = 410.7, a = 0.34, b = 0.28, evaluated for N = 1e8 ... 1e11 parameters at D = 20 N tokens:
the reducible part shrinks by a constant factor per 10x, the irreducible E stays.
Facts about real systems are quoted with references in the notes.
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
SERIES = "Scaling Laws"

E, A, B, a, b = 1.69, 406.4, 410.7, 0.34, 0.28
Lf = lambda N, D: E + A / N ** a + B / D ** b
NS = [1e8, 1e9, 1e10, 1e11]
LOSS = {N: Lf(N, 20 * N) for N in NS}
RED_ = {N: LOSS[N] - E for N in NS}
RATIO = RED_[1e9] / RED_[1e10]
X = np.logspace(0, 3, 40)
Y = 10 * X ** -0.5
print({N: round(LOSS[N], 3) for N in NS}, {N: round(RED_[N], 3) for N in NS}, RATIO)
assert [round(LOSS[N], 2) for N in NS] == [3.49, 2.58, 2.13, 1.91]
assert 1.95 < RATIO < 2.05 and round(Y[-1], 2) == 0.32

SCENES = [
    ("title", [
        "过去几年，大模型越做越大，一个重要的原因是人们发现：模型的损失随着规模的增长，会以一种非常规律的方式下降。这个规律叫缩放定律。这个系列我们一起在小模型上看它是怎么来的、能用来做什么、又有哪些局限。",
    ]),
    ("powerlaw", [
        "缩放定律说的是幂律：损失约等于某个常数，乘以规模的负几次方。比如这个例子，规模每增加一百倍，结果变成原来的十分之一。",
        "在普通坐标里，这是一条一开始急剧下降、后来越来越平的曲线，看起来好像很快就没有收益了。但如果横轴和纵轴都取对数，它就变成了一条直线，直线的斜率就是那个指数。",
    ]),
    ("loglog", [
        "对数坐标的好处是，它让我们看清楚每一次十倍增长带来的收益。只要直线不弯，规模每扩大十倍，损失就按同样的比例下降。这让人们可以用小模型的结果，画出直线，去预测大得多的模型。",
    ]),
    ("irreducible", [
        "实际的损失不会一直降到零。语言本身有不可预测的部分，比如同一句话的下一个词本来就有好几种合理的说法。所以缩放定律通常写成：一个不可约的常数，加上随规模下降的幂律项。",
        f"以龙猫论文拟合的公式为例，假设每个参数配二十个训练词元，模型从一亿参数长到十亿、百亿、千亿，损失从约{dec_zh(LOSS[1e8], 2)}降到{dec_zh(LOSS[1e9], 2)}、{dec_zh(LOSS[1e10], 2)}、{dec_zh(LOSS[1e11], 2)}。可约的部分每十倍缩小约一半，而一点六九的常数部分始终都在。",
    ]),
    ("plan", [
        "这个系列接下来会在莎士比亚文本的小模型上，分别看参数量、数据量和计算量的缩放，计算最优的分配，用小模型外推大模型，超参数怎样随规模变化，数据不够时怎么办，所谓的涌现能力，以及推理时的计算缩放。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，缩放定律是幂律：规模每扩大十倍，损失按固定比例下降。",
        "第二，在对数坐标里幂律是一条直线，斜率就是指数。",
        "第三，实际损失还有一个不可约的常数部分。下一期，我们在小模型上测量参数量的缩放。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "什么是缩放定律", "幂律与对数坐标", SERIES)


def sc_powerlaw(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "幂律：y = C · x^(-α)（示例：y = 10 · x^(-0.5)）")
    tv.line_chart(d, (120, 200, 600, 560), X, [("线性坐标", Y, CYAN)], xticks=[(1, "1"), (500, "500"), (1000, "1000")], xlabel="规模")
    d.text((120, 165), "普通坐标", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        tv.line_chart(d, (700, 200, 1180, 560), X, [("对数坐标", Y, GREEN)], xlog=True, ylog=True,
                      xticks=[(1, "1"), (10, "10"), (100, "100"), (1000, "1000")], xlabel="规模（对数）")
        d.text((700, 165), "双对数坐标：直线，斜率 = -α", font=F_SMALL, fill=GREEN)


def sc_loglog(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "每十倍，同样的比例")
    tv.line_chart(d, (160, 200, 820, 580), X, [("y", Y, GREEN)], xlog=True, ylog=True,
                  xticks=[(1, "1"), (10, "10"), (100, "100"), (1000, "1000")], xlabel="规模（对数）")
    for v in (1, 10, 100, 1000):
        d.text((900, 220 + int(np.log10(v)) * 60), f"x = {v:<5} y = {10 * v ** -0.5:.2f}", font=vk.font(24, mono=True), fill=TEXT)
    vk.text_c(d, (1010, 520), "小模型画出直线 → 外推大模型", F_SMALL, AMBER)


def sc_irreducible(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "不可约损失 + 幂律项")
    tv.fpaste(c, r"$L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$", 640, 190, 44)
    if seg >= 1:
        d.text((140, 270), "Chinchilla 拟合：E = 1.69，A = 406.4，B = 410.7，α = 0.34，β = 0.28；D = 20 N", font=F_SMALL, fill=MUTED)
        labs = ["1 亿", "10 亿", "100 亿", "1000 亿"]
        x0, y0, x1, y1 = 180, 330, 900, 620
        w = (x1 - x0) / 4
        for i, N in enumerate(NS):
            hE = E / 3.9 * (y1 - y0)
            hR = RED_[N] / 3.9 * (y1 - y0)
            xa = x0 + i * w + 30
            d.rectangle([xa, y1 - hE, xa + w - 60, y1], fill=MUTED)
            d.rectangle([xa, y1 - hE - hR, xa + w - 60, y1 - hE], fill=GREEN)
            d.text((xa + (w - 60) / 2, y1 - hE - hR - 16), f"{LOSS[N]:.2f}", font=vk.font(20, mono=True), fill=TEXT, anchor="mm")
            d.text((xa + (w - 60) / 2, y1 + 20), labs[i], font=F_SMALL, fill=MUTED, anchor="mm")
        d.text((960, 430), "■ 可约部分", font=F_SMALL, fill=GREEN)
        d.text((960, 480), "■ 不可约 E = 1.69", font=F_SMALL, fill=MUTED)


def sc_plan(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "这个系列")
    items = ["参数量", "数据量", "计算最优", "外推", "超参数", "数据受限", "涌现", "推理时计算"]
    for i, t in enumerate(items):
        x = 160 + (i % 4) * 250
        y = 230 + (i // 4) * 130
        d.rounded_rectangle([x, y, x + 210, y + 80], 14, outline=[CYAN, GREEN, AMBER, VIOLET][i % 4], width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 105, y + 40), f"{i + 2:02d} {t}", F_SMALL, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("幂律", "每十倍，按固定比例下降"), ("对数坐标", "直线，斜率是指数"),
                      ("不可约", "损失还有常数下限")], seg, "下一期：参数量的缩放")


RENDER = {"title": sc_title, "powerlaw": sc_powerlaw, "loglog": sc_loglog, "irreducible": sc_irreducible, "plan": sc_plan, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "scl-powerlaw")
