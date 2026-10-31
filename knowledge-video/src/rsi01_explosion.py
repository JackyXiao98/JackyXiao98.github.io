"""Recursive Self-Improvement 01 — 什么是递归自我改进：从“智能爆炸”说起.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python rsi01_explosion.py <out_dir>

Real computations: the toy growth law dI/dt = I^alpha (I(0) = 1) for alpha = 0.5, 1, 1.5, integrated
numerically (RK4) and checked against the closed forms; the finite blow-up time for alpha > 1.
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
SERIES = "Recursive Self-Improvement"

ALPHAS = [0.5, 1.0, 1.5]
T_END, DT = 1.9, 1e-4


def rk4(alpha):
    f = lambda I: I ** alpha
    I, out = 1.0, [1.0]
    for _ in range(int(round(T_END / DT))):
        k1 = f(I)
        k2 = f(I + DT / 2 * k1)
        k3 = f(I + DT / 2 * k2)
        k4 = f(I + DT * k3)
        I += DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(I)
    return np.array(out)


TS = np.linspace(0, T_END, int(round(T_END / DT)) + 1)
CURVES = {a: rk4(a) for a in ALPHAS}
CLOSED = {0.5: (1 + TS / 2) ** 2, 1.0: np.exp(TS), 1.5: (1 - TS / 2) ** -2}
END = {a: float(CURVES[a][-1]) for a in ALPHAS}
T_BLOW = 1 / (1.5 - 1)
for a in ALPHAS:
    assert np.allclose(CURVES[a], CLOSED[a], rtol=1e-6)
assert abs(END[0.5] - 3.8025) < 1e-3 and abs(END[1.0] - 6.686) < 1e-3 and abs(END[1.5] - 400) < 0.01 and T_BLOW == 2

SCENES = [
    ("title", [
        "欢迎来到递归自我改进系列。这一期从一个六十年前的设想说起，看看人工智能改进自己，会以什么样的速度发生。",
    ]),
    ("good", [
        "一九六五年，数学家古德写道，设想一台在所有智力活动上都超过人类的机器。设计机器也是一种智力活动，所以它能设计出更好的机器。",
        "更好的机器又能设计出再好一点的机器，这样循环下去，就会出现所谓的智能爆炸。这就是递归自我改进的基本想法，改进者本身也在被改进。",
    ]),
    ("model", [
        "这个循环到底会不会爆炸，取决于一个关键问题，能力每提高一点，改进的速度提高多少。",
        "我们用一个最简单的模型来描述。能力的增长速度，等于能力的阿尔法次方。阿尔法刻画了改进的收益，小于一说明收益递减，等于一说明收益不变，大于一说明收益递增。",
    ]),
    ("curves", [
        "我们从能力等于一出发，数值求解三种情况，并和精确解做了核对。",
        f"阿尔法等于零点五时，到时间一点九，能力只增长到{dec_zh(END[0.5], 1)}，是多项式增长。阿尔法等于一时是指数增长，达到{dec_zh(END[1.0], 1)}。",
        f"阿尔法等于一点五时，能力已经达到{int_zh(round(END[1.5]))}，而且在时间等于二时趋于无穷，这就是有限时间内的爆炸。三条曲线只差在一个指数上，结局却完全不同。",
    ]),
    ("questions", [
        "所以真正的问题，不是人工智能能不能改进自己，而是改进的收益是递增还是递减，以及有没有别的瓶颈拖慢它。",
        "接下来的几期，我们会依次看到算力和数据的瓶颈，自我训练、自我博弈和验证者这些具体的改进方法，以及模型崩溃、奖励黑客等失效模式。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，递归自我改进是指改进者本身也被改进，形成一个循环。",
        "第二，在简单的增长模型里，收益递减、收益不变和收益递增，分别对应多项式增长、指数增长和有限时间爆炸。",
        "第三，现实中的关键，是改进的收益和瓶颈。下一期，我们看瓶颈在哪里。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "什么是递归自我改进", "从“智能爆炸”说起", SERIES)


def sc_good(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "I. J. Good（1965）")
    vk.text_c(d, (640, 160), "“第一台超智能机器，是人类需要做的最后一项发明”", F_BODY, TEXT)
    vk.text_c(d, (640, 200), "（大意，Good, 1965）", F_SMALL, MUTED)
    if seg >= 1:
        cx, cy, r = 640, 420, 140
        labs = [("机器", CYAN), ("设计更好的机器", VIOLET), ("更好的机器", GREEN), ("改进者也被改进", AMBER)]
        for i, (t, col) in enumerate(labs):
            a = -np.pi / 2 + i * np.pi / 2
            x, y = cx + r * np.cos(a), cy + r * np.sin(a)
            vk.text_c(d, (x, y), t, F_SMALL, col)
            a2 = a + np.pi / 2
            vk.arrow(d, (cx + r * 0.75 * np.cos(a + 0.45), cy + r * 0.75 * np.sin(a + 0.45)),
                     (cx + r * 0.75 * np.cos(a2 - 0.45), cy + r * 0.75 * np.sin(a2 - 0.45)), MUTED, 2)


def sc_model(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "一个最简单的增长模型")
    tv.fpaste(c, r"$\frac{dI}{dt}=I^{\alpha},\qquad I(0)=1$", 640, 200, 40)
    if seg >= 1:
        for i, (t, sub, col) in enumerate([("α < 1", "收益递减", GREEN), ("α = 1", "收益不变", CYAN), ("α > 1", "收益递增", RED)]):
            x = 200 + i * 300
            d.rounded_rectangle([x, 330, x + 260, 440], 14, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 130, 365), t, vk.font(30), col)
            vk.text_c(d, (x + 130, 410), sub, F_SMALL, TEXT)


def sc_curves(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "数值求解（RK4，与精确解一致）")
    k = 1 if seg == 0 else 3
    step = 100
    series = [(f"α = {a}", CURVES[a][::step], col) for a, col in zip(ALPHAS, [GREEN, CYAN, RED])][:k if seg < 2 else 3]
    if seg == 1:
        series = series[:2]
    tv.line_chart(d, (140, 160, 760, 560), TS[::step], series, ylog=True, ylim=(0, np.log10(500)), xlabel="时间 t",
                  ylabel="能力 I（对数）", legend_xy=(160, 220), xticks=[(0, "0"), (0.5, "0.5"), (1, "1"), (1.5, "1.5"), (1.9, "1.9")])
    rows = [(0.5, r"$(1+t/2)^2$", GREEN), (1.0, r"$e^{t}$", CYAN), (1.5, r"$(1-t/2)^{-2}$", RED)]
    for i, (a, tex, col) in enumerate(rows[: (2 if seg == 1 else 3) if seg >= 1 else 1]):
        y = 200 + i * 110
        tv.fpaste(c, tex, 870, y, 28, "#%02x%02x%02x" % col)
        d.text((1000, y), f"I(1.9) = {END[a]:.1f}", font=vk.font(22, mono=True), fill=col, anchor="lm")
    if seg >= 2:
        vk.text_c(d, (1000, 560), "α = 1.5：t → 2 时 I → ∞", F_BODY, RED)


def sc_questions(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "真正的问题")
    vk.text_c(d, (640, 200), "改进的收益：递增还是递减？", vk.font(34), AMBER)
    vk.text_c(d, (640, 260), "有没有别的瓶颈拖慢它？", vk.font(34), AMBER)
    if seg >= 1:
        items = ["02 瓶颈", "03 自训练", "04 自我博弈", "05 验证者", "06 失效模式", "07 改进代码", "08 最新进展", "09 安全"]
        for i, t in enumerate(items):
            x, y = 170 + (i % 4) * 245, 360 + (i // 4) * 80
            d.rounded_rectangle([x, y, x + 215, y + 60], 10, outline=(30, 41, 59), width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 107, y + 30), t, F_SMALL, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("定义", "改进者本身也被改进"), ("增长模型", "α<1 多项式 · α=1 指数 · α>1 爆炸"),
                      ("关键", "改进的收益与瓶颈")], seg, "下一期：反馈回路 —— 瓶颈在哪里")


RENDER = {"title": sc_title, "good": sc_good, "model": sc_model, "curves": sc_curves, "questions": sc_questions,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rsi-explosion")
