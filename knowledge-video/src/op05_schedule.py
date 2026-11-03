"""Optimization 05 — 学习率调度：预热、余弦与 WSD.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] OP_CACHE=<dir> TF_CACHE=<dir> python op05_schedule.py <out_dir>

Real computations (JAX):
- opx.schedules: a 2-layer Transformer (d = 64) learning a 12-token second-order Markov language (sequences of
  10 tokens; loss on positions 2-9; floor = the teacher's own entropy on the test set), Adam, peak lr 1e-2,
  batch 64, 1,000 steps, 50 warmup steps; constant / cosine to 0 / WSD (linear decay to 0 over the last 20%).
- opx.postln: 6-layer pre-LN addition Transformer, Adam lr 3e-3, 800 steps, no warmup vs 200 warmup steps;
  steps until test exact-match >= 99%.
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

SC = opx.cached(CACHE, "sched", opx.schedules)
FLOOR = SC["constant"]["floor"]
H = {k: SC[k]["hist"] for k in SC}
ST = H["constant"][:, 0]
L = {k: H[k][:, 1] for k in H}
at = lambda k, s: float(L[k][list(ST).index(s)])
PL = opx.cached(CACHE, "postln", opx.postln)
W0, W200 = opx._first(PL[(False, 0)]["hist"]), opx._first(PL[(False, 200)]["hist"])
print(FLOOR, {k: (at(k, 800), at(k, 1000)) for k in L}, W0, W200)
assert abs(FLOOR - 1.0535) < 1e-3 and at("wsd", 800) == at("constant", 800) and abs(at("constant", 1000) - 1.097) < 1e-3
assert abs(at("cosine", 1000) - 1.0668) < 1e-3 and abs(at("wsd", 1000) - 1.0705) < 1e-3 and abs(at("wsd", 800) - 1.1014) < 1e-3
assert W0 == 440 and W200 == 280


def lr_curve(kind, n=1000, peak=1e-2, w=50):
    t = np.arange(n + 1)
    warm = np.minimum(t / w, 1.0)
    if kind == "constant":
        return peak * warm
    if kind == "cosine":
        return np.where(t < w, peak * t / w, peak * 0.5 * (1 + np.cos(np.pi * (t - w) / (n - w))))
    dec = int(0.2 * n)
    return np.where(t < w, peak * t / w, np.where(t < n - dec, peak, peak * (n - t) / dec))


SCENES = [
    ("title", [
        "前面我们都用一个固定的学习率。实际训练大模型时，学习率会随着训练不断变化。这一期看学习率调度，预热、余弦退火，以及近年流行的预热稳定衰减。",
    ]),
    ("why", [
        "学习率大，走得快，但随机梯度的噪声会让参数在低谷附近来回晃，降不到最低。学习率小，晃得少，但走得慢。所以一个自然的想法是，先大后小。",
    ]),
    ("warmup", [
        "但在训练的最开始，通常反过来，先用很小的学习率，在几百或几千步里逐渐升到最大值，这叫预热。原因是一开始参数是随机的，梯度很大且不稳定，亚当的二阶矩估计也还不准，直接用大学习率容易训练崩溃或者走偏。",
        f"我们在一个六层的加法模型上比较。不预热时，第{int_zh(W0)}步达到百分之九十九；用两百步预热时，第{int_zh(W200)}步就达到了。",
    ]),
    ("schedules", [
        "预热之后怎样下降？最常见的是余弦退火，学习率沿着余弦曲线，从最大值平滑地降到接近零。它的缺点是必须事先知道总的训练步数。",
        "近年流行的另一种是预热稳定衰减。预热以后保持一个恒定的学习率，直到最后一小段，比如最后百分之二十，再快速降到零。",
    ]),
    ("experiment", [
        f"我们训练一个小变换器去学习一种十二个词元的人造语言，它的损失有一个无法再低的下限，约{dec_zh(FLOOR, 3)}。三种调度都训练一千步。",
        f"恒定学习率最后停在{dec_zh(at('constant', 1000), 3)}，噪声让它降不下去。余弦退火最后是{dec_zh(at('cosine', 1000), 3)}。",
        f"预热稳定衰减在前八百步和恒定学习率完全一样，到了衰减阶段，损失迅速下降，从{dec_zh(at('wsd', 800), 3)}降到{dec_zh(at('wsd', 1000), 3)}，几乎追上余弦。",
    ]),
    ("wsd", [
        "预热稳定衰减的好处是灵活。稳定阶段可以一直训练下去，在任何时候想要一个模型，就从当前的检查点分出一段衰减，不需要事先定好总步数，也方便在训练中继续加数据。",
        "衰减时损失的迅速下降，也说明了前面的高学习率阶段，模型一直被噪声挡在低谷上方。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，学习率先大后小，大的阶段走得快，小的阶段降噪。",
        "第二，预热让训练开头更稳定，在我们的实验里也更快。",
        "第三，预热稳定衰减在最后一段才降学习率，效果接近余弦，却更灵活。下一期，我们看初始化和归一化。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "学习率调度", "预热、余弦与 WSD", SERIES)


def sc_why(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "先大后小")
    for i, (a, b, col) in enumerate([("学习率大", "走得快 · 在低谷附近来回晃", RED), ("学习率小", "晃得少 · 走得慢", CYAN)]):
        x = 140 + i * 540
        d.rounded_rectangle([x, 220, x + 460, 340], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 230, 260), a, F_BODY, col)
        vk.text_c(d, (x + 230, 305), b, F_SMALL, TEXT)
    vk.text_c(d, (640, 450), "→ 先用大学习率，再逐渐减小", F_BODY, AMBER)


def sc_warmup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "预热（warmup）")
    tv.line_chart(d, (140, 180, 620, 450), np.arange(1001), [("学习率", lr_curve("constant") * 100, AMBER)], ylim=(0, 1.2),
                  xlabel="步数", legend_xy=(400, 300), xticks=[(0, "0"), (500, "500"), (1000, "1000")])
    d.text((700, 200), "开头：参数随机、梯度大且不稳定", font=F_SMALL, fill=TEXT)
    d.text((700, 240), "Adam 的二阶矩估计还不准", font=F_SMALL, fill=TEXT)
    d.text((700, 280), "→ 大学习率容易崩溃或走偏", font=F_SMALL, fill=RED)
    if seg >= 1:
        tv.bar_chart(d, (700, 380, 1180, 600), ["不预热", "预热 200 步"], [W0, W200], [RED, GREEN], fmt="{:.0f}", vmax=520)
        d.text((700, 340), "6 层加法模型：达到 99% 的步数", font=F_SMALL, fill=MUTED)


def sc_schedules(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "余弦退火 vs 预热-稳定-衰减（WSD）")
    series = [("余弦", lr_curve("cosine") * 100, CYAN)] + ([("WSD（最后 20% 衰减）", lr_curve("wsd") * 100, GREEN)] if seg >= 1 else [])
    tv.line_chart(d, (160, 180, 1120, 540), np.arange(1001), series, ylim=(0, 1.2), xlabel="步数", ylabel="学习率（× 0.01）",
                  legend_xy=(240, 470), xticks=[(0, "0"), (500, "500"), (800, "800"), (1000, "1000")])
    vk.text_c(d, (640, 610), "余弦：必须事先知道总步数" if seg == 0 else "WSD：恒定，直到最后一小段才快速降到 0", F_BODY, AMBER)


COLS = {"constant": RED, "cosine": CYAN, "wsd": GREEN}
NAMES = {"constant": "恒定", "cosine": "余弦", "wsd": "WSD"}


def sc_experiment(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "12 个词元的人造语言 · Adam · 1000 步")
    keys = [["constant", "cosine"], ["constant", "cosine"], ["constant", "cosine", "wsd"]][min(seg, 2)]
    sel = ST >= 200
    series = [(NAMES[k], L[k][sel], COLS[k]) for k in keys]
    if seg >= 1:
        tv.line_chart(d, (140, 170, 800, 580), ST[sel], series, ylim=(1.04, 1.2), xlabel="步数", ylabel="测试损失",
                      legend_xy=(860, 560), xticks=[(s, str(s)) for s in (200, 400, 600, 800, 1000)])
        y = 580 - (FLOOR - 1.04) / 0.16 * 410
        d.line([(140, y), (800, y)], fill=MUTED, width=1)
        d.text((150, y - 26), f"下限 {FLOOR:.3f}", font=F_SMALL, fill=MUTED)
        for i, k in enumerate(keys):
            d.text((860, 250 + i * 50), NAMES[k], font=F_SMALL, fill=COLS[k])
            d.text((1180, 250 + i * 50), f"{at(k, 1000):.3f}", font=vk.font(24, mono=True), fill=COLS[k], anchor="ra")
        d.text((860, 210), "第 1000 步", font=F_SMALL, fill=MUTED)
    else:
        vk.text_c(d, (640, 350), f"损失下限（老师的熵）≈ {FLOOR:.3f}", F_BODY, MUTED)
    if seg >= 2:
        d.text((860, 450), f"WSD 第 800 步：{at('wsd', 800):.3f}", font=F_SMALL, fill=GREEN)
        d.text((860, 490), "衰减阶段迅速下降", font=F_SMALL, fill=GREEN)


def sc_wsd(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "WSD 的灵活性（MiniCPM, 2024; Hägele et al., 2024）")
    t = np.arange(1001)
    main = np.where(t < 50, t / 50, 1.0)
    tv.line_chart(d, (160, 180, 1120, 470), t, [("稳定阶段", main, CYAN)], ylim=(0, 1.2), xlabel="步数")
    for start, col in ((400, GREEN), (700, AMBER)):
        xs = np.arange(start, start + 101)
        x0 = 160 + start / 1000 * 960
        x1 = 160 + (start + 100) / 1000 * 960
        d.line([(x0, 470 - 1 / 1.2 * 290), (x1, 470)], fill=col, width=3)
    vk.text_c(d, (640, 530), "随时从检查点分出一段衰减 · 不必事先定好总步数", F_BODY, GREEN)
    if seg >= 1:
        vk.text_c(d, (640, 600), "衰减时的骤降：高学习率阶段一直被噪声挡在低谷上方", F_SMALL, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 6, [("先大后小", "大的走得快，小的降噪"), ("预热", f"6 层模型：{W0} → {W200} 步"),
                      ("WSD", "接近余弦，更灵活")], seg, "下一期：初始化与归一化")


RENDER = {"title": sc_title, "why": sc_why, "warmup": sc_warmup, "schedules": sc_schedules, "experiment": sc_experiment, "wsd": sc_wsd, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "op-schedule")
