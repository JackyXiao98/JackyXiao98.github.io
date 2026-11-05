"""State Space Models 04 — 长程记忆：特征值与衰减.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python ssm04_memory.py <out_dir>

Computations (numpy): a diagonal state h_t = a h_{t-1} + b u_t keeps a^L of an input after L steps; half-life
ln 2 / (-ln a). For a discretised channel a = exp(-dt * n), n = 1..16 (A = -(1..16) as in the series' toy layers):
(1) memory after 10 / 100 / 1,000 steps for a = 0.5, 0.9, 0.99, 0.999; (2) over the 128 x 16 channels of one
layer, the share whose half-life exceeds 100 / 1,000 steps for step sizes initialised log-uniformly in
[0.001, 0.1] (the usual S4D / Mamba choice, same grid as the toy layers) vs in [0.5, 1].
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
SERIES = "State Space Models"

AS = [0.5, 0.9, 0.99, 0.999]
KEEP = {a: {L: a ** L for L in (10, 100, 1000)} for a in AS}
HALF = {a: np.log(2) / -np.log(a) for a in AS}
n = np.arange(1, 17)


def halflives(lo, hi, E=128):
    dt = np.exp(np.random.default_rng(0).uniform(np.log(lo), np.log(hi), E))
    return (np.log(2) / (dt[:, None] * n[None])).ravel()


HS = halflives(1e-3, 1e-1)
HF = halflives(0.5, 1.0)
SHARE = {k: {L: float((h > L).mean()) for L in (100, 1000)} for k, h in (("slow", HS), ("fast", HF))}
print(KEEP, HALF, SHARE, np.median(HS), np.median(HF))
assert round(HALF[0.99]) == 69 and round(HALF[0.999]) == 693 and round(KEEP[0.99][100], 3) == 0.366 and KEEP[0.9][100] < 1e-4
assert SHARE["fast"][100] == 0 and round(SHARE["slow"][100], 3) == 0.067 and round(HS.max()) == 684

SCENES = [
    ("title", [
        "传统的循环网络有一个老问题：记不住很久以前的事。状态空间模型能处理很长的序列，关键在于它的状态是怎样衰减的。这一期看状态矩阵的特征值，怎样决定一个模型能记多久。",
    ]),
    ("decay", [
        "把状态矩阵对角化以后，每个状态维度都在做同一件事：每一步把旧的值乘以一个系数，再加上新的输入。一个输入经过很多步以后，还剩下这个系数的步数次方。",
        f"系数是零点九时，一百步后只剩十万分之几，几乎全忘了；系数是零点九九时，一百步后还剩约{pct_zh(KEEP[0.99][100])}，半衰期约{int_zh(round(HALF[0.99]))}步；系数是零点九九九时，半衰期约{int_zh(round(HALF[0.999]))}步。系数越接近一，记忆越长。",
    ]),
    ("vanish", [
        "这也解释了循环网络训练时的梯度消失：梯度往回传播时同样要乘以这些系数，系数小于一，梯度就指数级衰减，模型学不到远处的依赖；系数大于一，又会爆炸。",
        "状态空间模型通过参数化保证系数始终在零和一之间：系数等于负的衰减率乘以步长再取指数。这样状态永远不会爆炸，而记忆的长短由步长和衰减率决定。",
    ]),
    ("init", [
        f"初始化因此非常重要。我们的小模型里，每层有一百二十八个通道、每个通道十六个状态维度。如果把步长初始化在零点零零一到零点一之间，约{pct_zh(SHARE['slow'][100])}的状态维度半衰期超过一百步，最长的约{int_zh(round(HS.max()))}步；如果初始化在零点五到一之间，没有一个能记住一百步，最长的半衰期还不到两步。",
        "所以模型同时拥有从几步到上千步的各种时间尺度，可以从中挑选需要的。结构化状态空间模型最初用一种叫做河马的特殊初始化，从数学上保证状态能近似地记住整段历史，后来发现对角的简化版本配合合适的步长范围，也能做得很好。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，每个状态维度的衰减系数决定记忆的长短，系数越接近一，记得越久。",
        "第二，参数化保证系数在零和一之间，避免梯度爆炸。",
        "第三，用跨越多个数量级的步长初始化，让模型同时拥有短期和长期的记忆。下一期，我们看曼巴的选择性。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "长程记忆", "特征值与衰减", SERIES)


def sc_decay(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "输入经过 L 步后还剩 a^L")
    Ls = np.arange(0, 1001, 10)
    cols = [RED, AMBER, CYAN, GREEN]
    ser = [(f"a = {a}", a ** Ls, cols[i]) for i, a in enumerate(AS)]
    tv.line_chart(d, (140, 200, 780, 580), Ls, ser, ylim=(0, 1), xticks=[(0, "0"), (500, "500"), (1000, "1000")], xlabel="步数 L", legend_xy=(840, 220))
    d.text((140, 165), "剩下的比例", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        f = vk.font(22, mono=True)
        d.text((840, 380), "a       半衰期", font=F_SMALL, fill=MUTED)
        for i, a in enumerate(AS):
            d.text((840, 415 + i * 38), f"{a:<7} {HALF[a]:>6.0f} 步", font=F_SMALL, fill=cols[i])


def sc_vanish(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "梯度也乘以同样的系数")
    vk.text_c(d, (640, 210), "a < 1：梯度指数衰减（学不到远处）  ·  a > 1：爆炸", F_BODY, RED)
    if seg >= 1:
        tv.fpaste(c, r"$a=e^{-\Delta\cdot\lambda},\quad \Delta>0,\ \lambda>0\ \Rightarrow\ 0<a<1$", 640, 340, 40)
        vk.text_c(d, (640, 450), "记忆长短由步长 Δ 与衰减率 λ 决定", F_BODY, GREEN)


def sc_init(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "128 个通道 × 16 个状态维度的半衰期")
    edges = np.logspace(-1, 4, 26)
    for k, (h, col, lab, y0) in enumerate(((HS, GREEN, "步长 ∈ [0.001, 0.1]", 200), (HF, RED, "步长 ∈ [0.5, 1]", 410))):
        cnt, _ = np.histogram(h, edges)
        for i, v in enumerate(cnt):
            x = 140 + i * 26
            hgt = v / cnt.max() * 150 if cnt.max() else 0
            d.rectangle([x, y0 + 160 - hgt, x + 22, y0 + 160], fill=col)
        d.text((140, y0 - 30), lab, font=F_SMALL, fill=col)
        d.text((860, y0 + 60), f"> 100 步：{SHARE['slow' if k == 0 else 'fast'][100]:.0%}", font=F_SMALL, fill=col)
        d.text((860, y0 + 100), f"最长：{(HS if k == 0 else HF).max():.0f} 步", font=F_SMALL, fill=col)
    for v in (1, 10, 100, 1000):
        x = 140 + (np.log10(v) + 1) / 5 * 26 * 25
        d.text((x, 600), str(v), font=vk.font(18, mono=True), fill=MUTED, anchor="mm")
    d.text((800, 625), "半衰期（步，对数）", font=F_SMALL, fill=MUTED, anchor="rm")
    if seg >= 1:
        d.text((860, 370), "S4：HiPPO 初始化", font=F_SMALL, fill=AMBER)
        d.text((860, 405), "S4D / Mamba：对角 + 步长范围", font=F_SMALL, fill=AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("衰减系数", "越接近 1，记得越久"), ("参数化", "保证 0 < a < 1"),
                      ("初始化", "跨越多个数量级的时间尺度")], seg, "下一期：选择性")


RENDER = {"title": sc_title, "decay": sc_decay, "vanish": sc_vanish, "init": sc_init, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "ssm-memory")
