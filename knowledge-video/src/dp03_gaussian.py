"""Differential Privacy 03 — 高斯机制与 (ε, δ)-差分隐私.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python dp03_gaussian.py <out_dir>

Real computations: privacy-loss curves of the Laplace and Gaussian mechanisms; the exact
(eps, delta) curve of the Gaussian mechanism (Balle & Wang, 2018) and the noise it needs versus the
classic calibration; a "release each record with probability delta" mechanism on 1,000,000 people;
per-coordinate noise for a 100-dimensional query (L1 vs L2 sensitivity).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw
from scipy.optimize import brentq

import dpx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Differential Privacy"

# 1. privacy loss ln p_D(x)/p_D'(x), neighbours 0 vs 1, Laplace b = 1 and Gaussian sigma = 1
XS = np.linspace(-3, 4, 400)
LOSS_LAP = np.abs(XS - 1) - np.abs(XS)          # ln of Lap(0,1)/Lap(1,1) densities
LOSS_GAU = ((XS - 1) ** 2 - XS ** 2) / 2        # ln of N(0,1)/N(1,1) densities
assert np.abs(LOSS_LAP).max() <= 1 + 1e-12 and np.abs(LOSS_GAU).max() > 3

# 2. calibration for eps = 1, delta = 1e-5, sensitivity 1
EPS, DELTA = 1.0, 1e-5
SIG_CLASSIC = float(dpx.classic_sigma(EPS, DELTA))
SIG_EXACT = brentq(lambda s: dpx.gaussian_delta(EPS, s) - DELTA, 0.5, 20)
SAVE = 1 - SIG_EXACT / SIG_CLASSIC
EPS_GRID = np.linspace(0.05, 1.5, 60)
DELTA_CURVE = dpx.gaussian_delta(EPS_GRID, SIG_EXACT)
assert DELTA_CURVE.min() > 1e-9
print(f"sigma classic {SIG_CLASSIC:.2f} exact {SIG_EXACT:.2f} save {SAVE:.1%}")
assert abs(SIG_CLASSIC - 4.84) < 0.01 and abs(SIG_EXACT - 3.73) < 0.01

# 3. why delta must be << 1/n: release each record with probability delta
N_POP, D_BAD = 1_000_000, 1e-3
LEAKED = int(np.random.default_rng(0).binomial(N_POP, D_BAD))
assert 900 < LEAKED < 1100

# 4. a 100-dimensional query, each person changes every coordinate by at most 1
DIM = 100
L1, L2 = DIM, np.sqrt(DIM)
SD_LAP = np.sqrt(2) * L1 / EPS
SD_GAU = L2 * SIG_EXACT
assert L2 == 10 and SD_LAP / SD_GAU > 3.5

SCENES = [
    ("title", [
        "上一期我们学了拉普拉斯机制。这一期介绍另一种常用的加噪方法，高斯机制，以及它带来的稍微放松的定义，带德尔塔的差分隐私。",
    ]),
    ("loss", [
        "先定义隐私损失。看到一个输出以后，它在有某个人和没有某个人两种情况下的概率之比，取对数，就是这个输出的隐私损失。",
        "对拉普拉斯噪声，不管输出是什么，隐私损失都被限制在正负艾普西隆之间，这就是纯粹的差分隐私。",
        "换成高斯噪声，隐私损失是一条直线，输出越极端，损失越大，没有上限。不过，出现极端输出的概率非常小。",
    ]),
    ("delta", [
        "所以我们放松定义，允许保证以很小的概率德尔塔不成立，这就是带德尔塔的差分隐私。",
        f"德尔塔必须远小于一除以人数。举个反例，一个算法以千分之一的概率，直接公开每一条记录，它在形式上满足德尔塔等于千分之一。在一百万人的数据上模拟一次，有{big_zh(LEAKED)}个人的数据被完整公开了。",
        "因此实践中，德尔塔通常取十的负五次方，或者更小。",
    ]),
    ("calibrate", [
        "高斯噪声要加多大？经典的公式给出，艾普西隆等于一、德尔塔等于十的负五次方时，标准差约为敏感度的四点八四倍。",
        f"但这个公式偏保守。直接计算高斯机制精确的艾普西隆和德尔塔曲线，只需要{dec_zh(SIG_EXACT)}倍，噪声少了{pct_zh(SAVE)}。",
    ]),
    ("vector", [
        "高斯机制最大的优势，在于高维的查询。它的敏感度用欧几里得长度来衡量，而拉普拉斯机制用的是各个坐标绝对值之和。",
        f"假设一次查询返回一百个数，每个人最多让每个数改变一。绝对值之和的敏感度是一百，欧几里得长度只有十。同样的隐私参数下，拉普拉斯噪声每个坐标的标准差约为{int_zh(round(SD_LAP))}，高斯噪声只有约{int_zh(round(SD_GAU))}。",
        "训练神经网络时，梯度有成千上万维，所以后面讲到的差分隐私训练，用的都是高斯机制。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，高斯噪声的隐私损失没有上限，需要允许一个很小的失败概率德尔塔。",
        "第二，德尔塔要远小于一除以人数，通常取十的负五次方或更小。",
        "第三，高斯机制按欧几里得敏感度加噪，在高维查询中噪声小得多。下一期，我们讨论多次查询时，隐私损失怎样累积。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "高斯机制与 (ε, δ)-差分隐私", "隐私损失 · δ 的含义 · 高维查询", SERIES)


def sc_loss(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "隐私损失")
    tv.fpaste(c, r"$\mathcal{L}(o)=\ln\frac{\Pr[M(D)=o]}{\Pr[M(D')=o]}$", 640, 150, 30)
    if seg >= 1:
        series = [("拉普拉斯（b = 1）", LOSS_LAP, CYAN)] + ([("高斯（σ = 1）", LOSS_GAU, RED)] if seg >= 2 else [])
        tv.line_chart(d, (200, 230, 1080, 580), XS, series, ylim=(-4, 5.5), xlabel="输出 o（D: 0，D′: 1）", ylabel="隐私损失",
                      legend_xy=(840, 250), xticks=[(x, str(x)) for x in range(-3, 5)])
        for v in (1, -1):
            y = 580 - (v + 4) / 9.5 * 350
            d.line([(200, y), (1080, y)], fill=(71, 85, 105), width=1)
            d.text((1090, y), "+ε" if v > 0 else "-ε", font=F_SMALL, fill=MUTED, anchor="lm")


def sc_delta(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "(ε, δ)-差分隐私")
    tv.fpaste(c, r"$\Pr[M(D)\in S]\ \leq\ e^{\varepsilon}\Pr[M(D')\in S]+\delta$", 640, 160, 34)
    if seg >= 1:
        d.rounded_rectangle([200, 240, 1080, 470], 14, outline=RED, width=2, fill=(12, 19, 34))
        vk.text_c(d, (640, 280), "反例：以概率 δ 公开每一条完整记录", F_BODY, RED)
        vk.text_c(d, (640, 340), f"n = {N_POP:,} 人，δ = {D_BAD:g}", F_BODY, TEXT)
        vk.text_c(d, (640, 410), f"一次模拟：{LEAKED:,} 人的数据被完整公开", vk.font(32), AMBER)
    if seg >= 2:
        vk.text_c(d, (640, 540), "要求 δ ≪ 1/n，实践中常取 δ = 1e-5 或更小", F_BODY, GREEN)


def sc_calibrate(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "噪声要加多大（ε = 1，δ = 1e-5，Δ₂ = 1）".replace("₂", "2"))
    tv.fpaste(c, r"$\sigma_{\mathrm{classic}}=\frac{\sqrt{2\ln(1.25/\delta)}}{\varepsilon}\,\Delta_2=4.84$", 360, 180, 28)
    if seg >= 1:
        tv.fpaste(c, r"$\delta(\varepsilon)=\Phi\!\left(\frac{\Delta}{2\sigma}-\frac{\varepsilon\sigma}{\Delta}\right)-e^{\varepsilon}\,\Phi\!\left(-\frac{\Delta}{2\sigma}-\frac{\varepsilon\sigma}{\Delta}\right)$", 360, 280, 22)
        tv.line_chart(d, (700, 150, 1160, 520), EPS_GRID, [(f"σ = {SIG_EXACT:.2f}", DELTA_CURVE, CYAN)], ylog=True,
                      xlabel="ε", ylabel="δ（对数）", xticks=[(0.5, "0.5"), (1, "1"), (1.5, "1.5")], legend_xy=(940, 170))
        x1 = 700 + (1 - EPS_GRID[0]) / (EPS_GRID[-1] - EPS_GRID[0]) * 460
        d.line([(x1, 150), (x1, 520)], fill=AMBER, width=1)
        d.text((x1 - 8, 330), "ε = 1 → δ = 1e-5", font=F_SMALL, fill=AMBER, anchor="ra")
        tv.bar_chart(d, (150, 360, 600, 580), ["经典公式", "精确计算"], [SIG_CLASSIC, SIG_EXACT], [MUTED, CYAN], fmt="σ = {:.2f}",
                     vmax=6)
        vk.text_c(d, (930, 580), f"精确 σ 少 {SAVE:.0%} 的噪声（Balle & Wang, 2018）", F_SMALL, GREEN)


def sc_vector(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, f"高维查询：{DIM} 个数，每人让每个数最多变 1")
    tv.fpaste(c, r"$\Delta_1=\sum_i|\Delta_i|=100,\qquad \Delta_2=\sqrt{\sum_i\Delta_i^2}=10$", 640, 160, 30)
    if seg >= 1:
        tv.bar_chart(d, (360, 240, 920, 520), ["拉普拉斯（Δ1 = 100）", "高斯（Δ2 = 10）"], [SD_LAP, SD_GAU], [VIOLET, CYAN],
                     fmt="{:.1f}", vmax=SD_LAP * 1.2)
        vk.text_c(d, (640, 580), "每个坐标的噪声标准差（ε = 1；高斯另有 δ = 1e-5）", F_SMALL, MUTED)
    if seg >= 2:
        vk.text_c(d, (640, 630), "神经网络的梯度有成千上万维 → DP-SGD 用高斯机制", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("隐私损失", "高斯噪声的损失无上限 → 需要 δ"), ("δ", "δ ≪ 1/n，常取 1e-5 或更小"),
                      ("高维", "按 L2 敏感度加噪，噪声小得多")], seg, "下一期：组合定理与隐私预算")


RENDER = {"title": sc_title, "loss": sc_loss, "delta": sc_delta, "calibrate": sc_calibrate, "vector": sc_vector,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "dp-gaussian")
