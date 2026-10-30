"""Differential Privacy 05 — 更紧的记账：Rényi 差分隐私与子采样放大.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python dp05_accounting.py <out_dir>

Real computations: amplification by subsampling for pure DP; the Renyi-DP accountant for the
Poisson-subsampled Gaussian (dpx.py) on an MNIST-sized DP-SGD setting (60,000 examples, batch 256,
sigma = 1.1, 60 epochs, delta = 1e-5), compared with per-step conversion + basic composition and
with no subsampling; eps as a function of the number of epochs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import dpx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Differential Privacy"

# 1. subsampling, pure DP: eps' = ln(1 + q (e^eps - 1))
Q_DEMO, EPS_DEMO = 0.01, 1.0
EPS_AMP = float(np.log(1 + Q_DEMO * (np.exp(EPS_DEMO) - 1)))
assert abs(EPS_AMP - 0.0170) < 1e-3

# 2. MNIST-sized DP-SGD
N, B, SIG, EPOCHS, DELTA = 60000, 256, 1.1, 60, 1e-5
Q = B / N
STEPS = int(round(EPOCHS / Q))
RDP1 = dpx.rdp_subsampled_gaussian(Q, SIG)
EPS_RDP, ORDER = dpx.rdp_to_eps(STEPS * RDP1, DELTA)
EPS_NOAMP = dpx.dpsgd_eps(1.0, SIG, STEPS, DELTA)
EPS_STEP = dpx.rdp_to_eps(RDP1, DELTA / STEPS)[0]
EPS_NAIVE = STEPS * EPS_STEP
EP_GRID = np.array([1, 2, 5, 10, 20, 30, 40, 60, 80, 100])
EPS_CURVE = np.array([dpx.dpsgd_eps(Q, SIG, int(round(e / Q)), DELTA) for e in EP_GRID])
print(STEPS, EPS_RDP, ORDER, EPS_NOAMP, EPS_STEP, EPS_NAIVE, dict(zip(EP_GRID, np.round(EPS_CURVE, 2))))
assert STEPS == 14062 and abs(EPS_RDP - 3.01) < 0.01 and EPS_NOAMP > 10000 and EPS_NAIVE > 20000
assert abs(EPS_CURVE[EP_GRID == 1][0] - 1.03) < 0.01 and abs(EPS_CURVE[EP_GRID == 100][0] - 3.92) < 0.01

SCENES = [
    ("title", [
        "训练一个模型要迭代上万步，如果用上一期的组合定理来记账，隐私预算很快就会用完。这一期介绍两个关键工具，子采样放大，和瑞尼差分隐私。",
    ]),
    ("subsample", [
        "先看子采样放大。训练时每一步只随机抽取一小批数据，每个人只有很小的概率被抽中。",
        f"如果一个人这一步没有被抽中，这一步就完全不会泄露他的信息。可以证明，对一个满足艾普西隆等于一的算法，如果每个人被抽中的概率是百分之一，隐私损失会降到约{dec_zh(EPS_AMP, 3)}，几乎缩小了六十倍。",
    ]),
    ("renyi", [
        "第二个工具是瑞尼差分隐私。它不再只看最坏情况下的概率比，而是用瑞尼散度，衡量有没有某个人时，两个输出分布整体上相差多少。",
        "对高斯机制，它的值有一个简单的公式，阶数阿尔法除以二倍的方差。更重要的是，多步组合时，瑞尼差分隐私可以直接相加，而且很紧。最后再一次性换算成常用的艾普西隆和德尔塔。",
    ]),
    ("compare", [
        f"我们来看一个典型的设置，六万个训练样本，每批二百五十六个，噪声系数一点一，训练六十轮，一共{big_zh(STEPS)}步，德尔塔取十的负五次方。",
        f"如果每一步先换算成艾普西隆，再按基本组合相加，总的艾普西隆超过两万。如果用瑞尼差分隐私，但不考虑子采样，也有一万多。",
        f"把两者结合起来，总的艾普西隆只有{dec_zh(EPS_RDP)}。同样的训练过程，记账方法不同，结论可以相差几千倍。",
    ]),
    ("epochs", [
        f"在这个设置下，训练一轮的艾普西隆约为{dec_zh(EPS_CURVE[0])}，训练六十轮是{dec_zh(EPS_RDP)}，一百轮是{dec_zh(EPS_CURVE[-1])}。",
        "隐私损失随训练步数增长得越来越慢，大致和步数的平方根同步。这就是在大规模训练中，差分隐私能够实际可行的原因。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，每一步只随机抽取少量数据，可以大幅放大隐私保护。",
        "第二，瑞尼差分隐私在多步组合时可以直接相加，最后换算成艾普西隆和德尔塔，比传统方法紧得多。",
        "第三，两者结合，让上万步的训练也只需要个位数的艾普西隆。下一期，我们正式介绍差分隐私的随机梯度下降，并动手训练一个模型。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "更紧的记账", "Rényi 差分隐私与子采样放大", SERIES)


def sc_subsample(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "子采样放大")
    rng = np.random.default_rng(3)
    pick = rng.random(400) < 0.05
    for i in range(400):
        x, y = 150 + (i % 40) * 13, 150 + (i // 40) * 13
        d.rectangle([x, y, x + 10, y + 10], fill=CYAN if (pick[i] and seg >= 0) else (40, 52, 70))
    d.text((150, 290), "每一步：每人以概率 q 被抽中（青色）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        tv.fpaste(c, r"$\varepsilon'=\ln\left(1+q\,(e^{\varepsilon}-1)\right)$", 950, 200, 32)
        tv.fpaste(c, r"$\varepsilon=1,\ q=0.01\ \Rightarrow\ \varepsilon'\approx 0.017$", 950, 290, 28, "#fbbf24")
        vk.text_c(d, (640, 420), "没被抽中的人，这一步完全不泄露信息", F_BODY, TEXT)


def sc_renyi(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "Rényi 差分隐私（Mironov, 2017）")
    tv.fpaste(c, r"$D_\alpha\big(M(D)\,\|\,M(D')\big)=\frac{1}{\alpha-1}\ln\mathbb{E}_{o\sim M(D')}\left[\left(\frac{p_D(o)}{p_{D'}(o)}\right)^{\alpha}\right]\leq\varepsilon(\alpha)$".replace(r"\big", ""), 640, 160, 26)
    if seg >= 1:
        rows = [(r"$\text{Gaussian: }\varepsilon(\alpha)=\frac{\alpha\,\Delta_2^2}{2\sigma^2}$", CYAN),
                (r"$\text{composition: }\varepsilon_{\mathrm{total}}(\alpha)=\sum_t\varepsilon_t(\alpha)$", VIOLET),
                (r"$\text{convert: }\varepsilon=\min_\alpha\ \varepsilon_{\mathrm{total}}(\alpha)+\frac{\ln(1/\delta)}{\alpha-1}$", GREEN)]
        for i, (tex, col) in enumerate(rows):
            tv.fpaste(c, tex, 640, 270 + i * 90, 28, "#%02x%02x%02x" % col)


def sc_compare(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "同一次训练，三种记账方法")
    vk.text_c(d, (640, 110), f"N = {N:,}，batch {B}，σ = {SIG}，{EPOCHS} 轮（{STEPS:,} 步），δ = 1e-5", F_SMALL, MUTED)
    if seg == 0:
        tv.fpaste(c, r"$q=\frac{256}{60000}\approx 0.0043$", 640, 320, 32)
        return
    vals = [EPS_NAIVE, EPS_NOAMP] + ([EPS_RDP] if seg >= 2 else [])
    labs = ["逐步换算 + 基本组合", "RDP，不考虑子采样", "RDP + 子采样"][:len(vals)]
    x0, x1, y0, y1 = 200, 1080, 170, 560
    gw = (x1 - x0) / 3
    lo, hi = 0, np.log10(120000)
    d.line([(x0, y1), (x1, y1)], fill=(71, 85, 105), width=2)
    for i, (v, lab, col) in enumerate(zip(vals, labs, [RED, AMBER, GREEN])):
        bx = x0 + i * gw + gw * 0.2
        top = y1 - (np.log10(v) - lo) / (hi - lo) * (y1 - y0)
        d.rectangle([bx, top, bx + gw * 0.6, y1], fill=col)
        d.text((bx + gw * 0.3, top - 18), f"ε = {v:,.0f}" if v > 100 else f"ε = {v:.2f}", font=vk.font(22, mono=True), fill=col, anchor="mm")
        d.text((bx + gw * 0.3, y1 + 24), lab, font=F_SMALL, fill=MUTED, anchor="mm")
    d.text((x1, y0 - 20), "总 ε（对数刻度）", font=F_SMALL, fill=MUTED, anchor="ra")


def sc_epochs(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "ε 随训练轮数的变化（RDP + 子采样）")
    tv.line_chart(d, (180, 160, 1100, 540), EP_GRID, [("ε（δ = 1e-5）", EPS_CURVE, CYAN)], ylim=(0, 4.5),
                  upto=len(EP_GRID) if seg >= 1 else max(2, int(len(EP_GRID) * vk.ease(su))), xlabel="训练轮数", ylabel="ε",
                  xticks=[(e, str(e)) for e in [1, 10, 20, 40, 60, 80, 100]])
    for e in [1, 60, 100]:
        if seg >= 1 or e == 1:
            v = EPS_CURVE[EP_GRID == e][0]
            x = 180 + (e - 1) / 99 * 920
            y = 540 - v / 4.5 * 380
            d.text((x + 8, y - 30), f"{v:.2f}", font=vk.font(22, mono=True), fill=AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("子采样", "每步只抽少量数据，放大隐私"), ("RDP", "多步直接相加，最后换算 (ε, δ)"),
                      ("结合", "上万步训练，ε 只有个位数")], seg, "下一期：DP-SGD —— 动手训练一个差分隐私模型")


RENDER = {"title": sc_title, "subsample": sc_subsample, "renyi": sc_renyi, "compare": sc_compare, "epochs": sc_epochs,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "dp-accounting")
