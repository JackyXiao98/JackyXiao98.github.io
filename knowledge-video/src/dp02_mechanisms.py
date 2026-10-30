"""Differential Privacy 02 — 随机应答与拉普拉斯机制.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python dp02_mechanisms.py <out_dir>

Real computations: randomized response on a simulated survey of 10,000 people (1,000 repeats);
the Laplace mechanism for a counting query at the same eps = ln 3 (error in people); mean absolute
error of Laplace noise across eps; sensitivity of a capped salary sum.
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
SERIES = "Differential Privacy"
rng = np.random.default_rng(0)

# randomized response: heads -> truth, tails -> second coin
N, P_TRUE, REPS = 10000, 0.2, 1000
EPS_RR = np.log(3)
truth = rng.random(N) < P_TRUE
TRUE_COUNT = int(truth.sum())


def rr(truth):
    c1 = rng.random(truth.size) < 0.5
    c2 = rng.random(truth.size) < 0.5
    return np.where(c1, truth, c2)


ANS = rr(truth)
F_YES = float(ANS.mean())
EST = 2 * F_YES - 0.5
ESTS = np.array([2 * rr(truth).mean() - 0.5 for _ in range(REPS)])
RR_SD_PEOPLE = float(ESTS.std() * N)
assert abs(EST - TRUE_COUNT / N) < 0.02 and 80 < RR_SD_PEOPLE < 110

# Laplace for the same count, same eps
LAP = rng.laplace(0, 1 / EPS_RR, REPS)
LAP_SD = float(LAP.std())
assert 1.1 < LAP_SD < 1.5  # sqrt(2)/ln3 = 1.29

# mean |noise| vs eps
EPSS = np.array([0.1, 0.2, 0.5, 1, 2, 5])
MAE = np.array([np.abs(rng.laplace(0, 1 / e, 200000)).mean() for e in EPSS])
assert np.allclose(MAE, 1 / EPSS, rtol=0.02)

# sensitivity of a capped salary sum, average over 10,000 people
CAP, EPS_S = 50000, 1.0
AVG_NOISE = CAP / (N * EPS_S)
assert AVG_NOISE == 5

SCENES = [
    ("title", [
        "上一期我们看到，差分隐私的关键是加入随机噪声。这一期介绍两种最基本的加噪方法，随机应答和拉普拉斯机制。",
    ]),
    ("rr", [
        "先看随机应答。假设要调查一个敏感问题，比如你有没有作弊过。",
        "每个人先偷偷抛一枚硬币。正面朝上，就如实回答；反面朝上，就再抛一次，正面答是，反面答否。",
        "这样，真正作弊的人有四分之三的概率答是，没作弊的人有四分之一的概率答是。任何一个人的回答都可以推说是硬币决定的，可以证明这满足差分隐私，艾普西隆等于三的自然对数，约为一点一。",
    ]),
    ("estimate", [
        f"那还能统计出真实比例吗？可以。我们模拟了一万个人，其中真实作弊的有{big_zh(TRUE_COUNT)}人，答是的比例是{pct_zh(F_YES, 1)}。",
        f"因为答是的比例等于四分之一加上二分之一乘真实比例，反推回去，估计的真实比例是{pct_zh(EST, 1)}，和真实比例{pct_zh(TRUE_COUNT / N, 1)}很接近。",
        f"重复模拟一千次，估计值的标准差，换算成人数大约是{int_zh(round(RR_SD_PEOPLE))}人。",
    ]),
    ("laplace", [
        "第二种方法是拉普拉斯机制。这时由一个可信的机构收集真实数据，只在发布结果时加噪声。",
        "加多少噪声，由敏感度决定，也就是一个人最多能让结果改变多少。计数查询的敏感度是一，所以加上尺度为艾普西隆分之一的拉普拉斯噪声。",
        f"同样取艾普西隆等于三的自然对数，噪声的标准差只有大约{dec_zh(LAP_SD, 1)}人，而随机应答是{int_zh(round(RR_SD_PEOPLE))}人。原因是随机应答在每个人那里都加了噪声，误差会累积。",
    ]),
    ("tradeoff", [
        "艾普西隆越小，噪声越大。拉普拉斯噪声的平均绝对误差正好是艾普西隆分之一，我们用二十万次抽样验证了这一点。",
        f"对于求和这样的查询，敏感度取决于一个人最多能贡献多少。比如把月薪截断在五万元，对一万人的平均工资加噪声，平均误差只有{int_zh(AVG_NOISE)}元。人越多，单个人的影响越小，需要的噪声相对越少。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，随机应答让每个人在本地加噪声，不需要信任收集者，但误差较大。",
        "第二，拉普拉斯机制按敏感度除以艾普西隆加噪声，误差小得多，但需要可信的收集者。",
        "第三，噪声大小和隐私强度此消彼长。下一期，我们介绍高斯机制，以及带德尔塔的差分隐私。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "随机应答与拉普拉斯机制", "两种最基本的加噪方法", SERIES)


def coin(d, x, y, txt, col):
    d.ellipse([x - 46, y - 46, x + 46, y + 46], outline=col, width=3, fill=(12, 19, 34))
    vk.text_c(d, (x, y), txt, vk.font(18), col)


def sc_rr(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "随机应答（Warner, 1965）")
    vk.text_c(d, (640, 140), "“你作弊过吗？”", F_BODY, TEXT)
    if seg >= 1:
        coin(d, 640, 230, "硬币 1", AMBER)
        vk.arrow(d, (610, 260), (420, 330), MUTED, 2)
        vk.arrow(d, (670, 260), (860, 330), MUTED, 2)
        vk.text_c(d, (420, 360), "正面：如实回答", F_BODY, CYAN)
        coin(d, 860, 380, "硬币 2", AMBER)
        vk.text_c(d, (760, 460), "正面 → 是", F_SMALL, TEXT)
        vk.text_c(d, (960, 460), "反面 → 否", F_SMALL, TEXT)
    if seg >= 2:
        tv.fpaste(c, r"$\Pr[\text{yes}\mid\text{cheated}]=\frac{3}{4},\quad \Pr[\text{yes}\mid\text{not}]=\frac{1}{4},\quad \varepsilon=\ln\frac{3/4}{1/4}=\ln 3\approx 1.10$", 640, 560, 24)


def sc_estimate(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, f"模拟：{N:,} 人，真实比例 20%")
    k = 400
    for i in range(k):
        x, y = 130 + (i % 40) * 12, 150 + (i // 40) * 12
        col = (RED if truth[i] else (40, 52, 70)) if seg == 0 else ((CYAN if ANS[i] else (40, 52, 70)))
        d.rectangle([x, y, x + 9, y + 9], fill=col)
    d.text((130, 280), "前 400 人：" + ("真实情况（红 = 作弊）" if seg == 0 else "回答（青 = 答“是”）"), font=F_SMALL, fill=MUTED)
    rows = [("真实作弊人数", f"{TRUE_COUNT:,}（{TRUE_COUNT / N:.1%}）", TEXT),
            ("答“是”的比例", f"{F_YES:.1%}", CYAN)]
    if seg >= 1:
        rows.append(("估计 = 2 × 比例 − 0.5".replace("−", "-"), f"{EST:.1%}", GREEN))
    for i, (a, b, col) in enumerate(rows):
        d.text((720, 160 + i * 60), a, font=F_SMALL, fill=MUTED)
        d.text((1180, 160 + i * 60), b, font=F_BODY, fill=col, anchor="ra")
    if seg >= 2:
        h, edges = np.histogram(ESTS * N, bins=30)
        x0, x1, y0, y1 = 140, 640, 340, 580
        for j, v in enumerate(h):
            xa = x0 + j * (x1 - x0) / len(h)
            d.rectangle([xa, y1 - v / h.max() * (y1 - y0), xa + (x1 - x0) / len(h) - 2, y1], fill=CYAN)
        d.text((x0, y1 + 10), f"{edges[0]:.0f}", font=vk.font(18, mono=True), fill=MUTED)
        d.text((x1, y1 + 10), f"{edges[-1]:.0f}", font=vk.font(18, mono=True), fill=MUTED, anchor="ra")
        vk.text_c(d, ((x0 + x1) / 2, y1 + 50), "1000 次模拟的估计人数", F_SMALL, MUTED)
        vk.text_c(d, (950, 450), f"标准差 ≈ {RR_SD_PEOPLE:.0f} 人", F_BODY, AMBER)


def sc_laplace(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "拉普拉斯机制（中心化）")
    tv.fpaste(c, r"$M(D)=f(D)+\mathrm{Lap}\!\left(\frac{\Delta f}{\varepsilon}\right),\qquad \Delta f=\max_{D\sim D'}|f(D)-f(D')|$", 640, 150, 28)
    if seg >= 1:
        vk.text_c(d, (640, 220), "计数查询：Δf = 1", F_BODY, TEXT)
    if seg >= 2:
        tv.bar_chart(d, (360, 300, 920, 560), ["随机应答（本地）", "拉普拉斯（中心化）"], [RR_SD_PEOPLE, LAP_SD], [CYAN, VIOLET],
                     fmt="{:.1f}", vmax=RR_SD_PEOPLE * 1.2)
        vk.text_c(d, (640, 620), f"同样 ε = ln 3：估计人数的标准差（{N:,} 人）", F_SMALL, MUTED)


def sc_tradeoff(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "隐私与准确性的权衡")
    tv.line_chart(d, (150, 160, 640, 540), EPSS, [("实测平均误差", MAE, CYAN), ("1/ε", 1 / EPSS, AMBER)], xlog=True, ylog=True,
                  ylim=(np.log10(0.08), np.log10(30)), xlabel="ε", ylabel="平均绝对误差", legend_xy=(440, 175), xticks=[(e, f"{e:g}") for e in EPSS])
    if seg >= 1:
        tv.fpaste(c, r"$\Delta=50000,\ n=10000,\ \varepsilon=1$", 940, 250, 26)
        tv.fpaste(c, r"$\mathrm{avg\ noise}=\frac{\Delta}{n\,\varepsilon}=5$", 940, 330, 30, "#fbbf24")
        vk.text_c(d, (940, 410), "月薪截断在 5 万元；平均工资的平均误差 5 元", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("随机应答", "本地加噪，不需信任收集者"), ("拉普拉斯", "噪声 = 敏感度 / ε，误差更小"),
                      ("权衡", "ε 越小越安全，噪声越大")], seg, "下一期：高斯机制与 (ε, δ)-差分隐私")


RENDER = {"title": sc_title, "rr": sc_rr, "estimate": sc_estimate, "laplace": sc_laplace, "tradeoff": sc_tradeoff,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "dp-mechanisms")
