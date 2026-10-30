"""Differential Privacy 08 — 实际应用：从人口普查到大模型.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python dp08_practice.py <out_dir>

Real computations: local (randomized response, eps = 1) vs central (Laplace, eps = 1) estimation of
how many of n people have a property, for n = 10^3 ... 10^6 (200 repeats each). Facts about real
deployments are quoted with references in the notes.
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

EPS, P_TRUE, REPS = 1.0, 0.3, 200
NS = np.array([1000, 10000, 100000, 1000000])
KEEP = np.exp(EPS) / (1 + np.exp(EPS))  # randomized response: report the truth with this probability


def local_sd(n):
    k = rng.binomial(n, P_TRUE)
    # each of the k "yes" reports yes w.p. KEEP, each of the n-k "no" reports yes w.p. 1-KEEP
    yes = rng.binomial(k, KEEP, REPS) + rng.binomial(n - k, 1 - KEEP, REPS)
    est = (yes - n * (1 - KEEP)) / (2 * KEEP - 1)
    return float(np.std(est - k))


LOCAL = np.array([local_sd(n) for n in NS])
CENTRAL = np.full(len(NS), float(np.std(rng.laplace(0, 1 / EPS, 100000))))
RATIO = LOCAL[-1] / CENTRAL[-1]
print(LOCAL, CENTRAL, RATIO)
assert 1.35 < CENTRAL[0] < 1.48 and 800 < LOCAL[-1] < 1100 and 500 < RATIO < 800

SCENES = [
    ("title", [
        "这是差分隐私系列的最后一期。我们来看它在现实中的应用，从人口普查、手机，到今天的大模型。",
    ]),
    ("models", [
        "部署差分隐私有两种模式。中心化模式由一个可信的机构收集原始数据，只在发布结果时加噪声；本地模式让每个人在自己的设备上先加噪声，再上传。",
        f"我们用同样的艾普西隆等于一，比较两种模式统计人数的误差。中心化模式的误差始终只有一点几个人，和总人数无关。",
        f"本地模式的误差随人数的平方根增长，一百万人时约为{big_zh(round(LOCAL[-1], -2))}人，是中心化模式的{int_zh(int(RATIO // 100 * 100))}多倍。所以本地模式只适合非常大的用户群，用来统计常见的趋势。",
    ]),
    ("census", [
        "最著名的中心化应用，是美国二零二零年的人口普查。人口普查局发现，用以前的发布方式，可以从公开的统计表里重建出大量个人记录。",
        "于是他们第一次用差分隐私发布普查结果，对各级地区的统计数字加入经过精心设计的噪声，同时保证州的总人数等关键数字准确。",
    ]),
    ("devices", [
        "本地模式的代表是科技公司的数据收集。谷歌在浏览器里用本地差分隐私统计用户设置，苹果在手机上用它统计常用的表情和新词。",
        "这些场景的用户数以亿计，正好弥补了本地模式误差大的缺点，公司只能看到群体的趋势，看不到任何一个人的原始数据。",
    ]),
    ("llm", [
        "到了大模型时代，隐私问题更加突出。研究者已经能从语言模型和图像扩散模型中，提取出训练数据里的原文和图片，扩散模型系列里有一期专门讲过。",
        "差分隐私提供了一种解决方案。常见的做法是先在公开数据上预训练，再用差分隐私的随机梯度下降，在私有数据上微调，这时准确率的损失要小得多。还可以用差分隐私训练的模型生成合成数据，再拿去自由使用。",
    ]),
    ("recap", [
        "回顾整个系列。我们从差分攻击出发，定义了差分隐私，学习了随机应答、拉普拉斯和高斯机制，又讲了组合定理、隐私预算和更紧的瑞尼记账。",
        "然后用差分隐私的随机梯度下降训练模型，用成员推断攻击检验它，最后看到了它在现实中的应用。希望这个系列能帮助你理解，怎样在利用数据的同时保护每一个人。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "实际应用", "从人口普查到大模型", SERIES)


def sc_models(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "中心化 vs 本地差分隐私")
    for i, (t, sub, col) in enumerate([("中心化", "可信机构收集原始数据，发布时加噪", VIOLET),
                                       ("本地", "每个人在设备上先加噪，再上传", CYAN)]):
        x = 140 + i * 520
        d.rounded_rectangle([x, 130, x + 480, 220], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((x + 24, 155), t, font=F_BODY, fill=col, anchor="lm")
        d.text((x + 24, 196), sub, font=F_SMALL, fill=MUTED, anchor="lm")
    if seg >= 1:
        series = [("中心化（拉普拉斯）", CENTRAL, VIOLET)] + ([("本地（随机应答）", LOCAL, CYAN)] if seg >= 2 else [])
        tv.line_chart(d, (200, 270, 1000, 590), NS, series, xlog=True, ylog=True, ylim=(np.log10(0.5), np.log10(5000)),
                      xlabel="人数 n", ylabel="统计人数的误差（标准差，ε = 1）", legend_xy=(230, 330),
                      xticks=[(1000, "1K"), (10000, "10K"), (100000, "100K"), (1000000, "1M")])
        d.text((1010, 590 - (np.log10(CENTRAL[-1]) - np.log10(0.5)) / 4 * 320), f"{CENTRAL[-1]:.1f}", font=vk.font(20, mono=True), fill=VIOLET, anchor="lm")
    if seg >= 2:
        d.text((1010, 590 - (np.log10(LOCAL[-1]) - np.log10(0.5)) / 4 * 320), f"{LOCAL[-1]:,.0f}", font=vk.font(20, mono=True), fill=CYAN, anchor="lm")


def sc_census(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "美国 2020 年人口普查")
    rows = [("问题", "旧的发布方式：可从统计表重建大量个人记录（重建攻击）", RED, 0),
            ("方案", "TopDown 算法：各级地区的统计数字加入噪声", CYAN, 1),
            ("约束", "州总人数等关键数字保持准确", GREEN, 1)]
    for i, (k, v, col, s) in enumerate(rows):
        if seg >= s:
            y = 170 + i * 110
            d.rounded_rectangle([150, y, 1130, y + 80], 12, outline=col, width=2, fill=(12, 19, 34))
            d.text((180, y + 40), k, font=F_BODY, fill=col, anchor="lm")
            d.text((300, y + 40), v, font=F_SMALL, fill=TEXT, anchor="lm")
    vk.text_c(d, (640, 560), "中心化差分隐私的代表性应用（Abowd et al., 2022）", F_SMALL, MUTED)


def sc_devices(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "本地差分隐私：浏览器与手机")
    for i, (t, sub, col) in enumerate([("Google Chrome", "RAPPOR（2014）：统计浏览器设置等", CYAN),
                                       ("Apple iOS / macOS", "统计常用表情、新词（2016 起）", VIOLET)]):
        x = 140 + i * 520
        d.rounded_rectangle([x, 160, x + 480, 300], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 240, 205), t, F_BODY, col)
        vk.text_c(d, (x + 240, 260), sub, F_SMALL, TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 400), "用户数以亿计 → 抵消本地模式的大误差", F_BODY, AMBER)
        vk.text_c(d, (640, 460), "公司只看到群体趋势，看不到任何人的原始数据", F_BODY, TEXT)


def sc_llm(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "大模型时代")
    rows = [("风险", "可从语言模型、扩散模型中提取训练数据（Carlini et al., 2021, 2023）", RED, 0),
            ("微调", "公开数据预训练 + 私有数据上 DP-SGD 微调，精度损失小", CYAN, 1),
            ("合成数据", "DP 训练的生成模型 → 合成数据可自由使用（后处理不额外消耗隐私）", GREEN, 1)]
    for i, (k, v, col, s) in enumerate(rows):
        if seg >= s:
            y = 160 + i * 120
            d.rounded_rectangle([110, y, 1170, y + 90], 12, outline=col, width=2, fill=(12, 19, 34))
            d.text((140, y + 45), k, font=F_BODY, fill=col, anchor="lm")
            d.text((290, y + 45), v, font=F_SMALL, fill=TEXT, anchor="lm")


EPS_ = ["什么是差分隐私", "随机应答等机制", "高斯机制与 δ", "组合与预算", "Rényi 记账与放大", "DP-SGD", "成员推断与审计", "实际应用"]


def sc_recap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "系列回顾")
    k = 5 if seg == 0 else 8
    for i, t in enumerate(EPS_):
        x, y = 170 + (i % 4) * 245, 200 + (i // 4) * 170
        on = i < k
        d.rounded_rectangle([x, y, x + 215, y + 120], 14, outline=CYAN if on else (30, 41, 59), width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 107, y + 36), f"{i + 1:02d}", vk.font(26, mono=True), CYAN if on else MUTED)
        vk.text_c(d, (x + 107, y + 82), t, F_SMALL, TEXT if on else MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 590), "感谢观看 · Differential Privacy 系列完", F_BODY, VIOLET)


RENDER = {"title": sc_title, "models": sc_models, "census": sc_census, "devices": sc_devices, "llm": sc_llm,
          "recap": sc_recap}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "dp-practice")
