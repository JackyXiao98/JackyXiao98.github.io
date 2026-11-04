"""MoE 09 — MoE 的权衡与前沿.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MOE_CACHE=<dir> python moe09_tradeoffs.py <out_dir>

Recap of this series' toy experiments (all read from the moex caches of episodes 01-07) and the trend in public
MoE models (expert count, experts per token, activated share; figures from their reports, see notes).
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import moex
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Mixture of Experts"
CACHE = os.environ.get("MOE_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "moecache"))

SC = moex.cached(CACHE, "scale", moex.exp_scale)
BA = moex.cached(CACHE, "balance", moex.exp_balance)
CP = moex.cached(CACHE, "capacity", moex.exp_capacity)
FI = moex.cached(CACHE, "fine", moex.exp_fine)
ST = moex.cached(CACHE, "stability", moex.exp_stability)
RECAP = [("01", "MoE 8 选 1 vs 稠密", f"{SC['moe8']['final']:.3f} vs {SC['dense128']['final']:.3f}", "≈ 8 倍计算的稠密"),
         ("03", "不加均衡", f"{BA['none']['final']:.3f}", "一个专家包揽 77%"),
         ("04", "容量系数 1.0", f"{CP[1.0]['final']:.3f}", "丢掉 4% 的词元"),
         ("06", "细粒度 32 选 4", f"{FI['32x32_top4']['final']:.3f}", "同样的计算"),
         ("07", "从稠密升级", f"{ST['upcycle']['final']:.3f}", "1,000 步，从头 1.789")]
MODELS = [("Mixtral 8x7B", "2023.12", 8, 2, 12.9 / 46.7), ("DeepSeek-V3", "2024.12", 256, 8, 37 / 671),
          ("Qwen3-235B", "2025.04", 128, 8, 22 / 235), ("Kimi K2", "2025.07", 384, 8, 32 / 1040)]
print(RECAP)
assert round(SC["moe8"]["final"], 3) == 1.686 and round(SC["dense128"]["final"], 3) == 1.794 and round(BA["none"]["final"], 3) == 1.738
assert round(CP[1.0]["final"], 3) == 1.735 and round(FI["32x32_top4"]["final"], 3) == 1.670 and round(ST["upcycle"]["final"], 3) == 1.750
assert round(ST["moe_scratch"]["final"], 3) == 1.789

SCENES = [
    ("title", [
        "这是混合专家系列的最后一期。我们先回顾这个系列在小模型上看到的现象，再总结混合专家的权衡，最后看看今天的大模型往哪个方向走。",
    ]),
    ("recap", [
        "在八种语言的小任务上，八个专家每次选一个，用小模型的计算量，达到了八倍计算的稠密模型的效果。",
        "不加均衡，一个专家会包揽四分之三以上的词元；容量设得太紧，会丢掉词元；把专家切细、多选几个，损失更低；从稠密模型升级，比从头训练快得多。",
    ]),
    ("tradeoffs", [
        "混合专家的好处很明确：在相同的计算下，可以有多得多的参数，学到更多的知识。",
        "代价也不少。显存要放下全部参数；专家分布在多张卡上，需要全对全通信；负载要均衡，否则又慢又浪费；训练更容易不稳定；稀疏模型在小数据上微调时也更容易过拟合；批量大了以后，读取权重上的优势会消失。",
    ]),
    ("when", [
        "所以混合专家最适合的是大规模的训练和服务：有很多显卡，有很多并发的请求，计算是主要成本。",
        "反过来，在一张卡或者手机上运行，或者显存才是瓶颈的时候，同样效果的稠密小模型往往更划算。",
    ]),
    ("trend", [
        "再看趋势。法国公司的八乘七十亿模型是八个专家选两个，激活约四分之一的参数。一年后，深度求索第三代用了二百五十六个专家选八个，只激活约百分之五点五。月之暗面的万亿参数模型，三百八十四个专家选八个，只激活约百分之三。",
        "专家越来越多、越来越小，激活的比例越来越低。模型的总参数在增长，但每个词元的计算增长得慢得多。",
    ]),
    ("frontier", [
        "前沿上还有很多活跃的方向。比如让专家来挑选词元，而不是词元挑选专家，天然保证负载均衡；比如根据词元的难度，动态决定用几个专家；以及在通信和推理系统上的大量优化。",
        "这个系列就到这里。混合专家用一个简单的想法，把参数和计算分开，让模型可以在可控的计算下继续变大。希望这个系列能帮你理解它背后的取舍。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 9, "MoE 的权衡与前沿", "系列回顾", SERIES)


def sc_recap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "本系列的小实验（验证损失，下限 1.557）")
    n = 1 if seg == 0 else len(RECAP)
    f = vk.font(22, mono=True)
    for i, (ep, a, b, note) in enumerate(RECAP[:n]):
        y = 180 + i * 85
        col = [GREEN, RED, AMBER, CYAN, VIOLET][i]
        d.rounded_rectangle([110, y, 1170, y + 64], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((140, y + 32), f"第 {ep} 期", font=F_SMALL, fill=MUTED, anchor="lm")
        d.text((260, y + 32), a, font=F_BODY, fill=col, anchor="lm")
        d.text((640, y + 32), b, font=f, fill=TEXT, anchor="lm")
        d.text((900, y + 32), note, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_tradeoffs(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "权衡")
    d.rounded_rectangle([110, 180, 560, 560], 14, outline=GREEN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (335, 220), "好处", F_BODY, GREEN)
    vk.text_c(d, (335, 300), "相同计算，更多参数", F_SMALL, TEXT)
    vk.text_c(d, (335, 345), "学到更多知识", F_SMALL, TEXT)
    if seg >= 1:
        d.rounded_rectangle([640, 180, 1170, 620], 14, outline=RED, width=2, fill=(12, 19, 34))
        vk.text_c(d, (905, 220), "代价", F_BODY, RED)
        for i, t in enumerate(["显存：放下全部参数", "通信：全对全", "负载均衡", "训练不稳定", "小数据微调易过拟合", "大批量时读取优势消失"]):
            vk.text_c(d, (905, 290 + i * 52), t, F_SMALL, TEXT)


def sc_when(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "什么时候用 MoE？")
    d.rounded_rectangle([110, 200, 620, 420], 14, outline=GREEN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (365, 240), "适合", F_BODY, GREEN)
    for i, t in enumerate(["大规模训练与服务", "很多显卡、很多并发请求", "计算是主要成本"]):
        vk.text_c(d, (365, 300 + i * 40), t, F_SMALL, TEXT)
    if seg >= 1:
        d.rounded_rectangle([660, 200, 1170, 420], 14, outline=AMBER, width=2, fill=(12, 19, 34))
        vk.text_c(d, (915, 240), "不一定划算", F_BODY, AMBER)
        for i, t in enumerate(["单卡或端侧部署", "显存是瓶颈", "低并发、小批量"]):
            vk.text_c(d, (915, 300 + i * 40), t, F_SMALL, TEXT)


def sc_trend(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "趋势：更多更小的专家，更低的激活比例")
    labs = ["Mixtral", "DS-V3", "Qwen3", "K2"]
    tv.bar_chart(d, (110, 220, 600, 560), labs, [m[2] for m in MODELS], [CYAN, AMBER, VIOLET, GREEN], fmt="{:.0f}", vmax=420)
    d.text((110, 180), "路由专家数（每个词元选 2 / 8 / 8 / 8）", font=F_SMALL, fill=MUTED)
    tv.bar_chart(d, (690, 220, 1180, 560), labs, [m[4] * 100 for m in MODELS], [CYAN, AMBER, VIOLET, GREEN], fmt="{:.1f}%", vmax=32)
    d.text((690, 180), "每个词元激活的参数比例", font=F_SMALL, fill=MUTED)
    for i, m in enumerate(MODELS):
        d.text((110 + (i + 0.5) * 490 / 4, 618), m[1], font=vk.font(16, mono=True), fill=MUTED, anchor="mm")
        d.text((690 + (i + 0.5) * 490 / 4, 618), m[1], font=vk.font(16, mono=True), fill=MUTED, anchor="mm")


def sc_frontier(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "前沿方向")
    rows = [("专家挑选词元", "Expert Choice（2022）：天然负载均衡", CYAN),
            ("动态专家数", "按词元难度决定用几个专家", AMBER),
            ("系统优化", "通信与计算重叠 · 推理服务", GREEN)]
    for i, (a, b, col) in enumerate(rows):
        y = 190 + i * 100
        d.rounded_rectangle([140, y, 1140, y + 72], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((170, y + 36), a, font=F_BODY, fill=col, anchor="lm")
        d.text((480, y + 36), b, font=F_SMALL, fill=TEXT, anchor="lm")
    if seg >= 1:
        vk.text_c(d, (640, 560), "把参数和计算分开：在可控的计算下让模型继续变大", F_BODY, VIOLET)


RENDER = {"title": sc_title, "recap": sc_recap, "tradeoffs": sc_tradeoffs, "when": sc_when, "trend": sc_trend, "frontier": sc_frontier}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "moe-tradeoffs")
