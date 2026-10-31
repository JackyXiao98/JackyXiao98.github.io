"""Recursive Self-Improvement 07 — 改进自己的代码：从 STOP 到 Darwin Gödel Machine.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RSI_CACHE=<dir> python rsi07_code.py <out_dir>

Real computations (rsix.binpack_search): a (1+8) evolutionary search over priority functions for
online bin packing (300 Weibull-sized items per instance, capacity 100), starting from best fit and
accepting a child only if it packs the visible instances better; selection on 3 vs 30 visible
instances, scored on 50 held-out instances (excess bins over the lower bound, %).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import rsix
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Recursive Self-Improvement"
CACHE = os.environ.get("RSI_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "rsicache"))

B3 = rsix.cached(CACHE, "bp", rsix.binpack_search, n_train=3)
B30 = rsix.cached(CACHE, "bp", rsix.binpack_search, n_train=30)
FF, BF = B30["first_fit_test"], B30["best_fit_test"]
H3 = B3["hist"]
LAST_GAIN = H3[-2][1] - H3[-1][1]
print(FF, BF, B30["train"], B30["test"], B3["train"], B3["test"], H3)
assert FF > BF > B30["test"] and B30["best_fit_train"] > B30["train"]
assert B3["train"] < B3["best_fit_train"] and B3["test"] > BF and LAST_GAIN < 0.01 and 0.75 < H3[-1][2] - H3[-2][2] < 0.8

SCENES = [
    ("title", [
        "这一期看一种更直接的自我改进，人工智能修改代码，包括修改负责改进的那段代码本身。",
    ]),
    ("stop", [
        "二零二三年的一项工作叫自学优化器。研究者写了一个改进程序，它调用语言模型去改进别的程序。然后，让这个改进程序去改进它自己。",
        "改进以后的改进程序，在其他任务上也表现更好。这是递归自我改进在代码层面的一个小例子，不过被改进的只是外层的程序，语言模型本身没有变。",
    ]),
    ("evolve", [
        "二零二五年，谷歌深度思维的阿尔法进化，把语言模型和进化搜索结合起来。语言模型提出代码修改，自动评估器打分，好的版本留下来继续演化。",
        "它找到了一个用四十八次乘法完成四乘四复数矩阵乘法的算法，还改进了数据中心的调度策略和训练用的计算核。同一年的达尔文哥德尔机，让编程智能体反复改写自己的代码，在软件工程基准上的成绩从百分之二十提高到百分之五十。",
    ]),
    ("toy", [
        "我们来做一个可以运行的小例子。任务是装箱，物品一个接一个到来，要放进容量为一百的箱子，箱子越少越好。",
        "程序是一个打分函数，决定每个物品放进哪个箱子。我们从经典的最佳适配出发，每一代随机修改八个版本，只有在训练用的装箱实例上更好，才接受修改。",
        f"在三十个训练实例上搜索，测试集上多用的箱子比例，从最佳适配的{pct_zh(BF / 100, 2)}降到{pct_zh(B30['test'] / 100, 2)}。作为对比，最简单的首次适配是{pct_zh(FF / 100, 2)}。",
    ]),
    ("overfit", [
        "但如果只用三个训练实例来挑选，会发生什么？",
        f"训练实例上的成绩从{pct_zh(B3['best_fit_train'] / 100, 2)}提高到{pct_zh(B3['train'] / 100, 2)}，看起来不错。可在没见过的测试实例上，反而从{pct_zh(BF / 100, 2)}变差到{pct_zh(B3['test'] / 100, 2)}。",
        "最后一次被接受的修改，只让训练成绩提高了不到零点零一个百分点，却让测试成绩变差了将近零点八个百分点。只看可见的分数来挑选改进，就会改进到分数本身，而不是真正的能力。这就是为什么最新的系统，坚持用被改进者看不到的隐藏测试来决定接受哪些修改。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，人工智能可以通过修改代码改进程序，包括负责改进的程序本身。",
        "第二，进化式代码搜索已经在数学和工程问题上找到了新的结果。",
        "第三，挑选改进的评估必须独立、隐藏，否则会对分数过拟合。下一期，我们看二零二六年的最新进展。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "改进自己的代码", "从 STOP 到 Darwin Gödel Machine", SERIES)


def sc_stop(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "STOP：Self-Taught Optimizer（Zelikman et al., 2023）")
    d.rounded_rectangle([160, 180, 560, 280], 14, outline=CYAN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (360, 215), "改进程序 I", vk.font(30), CYAN)
    vk.text_c(d, (360, 255), "调用语言模型改进程序 P", F_SMALL, TEXT)
    d.rounded_rectangle([720, 180, 1120, 280], 14, outline=MUTED, width=2, fill=(12, 19, 34))
    vk.text_c(d, (920, 230), "被改进的程序 P", F_BODY, TEXT)
    vk.arrow(d, (565, 230), (715, 230), MUTED, 3)
    if seg >= 0:
        d.arc([260, 280, 460, 400], 0, 180, fill=AMBER, width=3)
        vk.text_c(d, (360, 420), "I 改进 I 自己 → I′", F_BODY, AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 520), "I′ 在其他任务上也更好；但语言模型本身没有变", F_BODY, TEXT)


def sc_evolve(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "进化式代码搜索")
    steps = [("语言模型提出修改", CYAN), ("自动评估打分", AMBER), ("留下好的版本", GREEN)]
    for i, (t, col) in enumerate(steps):
        x = 150 + i * 340
        d.rounded_rectangle([x, 150, x + 300, 220], 12, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 150, 185), t, F_BODY, col)
        if i < 2:
            vk.arrow(d, (x + 305, 185), (x + 335, 185), MUTED, 2)
    if seg >= 1:
        rows = [("AlphaEvolve（DeepMind, 2025）", "4×4 复数矩阵乘法：48 次乘法；数据中心调度；训练计算核", VIOLET),
                ("Darwin Gödel Machine（Zhang et al., 2025）", "改写自己的代码：SWE-bench 20.0% → 50.0%", CYAN)]
        for i, (a, b, col) in enumerate(rows):
            y = 280 + i * 120
            d.rounded_rectangle([120, y, 1160, y + 90], 12, outline=col, width=2, fill=(12, 19, 34))
            d.text((150, y + 28), a, font=F_BODY, fill=col, anchor="lm")
            d.text((150, y + 64), b, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "可运行的小例子：在线装箱（容量 100）")
    for i, sz in enumerate([62, 35, 48, 20, 71, 15]):
        d.rectangle([150 + i * 70, 220 - sz, 200 + i * 70, 220], fill=CYAN)
    d.text((150, 230), "物品依次到来", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.text((620, 150), "程序 = 打分函数：物品放进哪个箱子", font=F_SMALL, fill=TEXT)
        d.text((620, 190), "初始：最佳适配；每代随机修改 8 个版本", font=F_SMALL, fill=TEXT)
        d.text((620, 230), "只接受在训练实例上更好的修改", font=F_SMALL, fill=AMBER)
    if seg >= 2:
        tv.bar_chart(d, (260, 330, 1020, 580), ["首次适配", "最佳适配（起点）", "进化后（30 个训练实例）"], [FF, BF, B30["test"]],
                     [MUTED, CYAN, GREEN], fmt="{:.2f}%", vmax=FF * 1.25)
        vk.text_c(d, (640, 640), "50 个测试实例：比下界多用的箱子比例", F_SMALL, MUTED)


def sc_overfit(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "只用 3 个训练实例来挑选")
    if seg >= 1:
        gens = [h[0] for h in H3]
        tv.line_chart(d, (160, 170, 820, 540), gens, [("训练实例（可见）", [h[1] for h in H3], CYAN),
                                                      ("测试实例（隐藏）", [h[2] for h in H3], RED)],
                      ylim=(4.8, 6.4), xlabel="被接受修改的代数", ylabel="多用箱子 %", legend_xy=(560, 190),
                      xticks=[(g, str(g)) for g in gens])
        d.text((880, 200), "训练", font=F_SMALL, fill=CYAN)
        d.text((880, 230), f"{B3['best_fit_train']:.2f}% → {B3['train']:.2f}%", font=vk.font(24, mono=True), fill=CYAN)
        d.text((880, 290), "测试", font=F_SMALL, fill=RED)
        d.text((880, 320), f"{BF:.2f}% → {B3['test']:.2f}%", font=vk.font(24, mono=True), fill=RED)
    if seg >= 2:
        vk.text_c(d, (640, 640), f"最后一次修改：训练只好 {LAST_GAIN:.3f} 个百分点，测试差了 {H3[-1][2] - H3[-2][2]:.2f} 个百分点", F_SMALL, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("改代码", "包括改进“改进程序”本身"), ("进化搜索", "已在数学与工程上找到新结果"),
                      ("评估", "独立、隐藏，否则过拟合分数")], seg, "下一期：2026 年的最新进展")


RENDER = {"title": sc_title, "stop": sc_stop, "evolve": sc_evolve, "toy": sc_toy, "overfit": sc_overfit,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rsi-code")
