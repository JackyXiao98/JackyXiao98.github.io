"""Scaling Laws 08 — 涌现能力：真的突然出现吗.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] SCL_CACHE=<dir> SHAKESPEARE=<file> python scl08_emerge.py <out_dir>

Real computations (sclx.exp_emerge): the width 16 / 32 / 64 / 128 models of episode 02 on 300 validation prompts of
64 characters: per-character accuracy (teacher forcing) - a continuous metric - and exact match of the next k
characters generated greedily (k = 1 ... 8) - an all-or-nothing metric. Illustration: if each character were
right independently with probability p, exact match of k characters would be p^k.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import sclx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Scaling Laws"
CACHE = os.environ.get("SCL_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "sclcache"))

EM = sclx.cached(CACHE, "emerge", sclx.exp_emerge, cache_dir=CACHE)
WS = [16, 32, 64, 128]
ACC = {w: EM[w]["acc_tf"] for w in WS}
EX6 = {w: EM[w]["exact"][5] for w in WS}
EX3 = {w: EM[w]["exact"][2] for w in WS}
PX = np.linspace(0.3, 0.9, 61)
print(ACC, EX6, EX3)
assert [round(ACC[w], 2) for w in WS] == [0.41, 0.47, 0.51, 0.53]
assert EX6[16] == 0 and round(EX6[32], 3) == 0.010 and round(EX6[128], 3) == 0.023
assert round(0.6 ** 8, 3) == 0.017 and round(0.9 ** 8, 2) == 0.43

SCENES = [
    ("title", [
        "有人报告说，某些能力在模型较小时完全没有，到了某个规模忽然出现，这被称为涌现能力。它们真的是突然出现的吗？二零二三年的一篇论文提出了一个不同的看法：这可能和我们用的衡量指标有关。",
    ]),
    ("metric", [
        "关键在于指标是连续的，还是全对才算对。比如要模型续写八个字符，如果每个字符答对的概率是零点六，八个全对的概率只有不到百分之二；每个字符的正确率提高到零点九，八个全对的概率就跳到百分之四十多。",
        "每个字符的正确率是平滑上升的，但全对的概率是它的八次方，在正确率低的时候几乎是零，然后快速上升，看起来就像突然出现了一样。",
    ]),
    ("toy", [
        f"我们用第二期的四个模型验证。每个字符的正确率随模型变大平滑地上升：从{pct_zh(ACC[16])}、{pct_zh(ACC[32])}、{pct_zh(ACC[64])}到{pct_zh(ACC[128])}。",
        f"但如果要求一口气续写对六个字符，最小的模型是零，宽度三十二的约{pct_zh(EX6[32], 1)}，最大的约{pct_zh(EX6[128], 1)}。在这个指标下，能力好像是从无到有、突然冒出来的，而底下的每个字符的进步一直是平稳的。",
    ]),
    ("debate", [
        "这并不是说大模型没有质的变化。有些任务确实需要多个步骤都做对，全对的指标本身就是用户关心的。但它提醒我们：看到突然的跳跃时，先换一个连续的指标看一看，比如每个字符的概率，或者答案的对数概率，往往能看到一条平滑的曲线，也就能更早地预测它什么时候会出现。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，全对才算对的指标，会把平滑的进步放大成突然的跳跃。",
        "第二，在我们的实验里，每个字符的正确率平滑上升，六个字符全对却从零开始突然出现。",
        "第三，换成连续的指标，更容易预测能力何时出现。最后一期，我们看推理时的计算缩放。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "涌现能力", "真的突然出现吗", SERIES)


def sc_metric(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "示例：每个字符独立答对的概率 p，八个全对 = p^8")
    tv.line_chart(d, (140, 200, 820, 580), PX, [("每个字符 p", PX, CYAN), ("八个全对 p^8", PX ** 8, RED)], ylim=(0, 1),
                  xticks=[(0.3, "0.3"), (0.6, "0.6"), (0.9, "0.9")], xlabel="每个字符的正确率", legend_xy=(880, 220))
    f = vk.font(22, mono=True)
    d.text((880, 340), f"p = 0.6 → {0.6 ** 8:.3f}", font=f, fill=TEXT)
    d.text((880, 380), f"p = 0.9 → {0.9 ** 8:.2f}", font=f, fill=TEXT)


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "四个模型（300 个提示）")
    labs = [f"宽度 {w}" for w in WS]
    tv.bar_chart(d, (110, 220, 590, 560), labs, [ACC[w] * 100 for w in WS], [CYAN] * 4, fmt="{:.0f}%", vmax=65)
    d.text((110, 180), "每个字符的正确率（连续）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        tv.bar_chart(d, (690, 220, 1170, 560), labs, [EX6[w] * 100 for w in WS], [RED] * 4, fmt="{:.1f}%", vmax=2.8)
        d.text((690, 180), "续写 6 个字符全对（全有或全无）", font=F_SMALL, fill=MUTED)


def sc_debate(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "看到跳跃时，换一个连续的指标")
    for i, (t, col) in enumerate([("全对指标：用户关心，但会放大跳跃", AMBER), ("连续指标：每个字符的概率、答案的对数概率", GREEN),
                                  ("→ 更早预测能力何时出现", CYAN)]):
        vk.text_c(d, (640, 230 + i * 80), t, F_BODY, col)
    vk.text_c(d, (640, 520), "Schaeffer, Miranda, Koyejo（NeurIPS 2023）", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 4, [("指标", "全对指标放大跳跃"), ("实验", "逐字平滑，6 字全对突然出现"),
                      ("建议", "用连续指标预测")], seg, "最后一期：推理时计算")


RENDER = {"title": sc_title, "metric": sc_metric, "toy": sc_toy, "debate": sc_debate, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "scl-emerge")
