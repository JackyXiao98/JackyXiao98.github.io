"""Scaling Laws 09 — 推理时计算的缩放与前沿.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] SCL_CACHE=<dir> SHAKESPEARE=<file> python scl09_testtime.py <out_dir>

Real computations (sclx.exp_emerge): for the width 16 / 32 / 64 / 128 models, 32 temperature-1 samples of the next
6 characters for each of 300 validation prompts; pass@k = probability that at least one of k samples equals the
true continuation (unbiased estimator 1 - C(n - c, k) / C(n, k)). Inference compute per attempt ~ non-embedding
parameters x number of samples. Recap numbers from the caches of episodes 02-08.
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
KS = [1, 2, 4, 8, 16, 32]
PK = {w: [EM[w]["passk"][k] for k in KS] for w in WS}
NP = {w: EM[w]["N"] for w in WS}
print(PK)
assert round(PK[128][0], 3) == 0.018 and round(PK[64][2], 3) == 0.021 and NP[64] * 4 == NP[128]
assert round(PK[32][-1], 3) == 0.033 and round(PK[128][-1], 3) == 0.083 and round(PK[16][-1], 2) == 0.01

SCENES = [
    ("title", [
        "前面几期讲的都是训练时的缩放：更大的模型、更多的数据、更多的计算。最后一期看另一个方向：在推理的时候多花计算，能不能换来更好的结果。",
    ]),
    ("passk", [
        "最简单的做法是多采样几次，再挑一个对的。我们让每个模型对三百个提示，各随机续写六个字符三十二次，看至少有一次完全写对的概率。",
        f"采样次数越多，成功率越高。最大的模型只试一次，成功率约{pct_zh(PK[128][0], 1)}；试三十二次，提高到约{pct_zh(PK[128][-1], 1)}。",
    ]),
    ("tradeoff", [
        f"更有意思的是和模型大小的比较。宽度六十四的模型参数只有最大模型的四分之一，让它试四次，推理的总计算量和大模型试一次差不多，成功率约{pct_zh(PK[64][2], 1)}，和大模型试一次的{pct_zh(PK[128][0], 1)}差不多，甚至略高。",
        "也就是说，推理时的计算和训练时的模型规模，在一定程度上可以互相替代。但这有一个前提：我们得有办法判断哪一次是对的。这里我们直接和正确答案比较，真实任务里则需要验证器、单元测试或者投票。",
    ]),
    ("frontier", [
        "近年的推理模型把这个想法推得更远：不只是多采样，而是让模型在回答之前生成很长的思考过程，用强化学习训练它更好地使用这些推理计算。研究发现，在一些任务上，增加推理计算的效果，可以媲美把模型放大很多倍。",
    ]),
    ("recap", [
        "回顾这个系列。损失随参数、数据和计算按幂律下降；计算最优要求模型和数据一起增长；可以用小模型外推大模型，但要条件一致；数据不够时，重复的收益递减；看似突然的涌现，常常是指标造成的；而推理时的计算，正在成为新的缩放维度。",
        "缩放定律让大模型的研发从碰运气变成了可以规划的工程。希望这个系列能帮你理解这些曲线背后的含义。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 9, "推理时计算的缩放", "与系列回顾", SERIES)


def sc_passk(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "pass@k：k 次采样至少一次完全写对（续写 6 个字符）")
    cols = [MUTED, VIOLET, CYAN, GREEN]
    ser = [(f"宽度 {w}", np.array(PK[w]) * 100, cols[i]) for i, w in enumerate(WS)]
    tv.line_chart(d, (140, 200, 820, 580), KS, ser, xlog=True, ylim=(0, 9), xticks=[(k, str(k)) for k in KS], xlabel="采样次数 k", legend_xy=(880, 220))
    d.text((140, 165), "成功率（%）", font=F_SMALL, fill=MUTED)


def sc_tradeoff(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "同样的推理计算：小模型多试 vs 大模型试一次")
    tv.bar_chart(d, (200, 230, 800, 560), ["宽度 64 × 4 次", "宽度 128 × 1 次"], [PK[64][2] * 100, PK[128][0] * 100], [CYAN, GREEN], fmt="{:.1f}%", vmax=2.6)
    d.text((200, 190), "成功率（推理计算：98k × 4 ≈ 393k × 1）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.text((860, 300), "前提：能判断哪次是对的", font=F_SMALL, fill=AMBER)
        d.text((860, 345), "验证器 · 单元测试 · 投票", font=F_SMALL, fill=TEXT)


def sc_frontier(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "推理模型：更长的思考过程")
    for i, (t, col) in enumerate([("回答前生成长的推理过程", CYAN), ("用强化学习训练如何使用推理计算", GREEN),
                                  ("Snell et al.（2024）：推理计算可媲美放大模型", AMBER)]):
        vk.text_c(d, (640, 230 + i * 80), t, F_BODY if i < 2 else F_SMALL, col)


def sc_recap(c, u, seg, su):
    tv.summary(c, 4, [("幂律", "损失随参数、数据、计算下降"), ("计算最优", "模型与数据一起增长"), ("外推", "条件一致才可靠"),
                      ("涌现", "常是指标造成的"), ("推理时计算", "新的缩放维度")], 5)


RENDER = {"title": sc_title, "passk": sc_passk, "tradeoff": sc_tradeoff, "frontier": sc_frontier, "recap": sc_recap}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "scl-testtime")
