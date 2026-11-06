"""Long Context 01 — 长上下文为什么难.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] LC_CACHE=<dir> SHAKESPEARE=<file> python lc01_why.py <out_dir>

Real computations (lcx.exp_extrap): for the RoPE model trained on 64-character windows, the validation loss at
each position inside the training window (more context -> lower loss): positions 0-7 vs 32-63; and the loss beyond
the training length (positions 128-510). Attention cost and KV-cache growth are arithmetic (episode facts in
notes).
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import lcx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Long Context"
CACHE = os.environ.get("LC_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "lccache"))

C = lcx.cached(CACHE, "extrap", lcx.exp_extrap)["rope"]["curve"]
EARLY, LATE, FAR = float(C[0:8].mean()), float(C[32:64].mean()), float(C[128:511].mean())
print(EARLY, LATE, FAR)
assert round(EARLY, 2) == 1.94 and round(LATE, 2) == 1.69 and round(FAR, 1) == 3.6

SCENES = [
    ("title", [
        "几年前，大模型的上下文只有几千个词元；今天，很多模型宣称支持十几万甚至上百万。能读整本书、整个代码库，听起来很美好。但长上下文远比看上去难。这个系列讲长上下文：难在哪里，怎样扩展，以及模型是不是真的用好了它。",
    ]),
    ("value", [
        f"先看上下文有多大用。我们在莎士比亚文本上训练一个小模型，每个片段六十四个字符。片段开头，模型几乎没有上下文，前八个位置的平均损失是{dec_zh(EARLY, 2)}；读到后半段，有了几十个字符的上下文，损失降到{dec_zh(LATE, 2)}。上下文越长，预测越准。",
    ]),
    ("three", [
        "但把上下文拉长，会遇到三个问题。第一是计算：注意力的计算随长度平方增长，缓存随长度线性增长，这在推理加速和状态空间模型两个系列里讲过。",
        f"第二是位置：模型只在一定长度上训练过，超过这个长度，它会遇到从没见过的位置。同一个小模型，测试时如果超过训练长度，后面的平均损失升到约{dec_zh(FAR, 1)}，比片段开头几乎没有上下文时还差得多。",
        "第三是数据和使用：真正很长、又需要从头读到尾才能理解的训练数据很少；而模型即使能读进很长的文本，也未必真的会用远处的信息。",
    ]),
    ("plan", [
        "接下来几期，我们先看位置编码，以及模型在训练长度之外为什么会失败；再看两类扩展上下文的方法；然后看长上下文的训练、模型是否迷失在中间、怎样评估，最后是系统层面的方法。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，上下文越长，预测越准，这是长上下文的价值。",
        "第二，难点有三个：计算、没见过的位置、以及数据和使用。",
        "第三，模型能读进长文本，不等于能用好它。下一期，我们看位置编码。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "长上下文为什么难", "计算、位置与使用", SERIES)


def sc_value(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "上下文越长，预测越准（训练长度 64 的小模型）")
    sm = np.convolve(np.pad(C[:64], (2, 1), mode="edge"), np.ones(4) / 4, mode="valid")
    tv.line_chart(d, (140, 200, 800, 580), np.arange(64), [("每个位置的损失", sm, GREEN)], ylim=(1.5, 2.6), xticks=[(0, "0"), (32, "32"), (63, "63")], xlabel="位置（前面有多少字符）")
    d.text((140, 165), "验证损失", font=F_SMALL, fill=MUTED)
    d.text((860, 300), f"位置 0–7：{EARLY:.2f}", font=F_BODY, fill=RED)
    d.text((860, 360), f"位置 32–63：{LATE:.2f}", font=F_BODY, fill=GREEN)


def sc_three(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "三个难点")
    items = [("计算", "注意力平方增长 · 缓存线性增长", CYAN), ("位置", f"超出训练长度：损失 {LATE:.2f} → {FAR:.1f}", AMBER), ("数据与使用", "长文档少 · 读进去 ≠ 用得好", VIOLET)]
    for i, (a, b, col) in enumerate(items[:seg + 1]):
        y = 200 + i * 120
        d.rounded_rectangle([140, y, 1140, y + 90], 14, outline=col, width=2, fill=(12, 19, 34))
        d.text((180, y + 45), a, font=F_BODY, fill=col, anchor="lm")
        d.text((420, y + 45), b, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_plan(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "这个系列")
    items = ["位置编码", "长度外推", "ALiBi", "扩展 RoPE", "长上下文训练", "迷失在中间", "评估", "系统与前沿"]
    for i, t in enumerate(items):
        x = 160 + (i % 4) * 250
        y = 230 + (i // 4) * 130
        d.rounded_rectangle([x, y, x + 210, y + 80], 14, outline=[CYAN, GREEN, AMBER, VIOLET][i % 4], width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 105, y + 40), f"{i + 2:02d} {t}", F_SMALL, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 4, [("价值", "上下文越长，预测越准"), ("难点", "计算 · 位置 · 数据与使用"),
                      ("提醒", "读进去 ≠ 用得好")], seg, "下一期：位置编码")


RENDER = {"title": sc_title, "value": sc_value, "three": sc_three, "plan": sc_plan, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "lc-why")
