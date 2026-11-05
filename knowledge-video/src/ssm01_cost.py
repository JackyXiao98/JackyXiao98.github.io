"""State Space Models 01 — 注意力的代价：为什么要找替代.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python ssm01_cost.py <out_dir>

Computations: attention score/value multiply-adds for a whole sequence of length T grow as T^2 (per layer and head
dim: 2 T^2 d), per generated token as T; KV cache of the Llama-3-8B configuration (32 layers, 8 KV heads, head dim
128, bf16: 128 KiB per token) at 4K / 32K / 128K tokens; the recurrent state of an example Mamba-style
configuration (64 layers, inner width 5,120, state size 16, bf16; plus a width-4 convolution buffer) - constant in T.
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

KV_TOK = 2 * 32 * 8 * 128 * 2
KV = {T: KV_TOK * T for T in (4096, 32768, 131072)}
STATE = 64 * (5120 * 16 + 5120 * 3) * 2
print({T: v / 2 ** 30 for T, v in KV.items()}, STATE / 2 ** 20)
assert KV_TOK == 131072 and KV[4096] == 2 ** 29 and KV[131072] == 16 * 2 ** 30 and round(STATE / 2 ** 20, 1) == 11.9

SCENES = [
    ("title", [
        "变换器的核心是注意力：每个词元都要看前面所有的词元。这带来了强大的能力，也带来了随长度增长的代价。这个系列讲另一类模型：线性注意力和状态空间模型，它们想用固定大小的记忆来处理任意长的序列。",
    ]),
    ("quadratic", [
        "先看训练。一段序列，注意力要计算每一对词元之间的分数，计算量和序列长度的平方成正比。长度翻一倍，注意力的计算变成四倍；长度增加十倍，就变成一百倍。",
        "再看生成。用了键值缓存以后，每生成一个词元，要和前面所有的键做一次比较，所以每一步的代价随长度线性增长，而缓存本身也随长度线性增长。",
    ]),
    ("memory", [
        f"按第三代羊驼八十亿参数模型的配置，每个词元的键值缓存是一百二十八千字节。四千个词元的上下文需要半吉字节，三万两千个需要四吉字节，十二万八千个需要十六吉字节，比模型本身的权重还大。",
    ]),
    ("rnn", [
        "循环网络走的是另一条路：它只保留一个固定大小的状态。每读一个词元，就把它的信息揉进状态里，再根据状态产生输出。不管序列多长，每一步的计算和占用的内存都一样。",
        f"以一个六十四层的状态空间模型的示例配置估算，整个状态只有约{int_zh(round(STATE / 2 ** 20))}兆字节，和上下文长度无关。",
    ]),
    ("tradeoff", [
        "代价是什么？固定大小的状态是一种压缩。注意力可以随时回头看任何一个词元的原样，而循环状态只能记住它认为重要的部分。传统的循环网络还有两个问题：难以并行训练，而且记不住很久以前的事。",
        "这个系列接下来的几期，就是看线性注意力和状态空间模型怎样一步步解决这些问题，以及它们在哪里仍然比不上注意力。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，注意力的训练计算随长度平方增长，生成时的缓存随长度线性增长。",
        "第二，循环模型只保留固定大小的状态，每一步的代价不随长度变化。",
        "第三，代价是信息要被压缩进有限的状态里。下一期，我们看线性注意力。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "注意力的代价", "为什么要找替代", SERIES)


def sc_quadratic(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "注意力：每一对词元都要算分数")
    n = 8
    for i in range(n):
        for j in range(i + 1):
            d.rectangle([140 + j * 40, 200 + i * 40, 176 + j * 40, 236 + i * 40], fill=CYAN)
    d.text((140, 530), "T × T（因果：下三角）", font=F_SMALL, fill=MUTED)
    if seg == 0:
        Ts = [1, 2, 4, 8, 16]
        tv.bar_chart(d, (560, 220, 1150, 560), [f"T×{t}" for t in Ts], [t * t for t in Ts], [AMBER] * 5, fmt="{:.0f}×", vmax=290)
        d.text((560, 180), "训练时注意力的计算量（相对）", font=F_SMALL, fill=MUTED)
    else:
        Ts = [1, 2, 4, 8, 16]
        tv.bar_chart(d, (560, 220, 1150, 560), [f"T×{t}" for t in Ts], Ts, [VIOLET] * 5, fmt="{:.0f}×", vmax=18)
        d.text((560, 180), "生成时每一步的代价与缓存（相对）", font=F_SMALL, fill=MUTED)


def sc_memory(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "KV 缓存（Llama-3-8B 配置，每个词元 128 KB）")
    labs = ["4K", "32K", "128K"]
    tv.bar_chart(d, (200, 220, 800, 560), labs, [KV[T] / 2 ** 30 for T in KV], [AMBER, RED, RED], fmt="{:.1f} GB", vmax=18)
    y = 560 - 16 / 18 * 340
    d.line([(200, y), (800, y)], fill=MUTED, width=2)
    d.text((810, y), "权重 16 GB", font=F_SMALL, fill=MUTED, anchor="lm")
    d.text((200, 180), "单个请求的缓存", font=F_SMALL, fill=MUTED)


def sc_rnn(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "循环：固定大小的状态")
    for i in range(6):
        x = 140 + i * 170
        tv.token(d, x + 20, 380, f"x{i + 1}", CYAN, w=60, font=vk.font(20, mono=True))
        d.rounded_rectangle([x, 230, x + 100, 300], 12, outline=GREEN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 50, 265), "h", vk.font(26, mono=True), GREEN)
        vk.arrow(d, (x + 50, 375), (x + 50, 305), MUTED, 2)
        if i < 5:
            vk.arrow(d, (x + 105, 265), (x + 165, 265), GREEN, 3)
    if seg >= 1:
        vk.text_c(d, (640, 520), f"示例：64 层，状态约 {STATE / 2 ** 20:.0f} MB，与长度无关", F_BODY, GREEN)


def sc_tradeoff(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "取舍")
    d.rounded_rectangle([110, 190, 620, 460], 14, outline=CYAN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (365, 230), "注意力", F_BODY, CYAN)
    for i, t in enumerate(["随时回看任何词元的原样", "代价随长度增长"]):
        vk.text_c(d, (365, 300 + i * 50), t, F_SMALL, TEXT)
    d.rounded_rectangle([660, 190, 1170, 460], 14, outline=GREEN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (915, 230), "循环状态", F_BODY, GREEN)
    for i, t in enumerate(["代价固定", "信息要压缩", "传统 RNN：难并行、记不久"]):
        vk.text_c(d, (915, 300 + i * 50), t, F_SMALL, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("注意力", "训练平方增长，缓存线性增长"), ("循环", "固定大小状态"),
                      ("代价", "信息被压缩")], seg, "下一期：线性注意力")


RENDER = {"title": sc_title, "quadratic": sc_quadratic, "memory": sc_memory, "rnn": sc_rnn, "tradeoff": sc_tradeoff, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "ssm-cost")
