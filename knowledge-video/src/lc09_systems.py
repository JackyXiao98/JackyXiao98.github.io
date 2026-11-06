"""Long Context 09 — 系统与前沿.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] LC_CACHE=<dir> python lc09_systems.py <out_dir>

Computations: ring attention memory per device - a sequence of 1M tokens split over 8 / 64 devices; sliding-window
attention cost (window 4,096) vs full attention at 128K tokens (ratio of attended pairs). Recap numbers from the
lcx caches of episodes 01-08.
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

T, W = 131072, 4096
FULL = T * (T + 1) / 2
WIN = sum(min(i + 1, W) for i in range(T))
RATIO = FULL / WIN
PER = {n: 1_000_000 // n for n in (8, 64)}
print(RATIO, PER)
assert round(RATIO, 1) == 16.3 and PER[8] == 125000 and PER[64] == 15625

SCENES = [
    ("title", [
        "这是长上下文系列的最后一期。前面我们讲了位置编码、扩展方法、训练和评估。这一期看系统和架构层面的方法，最后回顾整个系列。",
    ]),
    ("ring", [
        "上下文长到一百万个词元时，一张显卡连一层的激活都放不下。环形注意力把序列切成很多块，分给多张显卡，每张卡保存自己那一块的查询，键和值的块沿着一个环在显卡之间传递，一边传一边用分块的在线柔性最大值累加结果。",
        f"这样，一百万个词元分到八张卡上，每张只负责十二万五千个；分到六十四张卡上，每张只有一万五千多个。计算和通信可以重叠，序列长度随显卡数量线性扩展。",
    ]),
    ("sparse", [
        f"另一条路是让注意力不再看全部。滑动窗口注意力只看最近的若干个词元，比如四千零九十六个。在十二万八千个词元的序列上，要计算的词元对只有全注意力的约百分之{dec_zh(1 / RATIO * 100, 1)}，也就是少了约{int_zh(round(RATIO))}倍。",
        "很多模型把滑动窗口层和全注意力层交替使用：大部分层只看局部，少数层负责全局。稀疏注意力的变体还会加上少数全局词元或随机连接，让信息能在多层之间传到远处。",
    ]),
    ("memory", [
        "第三条路是不把所有东西都放进上下文：用检索先从海量文本中挑出相关的片段，或者给模型一个可以读写的外部记忆，或者像状态空间模型那样，把历史压缩进一个固定大小的状态。这些方法各有取舍，常常和长上下文一起使用。",
    ]),
    ("recap", [
        "回顾这个系列。注意力本身不分先后，位置编码决定了模型在训练长度之外的表现；线性偏置靠少看远处来外推，在远处检索上吃了亏；旋转位置编码要配合插值、神经正切核缩放等方法和少量长文本训练才能扩展；而模型读得进去，不等于用得好，评估需要多跳和聚合类的任务。",
        "长上下文是计算、位置、数据和使用四方面的共同问题。希望这个系列能帮你理解，当一个模型说它支持一百万个词元时，背后意味着什么。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 9, "系统与前沿", "与系列回顾", SERIES)


def sc_ring(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "环形注意力（Ring Attention, 2023）")
    cx, cy, R = 380, 380, 170
    n = 8
    for i in range(n):
        a = 2 * np.pi * i / n - np.pi / 2
        x, y = cx + R * np.cos(a), cy + R * np.sin(a)
        d.rounded_rectangle([x - 45, y - 30, x + 45, y + 30], 10, outline=CYAN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x, y), f"卡 {i + 1}", F_SMALL, CYAN)
        a2 = 2 * np.pi * (i + 0.5) / n - np.pi / 2
        d.ellipse([cx + (R - 5) * np.cos(a2) - 4, cy + (R - 5) * np.sin(a2) - 4, cx + (R - 5) * np.cos(a2) + 4, cy + (R - 5) * np.sin(a2) + 4], fill=AMBER)
    vk.text_c(d, (cx, cy), "K、V 块沿环传递", F_SMALL, AMBER)
    if seg >= 1:
        f = vk.font(24, mono=True)
        d.text((720, 300), "100 万词元", font=F_BODY, fill=TEXT)
        d.text((720, 370), f"8 张卡：每张 {PER[8]:,}", font=F_SMALL, fill=GREEN)
        d.text((720, 420), f"64 张卡：每张 {PER[64]:,}", font=F_SMALL, fill=GREEN)


def sc_sparse(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "滑动窗口与稀疏注意力")
    n = 14
    w = 4
    for i in range(n):
        for j in range(i + 1):
            col = GREEN if i - j < w else (30, 41, 59)
            d.rectangle([140 + j * 28, 190 + i * 28, 164 + j * 28, 214 + i * 28], fill=col)
    d.text((140, 600), "窗口内（绿）", font=F_SMALL, fill=GREEN)
    d.text((620, 250), f"128K 词元，窗口 4,096：", font=F_SMALL, fill=TEXT)
    d.text((620, 290), f"词元对 ≈ 全注意力的 {1 / RATIO:.1%}", font=F_BODY, fill=GREEN)
    if seg >= 1:
        d.text((620, 380), "窗口层与全局层交替（如 Gemma 2）", font=F_SMALL, fill=CYAN)
        d.text((620, 420), "Longformer / BigBird：全局词元 + 随机连接", font=F_SMALL, fill=AMBER)


def sc_memory(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "不把所有东西放进上下文")
    for i, (t, col) in enumerate([("检索：先挑出相关片段（RAG）", CYAN), ("外部记忆：可读写", VIOLET), ("压缩进固定状态（SSM）", GREEN)]):
        vk.text_c(d, (640, 230 + i * 80), t, F_BODY, col)


def sc_recap(c, u, seg, su):
    tv.summary(c, 4, [("位置编码", "决定训练长度之外的表现"), ("ALiBi", "外推靠少看远处"), ("扩展 RoPE", "插值 / NTK / YaRN + 少量长文本"),
                      ("使用与评估", "读进去 ≠ 用得好"), ("系统", "环形注意力 · 稀疏 · 检索")], 5)


RENDER = {"title": sc_title, "ring": sc_ring, "sparse": sc_sparse, "memory": sc_memory, "recap": sc_recap}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "lc-systems")
