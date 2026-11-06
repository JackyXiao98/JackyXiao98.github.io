"""State Space Models 09 — 权衡与前沿.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] SSM_CACHE=<dir> python ssm09_tradeoffs.py <out_dir>

Recap of the series' measurements: KV cache vs fixed state (episode 01 arithmetic: Llama-3-8B-like config,
128K tokens -> 16 GiB; example 64-layer SSM state ~12 MiB); Tiny Shakespeare validation loss (ssmx.exp_lm);
selective copying (ssmx.exp_selcopy); multi-query associative recall vs state size (ssmx.exp_recall).
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import ssmx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "State Space Models"
CACHE = os.environ.get("SSM_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "ssmcache"))

KV128 = 2 * 32 * 8 * 128 * 2 * 131072
STATE = 64 * (5120 * 16 + 5120 * 3) * 2
LM = ssmx.cached(CACHE, "lm", ssmx.exp_lm)
SC = ssmx.cached(CACHE, "selcopy", ssmx.exp_selcopy)
RC = ssmx.cached(CACHE, "recall", ssmx.exp_recall)
LOSS = {k: LM[k]["final"]["val"] for k in ("attn", "mamba", "hybrid")}
SEL = {k: SC[k]["final"]["seq_acc"] for k in ("lti", "mamba")}
REC = {"attn": RC["attn"]["final"]["tok_acc"], "m16": RC[("mamba", 16)]["final"]["tok_acc"], "hybrid": RC["hybrid"]["final"]["tok_acc"]}
print(KV128 / 2 ** 30, STATE / 2 ** 20, LOSS, SEL, REC)
assert KV128 == 16 * 2 ** 30 and round(STATE / 2 ** 20) == 12
assert round(LOSS["attn"], 2) == 1.77 and round(LOSS["mamba"], 2) == 1.57
assert round(SEL["mamba"], 3) == 0.996 and round(SEL["lti"], 2) == 0.56
assert REC["attn"] > 0.99 and REC["hybrid"] > 0.99 and REC["m16"] < 0.9

SCENES = [
    ("title", [
        "这是状态空间模型系列的最后一期。我们把前面的实验放在一起，看注意力和状态空间模型各自的长处和短处，什么时候该用哪一种，以及这个方向正在往哪里走。",
    ]),
    ("table", [
        f"先看成本。十二万八千个词元时，一个八十亿参数的变换器需要十六吉字节的键值缓存，而示例配置的状态空间模型只有约十二兆字节的状态，生成每一步的代价也不随长度增长。训练时，两者现在都能并行。",
        f"再看能力。在我们的语言建模小实验里，选择性状态空间模型的损失是{dec_zh(LOSS['mamba'], 2)}，比注意力的{dec_zh(LOSS['attn'], 2)}还低；选择性复制上，它达到{pct_zh(SEL['mamba'], 1)}。但在联想回忆上，状态大小十六时只有{pct_zh(REC['m16'])}，而注意力和混合模型接近百分之百。",
    ]),
    ("when", [
        "所以选择取决于任务。序列非常长、需要流式处理、或者内存很紧张时，比如音频、基因组和端侧设备，状态空间模型很有吸引力。需要从上下文里精确地找出某个信息，比如检索、复制和上下文学习，注意力仍然更可靠。而大多数通用模型，混合架构是一个折中：大部分层用状态空间，少量层用注意力。",
    ]),
    ("frontier", [
        "前沿的工作大致有三个方向。第一，更好的状态更新，比如门控的增量规则，让固定大小的状态更会擦写。第二，混合架构，研究注意力层放多少、放在哪里。第三，把训练好的变换器蒸馏成线性的循环模型，复用已有的大模型，降低推理成本。",
    ]),
    ("recap", [
        "回顾整个系列：注意力的代价；线性注意力；状态空间模型的递推与卷积；长程记忆与特征值；曼巴的选择性；并行扫描；固定状态在回忆上的极限；以及混合架构。核心的取舍始终是同一个：固定大小的状态，换来了速度和内存，代价是要压缩信息。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，状态空间模型的生成成本和内存不随长度增长，在局部建模和选择性复制上表现很好。",
        "第二，固定大小的状态限制了精确回忆，这时注意力更可靠。",
        "第三，混合架构和更好的状态更新，是目前主要的方向。这个系列就到这里，感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 9, "权衡与前沿", "什么时候用状态空间模型", SERIES)


def sc_table(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "注意力 vs 状态空间模型（本系列的测量）")
    rows = [("128K 词元的记忆", "KV 缓存 16 GiB", "状态 ≈ 12 MiB"), ("生成每步", "随长度增长", "固定"), ("训练", "并行", "并行（扫描 / 对偶）")]
    if seg >= 1:
        rows += [("莎士比亚损失", f"{LOSS['attn']:.3f}", f"{LOSS['mamba']:.3f}"), ("选择性复制", f"{SC['attn']['final']['seq_acc']:.1%}", f"{SEL['mamba']:.1%}"),
                 ("联想回忆（N = 16）", f"{REC['attn']:.1%}", f"{REC['m16']:.1%}")]
    d.text((560, 175), "注意力", font=F_BODY, fill=CYAN, anchor="mm")
    d.text((900, 175), "选择性 SSM", font=F_BODY, fill=GREEN, anchor="mm")
    for i, (n, a_, b_) in enumerate(rows):
        y = 220 + i * 62
        d.line([(140, y - 8), (1140, y - 8)], fill=(51, 65, 85), width=1)
        d.text((150, y + 18), n, font=F_SMALL, fill=TEXT, anchor="lm")
        d.text((560, y + 18), a_, font=F_SMALL, fill=CYAN, anchor="mm")
        d.text((900, y + 18), b_, font=F_SMALL, fill=GREEN, anchor="mm")


def sc_when(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "什么时候用哪一种")
    items = [("状态空间模型", "超长序列 · 流式 · 内存紧张（音频、基因组、端侧）", GREEN),
             ("注意力", "精确检索 · 复制 · 上下文学习", CYAN), ("混合", "通用模型：大部分 SSM 层 + 少量注意力层", AMBER)]
    for i, (a_, b_, col) in enumerate(items):
        y = 210 + i * 120
        d.rounded_rectangle([140, y, 1140, y + 90], 14, outline=col, width=2, fill=(12, 19, 34))
        d.text((180, y + 45), a_, font=F_BODY, fill=col, anchor="lm")
        d.text((450, y + 45), b_, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_frontier(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "前沿方向")
    for i, (t, col) in enumerate([("更好的状态更新：门控增量规则（Gated DeltaNet）", GREEN), ("混合架构：注意力层放多少、放在哪里", AMBER),
                                  ("蒸馏：把 Transformer 转成线性循环模型", VIOLET)]):
        vk.text_c(d, (640, 240 + i * 90), t, F_BODY, col)


def sc_recap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "系列回顾")
    items = ["注意力的代价", "线性注意力", "递推与卷积", "长程记忆", "选择性", "并行扫描", "回忆与复制", "混合架构"]
    for i, t in enumerate(items):
        x = 160 + (i % 4) * 250
        y = 220 + (i // 4) * 120
        d.rounded_rectangle([x, y, x + 210, y + 80], 14, outline=[CYAN, GREEN, AMBER, VIOLET][i % 4], width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 105, y + 40), f"{i + 1:02d} {t}", F_SMALL, TEXT)
    vk.text_c(d, (640, 520), "固定大小的状态：换来速度与内存，代价是压缩信息", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("SSM", "成本不随长度增长"), ("注意力", "精确回忆更可靠"),
                      ("方向", "混合架构 · 更好的状态更新")], seg)


RENDER = {"title": sc_title, "table": sc_table, "when": sc_when, "frontier": sc_frontier, "recap": sc_recap, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "ssm-tradeoffs")
