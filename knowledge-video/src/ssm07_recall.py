"""State Space Models 07 — 能力边界：回忆与复制.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] SSM_CACHE=<dir> python ssm07_recall.py <out_dir>

Real computations (ssmx.exp_recall, JAX): multi-query associative recall - 16 (key, value) pairs (keys from 32,
values from 32 others), then the 16 keys again in random order, each followed by its value; loss and accuracy at
every query. 2-layer width-64 models with a previous-token embedding, trained 2,000 steps (batch 32): softmax
attention; selective SSM with state size N = 1, 4, 16, 64; hybrid (selective SSM N = 16 + attention). Accuracy on
512 test sequences: per query, and all 16 correct. Single seed.
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

RC = ssmx.cached(CACHE, "recall", ssmx.exp_recall)
NS = (1, 4, 16, 64)
TOK = {N: RC[("mamba", N)]["final"]["tok_acc"] for N in NS}
SEQ = {N: RC[("mamba", N)]["final"]["seq_acc"] for N in NS}
ATT, HYB = RC["attn"]["final"], RC["hybrid"]["final"]
ex_x, ex_y, ex_w = ssmx.recall_batch()(np.random.default_rng(3), 1)
print(TOK, SEQ, ATT, HYB)
assert ATT["seq_acc"] == 1.0 and HYB["seq_acc"] == 1.0
assert [round(TOK[N], 3) for N in NS] == [0.405, 0.191, 0.215, 0.864] and round(SEQ[64], 3) == 0.131 and SEQ[16] == 0.0

SCENES = [
    ("title", [
        "前几期看到，选择性状态空间模型在语言建模和选择性复制上都很好。但固定大小的状态有一个根本的限制：它能装下的信息是有限的。这一期，我们看这个限制在哪里显现出来。",
    ]),
    ("task", [
        "我们用多查询联想回忆来检验。序列先给出十六个键值对，然后把这十六个键按随机顺序再问一遍，每个键后面要答出它对应的值。模型必须把十六对都记在状态里，才能在之后任意顺序地取回。",
        "对注意力来说这很容易：查询的时候直接回头找到相同的键，读出后面的值。而状态空间模型只能依靠读到这里为止压缩下来的状态。",
    ]),
    ("result", [
        f"结果差别很大。注意力模型对每个查询都答对了，十六个全对的比例也是百分之百。状态大小为一、四和十六的选择性模型，只答对{pct_zh(TOK[4])}到{pct_zh(TOK[1])}，随机猜大约是百分之三。",
        f"把状态大小增加到六十四，答对的比例升到{pct_zh(TOK[64])}，但十六个全对仍然只有{pct_zh(SEQ[64])}。在我们的实验里，状态一到十六之间并没有随状态变大而变好，这和单一随机种子、训练只有两千步有关；但最大的状态明显最好。",
    ]),
    ("hybrid", [
        "如果只把其中一层换成注意力呢？一层选择性状态空间加一层注意力的混合模型，同样答对了百分之百。这说明问题不在于模型的大小，而在于固定状态本身：只要有一层能回头精确地看，回忆就解决了。",
    ]),
    ("theory", [
        "理论和大模型上的实验都支持这一点。有研究证明，要完整地复制一段长度为几千的文本，循环模型的状态必须随长度增长；而两层的变换器就能做到。在八十亿参数的规模上比较，纯曼巴模型在需要从上下文里复制和查找信息的任务上，也明显落后于变换器。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，联想回忆要求把所有的键值对都存进状态，固定大小的状态很难做到。",
        "第二，在我们的实验里，状态越大越好，但只有注意力和混合模型做到全对。",
        "第三，一层注意力就能补上精确回忆的能力。下一期，我们看混合架构。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "能力边界", "回忆与复制", SERIES)


def sc_task(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "多查询联想回忆：16 个键值对，然后按随机顺序提问")
    x = ex_x[0]
    cell = 18
    for i, t in enumerate(x):
        xx = 64 + i * cell
        is_key = 1 <= t <= 32
        q = i >= 32
        col = (CYAN if is_key else GREEN) if not q else (AMBER if is_key else (51, 65, 85))
        d.rectangle([xx, 250, xx + cell - 3, 290], fill=col)
    d.text((64, 310), "键（青）· 值（绿）", font=F_SMALL, fill=MUTED)
    d.text((64 + 32 * cell, 310), "查询的键（橙）· 要答出的值", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 430), "注意力：回头找相同的键，读出后面的值", F_BODY, CYAN)
        vk.text_c(d, (640, 500), "状态空间模型：只能靠压缩后的固定状态", F_BODY, GREEN)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "每个查询答对的比例（512 条测试序列）")
    labs = ["注意力"] + [f"N = {N}" for N in NS]
    vals = [ATT["tok_acc"] * 100] + [TOK[N] * 100 for N in NS]
    cols = [CYAN, GREEN, GREEN, GREEN, GREEN]
    if seg == 0:
        labs, vals, cols = labs[:4], vals[:4], cols[:4]
    tv.bar_chart(d, (140, 220, 900, 560), labs, vals, cols, fmt="{:.1f}%", vmax=110)
    d.text((140, 180), "选择性 SSM：N = 状态大小", font=F_SMALL, fill=MUTED)
    d.text((940, 300), "随机猜 ≈ 3%", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.text((940, 360), "16 个全对：", font=F_SMALL, fill=TEXT)
        d.text((940, 400), f"注意力 {ATT['seq_acc']:.0%}", font=F_SMALL, fill=CYAN)
        d.text((940, 440), f"N = 64：{SEQ[64]:.1%}", font=F_SMALL, fill=GREEN)
        d.text((940, 480), "N ≤ 16：0%", font=F_SMALL, fill=GREEN)


def sc_hybrid(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "把一层换成注意力")
    tv.bar_chart(d, (260, 220, 1020, 560), ["选择性 SSM（N = 16）", "混合（SSM + 注意力）", "注意力"],
                 [TOK[16] * 100, HYB["tok_acc"] * 100, ATT["tok_acc"] * 100], [GREEN, RED, CYAN], fmt="{:.1f}%", vmax=110)
    vk.text_c(d, (640, 630), "问题在固定状态：只要一层能回头看，回忆就解决了", F_SMALL, AMBER)


def sc_theory(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "理论与大模型上的证据")
    for i, (t, col) in enumerate([("复制长度为 L 的文本：循环模型的状态必须随 L 增长", GREEN),
                                  ("两层 Transformer 就能做到（Jelassi et al., 2024）", CYAN),
                                  ("8B 规模：纯 Mamba 在复制、查找类任务上落后（Waleffe et al., 2024）", AMBER)]):
        vk.text_c(d, (640, 240 + i * 85), t, F_BODY if i < 2 else F_SMALL, col)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("回忆", "所有键值对都要存进状态"), ("实验", "N = 64 最好，仍不及注意力"),
                      ("补救", "一层注意力就够")], seg, "下一期：混合架构")


RENDER = {"title": sc_title, "task": sc_task, "result": sc_result, "hybrid": sc_hybrid, "theory": sc_theory, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "ssm-recall")
