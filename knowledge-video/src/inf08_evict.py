"""LLM Inference 08 — 长上下文：滑动窗口与缓存驱逐.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] INF_CACHE=<dir> python inf08_evict.py <out_dir>

Real computations (infx.exp_evict, numpy): the trained toy model decodes 48 held-out sequences of 64 tokens with a
KV cache limited to 4 / 8 / 16 entries per layer (evicted after every step), with three policies: sliding window
(keep the most recent tokens); attention sink + window (always keep the first 4 tokens, fill the rest with the most
recent); heavy hitters (keep the recent half plus the tokens with the largest accumulated attention, as in H2O).
Mean next-token loss over positions 16-63, compared with the full cache. Also the average attention weight on the
first token (positions 16-63, full cache) vs the uniform value 1/(t+1).
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import infx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "LLM Inference"
CACHE = os.environ.get("INF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "infcache"))

E = infx.cached(CACHE, "evict", infx.exp_evict, cache_dir=CACHE)
FULL = E["full"]
POL = ["window", "sink", "heavy"]
BUD = [4, 8, 16]
print(round(FULL, 3), {(p, b): round(E[(p, b)], 3) for p in POL for b in BUD}, E["first_attn"], E["uniform_attn"])
assert round(FULL, 3) == 1.591
assert [round(E[("window", b)], 2) for b in BUD] == [2.04, 1.87, 1.71]
assert [round(E[("sink", b)], 2) for b in BUD] == [2.26, 1.75, 1.66]
assert [round(E[("heavy", b)], 2) for b in BUD] == [1.85, 1.66, 1.61]
assert E["first_attn"] < E["uniform_attn"] / 4

SCENES = [
    ("title", [
        "键值缓存随上下文线性增长。当上下文长到几十万个词元时，缓存可能放不下，每一步读缓存也越来越慢。一个自然的想法是：只保留一部分缓存，把其余的丢掉。这一期看丢哪些最合适。",
    ]),
    ("policies", [
        "我们比较三种策略。第一种是滑动窗口，只保留最近的若干个词元。第二种是在窗口之外，永远保留最开头的几个词元。第三种是重要性驱逐：除了最近的一半，再保留历史上被关注得最多的那些词元。",
        "第二种做法来自对大模型的一个观察：很多注意力头会把大量注意力放在最开头的几个词元上，即使它们没有什么含义，像一个汇聚点。只用滑动窗口把它们丢掉，模型就会突然崩溃；保留下来，就能稳定地处理很长的文本。",
    ]),
    ("result", [
        f"我们让小模型在每一步只保留四个、八个或十六个缓存，看后面位置的平均损失。完整缓存时是{dec_zh(FULL, 2)}。",
        f"缓存为八时，滑动窗口的损失是{dec_zh(E[('window', 8)], 2)}，保留开头四个的是{dec_zh(E[('sink', 8)], 2)}，重要性驱逐是{dec_zh(E[('heavy', 8)], 2)}，最接近完整缓存。缓存为十六时，三者分别是{dec_zh(E[('window', 16)], 2)}、{dec_zh(E[('sink', 16)], 2)}和{dec_zh(E[('heavy', 16)], 2)}。",
        f"但缓存只有四个时，保留开头四个就意味着完全没有最近的词元，损失反而最差，是{dec_zh(E[('sink', 4)], 2)}。在这个任务里，预测下一个词元最需要的是上一个词元。",
    ]),
    ("sink", [
        f"我们的小模型有汇聚点吗？统计一下，开头第一个词元平均只得到约{pct_zh(E['first_attn'], 1)}的注意力，比平均分配的{pct_zh(E['uniform_attn'], 1)}还低得多。所以小模型里并没有大模型那样的汇聚点，保留开头之所以有帮助，原因也不一样。",
        "这提醒我们，哪些缓存重要，取决于模型和任务。重要性驱逐直接看注意力，所以在我们的实验里最稳。但它要额外统计注意力，和快速注意力这样不输出注意力矩阵的实现不容易配合。",
    ]),
    ("real", [
        "真实系统里还有很多做法：只在部分层用全局注意力，其他层用滑动窗口；把旧的缓存量化到更低的精度；或者把缓存放到内存甚至硬盘上，需要时再取回。长上下文推理仍然是一个很活跃的方向。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，长上下文的缓存太大时，可以只保留一部分。",
        "第二，在我们的实验里，按注意力保留重要的词元损失最小，滑动窗口最差。",
        "第三，大模型里的注意力汇聚点让保留开头词元很重要，但这取决于模型。最后一期，我们看推理的权衡与前沿。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "长上下文", "滑动窗口与缓存驱逐", SERIES)


def keep_row(d, y, kind, label, col):
    d.text((120, y + 22), label, font=F_SMALL, fill=col, anchor="lm")
    n = 16
    rng = np.random.default_rng(4)
    heavy = {1, 5, 6, 9}
    for i in range(n):
        x = 440 + i * 46
        if kind == "window":
            on = i >= n - 8
        elif kind == "sink":
            on = i < 4 or i >= n - 4
        else:
            on = i >= n - 4 or i in heavy
        d.rectangle([x, y, x + 38, y + 44], fill=col if on else None, outline=col if on else (51, 65, 85), width=2)


def sc_policies(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "保留哪些缓存？（例：16 个词元，保留 8 个）")
    keep_row(d, 190, "window", "滑动窗口", RED)
    keep_row(d, 290, "sink", "开头 4 个 + 窗口", AMBER)
    keep_row(d, 390, "heavy", "最近一半 + 被关注最多的", GREEN)
    if seg >= 1:
        vk.text_c(d, (640, 540), "StreamingLLM（2023）：大模型把大量注意力放在开头几个词元上——“注意力汇聚点”", F_SMALL, AMBER)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "只保留 4 / 8 / 16 个缓存（位置 16–63 的平均损失）")
    cols = [RED, AMBER, GREEN]
    names = ["滑动窗口", "开头 4 个 + 窗口", "重要性驱逐"]
    xs = list(range(3))
    ser = [(names[i], [E[(p, b)] for b in BUD], cols[i]) for i, p in enumerate(POL)]
    tv.line_chart(d, (120, 200, 700, 580), xs, ser, ylim=(1.5, 2.35), xticks=[(i, str(b)) for i, b in enumerate(BUD)], xlabel="缓存大小", legend_xy=(760, 210))
    y_f = 580 - (FULL - 1.5) / 0.85 * 380
    d.line([(120, y_f), (700, y_f)], fill=MUTED, width=1)
    d.text((130, y_f - 6), f"完整缓存 {FULL:.3f}", font=F_SMALL, fill=MUTED, anchor="lb")
    d.text((120, 165), "验证损失", font=F_SMALL, fill=MUTED)
    f = vk.font(22, mono=True)
    d.text((820, 340), "        4      8      16", font=f, fill=MUTED)
    for i, p in enumerate(POL):
        d.text((760, 380 + i * 42), ["窗口", "开头", "重要"][i], font=F_SMALL, fill=cols[i])
        d.text((820, 380 + i * 42), "    " + "  ".join(f"{E[(p, b)]:.3f}" for b in BUD), font=f, fill=TEXT)


def sc_sink(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "小模型里有注意力汇聚点吗？")
    tv.bar_chart(d, (250, 220, 800, 560), ["第一个词元得到的注意力", "平均分配时"], [E["first_attn"] * 100, E["uniform_attn"] * 100], [CYAN, MUTED], fmt="{:.1f}%", vmax=3.4)
    if seg >= 1:
        d.text((860, 300), "没有汇聚点", font=F_BODY, fill=AMBER)
        d.text((860, 360), "哪些缓存重要：", font=F_SMALL, fill=TEXT)
        d.text((860, 395), "取决于模型和任务", font=F_SMALL, fill=TEXT)
        d.text((860, 450), "重要性驱逐要统计注意力，", font=F_SMALL, fill=RED)
        d.text((860, 485), "和 FlashAttention 不易配合", font=F_SMALL, fill=RED)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "真实系统的其他做法")
    for i, (t, col) in enumerate([("部分层全局注意力，其他层滑动窗口", CYAN), ("旧缓存量化到更低精度", GREEN), ("缓存放到内存 / 硬盘，需要时取回", AMBER)]):
        y = 200 + i * 100
        d.rounded_rectangle([200, y, 1080, y + 70], 12, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (640, y + 35), t, F_BODY, col)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("驱逐", "缓存太大时只保留一部分"), ("实验", "按注意力保留最好，窗口最差"),
                      ("汇聚点", "大模型里保留开头很重要")], seg, "最后一期：推理的权衡与前沿")


RENDER = {"title": sc_title, "policies": sc_policies, "result": sc_result, "sink": sc_sink, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "inf-evict")
