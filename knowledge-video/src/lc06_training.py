"""Long Context 06 — 长上下文训练：数据与渐进扩展.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] LC_CACHE=<dir> SHAKESPEARE=<file> python lc06_training.py <out_dir>

Real computations (lcx.exp_extend / exp_extrap): the RoPE model trained 2,000 steps on 64 characters, then 200
steps (10% more) on 256 characters: loss over positions 128-254 before and after; attention cost per training
sequence grows with length^2 (ratio 16 for 4x length).
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

R = lcx.cached(CACHE, "extend", lcx.exp_extend, cache_dir=CACHE)
BEFORE, AFTER = float(R["none"][128:255].mean()), float(R["ft"][128:255].mean())
SHORT = float(R["ft"][0:64].mean())
print(BEFORE, AFTER, SHORT)
assert round(BEFORE, 2) == 3.52 and round(AFTER, 2) == 1.68 and round(SHORT, 2) == 1.76

SCENES = [
    ("title", [
        "要让模型真正处理长文本，最终还是要在长文本上训练。但长序列的训练又贵、数据又少。这一期看大模型是怎样安排长上下文训练的。",
    ]),
    ("cost", [
        "先看代价。注意力的计算随长度平方增长，把训练长度扩大四倍，每条序列的注意力计算就是十六倍。如果从一开始就用很长的序列训练，大部分计算都会花在注意力上，而模型在早期主要学的是短距离的语法和常识，用不上这么长的上下文。",
    ]),
    ("progressive", [
        "所以常见的做法是渐进扩展：先在短序列上完成绝大部分预训练，最后再用一小部分计算，在越来越长的序列上继续训练，同时调整位置编码。",
        f"在我们的小实验里，模型先在六十四个字符上训练两千步，再在二百五十六个字符上多训练二百步，只增加十分之一的步数。后半段的损失从{dec_zh(BEFORE, 2)}降到{dec_zh(AFTER, 2)}，而前面短距离的表现基本不变。",
    ]),
    ("data", [
        "第二个难点是数据。网页上的大部分文档都很短；真正很长、又需要前后关联才能理解的文本，比如书、长篇报告、代码仓库，占比很小。只是把短文档拼接在一起，模型学不到真正的长距离依赖。",
        "因此长上下文训练会专门提高长文档的比例，比如按长度对数据重新采样，或者把同一个代码仓库的文件按依赖顺序拼起来，同时保留一部分短数据，避免模型在短任务上退化。",
    ]),
    ("real", [
        "第三代羊驼的技术报告描述了这样一个过程：先在八千个词元上预训练，然后分六个阶段逐步把上下文扩展到十二万八千，每个阶段都检查短任务的表现是否恢复，以及大海捞针测试是否通过，这一阶段大约用了八千亿个词元。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，长序列训练的注意力代价随长度平方增长，所以先短后长。",
        "第二，在我们的实验里，只多训练十分之一的步数，就把后半段的损失从三点五降到一点七。",
        "第三，长文档数据稀缺，需要专门采样和构造。下一期，我们看模型是否真的用好了长上下文。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "长上下文训练", "数据与渐进扩展", SERIES)


def sc_cost(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "训练长度 × 4 → 注意力计算 × 16")
    tv.bar_chart(d, (300, 220, 900, 560), ["长度 L", "长度 2L", "长度 4L"], [1, 4, 16], [GREEN, AMBER, RED], fmt="{:.0f}×", vmax=18)
    d.text((300, 180), "每条序列的注意力计算（相对）", font=F_SMALL, fill=MUTED)


def sc_progressive(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "渐进扩展：先短后长")
    d.rectangle([140, 200, 140 + 600, 250], fill=GREEN)
    d.rectangle([740, 200, 740 + 60, 250], fill=AMBER)
    d.text((140, 260), "2,000 步 · 长度 64", font=F_SMALL, fill=GREEN)
    d.text((740, 260), "+200 步 · 长度 256", font=F_SMALL, fill=AMBER)
    if seg >= 1:
        sm = lambda y: np.convolve(np.pad(y, (8, 7), mode="edge"), np.ones(16) / 16, mode="valid")
        tv.line_chart(d, (140, 330, 800, 620), np.arange(255), [("扩展前", np.minimum(sm(R["none"]), 4), RED), ("多训练 200 步", sm(R["ft"]), GREEN)],
                      ylim=(1.5, 4.1), xticks=[(0, "0"), (64, "64"), (128, "128"), (254, "254")], xlabel="位置", legend_xy=(860, 360))
        d.text((860, 480), f"位置 128–254：{BEFORE:.2f} → {AFTER:.2f}", font=F_SMALL, fill=TEXT)


def sc_data(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "长文档稀缺")
    for i, (t, col) in enumerate([("大部分网页文档很短", MUTED), ("书、长报告、代码仓库：占比很小", AMBER), ("拼接短文档 ≠ 长距离依赖", RED)]):
        vk.text_c(d, (640, 210 + i * 70), t, F_BODY, col)
    if seg >= 1:
        vk.text_c(d, (640, 460), "按长度重新采样 · 按依赖顺序拼接代码 · 保留短数据", F_SMALL, GREEN)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "Llama 3（2024）")
    steps = ["8K", "", "", "", "", "", "128K"]
    for i in range(7):
        x = 140 + i * 150
        h = 40 + i * 45
        d.rectangle([x, 560 - h, x + 110, 560], fill=[GREEN, CYAN, CYAN, CYAN, CYAN, CYAN, AMBER][i])
    d.text((140, 590), "8K 预训练", font=F_SMALL, fill=GREEN)
    d.text((1040, 590), "128K", font=F_SMALL, fill=AMBER)
    vk.text_c(d, (640, 200), "六个阶段逐步扩展 · 检查短任务与大海捞针 · 约 8,000 亿词元", F_SMALL, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("先短后长", "注意力代价随长度平方增长"), ("实验", "+10% 步数：3.52 → 1.68"),
                      ("数据", "长文档需要专门采样")], seg, "下一期：迷失在中间")


RENDER = {"title": sc_title, "cost": sc_cost, "progressive": sc_progressive, "data": sc_data, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "lc-training")
