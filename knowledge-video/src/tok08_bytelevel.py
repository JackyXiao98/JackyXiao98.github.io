"""Tokenization 08 — 不分词行不行：字节级模型与前沿.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] TOK_DATA=<dir> TOK_CACHE=<dir> python tok08_bytelevel.py <out_dir>

Real computations (tokx.exp_vocab, reused): on Tiny Shakespeare (ASCII English, so one byte = one character) the
same small Transformer trained on raw characters / bytes vs BPE with 4,000 pieces for the same 2,000 steps: bits per
character and sequence length. Attention cost per character for a fixed text length grows with the square of the
sequence length (ratio (1 / 0.341)^2). Series recap numbers come from the caches of episodes 01-07.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tokx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Tokenization"
CACHE = os.environ.get("TOK_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tokcache"))

R = tokx.cached(CACHE, "vocab", tokx.exp_vocab)
B, P = float(R[65]["bits_per_char"]), float(R[4000]["bits_per_char"])
LEN = 1 / R[4000]["tok_per_char"]
ATT = LEN ** 2
M = tokx.cached(CACHE, "multi", tokx.exp_multi)
print(B, P, LEN, ATT)
assert round(B, 2) == 2.51 and round(P, 2) == 2.25 and round(LEN, 1) == 2.9 and round(ATT, 1) == 8.6

SCENES = [
    ("title", [
        "这个系列看到了分词的种种问题：未知字符、拆错的数字、多语言的不公平。一个自然的问题是：能不能干脆不分词，让模型直接读字节？这是分词系列的最后一期。",
    ]),
    ("cost", [
        f"直接读字节，最大的代价是序列变长。在莎士比亚这样的英文文本上，一个字节就是一个字符，序列是四千词表分词结果的约{dec_zh(LEN, 1)}倍长。注意力的计算随序列长度平方增长，同样一段文字，代价约是{dec_zh(ATT, 1)}倍。",
        f"在我们的实验里，同样的小模型、同样训练两千步，直接读字符每个字符要{dec_zh(B, 2)}比特，而用四千词表的分词是{dec_zh(P, 2)}比特。同样的计算，不分词的模型学得更差。",
    ]),
    ("ideas", [
        "所以字节级模型都要想办法把序列缩短。早期的字节版文本到文本模型直接在字节上训练，用更深的编码器来补偿。后来的方法把字节分成固定长度的小块，先在块内建模，再在块之间建模。",
        "二零二四年元公司的字节潜在变换器更进一步：根据下一个字节有多难预测来动态分块。好预测的地方块很大，难预测的地方块很小。论文报告，在八十亿参数的规模上，它能达到和第三代羊驼相当的效果，推理时的计算还可以更少。",
    ]),
    ("recap", [
        "回顾这个系列。我们看到，子词分词在序列长度和词表大小之间取得平衡；字节对编码一步步合并最常见的一对；字节级分词消除了未知字符；而分词也带来了数字被随意拆分、多语言词元数相差好几倍等问题。",
        "分词是语言模型里最容易被忽略、却影响最广的一步。希望这个系列能让你在看到模型的奇怪错误时，多想一想：是不是分词的问题。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "不分词行不行", "字节级模型与前沿", SERIES)


def sc_cost(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "直接读字节：序列更长")
    tv.bar_chart(d, (140, 230, 560, 560), ["字节（字符）", "BPE 4000"], [LEN, 1.0], [RED, GREEN], fmt="{:.1f}×", vmax=3.3)
    d.text((140, 190), "序列长度（相对）", font=F_SMALL, fill=MUTED)
    d.text((620, 220), f"注意力代价 ≈ {ATT:.1f} 倍", font=F_BODY, fill=RED)
    if seg >= 1:
        tv.bar_chart(d, (700, 400, 1150, 630), ["字节（字符）", "BPE 4000"], [B, P], [RED, GREEN], fmt="{:.2f}", vmax=2.8)
        d.text((700, 360), "比特 / 字符（同样的模型，2,000 步）", font=F_SMALL, fill=MUTED)


def sc_ideas(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "把字节序列缩短")
    rows = [("ByT5（2022）", "直接在字节上训练，更深的编码器", CYAN), ("MEGABYTE（2023）", "固定长度分块：块内 + 块间两级建模", VIOLET)]
    if seg >= 1:
        rows.append(("BLT（Meta，2024）", "按预测难度动态分块；8B 规模与 Llama 3 相当", GREEN))
    for i, (a, b, col) in enumerate(rows):
        y = 190 + i * 100
        d.rounded_rectangle([110, y, 1170, y + 72], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((140, y + 36), a, font=F_BODY, fill=col, anchor="lm")
        d.text((520, y + 36), b, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_recap(c, u, seg, su):
    tv.summary(c, 3, [("子词", "序列长度与词表大小的平衡"), ("BPE", "合并最常见的一对"), ("字节级", "没有未知字符"),
                      ("问题", "数字拆分、多语言不公平")], 4)


RENDER = {"title": sc_title, "cost": sc_cost, "ideas": sc_ideas, "recap": sc_recap}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "tok-bytelevel")
