"""Tokenization 04 — 词表大小的权衡.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] TOK_DATA=<dir> TOK_CACHE=<dir> SHAKESPEARE=<file> python tok04_vocab.py <out_dir>

Real computations (tokx.exp_vocab, JAX): Tiny Shakespeare tokenised with characters (65) and with BPE learned on
the first 90% at 256 / 1,000 / 4,000 pieces. For each, the same 2-layer Transformer (d = 64, context 64 tokens) is
trained 2,000 steps (batch 32, Adam 3e-3, cosine) and evaluated on the last 10%: loss per token, tokens per
character, bits per character = loss per token x tokens per character / ln 2, embedding parameters (input + output,
2 x V x 64; the non-embedding part is 98,304), and characters covered by the 64-token context.
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
VS = [65, 256, 1000, 4000]
BPC = {v: float(R[v]["bits_per_char"]) for v in VS}
LT = {v: R[v]["loss_tok"] for v in VS}
TPC = {v: R[v]["tok_per_char"] for v in VS}
EMB = {v: R[v]["emb_params"] for v in VS}
CTXC = {v: R[v]["chars_in_context"] for v in VS}
print(BPC, LT, TPC, EMB, CTXC)
assert [round(BPC[v], 2) for v in VS] == [2.51, 2.41, 2.33, 2.25] and [round(LT[v], 2) for v in VS] == [1.74, 3.01, 3.79, 4.58]
assert EMB[65] == 8320 and EMB[4000] == 512128 and round(CTXC[4000]) == 188 and round(TPC[4000], 2) == 0.34

SCENES = [
    ("title", [
        "字节对编码可以一直合并下去，那么词表应该多大？大词表让序列变短，但也让模型多出很多参数。这一期在小模型上看这个权衡。",
    ]),
    ("pertoken", [
        "先注意一个陷阱：不同词表的损失不能直接比。词表越大，每个词元包含的文字越多，预测下一个词元本来就更难。",
        f"在我们的实验里，按字符分词时每个词元的损失是{dec_zh(LT[65], 2)}，词表四千时变成{dec_zh(LT[4000], 2)}，看起来差了很多。但这是因为一个词元平均代表约三个字符。公平的比较要换算成每个字符要多少比特。",
    ]),
    ("bpc", [
        f"换算以后，结果反过来了。同样的小模型、同样训练两千步，字符级每个字符要{dec_zh(BPC[65], 2)}比特；词表二百五十六是{dec_zh(BPC[256], 2)}，一千是{dec_zh(BPC[1000], 2)}，四千是{dec_zh(BPC[4000], 2)}。词表越大，压缩得越好。",
        f"原因有两个。一是同样六十四个词元的上下文，词表四千时能看到约{int_zh(round(CTXC[4000]))}个字符，是字符级的三倍。二是同样的训练步数，大词表模型读过的文字也更多。所以这里比的其实是同样计算量下的效果。",
    ]),
    ("params", [
        f"代价在参数上。输入和输出两个嵌入矩阵的大小都和词表成正比。字符级只有约{big_zh(round(EMB[65] / 1000))}千个嵌入参数，词表四千时有约{big_zh(round(EMB[4000] / 10000))}万个，是模型其余部分的五倍多。",
        "对于小模型，词表太大会让大部分参数都花在嵌入上；而且很多罕见的词元在训练中很少出现，学得不好。对于大模型，嵌入只占很小的比例，所以大模型往往用更大的词表。",
    ]),
    ("real", [
        "真实模型的词表一路在变大：生成式预训练模型第二代约五万，第三代羊驼约十二万八千，一些多语言模型有二十多万。二零二四年有研究提出，最优的词表大小也随着模型规模增长，许多模型的词表可能还偏小。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，比较不同词表时，要看每个字符的比特数，而不是每个词元的损失。",
        "第二，在我们的实验里，词表越大，同样计算量下压缩得越好。",
        "第三，代价是嵌入参数随词表线性增长，对小模型尤其明显。下一期，我们看字节级分词。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "词表大小的权衡", "序列长度 vs 嵌入参数", SERIES)


def sc_pertoken(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "每个词元的损失：不能直接比")
    tv.bar_chart(d, (140, 220, 700, 560), [str(v) for v in VS], [LT[v] for v in VS], [MUTED] * 4, fmt="{:.2f}", vmax=5.2)
    d.text((140, 180), "每个词元的损失（nats）", font=F_SMALL, fill=MUTED)
    d.text((700, 618), "词表大小", font=F_SMALL, fill=MUTED, anchor="rm")
    if seg >= 1:
        f = vk.font(22, mono=True)
        d.text((780, 240), "每个字符的词元数", font=F_SMALL, fill=TEXT)
        for i, v in enumerate(VS):
            d.text((780, 290 + i * 40), f"{v:>5}   {TPC[v]:.2f}", font=f, fill=TEXT)
        tv.fpaste(c, r"$\mathrm{bpc}=\frac{\mathrm{loss/token}\times\mathrm{tokens/char}}{\ln 2}$", 960, 470, 30)


def sc_bpc(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "换算成每个字符的比特数（越低越好）")
    tv.bar_chart(d, (140, 220, 700, 560), [str(v) for v in VS], [BPC[v] for v in VS], [CYAN, GREEN, GREEN, GREEN], fmt="{:.2f}", vmax=2.8)
    d.text((140, 180), "比特 / 字符（同样的模型，2,000 步）", font=F_SMALL, fill=MUTED)
    d.text((700, 618), "词表大小", font=F_SMALL, fill=MUTED, anchor="rm")
    if seg >= 1:
        tv.bar_chart(d, (800, 260, 1180, 560), [str(v) for v in VS], [CTXC[v] for v in VS], [VIOLET] * 4, fmt="{:.0f}", vmax=210)
        d.text((800, 220), "64 个词元能看到的字符数", font=F_SMALL, fill=MUTED)


def sc_params(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "代价：嵌入参数随词表线性增长")
    labs = [str(v) for v in VS]
    tv.bar_chart(d, (140, 220, 760, 560), labs, [EMB[v] / 1000 for v in VS], [AMBER] * 4, fmt="{:.0f}k", vmax=560)
    y = 560 - 98.304 / 560 * 340
    d.line([(140, y), (760, y)], fill=MUTED, width=2)
    d.text((770, y), "其余部分 98k", font=F_SMALL, fill=MUTED, anchor="lm")
    d.text((140, 180), "嵌入参数（输入 + 输出）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.text((860, 300), "小模型：参数大多花在嵌入上", font=F_SMALL, fill=RED)
        d.text((860, 340), "罕见词元学得不好", font=F_SMALL, fill=RED)
        d.text((860, 410), "大模型：嵌入占比很小", font=F_SMALL, fill=GREEN)
        d.text((860, 450), "→ 用更大的词表", font=F_SMALL, fill=GREEN)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "真实模型的词表")
    tv.bar_chart(d, (200, 220, 860, 560), ["GPT-2", "Llama 2", "Llama 3", "Gemma"], [50.257, 32.0, 128.256, 256.0], [CYAN, MUTED, GREEN, AMBER], fmt="{:.0f}k", vmax=290)
    d.text((920, 300), "Tao et al.（2024）：", font=F_SMALL, fill=TEXT)
    d.text((920, 340), "最优词表随模型规模增长", font=F_SMALL, fill=AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("公平比较", "看每个字符的比特数"), ("实验", "词表越大，压缩越好"),
                      ("代价", "嵌入参数线性增长")], seg, "下一期：字节级分词")


RENDER = {"title": sc_title, "pertoken": sc_pertoken, "bpc": sc_bpc, "params": sc_params, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "tok-vocab")
