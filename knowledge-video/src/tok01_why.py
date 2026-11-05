"""Tokenization 01 — 为什么需要分词：字符、词与子词.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] TOK_DATA=<dir> TOK_CACHE=<dir> python tok01_why.py <out_dir>

Real computations (tokx.exp_units): Tiny Shakespeare split into characters, into words (GPT-2-like pre-split that
keeps the leading space) and into BPE pieces learned on the first 90%: vocabulary size and number of tokens; share
of word occurrences in the last 10% never seen in the first 90% (out-of-vocabulary for a word-level vocabulary);
BPE (1,000 and 4,000 pieces) tokens on the held-out text and number of unknown tokens.
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

U = tokx.cached(CACHE, "units", tokx.exp_units)
TC = U["test_chars"]
R1, R4 = U["bpe1000_test_tokens"] / TC, U["bpe4000_test_tokens"] / TC
print(U["chars"], U["char_vocab"], U["words"], U["word_vocab"], U["unseen_words"], R1, R4, U["bpe1000_unk"], U["bpe4000_unk"])
assert U["chars"] == 1115394 and U["char_vocab"] == 65 and U["words"] == 294763 and U["word_vocab"] == 15057
assert round(U["unseen_words"], 3) == 0.047 and round(R1, 2) == 0.43 and round(R4, 2) == 0.34 and U["bpe4000_unk"] == 0

SCENES = [
    ("title", [
        "语言模型看到的并不是文字，而是一串整数编号，每个编号叫一个词元。把文字切成词元的过程叫分词。这个系列讲分词：它怎么做，为什么这样做，以及它带来的各种奇怪问题。第一期先看，为什么要切成现在这个样子。",
    ]),
    ("chars", [
        f"最简单的做法是按字符切。莎士比亚的这份文本有约一百一十万个字符，只用到{int_zh(U['char_vocab'])}种不同的字符。词表很小，任何文字都能表示。",
        "但序列很长：一百一十万个字符就是一百一十万个词元。模型的上下文能装下的文字就少，生成同样长的回答要走更多步，每一步还要从很底层的字母开始理解。",
    ]),
    ("words", [
        f"另一个极端是按词切。同样的文本只有约{big_zh(round(U['words'] / 10000))}万个词，序列短了将近四倍，但不同的词有一万五千多个，词表大得多。",
        f"更麻烦的是没见过的词。用前百分之九十的文本建词表，在最后百分之十里，有约{pct_zh(U['unseen_words'], 1)}的词出现时根本不在词表里，比如一些人名和不常见的变形。模型只能把它们当成未知词。",
    ]),
    ("subword", [
        f"子词分词是两者之间的折中：常见的词保留整词，少见的词拆成几块常见的片段。比如用字节对编码学一千个片段，每个字符平均只要约{dec_zh(R1, 2)}个词元；学到四千个片段，约{dec_zh(R4, 2)}个。而且最差也能退回单个字符，所以没有未知词。",
        "今天几乎所有的大语言模型都用子词分词，词表一般在三万到二十几万之间。下一期，我们一步步看最常用的字节对编码是怎样学出词表的。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，按字符切词表小、序列长；按词切序列短、词表大，还有未知词。",
        "第二，子词分词是折中：常见词整词保留，罕见词拆成片段。",
        "第三，它在压缩率和词表大小之间取得平衡，是今天大模型的标准做法。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "为什么需要分词", "字符、词与子词", SERIES)


def sc_chars(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "按字符切")
    s = "To be, or not"
    for i, ch in enumerate(s):
        tv.token(d, 140 + i * 66, 200, ch if ch != " " else "␣", CYAN, w=56, font=vk.font(22, mono=True))
    f = vk.font(26, mono=True)
    d.text((140, 320), "词表", font=F_BODY, fill=MUTED)
    d.text((340, 320), f"{U['char_vocab']}", font=f, fill=GREEN)
    d.text((140, 380), "词元数", font=F_BODY, fill=MUTED)
    d.text((340, 380), f"{U['chars']:,}", font=f, fill=RED if seg >= 1 else TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 520), "序列长 · 上下文装得少 · 生成要走更多步", F_BODY, RED)


def sc_words(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "按词切")
    for i, w in enumerate(["To", " be", ",", " or", " not"]):
        tv.token(d, 140 + i * 130, 200, w.replace(" ", "␣"), VIOLET, w=110, font=vk.font(22, mono=True))
    f = vk.font(26, mono=True)
    d.text((140, 320), "词表", font=F_BODY, fill=MUTED)
    d.text((340, 320), f"{U['word_vocab']:,}", font=f, fill=RED)
    d.text((140, 380), "词元数", font=F_BODY, fill=MUTED)
    d.text((340, 380), f"{U['words']:,}", font=f, fill=GREEN)
    if seg >= 1:
        d.text((700, 300), f"测试集里没见过的词：{U['unseen_words']:.1%}", font=F_BODY, fill=RED)
        for i, w in enumerate(U["unseen_examples"][:6]):
            d.text((700 + (i % 3) * 170, 360 + (i // 3) * 40), w.strip(), font=vk.font(20, mono=True), fill=TEXT)


def sc_subword(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "子词：常见词整词，罕见词拆成片段（BPE）")
    for i, (w, segs) in enumerate(list(U["examples"].items())[:6]):
        y = 175 + i * 52
        d.text((140, y + 8), w.strip(), font=vk.font(22, mono=True), fill=MUTED)
        x = 420
        for s in segs:
            wdt = 22 + 14 * len(s)
            tv.token(d, x, y, s.replace(" ", "␣"), GREEN, w=wdt, h=40, font=vk.font(20, mono=True))
            x += wdt + 8
    labs = ["字符", "1000", "4000", "词"]
    vals = [1.0, R1, R4, U["words"] / U["chars"]]
    tv.bar_chart(d, (820, 210, 1220, 520), labs, vals, [CYAN, GREEN, GREEN, VIOLET], fmt="{:.2f}", vmax=1.15)
    d.text((820, 175), "每个字符的词元数（BPE 片段数）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 620), "最差退回单个字符：没有未知词 · 大模型词表约 3 万 – 20 多万", F_SMALL, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 4, [("字符 vs 词", "词表小序列长 / 序列短词表大"), ("子词", "常见整词，罕见拆分"),
                      ("标准做法", "压缩率与词表大小的平衡")], seg, "下一期：BPE")


RENDER = {"title": sc_title, "chars": sc_chars, "words": sc_words, "subword": sc_subword, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "tok-why")
