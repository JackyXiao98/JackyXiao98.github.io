"""Tokenization 05 — 字节级分词：再也没有未知字符.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] TOK_DATA=<dir> TOK_CACHE=<dir> python tok05_bytes.py <out_dir>

Real computations (tokx.exp_bytes): two BPE tokenizers with 2,000 merges learned on Tiny Shakespeare + four files
of the Complete Tang Poems (traditional Chinese): character-level (base vocabulary = every character seen, 5,100)
and byte-level (base = the UTF-8 bytes seen, 144 of 256). Encodings of four sample strings (English, a Tang poem line
whose first character never occurs in the corpus, an emoji, Greek and accented Latin letters): number of tokens and
of unknown tokens; tokens per character on the last 20,000 characters of each corpus.
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

B = tokx.cached(CACHE, "bytes", tokx.exp_bytes)
print(B["char_vocab_base"], B["byte_vocab_base"], B["en_tok_per_char"], B["zh_tok_per_char"], B["zh_bytes_per_char"])
assert B["char_vocab_base"] == 5100 and B["byte_vocab_base"] == 144
assert B["zh"]["unk"] == 1 and B["emoji"]["unk"] == 1 and B["rare"]["unk"] == 7 and B["emoji"]["n_byte"] == 12
assert round(B["en_tok_per_char"], 2) == 0.42 and round(B["zh_tok_per_char"], 2) == 1.26 and round(B["zh_bytes_per_char"], 2) == 2.97

SCENES = [
    ("title", [
        "前面的字节对编码从字符开始合并。但世界上的字符有十几万个，表情符号还在不断增加。如果某个字符训练时没见过，就只能当成未知词。这一期讲字节级分词：从字节而不是字符开始。",
    ]),
    ("problem", [
        f"我们在莎士比亚和唐诗的混合语料上，从字符开始学字节对编码。光是起点的字符就有{big_zh(B['char_vocab_base'])}个，因为每个汉字都是一个字符。",
        "即使这样，碰到语料里没出现过的字，比如这句诗的第一个字，或者一个表情符号、几个希腊字母，都只能变成未知词。模型完全不知道这里原本写的是什么。",
    ]),
    ("bytes", [
        "字节级分词换一个起点：任何文字在计算机里都是一串字节，最多只有二百五十六种。从这二百五十六个字节开始合并，起点的词表很小，而且任何文本都能编码，再也没有未知词。",
        f"同样的句子，字节级编码没有一个未知词。没见过的那个字被拆成了两个字节片段，表情符号被拆成了四个字节。在我们的语料里，只出现了{int_zh(B['byte_vocab_base'])}种不同的字节。",
    ]),
    ("cost", [
        f"代价是，有些字会被拆成好几个词元。一个汉字在通用编码里占三个字节。我们的字节级词表只学了两千次合并，英文平均每个字符只要{dec_zh(B['en_tok_per_char'], 2)}个词元，中文却要{dec_zh(B['zh_tok_per_char'], 2)}个，因为很多汉字还没来得及被合并成一个整体。",
        "生成式预训练模型第二代开始就用字节级的字节对编码，今天的大多数模型也是如此。词表足够大、训练语料里中文足够多时，常用汉字都会成为单独的词元。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，从字符开始合并，起点词表很大，还会有未知字符。",
        "第二，字节级分词从二百五十六个字节开始，任何文本都能编码。",
        "第三，代价是语料中少见的文字会被拆成好几个词元。下一期，我们看分词带来的各种怪现象。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "字节级分词", "再也没有未知字符", SERIES)


def enc_row(d, y, toks, col, x0=360, unk_col=RED):
    f = vk.font(20, mono=True)
    x = x0
    for t in toks:
        s = t.replace(" ", "␣") if t else "?"
        is_unk = t == "<unk>" or t == "�"
        s = "UNK" if t == "<unk>" else ("◆" if t == "�" else s)
        wdt = 20 + 13 * len(s) if all(ord(ch) < 0x2E80 for ch in s) else 20 + 26 * len(s)
        tv.token(d, x, y, s, unk_col if t == "<unk>" else (AMBER if t == "�" else col), w=wdt, h=40, font=f if all(ord(ch) < 0x2E80 for ch in s) else F_SMALL)
        x += wdt + 6


def sc_problem(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, f"从字符开始：起点词表 {B['char_vocab_base']:,}，仍有未知字符")
    rows = ["zh", "emoji", "rare"] if seg >= 1 else ["en"]
    for i, k in enumerate(rows):
        y = 200 + i * 110
        d.text((120, y + 6), B[k]["text"].strip().replace("🍕", "[披萨表情]"), font=F_SMALL, fill=MUTED)
        enc_row(d, y + 44, B[k]["char"], CYAN, x0=120)


def sc_bytes(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "从字节开始：任何文本都能编码（◆ = 字节片段）")
    for i, k in enumerate(["zh", "emoji", "rare"]):
        y = 190 + i * 110
        d.text((120, y + 6), B[k]["text"].strip().replace("🍕", "[披萨表情]"), font=F_SMALL, fill=MUTED)
        enc_row(d, y + 44, B[k]["byte"] if seg >= 0 else B[k]["char"], GREEN, x0=120)
    if seg >= 1:
        vk.text_c(d, (640, 640), f"没有未知词 · 语料里只出现了 {B['byte_vocab_base']} 种字节（最多 256）", F_SMALL, AMBER)


def sc_cost(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "代价：少见的文字被拆成多个词元（2,000 次合并）")
    tv.bar_chart(d, (200, 220, 760, 560), ["英文", "中文"], [B["en_tok_per_char"], B["zh_tok_per_char"]], [CYAN, AMBER], fmt="{:.2f}", vmax=1.5)
    d.text((200, 180), "每个字符的词元数", font=F_SMALL, fill=MUTED)
    d.text((840, 300), f"一个汉字 = 3 个 UTF-8 字节", font=F_BODY, fill=TEXT)
    if seg >= 1:
        d.text((840, 380), "GPT-2 起：字节级 BPE", font=F_SMALL, fill=GREEN)
        d.text((840, 420), "大词表 + 足够中文语料：", font=F_SMALL, fill=GREEN)
        d.text((840, 455), "常用汉字成为单独词元", font=F_SMALL, fill=GREEN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 4, [("字符起点", "词表大，仍有未知字符"), ("字节起点", "256 个字节，任何文本都能编码"),
                      ("代价", "少见文字拆成多个词元")], seg, "下一期：分词带来的怪现象")


RENDER = {"title": sc_title, "problem": sc_problem, "bytes": sc_bytes, "cost": sc_cost, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "tok-bytes")
