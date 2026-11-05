"""Tokenization 06 — 分词带来的怪现象.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] TOK_DATA=<dir> TOK_CACHE=<dir> python tok06_quirks.py <out_dir>

Real computations (tokx.exp_numbers): (1) a BPE with 600 merges learned on 20,000 random integers below 100,000
(written with a leading space): how eight numbers are split; (2) BPE with 2,000 merges on Tiny Shakespeare: how
"strawberry", "Mississippi" and "bookkeeper" are split; (3) tokx.exp_units: " the" vs "the" vs " The" are different
pieces. Facts about real systems are quoted with references in the notes.
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

NU = tokx.cached(CACHE, "numbers", tokx.exp_numbers)
SEG, LET = NU["seg"], NU["letters"]
print(SEG, LET)
assert SEG[" 12345"] == [" 123", "45"] and SEG[" 31415"] == [" 3", "14", "15"] and SEG[" 1000"] == [" 1", "00", "0"]
assert SEG[" 2025"] == [" 20", "25"] and SEG[" 777"] == [" 7", "77"]
assert LET[" strawberry"] == [" stra", "w", "ber", "ry"] and LET[" Mississippi"] == [" M", "iss", "iss", "i", "pp", "i"]

SCENES = [
    ("title", [
        "大模型有一些让人意外的弱点：数不清一个单词里有几个某字母，多位数的加法常常出错，对空格和大小写很敏感。这些问题有不少和分词有关。这一期看分词带来的怪现象。",
    ]),
    ("letters", [
        "先看数字母。模型看到的不是字母，而是词元。在我们的分词器里，草莓这个英文单词被切成四块，其中好几个字母被藏在同一块里。模型要回答里面有几个某个字母，就得记住每个词元由哪些字母组成，这是它从来没有被直接教过的。",
        "另一个例子是一个美国州名，被切成六块，其中两个片段长得一模一样。人一眼能数清的字母，模型要经过一次间接的回忆。",
    ]),
    ("numbers", [
        "再看数字。我们在两万个随机整数上学了一个字节对编码。结果，一二三四五被切成一二三和四五，三一四一五却被切成三、一四、一五。一千被切成一、零零、零。",
        "也就是说，同样是五位数，切法完全不一样，同一个位置上的数字落在不同的词元里。做加法时，模型必须先弄清楚每个词元代表第几位，再去对齐、进位，这比按位处理难得多。",
    ]),
    ("fixes", [
        "真实的模型针对数字做了很多处理。第二代羊驼把每个数字都切成单独的词元；另一些分词器把数字按每三位一组从左往右切。二零二四年的一项研究发现，把数字从右往左每三位一组切分，让大模型的算术准确率明显提高，因为这样同一个词元总是对应同一组数位。",
    ]),
    ("space", [
        "空格和大小写也会带来差别。带空格的单词和不带空格的同一个单词，是两个完全不同的词元；首字母大写又是另一个。模型要分别学会它们的意思。一句话开头多一个空格，就可能让模型看到一串从没见过的组合。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，模型看到的是词元而不是字母，所以数字母需要间接回忆。",
        "第二，数字被不一致地切分，让算术更难；按位或从右往左分组能缓解。",
        "第三，空格和大小写会产生不同的词元。下一期，我们看多语言的不公平。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "分词带来的怪现象", "数字母、算术与空格", SERIES)


def row(d, y, label, segs, col, x0=420):
    d.text((140, y + 8), label, font=vk.font(24, mono=True), fill=MUTED)
    f = vk.font(24, mono=True)
    x = x0
    for s in segs:
        wdt = 24 + 16 * len(s)
        tv.token(d, x, y, s.replace(" ", "␣"), col, w=wdt, h=46, font=f)
        x += wdt + 8


def sc_letters(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "模型看到的是词元，不是字母")
    row(d, 210, "strawberry", LET[" strawberry"], CYAN)
    if seg >= 1:
        row(d, 310, "Mississippi", LET[" Mississippi"], AMBER)
        row(d, 410, "bookkeeper", LET[" bookkeeper"], VIOLET)
    vk.text_c(d, (640, 560), "数字母 = 回忆每个词元由哪些字母组成", F_BODY, TEXT)


def sc_numbers(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "数字的切分并不一致（BPE，学自 2 万个随机整数）")
    keys = [" 12345", " 12346", " 31415", " 99999", " 2025", " 1000", " 777"]
    for i, k in enumerate(keys):
        y = 175 + i * 62
        row(d, y, k.strip(), SEG[k], RED if k in (" 31415", " 1000") else GREEN, x0=380)
    if seg >= 1:
        d.text((820, 320), "同一位置的数字", font=F_SMALL, fill=AMBER)
        d.text((820, 360), "落在不同的词元里", font=F_SMALL, fill=AMBER)
        d.text((820, 420), "→ 对齐、进位更难", font=F_SMALL, fill=RED)


def sc_fixes(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "真实模型怎么处理数字")
    rows = [("每个数字单独切分", "1|2|3|4|5", "如 Llama 2", CYAN), ("从左往右每三位", "123 | 45", "部分分词器", AMBER),
            ("从右往左每三位", "12 | 345", "Singh & Strouse 2024：更准", GREEN)]
    for i, (a, b, cc, col) in enumerate(rows):
        y = 200 + i * 110
        d.rounded_rectangle([120, y, 1160, y + 80], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((150, y + 40), a, font=F_BODY, fill=col, anchor="lm")
        d.text((470, y + 40), b, font=vk.font(26, mono=True), fill=TEXT, anchor="lm")
        d.text((800, y + 40), cc, font=F_SMALL, fill=MUTED, anchor="lm")


def sc_space(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "空格与大小写：不同的词元")
    for i, (w, col) in enumerate([("␣the", GREEN), ("the", CYAN), ("␣The", AMBER), ("THE", VIOLET)]):
        tv.token(d, 200 + i * 230, 260, w, col, w=170, h=70, font=vk.font(30, mono=True))
    vk.text_c(d, (640, 420), "四个不同的编号，模型要分别学会它们的意思", F_BODY, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("字母", "看到的是词元，数字母要间接回忆"), ("数字", "切分不一致，算术更难"),
                      ("空格大小写", "产生不同的词元")], seg, "下一期：多语言的不公平")


RENDER = {"title": sc_title, "letters": sc_letters, "numbers": sc_numbers, "fixes": sc_fixes, "space": sc_space, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "tok-quirks")
