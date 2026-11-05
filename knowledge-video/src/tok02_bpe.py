"""Tokenization 02 — BPE：一步步合并最常见的一对.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] TOK_DATA=<dir> TOK_CACHE=<dir> python tok02_bpe.py <out_dir>

Real computations: (1) tokx.exp_toy_bpe - byte-pair encoding on the classic toy corpus {low: 5, lower: 2, newest: 6,
widest: 3} with an end-of-word marker "_", six merges (ties broken by the larger pair in lexicographic order);
(2) tokx.exp_units - BPE learned on the first 90% of Tiny Shakespeare (pre-split into words that keep their leading
space): the first 16 merges with their counts and the total number of tokens in the training text as the vocabulary
grows from 65 characters to about 4,000 pieces.
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

TOY = tokx.cached(CACHE, "toy_bpe", tokx.exp_toy_bpe)
U = tokx.cached(CACHE, "units", tokx.exp_units)
ST = TOY["steps"]
CURVE = U["bpe_curve"]
N0 = int(U["chars"] * 0.9)
print([(s["pair"], s["count"]) for s in ST], U["bpe_first"][:6], U["bpe_counts"][:6], CURVE[0], CURVE[3], CURVE[-1])
assert [s["pair"] for s in ST[:3]] == [("t", "_"), ("s", "t_"), ("e", "st_")] and ST[0]["count"] == 9 and ST[3]["pair"] == ("o", "w")
assert U["bpe_first"][0] == (" ", "t") and U["bpe_counts"][0] == 21589 and U["bpe_first"][10] == (" t", "he")
assert CURVE[3] == (1065, 387537) and CURVE[-1] == (4065, 305042)

SCENES = [
    ("title", [
        "上一期说到，子词分词在字符和词之间取得平衡。最常用的子词方法叫字节对编码。它的规则简单得出奇：反复找出最常见的相邻一对，把它们合并成一个新的片段。这一期我们一步步看。",
    ]),
    ("toy", [
        "我们用一个经典的小例子：语料里有五个词，四个词，加上表示词尾的下划线。一开始，每个词都拆成单个字母。",
        f"第一步，统计所有相邻的两个片段出现了多少次。屏幕上这一对，最后一个字母和词尾连在一起，出现了{int_zh(ST[0]['count'])}次，和其他几对并列最多，我们把它合并成一个新片段。",
        "然后重新统计，再合并最多的一对。几步之后，最高级的词尾成了一个片段，“低”这个词也成了一个片段。每合并一次，词表就多一个片段，文本里的片段总数也变少一些。",
    ]),
    ("shakespeare", [
        f"在莎士比亚的文本上做同样的事。最先被合并的是一个空格加一个常见字母，出现了{big_zh(U['bpe_counts'][0])}次。到了第十一步，英语里最常见的冠词，连同它前面的空格，已经成了一个完整的片段。",
        "我们保留每个词前面的空格，所以同一个词出现在句首和句中会是不同的片段。这也是生成式预训练模型第二代的做法。",
    ]),
    ("curve", [
        f"随着合并次数增加，训练文本的片段数一路下降。六十五个字符时，一百万个字符就是一百万个片段；词表到一千左右，降到约{big_zh(round(CURVE[3][1] / 10000))}万；到四千，约{big_zh(round(CURVE[-1][1] / 10000))}万。",
        "但下降越来越慢：前面的合并都是高频的组合，后面合并的越来越罕见，每一步省下的片段越来越少。词表多大合适，是下一个要权衡的问题。",
    ]),
    ("encode", [
        "学好以后怎么用？对一段新文本，先切成单个字符，再按照训练时合并的先后顺序，能合并的就合并，直到没有可用的规则。所以字节对编码的输出完全由学到的合并顺序决定。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，字节对编码反复合并最常见的相邻一对，每次给词表加一个片段。",
        "第二，在莎士比亚文本上，片段数随词表增大而下降，但越来越慢。",
        "第三，编码时按合并的先后顺序应用规则。下一期，我们看另外两种子词方法。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "BPE", "一步步合并最常见的一对", SERIES)


def show_words(d, seqs, y0, hi=None):
    f = vk.font(22, mono=True)
    for i, (w, sq) in enumerate(seqs.items()):
        y = y0 + i * 62
        d.text((120, y + 10), f"{w.rstrip('_')} × {TOY['corpus'][w.rstrip('_')]}", font=F_SMALL, fill=MUTED)
        x = 360
        for s in sq:
            wdt = 24 + 16 * len(s)
            col = GREEN if (hi and s == hi) else CYAN
            tv.token(d, x, y, s, col, w=wdt, h=44, font=f)
            x += wdt + 8


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "经典小例子（“_” = 词尾）")
    if seg == 0:
        show_words(d, {w: list(w) for w in ST[0]["seqs"]}, 190)
    elif seg == 1:
        s = ST[0]
        show_words(d, s["seqs"], 190, s["pair"][0] + s["pair"][1])
        d.text((860, 220), f"合并 {s['pair'][0]} + {s['pair'][1]}", font=F_BODY, fill=GREEN)
        d.text((860, 270), f"出现 {s['count']} 次", font=F_SMALL, fill=TEXT)
    else:
        k = min(len(ST) - 1, 1 + int(su * (len(ST) - 1)))
        s = ST[k]
        show_words(d, s["seqs"], 190, s["pair"][0] + s["pair"][1])
        for j in range(k + 1):
            p = ST[j]["pair"]
            d.text((860, 200 + j * 40), f"{j + 1}. {p[0]} + {p[1]}  ({ST[j]['count']})", font=vk.font(22, mono=True), fill=GREEN if j == k else TEXT)


def sc_shakespeare(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "莎士比亚：最先被合并的 16 对（␣ = 空格）")
    f = vk.font(22, mono=True)
    for i, ((a, b), cnt) in enumerate(zip(U["bpe_first"], U["bpe_counts"])):
        col, row = i // 8, i % 8
        x, y = 140 + col * 520, 175 + row * 52
        sa, sb = a.replace(" ", "␣").replace("\n", "↵"), b.replace(" ", "␣").replace("\n", "↵")
        d.text((x, y), f"{i + 1:>2}. {sa} + {sb}", font=f, fill=GREEN if i == 10 else TEXT)
        d.text((x + 300, y), f"{cnt:,}", font=f, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 620), "保留词前的空格：“␣the” 与 “the” 是不同的片段（GPT-2 的做法）", F_SMALL, AMBER)


def sc_curve(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "词表越大，片段越少，但越来越慢")
    xs = [65] + [v for v, _ in CURVE]
    ys = [N0] + [n for _, n in CURVE]
    tv.line_chart(d, (140, 200, 840, 580), xs, [("训练文本的片段数", np.array(ys) / 1e4, GREEN)], ylim=(0, 105),
                  xticks=[(65, "65"), (1065, "1065"), (2065, "2065"), (4065, "4065")], xlabel="词表大小")
    d.text((140, 165), "片段数（万）", font=F_SMALL, fill=MUTED)
    f = vk.font(22, mono=True)
    for i, (v, n) in enumerate([(65, N0), CURVE[3], CURVE[-1]]):
        d.text((900, 260 + i * 50), f"{v:>5} → {n:,}", font=f, fill=TEXT)
    if seg >= 1:
        vk.text_c(d, (1030, 470), "后面的合并越来越罕见", F_SMALL, AMBER)


def sc_encode(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "编码：按合并的先后顺序应用规则")
    w = " understanding"
    segs = U["examples"][w]
    f = vk.font(24, mono=True)
    stages = [list(w), segs]
    for r, sq in enumerate(stages):
        x = 160
        y = 220 + r * 140
        for s in sq:
            wdt = 24 + 15 * len(s)
            tv.token(d, x, y, s.replace(" ", "␣"), CYAN if r == 0 else GREEN, w=wdt, font=f)
            x += wdt + 8
    vk.arrow(d, (400, 285), (400, 345), MUTED, 3)
    d.text((430, 300), "按学到的顺序合并", font=F_SMALL, fill=MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("训练", "反复合并最常见的相邻一对"), ("压缩", "词表越大片段越少，但越来越慢"),
                      ("编码", "按合并顺序应用规则")], seg, "下一期：WordPiece 与 Unigram")


RENDER = {"title": sc_title, "toy": sc_toy, "shakespeare": sc_shakespeare, "curve": sc_curve, "encode": sc_encode, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "tok-bpe")
