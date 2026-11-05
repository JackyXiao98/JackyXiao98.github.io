"""Tokenization 03 — WordPiece 与 Unigram：另外两种思路.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] TOK_DATA=<dir> TOK_CACHE=<dir> python tok03_wpuni.py <out_dir>

Real computations (tokx.exp_wp_unigram, Tiny Shakespeare words with leading spaces):
- the 10 adjacent pairs with the largest count (what BPE merges first) vs the 10 with the largest WordPiece-style
  score count(ab) / (count(a) count(b)) among pairs seen at least 20 times;
- a simplified Unigram language-model tokenizer (seeded with the 6,000 most frequent substrings of length <= 8 of
  the 3,000 most frequent words, hard-EM with Viterbi segmentation, pruning 20% per round down to about 1,000
  pieces) vs BPE with 1,000 pieces: segmentations of five words.
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

R = tokx.cached(CACHE, "wp_unigram", tokx.exp_wp_unigram)
BC, BS = R["scores"]["by_count"], R["scores"]["by_score"]
SU, SB = R["seg_uni"], R["seg_bpe"]
print(BC[:5], BS[:5], R["uni_size"], SU, SB)
assert BC[0] == (" t", 23835) and BS[0][0] == "QU" and BS[0][1] == 216 and R["uni_size"] == 1063
assert SU[" kingdom"] == [" king", "dom"] and SB[" kingdom"] == [" king", "d", "om"]
assert SU[" understanding"] == [" under", "st", "and", "ing"] and SB[" understanding"] == [" u", "nder", "st", "and", "ing"]

SCENES = [
    ("title", [
        "字节对编码按出现次数合并，是最常用的方法，但不是唯一的。这一期看另外两种：谷歌在早期模型里用的词片方法，以及从大词表往下删减的一元语言模型方法。",
    ]),
    ("wordpiece", [
        "词片方法同样是一步步合并，但挑选的标准不同。它不看一对出现了多少次，而是看这一对出现的次数，除以两个片段各自出现次数的乘积。也就是说，它偏爱那些单独很少见、但几乎总是一起出现的组合。",
        f"在莎士比亚的文本上，按次数排第一的是空格加一个常见字母，出现了两万多次。按词片的分数排，前几名变成了大写字母组合，比如剧本里角色名的开头，这一对只出现了{int_zh(BS[0][1])}次，但它的两个字母单独都很少见，所以分数最高。",
    ]),
    ("unigram", [
        "一元语言模型方法反过来做：先准备一个很大的候选词表，比如所有常见的子串，给每个片段一个概率。一个词的切法，就是让各片段概率之积最大的那种切法，可以用动态规划求出来。",
        "然后反复做两件事：用当前的切法统计每个片段被用了多少次，更新概率；再删掉最没用的那一部分片段。我们从六千个候选开始，每轮删掉五分之一，几轮后剩下约一千个。",
    ]),
    ("compare", [
        "两种方法切出来的结果有什么不同？在同样约一千个片段的词表下，字节对编码把王国这个词切成三块，一元模型切成王和后缀两块；理解这个词，一元模型切出了“在下面”这个完整的前缀。",
        "一元模型的切分常常更接近词的构成，而且它对一个词可以给出多种切法的概率，训练时可以随机采样不同的切法，起到正则化的作用。",
    ]),
    ("real", [
        "这些方法在工具里经常一起出现。句子片段这个库支持字节对编码和一元模型两种算法，并且把空格也当成普通字符处理，不需要先按空格切词，所以对中文、日文这样不用空格的语言很方便。很多多语言模型都用它训练词表。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，词片方法按共同出现的程度挑选合并，而不是按次数。",
        "第二，一元模型方法从大词表出发，按概率切分，逐步删减。",
        "第三，在我们的实验里，一元模型的切分更接近词的构成。下一期，我们看词表应该多大。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "WordPiece 与 Unigram", "另外两种思路", SERIES)


def sc_wordpiece(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "WordPiece：按“共同出现的程度”挑选")
    tv.fpaste(c, r"$\mathrm{BPE}:\ \mathrm{count}(ab)\qquad \mathrm{WordPiece}:\ \frac{\mathrm{count}(ab)}{\mathrm{count}(a)\,\mathrm{count}(b)}$", 640, 190, 34)
    if seg >= 1:
        f = vk.font(22, mono=True)
        d.text((180, 270), "按次数（BPE 先合并）", font=F_SMALL, fill=CYAN)
        d.text((720, 270), "按 WordPiece 分数", font=F_SMALL, fill=AMBER)
        for i in range(6):
            a, cnt = BC[i]
            d.text((180, 320 + i * 44), f"{a.replace(' ', '␣'):<6}{cnt:>8,}", font=f, fill=TEXT)
            b, cnt2, sc = BS[i]
            d.text((720, 320 + i * 44), f"{b:<6}{cnt2:>6}   {sc:.1e}", font=f, fill=TEXT)


def sc_unigram(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "Unigram：从大词表出发，按概率切分，逐步删减")
    tv.fpaste(c, r"$\mathrm{seg}^*(w)=\arg\max_{s_1\cdots s_k=w}\ \prod_i p(s_i)$", 640, 190, 38)
    if seg >= 1:
        hist = R["uni_hist"]
        tv.bar_chart(d, (250, 320, 1030, 610), [f"第 {i + 1} 轮" for i in range(len(hist))], [h[0] for h in hist], [VIOLET] * len(hist), fmt="{:.0f}", vmax=2400)
        d.text((250, 285), "每轮 EM 之后的词表大小（6,000 个候选 → 约 1,000）", font=F_SMALL, fill=MUTED)


def seg_row(d, y, word, segs, col, x0):
    f = vk.font(22, mono=True)
    x = x0
    for s in segs:
        wdt = 22 + 14 * len(s)
        tv.token(d, x, y, s.replace(" ", "␣"), col, w=wdt, h=42, font=f)
        x += wdt + 6


def sc_compare(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "同样约 1,000 个片段：BPE vs Unigram")
    d.text((420, 165), "BPE", font=F_SMALL, fill=CYAN)
    d.text((820, 165), "Unigram", font=F_SMALL, fill=GREEN)
    for i, w in enumerate(SU):
        y = 210 + i * 70
        d.text((120, y + 8), w.strip(), font=vk.font(22, mono=True), fill=MUTED)
        seg_row(d, y, w, SB[w], CYAN, 420)
        seg_row(d, y, w, SU[w], GREEN, 820)
    if seg >= 1:
        vk.text_c(d, (640, 610), "Unigram 还能给出多种切法的概率 → 训练时采样（subword regularization）", F_SMALL, AMBER)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "SentencePiece")
    for i, (t, col) in enumerate([("支持 BPE 与 Unigram 两种算法", CYAN), ("空格当作普通字符（▁）：不需要先按空格切词", GREEN), ("适合中文、日文等不用空格的语言", AMBER)]):
        vk.text_c(d, (640, 230 + i * 80), t, F_BODY, col)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("WordPiece", "按共同出现程度合并"), ("Unigram", "大词表出发，概率切分，逐步删减"),
                      ("实验", "Unigram 切分更接近词的构成")], seg, "下一期：词表大小的权衡")


RENDER = {"title": sc_title, "wordpiece": sc_wordpiece, "unigram": sc_unigram, "compare": sc_compare, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "tok-wpuni")
