"""Transformer 01 — 词元化与嵌入：文字怎样变成向量.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python tf01_tokens.py <out_dir>

Real computations: byte-pair encoding run on a small corpus (Sennrich et al. style example) and
8-dimensional word vectors from PPMI co-occurrence statistics + SVD on a template corpus.
"""

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL, F_TITLE
from zhnum import int_zh, dec_zh

# ---------------------------------------------------------------------------
# BPE
# ---------------------------------------------------------------------------
CORPUS = ("low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3 + "slow " * 2 + "lowest " * 2 +
          "newer " * 3 + "wider " * 2).split()
WORDS0 = collections.Counter(tuple(w) + ("_",) for w in CORPUS)  # "_" marks the end of a word


def pair_counts(words):
    c = collections.Counter()
    for w, f in words.items():
        for a, b in zip(w, w[1:]):
            c[(a, b)] += f
    return c


def merge(words, pair):
    out = {}
    for w, f in words.items():
        n, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
                n.append(w[i] + w[i + 1])
                i += 2
            else:
                n.append(w[i])
                i += 1
        out[tuple(n)] = f
    return out


def n_tokens(ws):
    return sum(len(w) * f for w, f in ws.items())


STATES, MERGES = [WORDS0], []
for _ in range(10):
    pair, cnt = pair_counts(STATES[-1]).most_common(1)[0]
    MERGES.append((pair, cnt))
    STATES.append(merge(STATES[-1], pair))
NW = len(CORPUS)
TOK0, TOK10 = n_tokens(STATES[0]), n_tokens(STATES[-1])
print("BPE tokens", TOK0, "->", TOK10, "merges", [("".join(p), c) for p, c in MERGES])
assert (TOK0, TOK10, NW) == (149, 58, 25)

# ---------------------------------------------------------------------------
# Word vectors from co-occurrence (PPMI + SVD)
# ---------------------------------------------------------------------------
ANIMALS = {"cat": ["meows", "purrs", "sleeps", "hunts"], "dog": ["barks", "runs", "fetches", "sleeps"],
           "bird": ["sings", "flies", "nests", "hunts"], "horse": ["runs", "gallops", "jumps", "neighs"]}
FOODS = {"rice": ["white", "steamed", "soft", "fried"], "bread": ["soft", "fresh", "baked", "toasted"],
         "apple": ["sweet", "red", "fresh", "juicy"], "noodle": ["long", "hot", "fried", "spicy"]}
rng = np.random.default_rng(0)
SENTS = []
for _ in range(6000):
    if rng.random() < 0.5:
        a = rng.choice(list(ANIMALS))
        SENTS.append(rng.choice([f"my {a} is cute", f"the {a} {rng.choice(ANIMALS[a])}",
                                 f"a {a} {rng.choice(ANIMALS[a])} outside", f"i feed the {a}"]))
    else:
        f = rng.choice(list(FOODS))
        SENTS.append(rng.choice([f"i eat {f}", f"the {f} is {rng.choice(FOODS[f])}",
                                 f"{rng.choice(FOODS[f])} {f} for lunch", f"we cook {f}"]))
VOC = sorted({w for s in SENTS for w in s.split()})
IX = {w: i for i, w in enumerate(VOC)}
C = np.zeros((len(VOC), len(VOC)))
for s in SENTS:
    t = s.split()
    for i, w in enumerate(t):
        for j in range(max(0, i - 2), min(len(t), i + 3)):
            if j != i:
                C[IX[w], IX[t[j]]] += 1
P = C / C.sum()
pw = P.sum(1, keepdims=True)
PPMI = np.maximum(np.log(P / (pw @ pw.T) + 1e-12), 0)
U, SV, _ = np.linalg.svd(PPMI)
DIM = 16
EMB = U[:, :DIM] * SV[:DIM]


def cos(a, b):
    x, y = EMB[IX[a]], EMB[IX[b]]
    return float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))


C_CD, C_RB, C_CR = cos("cat", "dog"), cos("rice", "bread"), cos("cat", "rice")
print(f"cos cat-dog {C_CD:.2f} rice-bread {C_RB:.2f} cat-rice {C_CR:.2f}")
assert 0.75 < C_CD < 0.9 and 0.3 < C_RB < 0.45 and abs(C_CR) < 0.1
SHOW = ["cat", "dog", "bird", "horse", "rice", "bread", "apple", "noodle"]
ZH = {"cat": "猫", "dog": "狗", "bird": "鸟", "horse": "马", "rice": "米饭", "bread": "面包", "apple": "苹果", "noodle": "面条"}
SIM = np.array([[cos(a, b) for b in SHOW] for a in SHOW])

# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------
SCENES = [
    ("title", [
        "欢迎来到变换器系列。这个系列我们从零开始，一步一步搭出一个完整的变换器模型。",
        "第一期讲最开始的一步，文字怎样变成模型能够处理的向量。",
    ]),
    ("split", [
        "模型不能直接读文字，第一步是把文字切成一个个小单元，叫做词元。",
        "最简单的切法是按字符切，但这样序列很长；按整个词切，词表又太大，遇到没见过的词也无能为力。",
        "折中的办法叫字节对编码，从单个字符出发，反复把最常一起出现的相邻一对，合并成一个新的词元。",
    ]),
    ("bpe", [
        f"我们在一个小语料上真实地运行这个算法。语料里有{int_zh(NW)}个词，一开始按字符切，一共{int_zh(TOK0)}个词元。",
        "每一步都合并当前最常见的相邻对，逐渐合并出常见的词尾，以及整个常见的词。",
        f"合并十次以后，词元总数从{int_zh(TOK0)}降到{int_zh(TOK10)}，平均每个词从大约六个词元，减少到两个多。常见的词变成一个词元，少见的词仍然可以拆成片段来表示。",
    ]),
    ("embed", [
        "切好以后，每个词元对应词表里的一个编号，再通过一张嵌入表，查出它的向量。",
        "这张表就是一个矩阵，每一行是一个词元的向量，查表等价于用独热向量去乘这个矩阵。",
        "嵌入向量是在训练中学出来的，经常出现在相似上下文里的词，向量也会彼此靠近。",
    ]),
    ("geometry", [
        "我们用一个小语料来演示这一点。这里没有训练神经网络，而是用词与词的共现统计，算出每个词的十六维向量。",
        f"可以看到，猫和狗的余弦相似度是{dec_zh(C_CD)}，米饭和面包是{dec_zh(C_RB)}，而猫和米饭的相似度几乎为零。动物和食物在向量空间里，自然地分成了两组。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，词元化把文字切成词元，字节对编码通过反复合并常见的相邻对，来建立词表。",
        "第二，每个词元查嵌入表得到一个向量，查表等价于独热向量乘以矩阵。",
        "第三，嵌入是学出来的，上下文相似的词，向量也相近。",
        "下一期，我们讲变换器的核心，注意力机制。感谢大家的观看。",
    ]),
]

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
MONO = vk.font(22, mono=True)


def show_word(w):
    return "".join(w).replace("_", "·")


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "词元化与嵌入", "文字怎样变成向量")


def sc_split(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "词元：模型处理文字的最小单元")
    vk.text_c(d, (640, 150), "unbelievable", vk.font(40, mono=True), TEXT)
    rows = [("按字符", list("unbelievable"), MUTED, 1), ("按整词", ["unbelievable"], VIOLET, 1), ("子词（BPE）", ["un", "believ", "able"], CYAN, 2)]
    for i, (name, toks, col, need) in enumerate(rows):
        if seg >= need or (seg == 1 and i < 2):
            y = 230 + i * 100
            d.text((180, y + 22), name, font=F_SMALL, fill=col, anchor="rm")
            tv.tokens_row(d, 210, y, toks, col, font=MONO)
    if seg >= 1:
        d.text((1080, 252), "序列太长", font=F_SMALL, fill=MUTED, anchor="rm")
        d.text((1080, 352), "词表太大", font=F_SMALL, fill=MUTED, anchor="rm")
    if seg >= 2:
        d.text((1080, 452), "折中", font=F_SMALL, fill=CYAN, anchor="rm")
        vk.text_c(d, (640, 560), "（子词切分仅为示意）", F_SMALL, MUTED)


def sc_bpe(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "字节对编码（BPE）：真实运行 10 次合并")
    k = 0 if seg == 0 else (int(10 * vk.ease(min(1, su * 1.2))) if seg == 1 else 10)
    ws = sorted(STATES[k].items(), key=lambda x: -x[1])
    y = 130
    for w, f in ws:
        d.text((70, y + 20), f"×{f}", font=vk.font(20, mono=True), fill=MUTED, anchor="lm")
        tv.tokens_row(d, 120, y, [s.replace("_", "·") for s in w], CYAN, gap=4, font=vk.font(20, mono=True))
        y += 52
    d.text((820, 130), "合并记录", font=F_SMALL, fill=VIOLET)
    for i in range(k):
        (a, b), cnt = MERGES[i]
        d.text((820, 170 + i * 34), f"{i + 1:2d}. {a.replace('_', '·')} + {b.replace('_', '·')}  ({cnt})", font=vk.font(20, mono=True), fill=TEXT)
    vk.text_c(d, (640, 570), f"词元总数 {n_tokens(STATES[k])}（{NW} 个词，· 表示词尾）", F_BODY, GREEN if k == 10 else TEXT)


def sc_embed(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "嵌入表：编号 → 向量")
    tv.tokens_row(d, 120, 160, ["i", "eat", "rice"], CYAN, font=MONO)
    d.text((120, 230), "编号：", font=F_SMALL, fill=MUTED)
    d.text((200, 230), f"{IX['i']}   {IX['eat']}   {IX['rice']}", font=vk.font(22, mono=True), fill=TEXT)
    if seg >= 1:
        M = EMB[:12, :8]
        tv.heat(d, (700, 140, 1100, 470), M, labels_y=VOC[:12], cmap="violet")
        vk.text_c(d, (900, 500), "嵌入矩阵 E（每行一个词元，只画前 8 维）", F_SMALL, MUTED)
        tv.fpaste(c, r"$e_i=E^{\top}\mathbf{1}_i$", 360, 360, 34, "#a78bfa")
        vk.text_c(d, (360, 420), "查表 = 独热向量 × 矩阵", F_SMALL, MUTED)
    if seg >= 2:
        vk.text_c(d, (640, 560), "E 是可训练参数，随模型一起学习", F_BODY, GREEN)


def sc_geometry(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "相似的上下文 → 相近的向量")
    box = (380, 140, 860, 540)
    tv.heat(d, box, SIM, vmin=-0.2, vmax=1.0, labels_x=[ZH[w] for w in SHOW], labels_y=[ZH[w] for w in SHOW],
            fmt="{:.2f}", font_size=16)
    vk.text_c(d, (620, 575), f"余弦相似度（{DIM} 维向量，由共现统计 + SVD 得到）", F_SMALL, MUTED)
    if seg >= 1:
        cw = (box[2] - box[0]) / 8
        for k, (lab, col) in enumerate([("动物", CYAN), ("食物", GREEN)]):
            x0, y0 = box[0] + 4 * k * cw, box[1] + 4 * k * (box[3] - box[1]) / 8
            d.rectangle([x0 - 2, y0 - 2, x0 + 4 * cw, y0 + 4 * (box[3] - box[1]) / 8], outline=col, width=3)
            d.text((box[2] + 30, y0 + 2 * (box[3] - box[1]) / 8), lab, font=F_BODY, fill=col, anchor="lm")


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("词元化", "字节对编码：反复合并常见相邻对"), ("嵌入表", "编号查表得到向量 = 独热 × 矩阵"),
                      ("向量几何", "上下文相似的词，向量相近")], seg, "下一期：注意力机制 —— 查询、键与值")


RENDER = {"title": sc_title, "split": sc_split, "bpe": sc_bpe, "embed": sc_embed, "geometry": sc_geometry, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "tokens")
