"""MoE 05 — 专家学到了什么：分工与专业化.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MOE_CACHE=<dir> python moe05_special.py <out_dir>

Real computations (moex.exp_special, JAX): the balanced MoE of episode 01 (8 x 128, top-1, auxiliary loss 0.01,
2,000 steps). For all 256 x 32 held-out tokens we record the top-1 expert in each layer and measure the mutual
information (bits; at most log2 8 = 3) between the chosen expert and (a) the hidden language of the sequence,
(b) the current token id, overall and for positions 0-1 / 2-7 / 8-31; plus the language x expert routing table.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import moex
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Mixture of Experts"
CACHE = os.environ.get("MOE_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "moecache"))

SP = moex.cached(CACHE, "special", moex.exp_special)
LANG = np.repeat(SP["eval_lang"][:, None], moex.LEN, 1)
TOK = SP["eval_x"][:, :-1]


def mi(a, b, na, nb):
    J = np.zeros((na, nb))
    np.add.at(J, (a.ravel(), b.ravel()), 1)
    J /= J.sum()
    pa, pb = J.sum(1, keepdims=True), J.sum(0, keepdims=True)
    m = J > 0
    return float((J[m] * np.log2(J[m] / (pa @ pb)[m])).sum())


def table(a, R, na):
    M = np.zeros((na, 8))
    np.add.at(M, (a.ravel(), R.ravel()), 1)
    return M / M.sum(1, keepdims=True)


R1, R2 = SP["route"]
MI = {(l, w): mi(LANG if w == "lang" else TOK, R, 8 if w == "lang" else 32, 8) for l, R in ((1, R1), (2, R2)) for w in ("lang", "tok")}
SPANS = [(0, 2), (2, 8), (8, 32)]
MIP = {w: [mi((LANG if w == "lang" else TOK)[:, a:b], R2[:, a:b], 8 if w == "lang" else 32, 8) for a, b in SPANS] for w in ("lang", "tok")}
T2 = table(LANG, R2, 8)
T1 = table(TOK, R1, 32)
PUR2 = float(T2.max(1).mean())
print({k: round(v, 2) for k, v in MI.items()}, {k: [round(x, 2) for x in v] for k, v in MIP.items()}, round(PUR2, 3))
assert round(MI[(1, "tok")], 1) == 2.4 and round(MI[(1, "lang")], 2) == 0.11
assert round(MI[(2, "lang")], 1) == 1.8 and round(MI[(2, "tok")], 1) == 0.7 and round(PUR2, 2) == 0.77
assert [round(x, 1) for x in MIP["lang"]] == [0.2, 1.6, 2.1] and round(MIP["tok"][0], 1) == 1.5

SCENES = [
    ("title", [
        "混合专家有很多专家，人们自然会问：每个专家是不是成了某个领域的专家，比如一个管数学，一个管代码？这一期我们打开小模型的路由，看看专家到底按什么分工。",
    ]),
    ("measure", [
        "我们记录留出数据里每个词元在每一层被分给了哪个专家，再看这个选择和两样东西有多大关系：这段序列来自哪种语言，以及当前词元本身是哪一个。",
        "我们用互信息来衡量。八个专家，最多是三比特；如果专家的选择和这样东西无关，就是零。",
    ]),
    ("layer1", [
        f"先看第一层。专家的选择和当前词元的互信息是{dec_zh(MI[(1, 'tok')], 1)}比特，而和语言只有{dec_zh(MI[(1, 'lang')], 2)}比特。",
        "也就是说，第一层的路由器主要看这个词元是谁，把不同的词元分给不同的专家，几乎不管它来自哪种语言。这时模型还没来得及从上下文判断语言。",
    ]),
    ("layer2", [
        f"第二层完全不同。和语言的互信息升到{dec_zh(MI[(2, 'lang')], 1)}比特，和当前词元降到{dec_zh(MI[(2, 'tok')], 1)}比特。看这张表，大多数语言主要交给一个专家，平均约{pct_zh(PUR2)}的词元去了它的主要专家。",
        f"再按位置看。序列的前两个位置，上下文太少，第二层还是主要看词元；到了第八个位置以后，和语言的互信息达到{dec_zh(MIP['lang'][2], 1)}比特。模型先读懂这是哪种语言，然后才把词元交给对应的专家。",
    ]),
    ("real", [
        "真实模型里的情况更复杂。对稀疏专家变换器的分析发现，编码器里的专家常常按词元的类型分工，比如标点、动词、人名，而不是按领域。",
        "对法国公司八乘七十亿模型的分析也发现，不同领域的文本在专家分配上没有明显差别，路由更多地跟着语法和词元本身走。我们的小任务里语言之间差别非常干净，所以才出现这么清楚的分工。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，在我们的实验里，浅层按词元分工，深层按语言分工。",
        "第二，深层的分工要等模型从上下文里读出语言之后才出现。",
        "第三，真实大模型中的专家多按词元和语法分工，很少是整齐的领域专家。下一期，我们看细粒度专家和共享专家。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "专家学到了什么", "分工与专业化", SERIES)


def sc_measure(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "专家的选择和什么有关？")
    for i, (t, col) in enumerate([("序列的语言（8 种）", AMBER), ("当前词元（32 个）", CYAN)]):
        x = 180 + i * 520
        d.rounded_rectangle([x, 220, x + 400, 300], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 200, 260), t, F_BODY, col)
        vk.arrow(d, (x + 200, 310), (640, 400), MUTED, 3)
    d.rounded_rectangle([490, 410, 790, 480], 14, outline=VIOLET, width=2, fill=(12, 19, 34))
    vk.text_c(d, (640, 445), "被选中的专家", F_BODY, VIOLET)
    if seg >= 1:
        vk.text_c(d, (640, 570), "互信息：0（无关）… 3 比特（完全决定，8 个专家）", F_BODY, GREEN)


def mi_bars(d, box, layer):
    tv.bar_chart(d, box, ["和语言", "和当前词元"], [MI[(layer, "lang")], MI[(layer, "tok")]], [AMBER, CYAN], fmt="{:.2f}", vmax=3.2)


def sc_layer1(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "第一层：按词元分工")
    mi_bars(d, (120, 220, 520, 560), 1)
    d.text((120, 180), "互信息（比特）", font=F_SMALL, fill=MUTED)
    tv.heat(d, (640, 190, 1200, 620), T1.T, vmin=0, vmax=1, font_size=14)
    d.text((640, 160), "专家（行）× 当前词元 0–31（列）：分到的比例", font=F_SMALL, fill=MUTED)


def sc_layer2(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "第二层：按语言分工")
    if seg == 0:
        mi_bars(d, (120, 220, 520, 560), 2)
        d.text((120, 180), "互信息（比特）", font=F_SMALL, fill=MUTED)
        tv.heat(d, (700, 200, 1160, 600), T2, vmin=0, vmax=1, labels_x=[f"E{i + 1}" for i in range(8)],
                labels_y=[f"语言{i + 1}" for i in range(8)], fmt="{:.0%}", font_size=14)
    else:
        xs = [0, 1, 2]
        tv.line_chart(d, (160, 220, 760, 560), xs, [("和语言", MIP["lang"], AMBER), ("和当前词元", MIP["tok"], CYAN)],
                      ylim=(0, 3), xticks=[(0, "0–1"), (1, "2–7"), (2, "8–31")], xlabel="序列中的位置", legend_xy=(820, 240))
        d.text((160, 185), "第二层的互信息（比特）", font=F_SMALL, fill=MUTED)
        vk.text_c(d, (960, 420), "先读懂是哪种语言，", F_SMALL, TEXT)
        vk.text_c(d, (960, 460), "再交给对应的专家", F_SMALL, TEXT)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "真实模型里的专家")
    d.rounded_rectangle([140, 190, 1140, 300], 14, outline=CYAN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (640, 225), "ST-MoE（2022）：编码器专家多按词元类型分工", F_BODY, CYAN)
    vk.text_c(d, (640, 268), "标点 · 动词 · 人名 · 数字……而不是按领域", F_SMALL, TEXT)
    if seg >= 1:
        d.rounded_rectangle([140, 340, 1140, 450], 14, outline=AMBER, width=2, fill=(12, 19, 34))
        vk.text_c(d, (640, 375), "Mixtral（2024）：不同领域的专家分配没有明显差别", F_BODY, AMBER)
        vk.text_c(d, (640, 418), "路由更多跟着语法与词元本身", F_SMALL, TEXT)
        vk.text_c(d, (640, 530), "我们的玩具：语言之间差别非常干净 → 分工才这么清楚", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("浅层", "按当前词元分工"), ("深层", "读出语言之后按语言分工"),
                      ("真实模型", "多按词元与语法，少有领域专家")], seg, "下一期：细粒度专家与共享专家")


RENDER = {"title": sc_title, "measure": sc_measure, "layer1": sc_layer1, "layer2": sc_layer2, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "moe-special")
