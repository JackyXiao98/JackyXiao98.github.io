"""Interpretability 04 — 注意力头与归纳头.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MI_CACHE=<dir> python mi04_induction.py <out_dir>

Real computations (mix.induction, JAX): attention-only Transformers (d = 64, 2 heads per layer, 32-token
vocabulary, length 48) trained 4,000 steps on sequences made of a random segment of random length 8-24
tiled to fill the sequence; loss on positions whose next token is a copy; on a random 24-token sequence
repeated twice: previous-token score of layer-0 heads (attention from t to t-1) and induction score
(attention from t to t-23, the token after the previous occurrence). 2-layer vs 1-layer model.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import mix
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Interpretability"
CACHE = os.environ.get("MI_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "micache"))

I2 = mix.cached(CACHE, "ind", mix.induction, layers=2)
I1 = mix.cached(CACHE, "ind", mix.induction, layers=1)
ST = np.array([h["step"] for h in I2["hist"]])
L2 = np.array([h["rep"] for h in I2["hist"]])
L1 = np.array([h["rep"] for h in I1["hist"]])
at = lambda L, s: float(L[list(ST).index(s)])
PREV = I2["prev"][0][1]
IND = I2["ind"][1]
print(at(L2, 200), at(L2, 300), L2[-1], L1[-1], PREV, IND, I1["ind"])
assert abs(np.log(32) - 3.466) < 1e-3 and abs(at(L2, 200) - 2.98) < 0.01 and abs(at(L2, 300) - 1.01) < 0.01
assert abs(L2[-1] - 0.18) < 0.005 and abs(L1[-1] - 2.47) < 0.005 and abs(PREV - 0.81) < 0.005
assert abs(IND[0] - 0.87) < 0.005 and abs(IND[1] - 0.89) < 0.005 and max(I1["ind"][0]) < 0.1 and max(I2["prev"][0][0], *I2["ind"][0]) < 0.09
SEQ = I2["seq_example"]

SCENES = [
    ("title", [
        "这一期看注意力头能实现什么样的算法，重点是一种在几乎所有语言模型里都会出现的电路，归纳头。",
    ]),
    ("idea", [
        "归纳头做的事情很简单。如果前面出现过，甲后面跟着乙，那么当甲再次出现时，就预测下一个是乙。比如前文出现过哈利波特，再看到哈利，就猜下一个词是波特。",
        "这是语言模型根据上下文学习的一个基础能力。前文里出现过的任何模式，哪怕训练时从没见过，都可以照着复制。",
    ]),
    ("two", [
        "一个注意力头做不到这件事，它需要两个头配合。第一层的头先看前一个位置，把前一个词元的信息写进当前位置，叫前一词元头。",
        "第二层的头再去寻找，哪个位置的前一个词元，和我现在的词元相同。找到以后，把那个位置的词元复制过来，这就是归纳头。",
    ]),
    ("train", [
        f"我们训练一个只有注意力层的小模型。数据是随机词元，其中一段随机长度的片段不断重复。重复部分只能靠往回找来预测，随便猜的损失是{dec_zh(np.log(32), 2)}。",
        f"两层模型在前两百步几乎没有进展，然后在短短一百步内，损失从{dec_zh(at(L2, 200), 2)}降到{dec_zh(at(L2, 300), 2)}，最后降到{dec_zh(L2[-1], 2)}。这种突然的下降，叫相变。",
        f"作为对照，只有一层的模型训练同样的步数，损失只降到{dec_zh(L1[-1], 2)}，始终学不会。",
    ]),
    ("patterns", [
        f"把一段随机序列重复两遍，看两层模型的注意力。第一层有一个头，平均有{pct_zh(PREV)}的注意力放在前一个位置上，正是前一词元头。",
        f"第二层的两个头，在第二遍里都盯着第一遍中，当前词元下一个位置的词元，平均注意力分别是{pct_zh(IND[0])}和{pct_zh(IND[1])}。注意力图上的这条斜线，就是归纳头的指纹。",
    ]),
    ("real", [
        "二零二二年的一项研究发现，在各种大小的语言模型里，归纳头都会在训练的某个时刻集中形成，同时模型利用上下文的能力明显提升，损失曲线上也出现一个小小的凸起。",
        "这是第一次在真实的语言模型里，把一个具体的电路和一种能力对应起来。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，归纳头实现了，见过甲后面是乙，再见到甲就预测乙。",
        "第二，它需要前一词元头和归纳头两层配合，单层模型学不会。",
        "第三，归纳头往往在训练中突然形成，伴随损失的相变。下一期，我们看为什么单个神经元很难解释，叠加现象。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "注意力头与归纳头", "见过 A B，再见 A 就预测 B", SERIES)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "归纳头：[A][B] … [A] → [B]")
    words = ["…", "哈利", "波特", "说", "…", "…", "哈利", "?"]
    cols = [MUTED, AMBER, GREEN, MUTED, MUTED, MUTED, AMBER, GREEN]
    for i, (w, col) in enumerate(zip(words, cols)):
        tv.token(d, 150 + i * 125, 200, w, col, w=105, font=F_SMALL)
    vk.arrow(d, (962, 260), (402, 260), AMBER, 2)
    vk.text_c(d, (680, 290), "找到上次出现的“哈利” → 复制它后面的“波特”", F_SMALL, AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 420), "根据上下文学习：任何前文出现过的模式都能照着复制", F_BODY, TEXT)


def sc_two(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "两个头配合")
    toks = ["A", "B", "…", "A", "?"]
    for i, t in enumerate(toks):
        tv.token(d, 260 + i * 160, 470, t, CYAN, w=110)
    d.rounded_rectangle([140, 160, 600, 260], 14, outline=VIOLET, width=2, fill=(12, 19, 34))
    vk.text_c(d, (370, 195), "第 1 层：前一词元头", F_BODY, VIOLET)
    vk.text_c(d, (370, 235), "在 B 处写入“我前面是 A”", F_SMALL, TEXT)
    vk.arrow(d, (440, 465), (345, 465), VIOLET, 3)
    if seg >= 1:
        d.rounded_rectangle([680, 160, 1140, 260], 14, outline=GREEN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (910, 195), "第 2 层：归纳头", F_BODY, GREEN)
        vk.text_c(d, (910, 235), "在 ? 处找“前面是 A”的位置 → 复制 B", F_SMALL, TEXT)
        d.line([(795, 520), (795, 560), (475, 560), (475, 520)], fill=GREEN, width=3)
        vk.text_c(d, (635, 590), "复制 B", F_SMALL, GREEN)


def sc_train(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "训练：只有注意力层的小模型")
    series = [("两层", L2, GREEN)] + ([("一层", L1, RED)] if seg >= 2 else [])
    n = len(ST) if seg >= 1 else list(ST).index(200) + 1
    tv.line_chart(d, (140, 160, 840, 560), ST, series, ylim=(0, 4.4), upto=n, xlabel="训练步数", ylabel="重复部分的损失",
                  legend_xy=(600, 200), xticks=[(s, str(s)) for s in (0, 1000, 2000, 3000, 4000)])
    y = 560 - np.log(32) / 4.4 * 400
    d.line([(140, y), (840, y)], fill=MUTED, width=1)
    d.text((845, y), f"瞎猜 ln 32 = {np.log(32):.2f}", font=F_SMALL, fill=MUTED, anchor="lm")
    if seg >= 1:
        d.text((880, 300), "相变", font=F_BODY, fill=AMBER)
        d.text((880, 345), f"第 200 步：{at(L2, 200):.2f}", font=F_SMALL, fill=TEXT)
        d.text((880, 380), f"第 300 步：{at(L2, 300):.2f}", font=F_SMALL, fill=TEXT)
        d.text((880, 415), f"第 4000 步：{L2[-1]:.2f}", font=F_SMALL, fill=GREEN)
    if seg >= 2:
        d.text((880, 470), f"一层模型：{L1[-1]:.2f}", font=F_SMALL, fill=RED)


def sc_patterns(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "注意力图：随机序列重复两遍（48 个词元）")
    A0 = I2["attn_example"][0][1]
    tv.heat(d, (120, 190, 500, 570), A0, 0, 1)
    d.text((120, 150), f"第 1 层 · 前一词元头（{PREV:.0%}）", font=F_SMALL, fill=VIOLET)
    if seg >= 1:
        A1 = I2["attn_example"][1][0]
        tv.heat(d, (700, 190, 1080, 570), A1, 0, 1, cmap="green")
        d.text((700, 150), f"第 2 层 · 归纳头（{IND[0]:.0%} / {IND[1]:.0%}）", font=F_SMALL, fill=GREEN)
        vk.text_c(d, (890, 620), "斜线 = 看向上一遍里“下一个”词元", F_SMALL, AMBER)
    vk.text_c(d, (310, 620), "行：当前位置 · 列：看向的位置", F_SMALL, MUTED)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "真实语言模型（Olsson et al., 2022）")
    for i, t in enumerate(["各种大小的模型里，归纳头在训练中某一时刻集中形成", "同时：利用上下文的能力明显提升 · 损失曲线出现小凸起"]):
        vk.text_c(d, (640, 200 + i * 60), t, F_BODY, TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 400), "第一次把一个具体电路与一种能力对应起来", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 6, [("归纳头", "[A][B] … [A] → [B]"), ("两层配合", "前一词元头 + 归纳头"),
                      ("相变", "训练中突然形成")], seg, "下一期：叠加")


RENDER = {"title": sc_title, "idea": sc_idea, "two": sc_two, "train": sc_train, "patterns": sc_patterns, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "mi-induction")
