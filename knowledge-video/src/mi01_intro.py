"""Interpretability 01 — 什么是机制可解释性：打开神经网络的黑箱.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MI_CACHE=<dir> TF_CACHE=<dir> python mi01_intro.py <out_dir>

Real computations (mix.probes): the tfm.py addition Transformer (2 layers, 4 heads, d = 64, 100% on 2,000
held-out problems). Logistic-regression probes read "carry" (units of a + units of b >= 10) from the
residual stream at every layer (0 = embeddings, 1, 2) and position; trained on the 8,000 training
problems, accuracy on the 2,000 test problems.
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
TF_CACHE = os.environ.get("TF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tfcache"))

PR = mix.cached(CACHE, "probe", mix.probes, tf_cache=TF_CACHE)
A = PR["trained"]
TOK = ["3", "7", "+", "4", "5", "=", "0", "8"]
print(A, PR["base_rate"])
assert abs(PR["base_rate"] - 0.544) < 1e-3 and abs(A[0, 1] - 0.763) < 1e-3 and abs(A[0, 4] - 0.752) < 1e-3
assert A[1, 4] == 1.0 and A[2, 6] == 1.0 and A[1, 6] < 0.75 and A[0, 5] == A[0, 0]

SCENES = [
    ("title", [
        "欢迎来到机制可解释性系列。我们已经知道怎样训练一个模型，这个系列要问的是，模型内部到底是怎样算出答案的。",
    ]),
    ("why", [
        "通常我们只看模型的输入和输出，比如测试准确率。但准确率只告诉我们模型做对了，不告诉我们它是怎么做对的，是真的学会了规律，还是抓住了某个巧合。",
        "机制可解释性想要像逆向工程一样，把训练好的网络拆开，找出它内部表示了哪些概念，这些概念叫特征；以及特征之间怎样一步步计算，这些计算路径叫电路。",
    ]),
    ("resid", [
        "这个系列的主要研究对象，是变换器系列里那个做两位数加法的小模型。它有两层，测试准确率百分之百。",
        "在变换器里，每个位置都有一个向量，叫残差流。每一层都从残差流里读取信息，算出结果，再加回残差流。所以残差流就像一块共享的黑板，我们可以在任何一层、任何位置去看上面写了什么。",
    ]),
    ("probe", [
        "最简单的看法是线性探针。比如我们想知道，模型有没有算出个位相加是否进位。三十七加四十五，个位七加五等于十二，有进位。",
        "我们在训练集上，用某一层、某个位置的向量，训练一个线性分类器去预测是否进位，再在测试集上检验。如果准确率很高，说明这个信息以线性的方式写在了那里。",
    ]),
    ("result", [
        f"结果画成一张表，行是层，列是位置。在最开始的嵌入层，只有两个个位数字本身的位置有一点信息，准确率约{pct_zh(A[0, 1])}，而全部猜多数类就有{pct_zh(PR['base_rate'])}。",
        "经过第一层之后，在第二个数的个位这个位置，探针的准确率达到了百分之百。也就是说，模型在读到第二个个位数字时，就已经把进位算好了。",
        "到了第二层，进位信息又出现在答案的位置上，正是模型要写出十位数字的地方。一张探针表，就勾勒出了信息在模型里流动的路线。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，机制可解释性研究模型内部的特征和电路，而不只是输入和输出。",
        "第二，残差流是各层共享的黑板，线性探针可以读出上面写的信息。",
        "第三，加法模型在读到第二个个位数字时就算好了进位。下一期，我们看探针的陷阱，读得出来不等于模型在用。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "什么是机制可解释性", "打开神经网络的黑箱", SERIES)


def sc_why(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "从“做对了”到“怎么做对的”")
    d.rounded_rectangle([120, 190, 340, 290], 14, outline=CYAN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (230, 240), "输入", F_BODY, CYAN)
    d.rounded_rectangle([480, 170, 800, 310], 14, outline=MUTED, width=2, fill=(30, 36, 48))
    vk.text_c(d, (640, 240), "？ 黑箱 ？", F_BODY, MUTED)
    d.rounded_rectangle([940, 190, 1160, 290], 14, outline=GREEN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (1050, 240), "输出 · 准确率", F_SMALL, GREEN)
    vk.arrow(d, (345, 240), (475, 240), MUTED, 3)
    vk.arrow(d, (805, 240), (935, 240), MUTED, 3)
    if seg >= 1:
        for i, (a, b, col) in enumerate([("特征", "模型内部表示的概念", AMBER), ("电路", "特征之间一步步的计算", VIOLET)]):
            y = 400 + i * 90
            d.rounded_rectangle([300, y, 980, y + 70], 12, outline=col, width=2, fill=(12, 19, 34))
            d.text((330, y + 35), a, font=F_BODY, fill=col, anchor="lm")
            d.text((460, y + 35), b, font=F_SMALL, fill=TEXT, anchor="lm")


def _stream(d, x0=170, y0=170, hi=None, toks=TOK):
    """Grid: rows = residual stream after layer 0/1/2, columns = positions."""
    for t, tok in enumerate(toks):
        x = x0 + t * 120
        vk.text_c(d, (x + 45, y0 + 3 * 95 + 30), tok, vk.font(30, mono=True), TEXT)
        for l in range(3):
            y = y0 + (2 - l) * 95
            col = AMBER if hi and (l, t) in hi else (51, 65, 85)
            d.rounded_rectangle([x, y, x + 90, y + 60], 10, outline=col, width=2, fill=(12, 19, 34))
    for l, lab in enumerate(["嵌入", "第 1 层后", "第 2 层后"]):
        d.text((x0 - 20, y0 + (2 - l) * 95 + 30), lab, font=F_SMALL, fill=MUTED, anchor="rm")


def sc_resid(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "研究对象：Transformer 系列的加法模型（2 层，100%）")
    _stream(d)
    if seg >= 1:
        for t in range(8):
            x = 170 + t * 120 + 45
            for l in range(2):
                y = 170 + (2 - l) * 95
                vk.arrow(d, (x, y - 2), (x, y - 33), CYAN, 2)
        vk.text_c(d, (640, 610), "残差流：每层读取 → 计算 → 加回（共享的黑板）", F_BODY, CYAN)


def sc_probe(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "线性探针：这里写着“进位”吗？")
    vk.text_c(d, (500, 170), "37 + 45 = 082", vk.font(32, mono=True), TEXT)
    vk.text_c(d, (860, 170), "个位 7 + 5 = 12 → 有进位", F_BODY, AMBER)
    if seg >= 1:
        d.rounded_rectangle([200, 280, 500, 380], 14, outline=CYAN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (350, 330), "某层某位置的向量", F_SMALL, CYAN)
        vk.arrow(d, (505, 330), (640, 330), MUTED, 3)
        d.rounded_rectangle([645, 280, 905, 380], 14, outline=VIOLET, width=2, fill=(12, 19, 34))
        vk.text_c(d, (775, 330), "线性分类器", F_SMALL, VIOLET)
        vk.arrow(d, (910, 330), (1000, 330), MUTED, 3)
        vk.text_c(d, (1070, 330), "进位？", F_BODY, AMBER)
        vk.text_c(d, (640, 470), "训练集上训练 · 测试集上检验（8,000 / 2,000 道题）", F_SMALL, MUTED)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "探针准确率：行 = 层，列 = 位置")
    rows = 1 if seg == 0 else 2 if seg == 1 else 3
    M = np.full((3, 8), np.nan)
    M[:rows] = A[:rows] * 100
    tv.heat(d, (240, 190, 1080, 520), M[::-1], 50, 100, labels_x=TOK, fmt="{:.0f}", font_size=22)
    for l, lab in enumerate(["第 2 层后", "第 1 层后", "嵌入"]):
        d.text((225, 190 + (l + .5) * 110), lab, font=F_SMALL, fill=MUTED, anchor="rm")
    d.text((240, 545), f"%；全部猜多数类 = {PR['base_rate']:.1%}", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 600), "读到第二个个位数字（5）时，进位已经算好", F_BODY, AMBER)
    if seg >= 2:
        vk.text_c(d, (640, 650), "第 2 层：进位出现在要写十位数字的位置", F_BODY, GREEN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("机制可解释性", "研究内部的特征与电路"), ("残差流 + 探针", "共享的黑板，线性读出"),
                      ("加法模型", "第二个个位数字处算好进位")], seg, "下一期：探针——读得出来 ≠ 在用")


RENDER = {"title": sc_title, "why": sc_why, "resid": sc_resid, "probe": sc_probe, "result": sc_result, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "mi-intro")
