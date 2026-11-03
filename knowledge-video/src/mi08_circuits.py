"""Interpretability 08 — 电路与归因图.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MI_CACHE=<dir> TF_CACHE=<dir> python mi08_circuits.py <out_dir>

Real computations on the tfm.py addition model:
- mix.ablation: mean-ablate each of the 8 attention heads and 2 MLPs; teacher-forced accuracy of the
  hundreds / tens / units digit on the 2,000 test problems.
- mix.patching (component level): copy one component's clean output (layer-0 components at the "units of
  b" position 4, layer-1 components at the answer position 6) into the corrupted run; same score as in
  episode 03. Average attention of position 4 (layer 0) and position 6 (layer 1).
Facts about real systems are quoted with references in the notes.
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

AB = mix.cached(CACHE, "abl", mix.ablation, tf_cache=TF_CACHE)
PA = mix.cached(CACHE, "patch", mix.patching, tf_cache=TF_CACHE)
CP = PA["comp"]
A4, A6 = PA["attn4"][0], PA["attn6"][1]
KEYS = [("head", 0, h) for h in range(4)] + [("mlp", 0)] + [("head", 1, h) for h in range(4)] + [("mlp", 1)]
NAME = lambda k: f"头 {k[1]}.{k[2]}" if k[0] == "head" else f"MLP {k[1]}"
TENS = {k: AB["res"][k][1] for k in KEYS}
print({NAME(k): (round(TENS[k], 3), round(CP[k], 2)) for k in KEYS}, A4[0][1], A4[1][1], A6[0][4], A6[2][4])
assert AB["base"] == [1.0, 1.0, 1.0] and min(TENS, key=TENS.get) == ("mlp", 0) and abs(TENS[("mlp", 0)] - 0.1855) < 1e-4
assert abs(TENS[("mlp", 1)] - 0.585) < 1e-4 and min(TENS[("head", 1, h)] for h in range(4)) > 0.83
assert abs(CP[("head", 0, 0)] - 0.62) < 0.005 and abs(CP[("mlp", 0)] - 0.68) < 0.005
assert abs(CP[("head", 1, 0)] - 0.38) < 0.005 and abs(CP[("head", 1, 2)] - 0.47) < 0.005 and abs(CP[("mlp", 1)] - 0.89) < 0.005
assert abs(A4[0][1] - 0.47) < 0.005 and abs(A4[1][1] - 0.44) < 0.005 and abs(A6[0][4] - 0.41) < 0.005 and abs(A6[2][4] - 0.26) < 0.005

SCENES = [
    ("title", [
        "第三期我们知道了进位的信息在哪里流动。这一期更进一步，找出是哪些具体的部件完成了这个计算，也就是电路。",
    ]),
    ("ablate", [
        "第一个工具是消融。把某个部件的输出，换成它在大量输入上的平均值，相当于把它关掉，看准确率下降多少。我们对八个注意力头和两个前馈层分别做了消融。",
        f"影响最大的是第一层的前馈层，关掉它，十位数字的准确率从百分之百降到{pct_zh(TENS[('mlp', 0)], 1)}。第二层的前馈层也很重要，降到{pct_zh(TENS[('mlp', 1)], 1)}。",
        f"而第二层的每个注意力头单独关掉，十位数字的准确率都还在百分之八十三以上。单个头的消融影响不大，可能是几个头在做相似的事，互为备份。",
    ]),
    ("patch", [
        "第二个工具是在部件的层面做修补，只把某一个部件的干净输出换进损坏的运行，看能恢复多少，配合注意力看它在读哪里。",
        f"结果拼出了一条完整的路径。第一步，在第二个数的个位这个位置，第一层的两个注意力头去看第一个数的个位，注意力分别是{pct_zh(A4[0][1])}和{pct_zh(A4[1][1])}，把两个个位数字放到一起。",
        f"第二步，第一层的前馈层根据这两个数字算出进位，修补它能恢复{pct_zh(CP[('mlp', 0)])}。",
        f"第三步，在答案的位置，第二层的头零和头二回头去看第二个数的个位，把进位搬过来，修补分别恢复{pct_zh(CP[('head', 1, 0)])}和{pct_zh(CP[('head', 1, 2)])}。最后，第二层的前馈层把进位和十位数字结合，写出答案，修补它恢复{pct_zh(CP[('mlp', 1)])}。",
    ]),
    ("caveat", [
        "注意，这些部件各自的恢复比例加起来并不等于一，因为部件之间有冗余，也有非线性的配合。电路是一个有用的简化，而不是模型的全部。",
    ]),
    ("graphs", [
        "在大模型上，逐个部件去修补太慢了。二零二五年，研究者训练一个更容易解释的替代模型，去模仿大模型每一层的计算，然后画出一次预测从输入到输出的归因图。",
        "他们在一个商用模型里看到了多步推理，比如先想到达拉斯所在的州是得克萨斯，再得出首府是奥斯汀；也看到了写诗时提前规划押韵的词。有意思的是，模型做加法时，也有一条估计大致大小的路径和一条计算个位数字的路径并行工作。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，消融把部件换成平均值，看它有多重要。",
        "第二，部件层面的修补和注意力，拼出了加法模型计算进位的完整电路。",
        "第三，大模型上可以用替代模型画出归因图，但电路始终是一种简化。最后一期，我们看可解释性能用来做什么。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "电路与归因图", "是哪些部件完成了计算？", SERIES)


def sc_ablate(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "消融：把部件换成平均输出")
    d.text((120, 150), "十位数字的准确率（关掉一个部件后）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vals = [TENS[k] * 100 for k in KEYS]
        cols = [CYAN] * 4 + [RED] + [VIOLET] * 4 + [AMBER]
        tv.bar_chart(d, (110, 220, 1170, 560), [NAME(k) for k in KEYS], vals, cols, fmt="{:.0f}%", vmax=115)
        d.text((110, 610), "第 1 层（层号 0）", font=F_SMALL, fill=CYAN)
        d.text((640, 610), "第 2 层（层号 1）", font=F_SMALL, fill=VIOLET)
    if seg >= 2:
        vk.text_c(d, (640, 660), "单个头影响小 → 几个头互为备份", F_SMALL, AMBER)


def _box(d, x, y, w, h, t1, t2, col):
    d.rounded_rectangle([x, y, x + w, y + h], 12, outline=col, width=2, fill=(12, 19, 34))
    vk.text_c(d, (x + w / 2, y + h / 2 - 15), t1, F_SMALL, col)
    vk.text_c(d, (x + w / 2, y + h / 2 + 17), t2, vk.font(18), TEXT)


def sc_patch(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "进位电路（部件修补 + 注意力）")
    toks = ["3", "7", "+", "4", "5", "=", "0"]
    for i, t in enumerate(toks):
        tv.token(d, 120 + i * 150, 600, t, AMBER if i in (1, 4, 6) else MUTED, w=70)
    xs = {i: 155 + i * 150 for i in range(7)}
    if seg >= 1:
        _box(d, xs[4] - 110, 440, 220, 80, "头 0.0 / 0.1", f"看“7”：{A4[0][1]:.0%} / {A4[1][1]:.0%}", CYAN)
        vk.arrow(d, (xs[1], 595), (xs[4] - 112, 485), CYAN, 3)
    if seg >= 2:
        _box(d, xs[4] - 110, 300, 220, 80, "MLP 0：算出进位", f"修补恢复 {CP[('mlp', 0)]:.0%}", RED)
        vk.arrow(d, (xs[4], 438), (xs[4], 384), MUTED, 3)
    if seg >= 3:
        _box(d, xs[6] - 150, 300, 220, 80, "头 1.0 / 1.2：搬运", f"恢复 {CP[('head', 1, 0)]:.0%} / {CP[('head', 1, 2)]:.0%}", VIOLET)
        vk.arrow(d, (xs[4] + 112, 340), (xs[6] - 152, 340), VIOLET, 3)
        _box(d, xs[6] - 150, 160, 220, 80, "MLP 1：写出十位", f"恢复 {CP[('mlp', 1)]:.0%}", AMBER)
        vk.arrow(d, (xs[6] - 40, 298), (xs[6] - 40, 244), MUTED, 3)
        vk.arrow(d, (xs[6], 595), (xs[6], 384), MUTED, 2)


def sc_caveat(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "电路是一种简化")
    vals = [CP[k] * 100 for k in KEYS]
    tv.bar_chart(d, (110, 220, 1170, 540), [NAME(k) for k in KEYS], vals, [CYAN] * 5 + [VIOLET] * 5, fmt="{:.0f}%", vmax=110)
    d.text((110, 160), "单个部件修补的恢复比例（加起来 ≠ 100%）", font=F_SMALL, fill=MUTED)
    vk.text_c(d, (640, 620), "冗余 + 非线性的配合", F_BODY, AMBER)


def sc_graphs(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "大模型的归因图（Anthropic, 2025）")
    vk.text_c(d, (640, 170), "训练可解释的替代模型（跨层转码器）模仿每层计算 → 画出归因图", F_SMALL, TEXT)
    if seg >= 1:
        chain = ["达拉斯", "得克萨斯", "奥斯汀"]
        for i, t in enumerate(chain):
            x = 260 + i * 300
            d.rounded_rectangle([x - 100, 250, x + 100, 320], 12, outline=CYAN, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x, 285), t, F_BODY, CYAN)
            if i < 2:
                vk.arrow(d, (x + 105, 285), (x + 195, 285), MUTED, 3)
        vk.text_c(d, (640, 360), "多步推理：先想到州，再得出首府", F_SMALL, MUTED)
        vk.text_c(d, (640, 430), "写诗：提前规划押韵的词", F_BODY, VIOLET)
        vk.text_c(d, (640, 500), "加法：估计大致大小 ∥ 计算个位数字，两条路径并行", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("消融", "换成平均值，看部件多重要"), ("电路", "头 0.0/0.1 → MLP 0 → 头 1.0/1.2 → MLP 1"),
                      ("归因图", "大模型上的替代模型 · 仍是简化")], seg, "最后一期：用可解释性做什么", label_w=430)


RENDER = {"title": sc_title, "ablate": sc_ablate, "patch": sc_patch, "caveat": sc_caveat, "graphs": sc_graphs, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "mi-circuits")
