"""Long Context 04 — ALiBi：用线性偏置代替位置编码.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] LC_CACHE=<dir> SHAKESPEARE=<file> python lc04_alibi.py <out_dir>

Real computations: (1) lcx.exp_extrap - loss of the ALiBi model vs RoPE beyond the training length (episode 03);
(2) the ALiBi penalties: head slopes 1/2, 1/4, 1/8, 1/16, penalty slope x distance at distances 8 / 64 / 500;
(3) lcx.exp_needle - synthetic retrieval: a (marker, key, value) triple hidden among random noise tokens, query at
the end; models trained on length 128 with the needle at a uniformly random depth (1,500 steps), tested on 256
sequences of length 128 and 512 with the needle at depth 0 / 0.25 / 0.5 / 0.75 / 1 (0 = farthest from the query).
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import lcx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Long Context"
CACHE = os.environ.get("LC_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "lccache"))

EX = lcx.cached(CACHE, "extrap", lcx.exp_extrap)
ND = lcx.cached(CACHE, "needle", lcx.exp_needle)
FAR = {k: float(EX[k]["curve"][128:511].mean()) for k in ("rope", "alibi")}
SLOPES = [2.0 ** -(i + 1) for i in range(4)]
PEN = {dist: [s * dist for s in SLOPES] for dist in (8, 64, 500)}
print(FAR, ND, PEN)
assert round(FAR["alibi"], 2) == 1.70 and round(FAR["rope"], 1) == 3.6
assert ND["alibi"][512][0] < 0.75 and ND["alibi"][512][3] == 1.0 and min(ND["nope"][512][:4]) == 1.0 and round(ND["rope"][512][1], 2) == 0.73
assert all(x > 0.98 for k in ND for x in ND[k][128])

SCENES = [
    ("title", [
        "第三期的实验里，线性偏置是唯一能外推到八倍训练长度的方法。这一期看它是怎么做到的，以及我们在一个检索任务上发现的代价。",
    ]),
    ("idea", [
        "线性偏置不往词元的表示里加任何位置信息，而是直接在注意力分数上减去一个惩罚，惩罚和距离成正比：离得越远，分数扣得越多。不同的头用不同的斜率，有的头只看很近的词，有的头能看远一些。",
        f"在我们的模型里，四个头的斜率是二分之一、四分之一、八分之一和十六分之一。距离五百时，即使最平缓的头，分数也要扣掉约{int_zh(round(PEN[500][3]))}，相当于几乎看不到了。",
    ]),
    ("why", [
        "它为什么能外推？因为在训练长度以外，模型遇到的情况和训练时几乎一样：远处的词元本来就被压得几乎看不见，长出来的部分只是更多看不见的词元。模型实际上一直在用一个有效长度有限的局部窗口。",
        f"这也是它在语言建模上外推得很好的原因：下一个字符主要取决于附近的字符。我们的模型到第五百个位置损失仍是{dec_zh(FAR['alibi'], 2)}。",
    ]),
    ("needle", [
        "但如果任务需要远处的信息呢？我们设计一个检索任务：在一长串随机的干扰词元里藏一个键值对，最后给出键，要求模型说出值。三种模型都在长度一百二十八上训练，在长度五百一十二上测试，并把键值对放在不同的深度。",
        f"在训练长度上，三个模型几乎都是百分之百。到了长度五百一十二，线性偏置在键值对离得最远时只答对{pct_zh(ND['alibi'][512][0])}；旋转位置编码在四分之一深度处只有{pct_zh(ND['rope'][512][1])}。而完全不用位置编码的模型，在前四个深度都答对了百分之百。",
    ]),
    ("lesson", [
        "所以线性偏置的外推，靠的是少看远处。在需要精确检索远处信息的任务上，这个偏置反而成了障碍。困惑度能外推，不等于模型真的会用长上下文。这也是为什么今天的大模型很少直接使用线性偏置，而是用旋转位置编码加上下文扩展的方法。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，线性偏置按距离线性地降低注意力分数，不同的头用不同的斜率。",
        "第二，它能外推，是因为远处的词元本来就被压得几乎看不见。",
        "第三，在我们的检索实验里，它在远处答错得最多。下一期，我们看怎样扩展旋转位置编码。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "ALiBi", "用线性偏置代替位置编码", SERIES)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "注意力分数减去 斜率 × 距离")
    tv.fpaste(c, r"$s_{ij}=\frac{q_i\cdot k_j}{\sqrt{d}}-m_h\,(i-j)$", 640, 200, 42)
    cols = [RED, AMBER, CYAN, GREEN]
    xs = np.arange(0, 64)
    tv.line_chart(d, (140, 290, 800, 600), xs, [(f"斜率 1/{2 ** (i + 1)}", -SLOPES[i] * xs, cols[i]) for i in range(4)], ylim=(-32, 0),
                  xticks=[(0, "0"), (32, "32"), (63, "63")], xlabel="距离", legend_xy=(860, 310))
    d.text((140, 262), "加到分数上的偏置", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.text((860, 470), f"距离 500：最平缓的头也要扣 {PEN[500][3]:.0f}", font=F_SMALL, fill=RED)


def sc_why(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "为什么能外推：远处本来就看不见")
    for i in range(24):
        a = np.exp(-i / 6)
        col = tuple(int(30 + (c_ - 30) * a) for c_ in (52, 211, 153))
        d.rectangle([1050 - i * 38, 230, 1080 - i * 38, 290], fill=col)
    d.text((1080, 310), "当前词元", font=F_SMALL, fill=GREEN, anchor="rm")
    d.text((140, 310), "远处：权重≈0", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        tv.bar_chart(d, (300, 380, 800, 620), ["RoPE", "ALiBi"], [FAR["rope"], FAR["alibi"]], [CYAN, GREEN], fmt="{:.2f}", vmax=4.2)
        d.text((860, 480), "位置 128–510 的平均损失", font=F_SMALL, fill=MUTED)


def sc_needle(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "检索任务：在干扰词元中找键值对（训练 128，测试 512）")
    if seg == 0:
        for i in range(40):
            col = (51, 65, 85)
            if i in (9, 10, 11):
                col = AMBER
            if i >= 38:
                col = CYAN
            d.rectangle([140 + i * 25, 260, 160 + i * 25, 300], fill=col)
        d.text((140 + 9 * 25, 320), "键值对（深度可变）", font=F_SMALL, fill=AMBER)
        d.text((140 + 38 * 25, 320), "查询", font=F_SMALL, fill=CYAN)
        return
    names = {"rope": "RoPE", "alibi": "ALiBi", "nope": "无位置编码"}
    cols = {"rope": CYAN, "alibi": GREEN, "nope": MUTED}
    ser = [(names[k], [x * 100 for x in ND[k][512]], cols[k]) for k in ("rope", "alibi", "nope")]
    tv.line_chart(d, (140, 200, 800, 580), list(lcx.DEPTHS), ser, ylim=(60, 102), xticks=[(0, "0"), (0.5, "0.5"), (1, "1")], xlabel="键值对的深度（0 = 离查询最远）", legend_xy=(860, 220))
    d.text((140, 165), "长度 512 上的正确率（%）", font=F_SMALL, fill=MUTED)
    d.text((860, 360), "长度 128：三者都≈100%", font=F_SMALL, fill=TEXT)


def sc_lesson(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "困惑度能外推 ≠ 会用长上下文")
    for i, (t, col) in enumerate([("ALiBi 的外推：靠少看远处", AMBER), ("精确检索远处信息：偏置成了障碍", RED), ("今天的大模型：RoPE + 上下文扩展", CYAN)]):
        vk.text_c(d, (640, 230 + i * 80), t, F_BODY, col)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("ALiBi", "分数减去 斜率 × 距离"), ("外推", "远处本来就看不见"),
                      ("代价", "远处检索答错最多")], seg, "下一期：扩展 RoPE")


RENDER = {"title": sc_title, "idea": sc_idea, "why": sc_why, "needle": sc_needle, "lesson": sc_lesson, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "lc-alibi")
