"""Long Context 07 — 迷失在中间：长上下文真的用好了吗.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] LC_CACHE=<dir> python lc07_middle.py <out_dir>

Real computations (lcx.exp_needle): retrieval accuracy of the RoPE / ALiBi / no-position models (trained on length
128 with the needle at a uniformly random depth) as a function of the needle's depth, at length 128 and 512.
The U-shaped curve of real models is quoted from Liu et al. (see notes); our toy does not reproduce it.
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

ND = lcx.cached(CACHE, "needle", lcx.exp_needle)
RP = ND["rope"][512]
print(ND)
assert round(RP[0], 2) == 0.91 and round(RP[1], 2) == 0.73 and round(RP[2], 2) == 0.89 and RP[3] == RP[4] == 1.0

SCENES = [
    ("title", [
        "模型能读进很长的文本，是不是就能用好其中的信息？二零二三年的一篇论文发现了一个有名的现象：迷失在中间。这一期看它是什么，以及我们的小实验看到了什么。",
    ]),
    ("paper", [
        "论文的做法是：给模型二十篇文档，其中只有一篇包含问题的答案，然后改变这篇文档的位置。结果，答案在开头或结尾时，模型答得最好；在中间时，准确率明显下降，有时甚至比完全不给文档还差。准确率随位置画出来，是一条两头高、中间低的曲线。",
        "这说明，宣称支持的上下文长度，和模型能有效利用的长度，是两回事。模型对开头和结尾更敏感，可能和训练数据里重要信息常出现的位置，以及注意力的特性有关。",
    ]),
    ("toy", [
        f"我们的小模型也做了类似的测试：在干扰词元中藏一个键值对，改变它的深度。在训练长度一百二十八上，无论放在哪里，三个模型都几乎全对，没有迷失在中间。",
        f"把长度拉到五百一十二，旋转位置编码的模型在离查询最近的两个深度全对，在四分之一深度处最差，只有{pct_zh(RP[1])}，最远的开头处是{pct_zh(RP[0])}。我们看到的是离得越远越难，并没有出现两头高、中间低的形状。",
        "这个差别并不奇怪：我们的训练数据里，键值对的位置是完全均匀的，而真实模型的训练数据和指令数据有自己的位置偏好。这也提醒我们，位置偏差很大程度上来自训练，而不只是架构。",
    ]),
    ("fixes", [
        "怎样缓解？可以在长上下文训练中专门构造把关键信息放在中间的数据；也可以在使用时把最重要的内容放在提示的开头或结尾；或者用检索先把相关的片段挑出来，再交给模型，而不是把所有东西都塞进上下文。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，真实模型在长上下文中，对开头和结尾的信息用得最好，中间最差。",
        "第二，我们的小模型没有出现两头高、中间低的形状，而是离得越远越难，位置偏差很大程度上来自训练数据。",
        "第三，宣称的上下文长度不等于有效长度。下一期，我们看怎样评估长上下文。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "迷失在中间", "长上下文真的用好了吗", SERIES)


def sc_paper(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "Liu et al.（2023）：答案文档放在不同位置")
    for i in range(20):
        col = AMBER if i == 9 else (51, 65, 85)
        d.rectangle([140 + i * 50, 220, 180 + i * 50, 280], fill=col)
    d.text((140 + 9 * 50, 295), "答案", font=F_SMALL, fill=AMBER)
    xs = np.linspace(0, 1, 50)
    u_ = 0.55 + 0.25 * (2 * xs - 1) ** 2
    tv.line_chart(d, (300, 360, 900, 620), xs, [("示意：U 形", u_, CYAN)], ylim=(0.4, 0.9), xticks=[(0, "0"), (1, "1")], xlabel="答案的位置（0 = 开头，1 = 结尾）")
    d.text((940, 470), "示意图，非实测", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.text((940, 380), "宣称长度 ≠ 有效长度", font=F_SMALL, fill=RED)


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "我们的检索实验（训练长度 128，均匀深度）")
    names = {"rope": "RoPE", "alibi": "ALiBi", "nope": "无位置编码"}
    cols = {"rope": CYAN, "alibi": GREEN, "nope": MUTED}
    T = 128 if seg == 0 else 512
    ser = [(names[k], [x * 100 for x in ND[k][T]], cols[k]) for k in ("rope", "alibi", "nope")]
    tv.line_chart(d, (140, 200, 800, 580), list(lcx.DEPTHS), ser, ylim=(60, 102), xticks=[(0, "0"), (0.5, "0.5"), (1, "1")], xlabel="深度（0 = 离查询最远）", legend_xy=(860, 220))
    d.text((140, 165), f"长度 {T} 上的正确率（%）", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        d.text((860, 380), "没有 U 形：离得越远越难", font=F_SMALL, fill=AMBER)
        d.text((860, 420), "训练数据：深度完全均匀", font=F_SMALL, fill=TEXT)


def sc_fixes(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "缓解方法")
    for i, (t, col) in enumerate([("训练：构造把关键信息放在中间的数据", CYAN), ("使用：重要内容放在开头或结尾", GREEN), ("检索：先挑出相关片段再交给模型", AMBER)]):
        vk.text_c(d, (640, 230 + i * 80), t, F_BODY, col)


def sc_summary(c, u, seg, su):
    tv.summary(c, 4, [("真实模型", "开头结尾好，中间差"), ("小实验", "没有 U 形，离得越远越难"),
                      ("提醒", "宣称长度 ≠ 有效长度")], seg, "下一期：评估")


RENDER = {"title": sc_title, "paper": sc_paper, "toy": sc_toy, "fixes": sc_fixes, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "lc-middle")
