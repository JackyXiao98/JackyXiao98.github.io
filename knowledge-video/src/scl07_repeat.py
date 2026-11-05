"""Scaling Laws 07 — 数据受限：重复数据还值多少.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] SCL_CACHE=<dir> SHAKESPEARE=<file> python scl07_repeat.py <out_dir>

Real computations (sclx.exp_repeat, JAX): the width-64 model trained 3,000 steps (6.1 M characters read) drawing
from a unique pool of 1.0 M / 512k / 256k / 128k characters, i.e. 6.1 / 12 / 24 / 48 passes over the pool; final
validation loss and training loss.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import sclx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Scaling Laws"
CACHE = os.environ.get("SCL_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "sclcache"))

R = sclx.cached(CACHE, "repeat", sclx.exp_repeat)
EP = sorted(R)
VAL = {e: R[e]["val"] for e in EP}
TRN = {e: R[e]["hist"][-1]["train"] for e in EP}
UNIQ = {e: 3000 * 32 * 64 / e for e in EP}
print(VAL, TRN, UNIQ)
assert [round(VAL[e], 2) for e in EP] == [1.70, 1.91, 2.02, 2.29] and [round(TRN[e], 2) for e in EP] == [1.52, 1.47, 1.38, 1.25]

SCENES = [
    ("title", [
        "缩放定律告诉我们，模型越大，需要的数据越多。但高质量的文本总量是有限的，有估计认为，按现在的速度，几年之内公开的人类文本就会被用完。数据不够时，把同样的数据多读几遍，还有用吗？",
    ]),
    ("setup", [
        "我们固定计算量：宽度六十四的模型，读六百一十万个字符。但改变其中不重复的部分有多少：全部约一百万个字符，也就是重复约六遍；或者只有一半、四分之一、八分之一，分别重复十二遍、二十四遍和四十八遍。",
    ]),
    ("result", [
        f"不重复的数据越少，验证损失越高：重复六遍时是{dec_zh(VAL[EP[0]], 2)}，十二遍是{dec_zh(VAL[EP[1]], 2)}，二十四遍是{dec_zh(VAL[EP[2]], 2)}，四十八遍是{dec_zh(VAL[EP[3]], 2)}。",
        f"而训练损失却反过来越来越低，四十八遍时只有{dec_zh(TRN[EP[3]], 2)}。模型在反复看到的数据上越来越熟练，但这些熟练越来越不能推广到新的文本上。同样的计算，花在重复的数据上，价值越来越低。",
    ]),
    ("real", [
        "二零二三年的一项研究系统地测量了这件事：在数据受限时，重复四遍以内，效果几乎和全新的数据一样好；再多重复，收益就迅速下降，重复几十遍之后，再多的计算也几乎没有用了。",
        "我们的小实验已经在重复六遍以上的区间，所以每多重复一倍都有明显的损失。应对数据不足的其他方向还有：加入代码等其他类型的数据，更仔细地过滤和去重，以及用模型生成的合成数据，但合成数据的质量和多样性又是新的挑战。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，同样的计算，不重复的数据越少，效果越差。",
        "第二，重复让训练损失越来越低，验证损失越来越高。",
        "第三，研究发现重复几遍以内影响不大，再多收益迅速下降。下一期，我们看涌现能力。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "数据受限", "重复数据还值多少", SERIES)


def sc_setup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "同样读 610 万个字符，不重复的部分有多少？")
    for i, e in enumerate(EP):
        y = 200 + i * 90
        w = UNIQ[e] / UNIQ[EP[0]] * 700
        d.rectangle([200, y, 200 + w, y + 50], fill=GREEN)
        d.text((210 + w, y + 10), f"{UNIQ[e] / 1e4:.0f} 万字符 × {e:g} 遍", font=F_SMALL, fill=TEXT)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "重复越多：训练越低，验证越高")
    xs = list(range(len(EP)))
    ser = [("验证", [VAL[e] for e in EP], RED)] + ([("训练", [TRN[e] for e in EP], CYAN)] if seg >= 1 else [])
    tv.line_chart(d, (140, 200, 820, 580), xs, ser, ylim=(1.1, 2.4), xticks=[(i, f"{e:g}") for i, e in enumerate(EP)], xlabel="重复遍数", legend_xy=(880, 220))
    d.text((140, 165), "损失", font=F_SMALL, fill=MUTED)
    f = vk.font(22, mono=True)
    for i, e in enumerate(EP):
        d.text((880, 330 + i * 40), f"{e:>4g} 遍  {VAL[e]:.3f}", font=F_SMALL, fill=TEXT)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "真实研究")
    vk.text_c(d, (640, 220), "Muennighoff et al.（NeurIPS 2023）：重复 ≤ 4 遍 ≈ 新数据；再多收益迅速下降", F_SMALL, CYAN)
    if seg >= 1:
        for i, (t, col) in enumerate([("加入代码等其他数据", GREEN), ("更仔细的过滤与去重", AMBER), ("合成数据：质量与多样性的挑战", VIOLET)]):
            vk.text_c(d, (640, 330 + i * 70), t, F_BODY, col)


def sc_summary(c, u, seg, su):
    tv.summary(c, 4, [("同样计算", "独特数据越少越差"), ("重复", "训练更低，验证更高"),
                      ("研究", "几遍以内影响不大")], seg, "下一期：涌现能力")


RENDER = {"title": sc_title, "setup": sc_setup, "result": sc_result, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "scl-repeat")
