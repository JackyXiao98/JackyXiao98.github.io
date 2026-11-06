"""Long Context 05 — 扩展 RoPE：位置插值、NTK 与 YaRN.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] LC_CACHE=<dir> SHAKESPEARE=<file> python lc05_extend.py <out_dir>

Real computations (lcx.exp_extend, JAX): the RoPE model of episode 03 (trained on 64 characters) evaluated on
windows of 256 characters: (a) unchanged; (b) position interpolation - positions divided by 4; (c) NTK-aware
scaling - base 10,000 x 4^(d/(d-2)) = 48,761 for head dim 16; each without further training; then (d) 200 steps of
fine-tuning at length 256 (lr 1e-3) with unchanged positions, and (e) the same fine-tuning with interpolation.
Mean loss over positions 0-63, 64-127 and 128-254.
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

R = lcx.cached(CACHE, "extend", lcx.exp_extend, cache_dir=CACHE)
KEYS = ["none", "pi", "ntk", "ft", "ft_pi"]
NAMES = {"none": "直接用", "pi": "位置插值（不训练）", "ntk": "NTK 缩放（不训练）", "ft": "微调 200 步", "ft_pi": "插值 + 微调 200 步"}
COLS = {"none": RED, "pi": VIOLET, "ntk": AMBER, "ft": GREEN, "ft_pi": CYAN}
SEG = {k: [float(R[k][a:b].mean()) for a, b in ((0, 64), (64, 128), (128, 255))] for k in KEYS}
print({k: [round(x, 3) for x in v] for k, v in SEG.items()}, R["ntk_base"])
assert [round(x, 2) for x in SEG["none"]] == [1.75, 2.16, 3.52] and round(SEG["pi"][0], 2) == 3.65
assert [round(x, 2) for x in SEG["ntk"]] == [1.8, 1.79, 2.39] and [round(x, 2) for x in SEG["ft"]] == [1.76, 1.73, 1.68]
assert np.allclose(SEG["ft_pi"], [1.865, 1.842, 1.763], atol=1e-3) and round(R["ntk_base"]) == 48761

SCENES = [
    ("title", [
        "上一期看到，旋转位置编码不能直接外推。但今天的大模型大多先在几千个词元上训练，再扩展到几十万。这一期看扩展旋转位置编码的几种方法：位置插值、神经正切核缩放和它们的改进。",
    ]),
    ("pi", [
        "位置插值的想法很直接：既然模型只见过零到六十四的位置，那就把长序列的位置压缩回这个范围。长度是四倍，就把每个位置除以四。这样所有的旋转角度都在训练见过的范围内，只是相邻位置之间的角度变小了。",
        f"但压缩以后，模型从没见过这么小的间隔。我们的实验里，不经过训练直接插值，连原本的前六十四个位置都坏了，损失升到{dec_zh(SEG['pi'][0], 2)}。位置插值必须配合微调。",
    ]),
    ("ntk", [
        "神经正切核缩放换了一个思路：不动快的维度，只放慢慢的维度。具体做法是把旋转的底数调大。这样相邻词元之间的细节保持不变，而远距离的维度被拉伸到能覆盖更长的范围。",
        f"在我们的实验里，不经过训练就用这种缩放，前一百二十八个位置都保持在一点八左右，再往后升到{dec_zh(SEG['ntk'][2], 2)}，比直接使用的{dec_zh(SEG['none'][2], 2)}好得多。",
    ]),
    ("finetune", [
        f"再看微调。在长度二百五十六上微调二百步，不做插值，最后一段的损失降到{dec_zh(SEG['ft'][2], 2)}；先插值再微调，是{dec_zh(SEG['ft_pi'][2], 2)}。在我们的小模型上，直接微调反而更好。",
        "这和论文的结论不同。位置插值的论文发现，在大模型上把上下文扩展几倍时，直接微调学得很慢，插值后只需要一千步左右。我们的模型很小、扩展的倍数也小，可能正是这个差别。所以这里的结论只适用于小实验。",
    ]),
    ("yarn", [
        "实际系统中常用的是二零二三年的叫纱线的方法：对不同频率的维度区别对待，快的维度不插值，慢的维度插值，中间平滑过渡，并调整注意力的温度。它只用少量长文本微调，就能把上下文扩展十几倍到几十倍，被很多开源模型采用。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，位置插值把长位置压回训练范围，但必须配合微调。",
        "第二，神经正切核缩放只拉伸慢的维度，在我们的实验里不训练也能部分扩展。",
        "第三，实际系统常用纱线方法，加少量长文本微调。下一期，我们看长上下文怎样训练。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "扩展 RoPE", "位置插值、NTK 与 YaRN", SERIES)


def bars(d, keys, title):
    labs = ["0–63", "64–127", "128–254"]
    x0 = 120
    for j, lab in enumerate(labs):
        d.text((x0 + 400 + j * 220, 175), lab, font=F_SMALL, fill=MUTED, anchor="mm")
    f = vk.font(24, mono=True)
    for i, k in enumerate(keys):
        y = 220 + i * 70
        d.text((x0, y + 15), NAMES[k], font=F_SMALL, fill=COLS[k])
        for j in range(3):
            val = SEG[k][j]
            col = GREEN if val < 1.9 else (AMBER if val < 2.6 else RED)
            d.rounded_rectangle([x0 + 330 + j * 220, y, x0 + 470 + j * 220, y + 50], 8, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x0 + 400 + j * 220, y + 25), f"{val:.2f}", f, col)
    d.text((x0, 640), title, font=F_SMALL, fill=MUTED)


def sc_pi(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "位置插值：位置 ÷ 4")
    tv.fpaste(c, r"$m\ \rightarrow\ m/4$", 640, 520, 40)
    bars(d, ["none"] + (["pi"] if seg >= 1 else []), "长度 256 上的平均损失（训练长度 64）")


def sc_ntk(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "NTK 缩放：调大底数，只放慢慢的维度")
    tv.fpaste(c, r"$b'=b\cdot s^{d/(d-2)}=48{,}761$", 640, 540, 34)
    bars(d, ["none", "pi", "ntk"] if seg >= 1 else ["none", "pi"], "长度 256 上的平均损失（不训练）")


def sc_finetune(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "在长度 256 上微调 200 步")
    bars(d, ["none", "ntk", "ft", "ft_pi"], "长度 256 上的平均损失")
    if seg >= 1:
        vk.text_c(d, (640, 560), "大模型上：插值后只需约 1,000 步（Chen et al., 2023）", F_SMALL, AMBER)


def sc_yarn(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "YaRN（Peng et al., 2023）")
    for i, (t, col) in enumerate([("快的维度不插值，慢的维度插值，中间平滑过渡", CYAN), ("调整注意力的温度", GREEN),
                                  ("少量长文本微调：上下文扩展十几到几十倍", AMBER)]):
        vk.text_c(d, (640, 230 + i * 80), t, F_BODY, col)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("位置插值", "压回训练范围，需要微调"), ("NTK", "只拉伸慢的维度"),
                      ("YaRN", "分频处理 + 少量微调")], seg, "下一期：长上下文训练")


RENDER = {"title": sc_title, "pi": sc_pi, "ntk": sc_ntk, "finetune": sc_finetune, "yarn": sc_yarn, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "lc-extend")
