"""State Space Models 06 — 并行扫描：循环也能并行训练.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python ssm06_scan.py <out_dir>

Real computations (ssmx.scan_parallel / scan_sequential, numpy): the linear recurrence h_t = a_t h_{t-1} + b_t with
input-dependent a_t (uniform in [0.5, 1]) and b_t (normal), for T = 8 (worked example) and T = 1,024 / 65,536:
sequential loop (T steps) vs Hillis-Steele parallel prefix scan with the associative operator
(a1, b1) o (a2, b2) = (a1 a2, a2 b1 + b2) (log2 T steps); largest difference and number of steps.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import ssmx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "State Space Models"

RES = {}
for T in (8, 1024, 65536):
    r = np.random.default_rng(T)
    a, b = r.uniform(0.5, 1, T), r.normal(size=T)
    par, steps = ssmx.scan_parallel(a, b)
    RES[T] = dict(diff=float(np.abs(par - ssmx.scan_sequential(a, b)).max()), steps=steps)
print(RES)
assert RES[8]["steps"] == 3 and RES[1024]["steps"] == 10 and RES[65536]["steps"] == 16
assert max(v["diff"] for v in RES.values()) < 1e-10

SCENES = [
    ("title", [
        "循环模型的一大问题是训练慢：每一步都要等上一步算完，没法在整个序列上并行。第三期的卷积技巧只适用于参数不随时间变化的模型。而上一期我们看到，选择性的模型需要参数随输入变化，卷积就用不了了。这一期讲另一个工具：并行扫描。",
    ]),
    ("assoc", [
        "状态空间模型的递推是线性的：新状态等于一个系数乘以旧状态，再加上一项。即使系数每一步都不同，我们也可以把相邻的两步合并成一步：两个系数相乘，加的那一项也相应组合。",
        "关键在于，这个合并操作满足结合律：先合并前两步再合并第三步，和先合并后两步再合并第一步，结果一样。有了结合律，就可以像加法求和一样，用树的结构并行计算。",
    ]),
    ("tree", [
        f"以八步为例。第一轮，每一步和它左边相邻的一步合并；第二轮，和距离为二的合并；第三轮，和距离为四的合并。三轮之后，每个位置都得到了从开头到它为止的完整结果。八步只需要三轮。",
        f"一般地，长度为二的多少次方，就需要多少轮。一千零二十四步只需要{int_zh(RES[1024]['steps'])}轮，六万五千多步只需要{int_zh(RES[65536]['steps'])}轮。我们验证，并行扫描和逐步计算的结果完全一致，差别只是浮点舍入。",
    ]),
    ("hardware", [
        "并行扫描的总计算量比逐步计算多一些，但轮数少得多，能充分利用显卡的并行能力。曼巴模型还专门为显卡写了融合的扫描内核：把状态留在片上的快速存储里，不写回显存，这和快速注意力的思路一样。",
        "在我们的代码里，训练时用的也是这样的结合律扫描，所以即使参数随输入变化，也能在整段序列上并行训练。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，线性递推的合并操作满足结合律。",
        "第二，有了结合律，就能用并行扫描在对数轮数内算完整个序列，结果与逐步计算一致。",
        "第三，这让参数随输入变化的循环模型也能并行训练。下一期，我们看固定大小的状态能记住多少。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "并行扫描", "循环也能并行训练", SERIES)


def sc_assoc(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "线性递推的合并")
    tv.fpaste(c, r"$h_t=a_t\,h_{t-1}+b_t$", 640, 200, 42)
    tv.fpaste(c, r"$(a_1,b_1)\circ(a_2,b_2)=(a_1a_2,\ a_2b_1+b_2)$", 640, 300, 38)
    if seg >= 1:
        tv.fpaste(c, r"$\left[(x\circ y)\circ z\right]=\left[x\circ(y\circ z)\right]$", 640, 420, 38)
        vk.text_c(d, (640, 530), "结合律 → 可以像求和一样用树并行", F_BODY, GREEN)


def sc_tree(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "Hillis–Steele 扫描：8 步只要 3 轮")
    x0, y0, w, h = 160, 190, 110, 90
    for r in range(4):
        for i in range(8):
            x, y = x0 + i * w, y0 + r * h
            lo = max(0, i - (2 ** r - 1)) if r > 0 else i
            col = [CYAN, VIOLET, AMBER, GREEN][r]
            d.rounded_rectangle([x, y, x + 80, y + 50], 8, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 40, y + 25), f"{lo + 1}–{i + 1}" if lo != i else f"{i + 1}", vk.font(18, mono=True), col)
            if r < 3 and i + 2 ** r < 8:
                d.line([(x + 40, y + 50), (x + 40 + 2 ** r * w, y + h)], fill=(71, 85, 105), width=1)
        d.text((x0 - 20, y0 + r * h + 25), f"{r}", font=F_SMALL, fill=MUTED, anchor="rm")
    if seg >= 1:
        f = vk.font(22, mono=True)
        for i, T in enumerate((1024, 65536)):
            d.text((160 + i * 480, 580), f"T = {T:,}：{RES[T]['steps']} 轮，差别 {RES[T]['diff']:.0e}", font=F_SMALL, fill=GREEN)


def sc_hardware(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "在硬件上")
    for i, (t, col) in enumerate([("总计算略多，轮数少得多：充分并行", CYAN), ("Mamba：融合的扫描内核，状态留在片上存储", GREEN), ("与 FlashAttention 同样的思路", AMBER)]):
        vk.text_c(d, (640, 230 + i * 80), t, F_BODY, col)
    if seg >= 1:
        vk.text_c(d, (640, 520), "本系列的模型：jax.lax.associative_scan", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 4, [("结合律", "线性递推可以合并"), ("并行扫描", "log T 轮，结果一致"),
                      ("意义", "随输入变化的循环也能并行训练")], seg, "下一期：回忆与复制")


RENDER = {"title": sc_title, "assoc": sc_assoc, "tree": sc_tree, "hardware": sc_hardware, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "ssm-scan")
