"""Scaling Laws 06 — 超参数怎样随规模变化.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] SCL_CACHE=<dir> SHAKESPEARE=<file> python scl06_hparams.py <out_dir>

Real computations (sclx.exp_lr, JAX): widths 32 / 64 / 128 trained 1,500 steps with peak learning rates 1e-3 / 3e-3 /
1e-2 / 3e-2 (AdamW without weight decay, 100 warm-up steps, cosine), standard parameterisation; validation loss.
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

R = sclx.cached(CACHE, "lr", sclx.exp_lr)
DS, LRS = [32, 64, 128], [1e-3, 3e-3, 1e-2, 3e-2]
LV = {d: [R[(d, lr)]["val"] for lr in LRS] for d in DS}
BEST = {d: LRS[int(np.argmin(LV[d]))] for d in DS}
print(LV, BEST)
assert BEST == {32: 3e-2, 64: 1e-2, 128: 3e-3}
assert round(R[(128, 3e-2)]["val"], 2) == 2.45 and round(R[(32, 3e-2)]["val"], 2) == 1.87 and round(R[(128, 3e-3)]["val"], 2) == 1.69

SCENES = [
    ("title", [
        "训练大模型时，没法对每个规模都把学习率、批量这些超参数仔细调一遍，太贵了。所以一个关键问题是：超参数怎样随规模变化？这一期在小模型上看学习率。",
    ]),
    ("sweep", [
        "我们对宽度三十二、六十四和一百二十八的三个模型，各试四个学习率，从千分之一到百分之三，每个训练一千五百步。",
        f"最好的学习率随模型变大而变小：宽度三十二时是百分之三，六十四时是百分之一，一百二十八时是千分之三。宽度每翻一倍，最好的学习率大约降到原来的三分之一左右。",
    ]),
    ("danger", [
        f"这意味着，把小模型上调好的学习率直接搬到大模型上，可能会很糟。百分之三的学习率对宽度三十二的模型最好，损失{dec_zh(R[(32, 3e-2)]['val'], 2)}；但用在宽度一百二十八的模型上，损失是{dec_zh(R[(128, 3e-2)]['val'], 2)}，比它自己最好的{dec_zh(R[(128, 3e-3)]['val'], 2)}差了很多。",
    ]),
    ("fixes", [
        "有两种应对办法。一是像缩放定律一样，把最优学习率本身也拟合成规模的函数，再外推。深度求索的模型就为学习率和批量拟合了随计算量变化的幂律。",
        "二是改变参数化方式，让最优的超参数不随宽度变化，比如最大更新参数化。我们在优化系列里讲过它：在小模型上调好的学习率，可以直接用在宽得多的模型上。",
        "批量也是类似的。研究发现存在一个临界批量：小于它时，批量加倍，训练步数几乎减半；大于它时，收益很快消失。而且临界批量会随着损失降低而变大。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，在我们的实验里，最好的学习率随宽度增大而变小。",
        "第二，小模型上调好的超参数，不能直接搬到大模型上。",
        "第三，可以拟合超参数的缩放规律，或者用最大更新参数化让它不随宽度变化。下一期，我们看数据不够时怎么办。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "超参数怎样随规模变化", "学习率与批量", SERIES)


def sc_sweep(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "学习率扫描（1,500 步）")
    cols = {32: VIOLET, 64: CYAN, 128: GREEN}
    ser = [(f"宽度 {dd}", np.minimum(LV[dd], 2.5), cols[dd]) for dd in DS]
    tv.line_chart(d, (140, 200, 820, 580), LRS, ser, xlog=True, ylim=(1.6, 2.5), xticks=[(lr, f"{lr:g}") for lr in LRS], xlabel="峰值学习率（对数）", legend_xy=(880, 220))
    d.text((140, 165), "验证损失", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        for i, dd in enumerate(DS):
            d.text((880, 350 + i * 45), f"宽度 {dd}：最好 {BEST[dd]:g}", font=F_SMALL, fill=cols[dd])


def sc_danger(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "学习率 0.03：对小模型最好，对大模型很糟")
    tv.bar_chart(d, (200, 220, 860, 560), ["宽度 32 · 0.03", "宽度 128 · 0.03", "宽度 128 · 0.003"],
                 [R[(32, 3e-2)]["val"], R[(128, 3e-2)]["val"], R[(128, 3e-3)]["val"]], [GREEN, RED, GREEN], fmt="{:.2f}", vmax=2.7)
    d.text((200, 180), "验证损失", font=F_SMALL, fill=MUTED)


def sc_fixes(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "应对办法")
    rows = [("拟合超参数的缩放规律", "DeepSeek LLM（2024）：学习率与批量随计算量的幂律", CYAN)]
    if seg >= 1:
        rows.append(("改变参数化（μP）", "最优学习率不随宽度变化：小模型调好直接迁移", GREEN))
    if seg >= 2:
        rows.append(("临界批量", "McCandlish et al.（2018）：超过后收益消失，随训练变大", AMBER))
    for i, (a, b, col) in enumerate(rows):
        y = 190 + i * 110
        d.rounded_rectangle([110, y, 1170, y + 80], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((140, y + 40), a, font=F_BODY, fill=col, anchor="lm")
        d.text((520, y + 40), b, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_summary(c, u, seg, su):
    tv.summary(c, 4, [("学习率", "最优值随宽度变小"), ("迁移", "不能直接搬到大模型"),
                      ("办法", "拟合规律或 μP")], seg, "下一期：数据受限")


RENDER = {"title": sc_title, "sweep": sc_sweep, "danger": sc_danger, "fixes": sc_fixes, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "scl-hparams")
