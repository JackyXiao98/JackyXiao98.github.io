"""MoE 02 — 路由：每个词元去哪个专家.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MOE_CACHE=<dir> python moe02_routing.py <out_dir>

Real computations (moex.exp_experts / exp_scale, JAX): the toy of episode 01 with 2 / 4 / 8 / 16 experts of hidden
128 and top-1 routing (Switch auxiliary loss 0.01), and 8 experts with top-2 (gates renormalised over the two
selected experts); 2,000 steps each; held-out loss and per-token active parameters. E = 1 is the dense-128 model.
The router example (scores -> softmax -> top-k) uses illustrative numbers, labelled as such on screen.
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

SC = moex.cached(CACHE, "scale", moex.exp_scale)
EX = moex.cached(CACHE, "experts", moex.exp_experts)
LOSS = {1: SC["dense128"]["final"], 2: EX[(2, 1)]["final"], 4: EX[(4, 1)]["final"], 8: SC["moe8"]["final"], 16: EX[(16, 1)]["final"]}
PAR = {1: SC["dense128"]["params"], 2: EX[(2, 1)]["params"], 4: EX[(4, 1)]["params"], 8: SC["moe8"]["params"], 16: EX[(16, 1)]["params"]}
T2 = EX[(8, 2)]
EXAMPLE = np.array([0.3, -1.2, 2.1, 0.4, 1.6, -0.5, 0.0, -0.8])
PROB = np.exp(EXAMPLE) / np.exp(EXAMPLE).sum()
TOP2 = PROB[[2, 4]] / PROB[[2, 4]].sum()
print({k: round(v, 4) for k, v in LOSS.items()}, round(T2["final"], 4), T2["active"], np.round(PROB, 3), np.round(TOP2, 3))
assert [round(LOSS[e], 2) for e in (1, 2, 4, 8, 16)] == [1.79, 1.76, 1.71, 1.69, 1.69] and abs(LOSS[16] - LOSS[8]) < 0.001
assert round(T2["final"], 2) == 1.68 and T2["active"] == 105472 and SC["moe8"]["active"] == 72704 and PAR[16] == 565248
assert round(PROB[2], 2) == 0.45 and round(PROB[4], 2) == 0.27 and round(TOP2[0], 2) == 0.62

SCENES = [
    ("title", [
        "上一期我们看到，混合专家用更少的计算达到了更大的模型的效果。这一期我们打开路由器，看它是怎样决定每个词元去哪个专家的，以及专家数和每次选几个专家会有什么影响。",
    ]),
    ("router", [
        "路由器本身非常简单，就是一个线性层。它把词元的向量乘上一个矩阵，给每个专家一个分数，再用柔性最大值变成概率。",
        f"比如在这个例子里，八个专家的概率中，第三个最高，约{pct_zh(PROB[2])}，第五个第二，约{pct_zh(PROB[4])}。",
        f"如果每次选一个，就只交给第三个专家，并把它的输出乘上这个概率。如果每次选两个，就交给第三和第五个，两个概率重新归一化，权重分别约为{pct_zh(TOP2[0])}和{pct_zh(TOP2[1])}，再把两个专家的输出加权相加。",
    ]),
    ("grad", [
        "这里有一个巧妙的地方。选择是离散的，没法求导；但输出乘上了路由概率，所以概率可以得到梯度。如果某个专家处理得好，路由器就会学着把这类词元更多地交给它。",
    ]),
    ("experts", [
        "专家的数量重要吗？我们保持每个词元只用一个专家，计算量不变，把专家数从一增加到十六。",
        f"一个专家，也就是稠密模型，损失{dec_zh(LOSS[1], 2)}；两个专家{dec_zh(LOSS[2], 2)}；四个{dec_zh(LOSS[4], 2)}；八个{dec_zh(LOSS[8], 2)}。专家越多越好，而计算量几乎不变。",
        f"但到了十六个专家，损失还是{dec_zh(LOSS[16], 2)}，不再下降，而参数又多了将近一倍。我们的数据只有八种语言，八个专家已经足够分工。真实数据要复杂得多，所以大模型会用几十到几百个专家。",
    ]),
    ("topk", [
        f"每次选两个专家呢？八个专家选两个，损失是{dec_zh(T2['final'], 2)}，比只选一个稍好一点，但每个词元的计算也多了，激活参数从约七万三千增加到约十万五千。",
        "早期的稀疏专家工作认为至少要选两个，路由器才能学会比较；交换变换器发现只选一个也能训练得很好，而且最省计算。今天的大模型常常选八个，但每个专家更小，我们会在第六期讲。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，路由器是一个线性层加柔性最大值，选出得分最高的几个专家，输出按概率加权。",
        "第二，在计算量不变的情况下，增加专家数能降低损失，直到数据里没有更多可以分工的东西。",
        "第三，选更多专家略有提升，但计算也更多。下一期，我们看路由器不加约束时会出现的问题。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "路由", "每个词元去哪个专家", SERIES)


def sc_router(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "路由器：线性层 + softmax + top-k（示例数字）")
    tv.fpaste(c, r"$p=\mathrm{softmax}(W_r\,h)$", 640, 175, 38)
    labs = [f"E{i + 1}" for i in range(8)]
    cols = [GREEN if (i == 2 or (seg >= 2 and i == 4)) else MUTED for i in range(8)]
    tv.bar_chart(d, (120, 260, 700, 560), labs, PROB * 100, cols, fmt="{:.0f}%", vmax=50)
    if seg >= 2:
        d.text((780, 280), "选 1：只用 E3，输出 × 0.45", font=F_SMALL, fill=CYAN)
        d.text((780, 340), "选 2：E3、E5，重新归一化", font=F_SMALL, fill=GREEN)
        tv.fpaste(c, rf"$y={TOP2[0]:.2f}\,E_3(h)+{TOP2[1]:.2f}\,E_5(h)$", 980, 420, 30)


def sc_grad(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "路由器怎么学？")
    tv.fpaste(c, r"$y=\sum_{i\in \mathrm{top}k}\,p_i\,E_i(h)$", 640, 230, 46)
    vk.text_c(d, (640, 350), "选谁：离散，不可导", F_BODY, RED)
    vk.text_c(d, (640, 420), "乘上的概率 p：可导 → 路由器得到梯度", F_BODY, GREEN)


def sc_experts(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "专家数（每个词元都只用 1 个，计算量不变）")
    Es = [1, 2, 4, 8, 16] if seg >= 2 else [1, 2, 4, 8]
    cols = [MUTED, CYAN, CYAN, GREEN, AMBER]
    xs = np.log2(Es)
    tv.line_chart(d, (120, 200, 720, 560), list(range(len(Es))), [("损失", [LOSS[e] for e in Es], GREEN)],
                  ylim=(1.62, 1.82), xticks=[(i, str(e)) for i, e in enumerate(Es)], xlabel="专家数")
    d.text((120, 165), "验证损失", font=F_SMALL, fill=MUTED)
    f = vk.font(22, mono=True)
    d.text((800, 190), "专家数   损失     总参数", font=F_SMALL, fill=MUTED)
    for i, e in enumerate(Es):
        d.text((800, 235 + i * 45), f"{e:>3}", font=f, fill=cols[i])
        d.text((900, 235 + i * 45), f"{LOSS[e]:.3f}", font=f, fill=TEXT)
        d.text((1030, 235 + i * 45), f"{PAR[e] / 1e3:.0f}k", font=f, fill=TEXT)
    if seg >= 2:
        vk.text_c(d, (640, 640), "数据只有 8 种语言：8 个专家之后不再变好", F_BODY, AMBER)


def sc_topk(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "每次选几个？（8 个专家）")
    tv.bar_chart(d, (140, 220, 560, 560), ["选 1", "选 2"], [SC["moe8"]["final"], T2["final"]], [GREEN, CYAN], fmt="{:.3f}", vmax=1.8)
    d.text((140, 180), "验证损失", font=F_SMALL, fill=MUTED)
    tv.bar_chart(d, (700, 220, 1120, 560), ["选 1", "选 2"], [SC["moe8"]["active"] / 1e3, T2["active"] / 1e3], [GREEN, CYAN], fmt="{:.0f}k", vmax=130)
    d.text((700, 180), "每个词元激活的参数", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 640), "Shazeer 2017：k ≥ 2 · Switch 2021：k = 1 · 今天：更小的专家、k = 8", F_SMALL, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("路由器", "线性层 + softmax，选 top-k"), ("专家数", "越多越好，直到无可分工"),
                      ("选几个", "多选略好，计算也更多")], seg, "下一期：负载均衡")


RENDER = {"title": sc_title, "router": sc_router, "grad": sc_grad, "experts": sc_experts, "topk": sc_topk, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "moe-routing")
